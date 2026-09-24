#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerateLammpsDatafile_NoZPE.py
================================
Genere N data files LAMMPS pour des trajectoires d'incidence H2 sur surface,
SANS ZPE (pas de vibration interne). H2 est traite comme rotor rigide :
distance H-H = r_eq (0.7414 A par defaut, modifiable). Seule la vitesse de
translation du centre de masse est imposee a partir de (E, theta, phi).

Workflow analogue a celui de GenerateZPELammpsDatafile_ZPE.py :
   - on lit un LAMMPS_POSCAR construit en amont par l'utilisateur, qui contient
     la surface (W + eventuellement O) avec les IDs >= 3
   - on ajoute H1 (id 1) et H2 (id 2) au debut de la section Atoms
   - le header doit deja prevoir le total (N_surface + 2 atomes)

Pour chaque trajectoire :
   - position XY du CdM de H2 : aleatoire uniforme dans la cellule
   - hauteur Z du CdM         : z_top_W + height_offset (option --height)
                                 (base sur les atomes W type 2 uniquement, O exclu)
   - orientation H-H          : aleatoire isotrope (par defaut)
   - distance H-H             : r_eq (0.7414 A par defaut, option --hh-distance)
   - vitesse de translation   : v(E) avec
        vx =  v sin(theta) cos(phi)
        vy =  v sin(theta) sin(phi)
        vz = -v cos(theta)        (vers la surface)

Atomes fixes :
   - Detectes via le commentaire "# fixed" (ou "# frozen", "# F") en fin de
     ligne dans la section Atoms du LAMMPS_POSCAR.
   - Ou specifies via l'option --frozen-ids (ex: "35-42" ou "35:42" ou "35,36,37").
   - Un fichier groups.lmp est genere a cote, definissant les groupes
     'frozen' et 'mobile' (et 'H', 'W', 'O' par type) que in.simulation
     inclura via 'include groups.lmp'.

Convention angles : theta par rapport a la normale (axe Z), phi azimut dans XY.

Unites LAMMPS metal : A, eV, ps, amu, A/ps.

Usage typique :
   ./GenerateLammpsDatafile_NoZPE.py -p LAMMPS_POSCAR -n 1000 -E 0.1 -T 45 -P random
"""
import argparse
import math
import re
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# v[A/ps] = sqrt(2 E[eV] / m[amu]) * EV_AMU_TO_APS  (~ 98.227)
EV_AMU_TO_APS = math.sqrt(1.602176634e-19 / 1.66053906660e-27) * 1e-2

M_H = 1.00794   # amu (m_H2 = 2 * M_H)
R_HH_EQ = 0.7414  # A, distance d'equilibre H2


# ---------------------------------------------------------------------------
# Lecture LAMMPS data file
# ---------------------------------------------------------------------------
def read_lammps_data(path):
    """
    Lit un LAMMPS data file. Retourne un dict avec :
      - n_atoms_total : entier lu dans le header (peut > nb atomes listes
                        si on reserve des IDs pour H)
      - n_types       : nb de types
      - box           : (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)
      - masses        : {type: mass}
      - atoms         : liste de dicts {id, type, x, y, z, fixed (bool)}
    """
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    n_atoms_total = None
    n_types = None
    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0
    masses = {}
    atoms = []

    section = None
    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        # Headers de sections
        if s.startswith("Masses"):
            section = "masses";       continue
        if s.startswith("Atoms"):
            section = "atoms";        continue
        if s.startswith("Velocities"):
            section = "velocities";   continue
        if s.startswith("Bonds") or s.startswith("Angles"):
            section = "skip";         continue

        if section is None:
            # Header
            if "atoms" in s and "atom types" not in s:
                try:
                    n_atoms_total = int(s.split()[0])
                except ValueError:
                    pass
                continue
            if "atom types" in s:
                n_types = int(s.split()[0]);                          continue
            if "xlo" in s:
                p = s.split(); xlo, xhi = float(p[0]), float(p[1]);   continue
            if "ylo" in s:
                p = s.split(); ylo, yhi = float(p[0]), float(p[1]);   continue
            if "zlo" in s:
                p = s.split(); zlo, zhi = float(p[0]), float(p[1]);   continue
            if "xy xz yz" in s:
                p = s.split()
                xy, xz, yz = float(p[0]), float(p[1]), float(p[2]);   continue
            # Ligne de commentaire libre, on ignore
            continue

        # Sections data
        if section == "masses":
            data, _, _ = s.partition("#")
            p = data.split()
            if len(p) >= 2:
                masses[int(p[0])] = float(p[1])
            continue

        if section == "atoms":
            data, _, comment = raw.partition("#")
            p = data.split()
            if len(p) < 5:
                continue
            atom_id = int(p[0])
            atom_type = int(p[1])
            x, y, z = float(p[2]), float(p[3]), float(p[4])
            c = comment.strip().lower()
            is_fixed = bool(
                re.search(r"\b(fixed|frozen)\b", c)
                or c == "f"
                or c.startswith("f ")
                or c.endswith(" f")
            )
            atoms.append(
                {"id": atom_id, "type": atom_type,
                 "x": x, "y": y, "z": z, "fixed": is_fixed}
            )
            continue

        if section in ("velocities", "skip"):
            # On ignore : on reecrira les vitesses (H = v, autres = 0)
            continue

    if n_atoms_total is None or n_types is None or xlo is None:
        raise ValueError(f"Header LAMMPS data file incomplet : {path}")

    return {
        "n_atoms_total": n_atoms_total,
        "n_types": n_types,
        "box": (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz),
        "masses": masses,
        "atoms": atoms,
    }


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def velocity_magnitude(E_eV, m_amu):
    """Vitesse de translation en A/ps."""
    return math.sqrt(2.0 * E_eV / m_amu) * EV_AMU_TO_APS


def velocity_vector(E_eV, m_amu, theta_deg, phi_deg, rng):
    """v=(vx, vy, vz) en A/ps. phi peut etre 'random' ou un nombre."""
    v = velocity_magnitude(E_eV, m_amu)
    th = math.radians(float(theta_deg))
    if phi_deg == "random" or phi_deg is None:
        ph = rng.uniform(0.0, 2 * math.pi)
    else:
        ph = math.radians(float(phi_deg))
    return np.array([
         v * math.sin(th) * math.cos(ph),
         v * math.sin(th) * math.sin(ph),
        -v * math.cos(th),
    ])


def random_unit_vector(rng):
    """Vecteur unitaire isotrope sur la sphere."""
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2 * math.pi)
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * math.cos(phi), s * math.sin(phi), z])


def random_xy_in_cell(xy_vecs, rng):
    """Position XY uniforme dans le parallelogramme (a1_xy, a2_xy)."""
    u, w = rng.rand(), rng.rand()
    return u * xy_vecs[0] + w * xy_vecs[1]


def parse_ids(spec):
    """Parse '1,2,3,7-10,15' ou '35:42' en set d'IDs."""
    if not spec:
        return set()
    out = set()
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok or ":" in tok:
            sep = "-" if "-" in tok else ":"
            a, b = tok.split(sep)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def ids_to_ranges(ids):
    """Liste triee d'IDs -> liste de tuples (debut, fin) pour LAMMPS."""
    if not ids:
        return []
    ids = sorted(ids)
    out = []
    s = p = ids[0]
    for v in ids[1:]:
        if v == p + 1:
            p = v
        else:
            out.append((s, p))
            s = p = v
    out.append((s, p))
    return out


def ids_str(ids):
    return " ".join(f"{a}" if a == b else f"{a}:{b}" for a, b in ids_to_ranges(ids))


# ---------------------------------------------------------------------------
# Ecriture
# ---------------------------------------------------------------------------
def write_data_file(out_path, info, atoms_with_H, velocities, comment_header):
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = info["box"]
    with open(out_path, "w") as f:
        f.write(comment_header + "\n\n")
        f.write(f"{info['n_atoms_total']} atoms\n")
        f.write(f"{info['n_types']} atom types\n\n")
        f.write(f"{xlo} {xhi} xlo xhi\n")
        f.write(f"{ylo} {yhi} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi\n")
        if abs(xy) > 1e-12 or abs(xz) > 1e-12 or abs(yz) > 1e-12:
            f.write(f"{xy} {xz} {yz} xy xz yz\n")
        f.write("\nMasses\n\n")
        for t in sorted(info["masses"]):
            f.write(f"{t} {info['masses'][t]}\n")
        f.write("\nAtoms # atomic\n\n")
        for a in atoms_with_H:
            f.write(f"{a['id']} {a['type']} "
                    f"{a['x']:.10f} {a['y']:.10f} {a['z']:.10f}\n")
        f.write("\nVelocities\n\n")
        for a, v in zip(atoms_with_H, velocities):
            f.write(f"{a['id']} {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")


def write_groups_lmp(out_path, atoms_with_H, n_types):
    """Definitions de groupes : H / W / O (par type) + frozen / mobile."""
    # Convention par defaut : H=1 ; si n_types=2 -> W=2 ; si n_types>=3 -> W=2, O=3
    type_to_name = {1: "H"}
    if n_types == 2:
        type_to_name[2] = "W"
    else:
        type_to_name[2] = "W"
        type_to_name[3] = "O"
    with open(out_path, "w") as f:
        f.write("# Groupes generes par GenerateLammpsDatafile_NoZPE.py\n\n")
        for t in sorted({a["type"] for a in atoms_with_H}):
            name = type_to_name.get(t, f"type{t}")
            f.write(f"group {name} type {t}\n")
        f.write("\n")
        frozen_ids = sorted(a["id"] for a in atoms_with_H if a.get("fixed"))
        if frozen_ids:
            f.write(f"group frozen id {ids_str(frozen_ids)}\n")
        else:
            f.write("group frozen empty\n")
        f.write("group mobile subtract all frozen\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(info, out_dir, n_traj, E_eV, theta, phi, height_offset,
             prefix="data_POSCAR_", seed=None, hh_distance=R_HH_EQ,
             hh_orientation="random", extra_frozen=None):
    rng = np.random.RandomState(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if extra_frozen:
        eset = set(extra_frozen)
        for a in info["atoms"]:
            if a["id"] in eset:
                a["fixed"] = True

    surface_atoms = info["atoms"]

    # Verifier qu'il n'y a pas deja d'atomes avec id 1 ou 2 dans le data file
    used_ids = {a["id"] for a in surface_atoms}
    if 1 in used_ids or 2 in used_ids:
        raise ValueError(
            "Le LAMMPS_POSCAR contient deja des atomes avec id 1 ou 2. "
            "Reserve les IDs 1 et 2 pour H1 et H2 (commence la surface a id 3)."
        )

    # Coherence header : n_atoms_total = n_surface + 2
    n_expected = len(surface_atoms) + 2
    if info["n_atoms_total"] != n_expected:
        print(
            f"ATTENTION: header annonce {info['n_atoms_total']} atomes, "
            f"mais surface = {len(surface_atoms)} + 2 H = {n_expected}. "
            f"On reecrira le header a {n_expected}.",
            file=sys.stderr,
        )
        info["n_atoms_total"] = n_expected

    # Type 1 (H) doit avoir une masse
    if 1 not in info["masses"]:
        info["masses"][1] = M_H
        print(f"NOTE: type 1 (H) absent de Masses, ajout automatique m_H = {M_H}",
              file=sys.stderr)


    # Boite
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = info["box"]
    
    # Hauteur du W le plus haut (type 2 uniquement, O exclu)
    w_atoms = [a for a in surface_atoms if a["type"] == 2]
    if not w_atoms:
        raise ValueError("Aucun atome de type W (type 2) trouve dans le data file.")
    z_mid = 0.5 * (zhi - zlo)          # milieu de la boite
    w_slab = [a for a in w_atoms if a["z"] < z_mid]
    if not w_slab:
        raise ValueError("Aucun W sous z_mid : verifie l'orientation du slab / la boite.") 

    z_top_surface = max(a["z"] for a in w_slab)
    z_H2 = z_top_surface + float(height_offset)

    a1_xy = np.array([xhi - xlo, 0.0])
    a2_xy = np.array([xy, yhi - ylo])
    if zhi - z_H2 < 1.0:
        print(
            f"ATTENTION: H2 a z = {z_H2:.2f} A, proche du toit de la boite "
            f"(zhi = {zhi:.2f}). Augmente la hauteur de la box.",
            file=sys.stderr,
        )
    s3 = (z_H2 - zlo) / (zhi - zlo)
    origin_xy = np.array([xlo + xz * s3, ylo + yz * s3])  # ← correction
    m_H2 = 2.0 * M_H
    v_mag = velocity_magnitude(E_eV, m_H2)

    n_fixed = sum(1 for a in surface_atoms if a["fixed"])
    n_mobile = len(surface_atoms) - n_fixed

    for i in range(1, n_traj + 1):
        # Position CdM
        #xy_cm = random_xy_in_cell((a1_xy, a2_xy), rng) + np.array([xlo, ylo])
        xy_cm = random_xy_in_cell((a1_xy, a2_xy), rng) + origin_xy 
        cm = np.array([xy_cm[0], xy_cm[1], z_H2])

        # Orientation H-H
        if hh_orientation == "random":
            d = random_unit_vector(rng)
        else:
            d = np.asarray(hh_orientation, dtype=float)
            d = d / np.linalg.norm(d)
        pH1 = cm - 0.5 * hh_distance * d
        pH2 = cm + 0.5 * hh_distance * d

        # Vitesse (commune aux 2 atomes -> translation pure)
        v = velocity_vector(E_eV, m_H2, theta, phi, rng)

        H1 = {"id": 1, "type": 1, "x": pH1[0], "y": pH1[1], "z": pH1[2], "fixed": False}
        H2 = {"id": 2, "type": 1, "x": pH2[0], "y": pH2[1], "z": pH2[2], "fixed": False}
        all_atoms = [H1, H2] + surface_atoms

        velocities = [v, v] + [np.zeros(3)] * len(surface_atoms)

        header = (
            f"# data file no-ZPE, traj {i}/{n_traj}\n"
            f"# E = {E_eV} eV, theta = {theta} deg, phi = {phi}\n"
            f"# v_CdM = ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}) A/ps,  |v| = {v_mag:.6f} A/ps\n"
            f"# r_HH = {hh_distance} A, z_H2_CdM = {z_H2:.4f} A"
        )

        out_path = out_dir / f"{prefix}{i}.lammps"
        write_data_file(out_path, info, all_atoms, velocities, header)

    # groups.lmp commun (basé sur la surface + H1/H2 mobiles)
    template_atoms = [
        {"id": 1, "type": 1, "fixed": False},
        {"id": 2, "type": 1, "fixed": False},
    ] + surface_atoms
    write_groups_lmp(out_dir / "groups.lmp", template_atoms, info["n_types"])

    return {
        "n_traj": n_traj,
        "v_mag": v_mag,
        "z_top_surface": z_top_surface,
        "z_H2": z_H2,
        "n_fixed": n_fixed,
        "n_mobile": n_mobile,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-p", "--datafile", required=True,
                    help="LAMMPS_POSCAR (data file) de reference (surface seule, sans H)")
    ap.add_argument("-n", "--numgen", type=int, default=30000,
                    help="nombre de data files a generer [30000]")
    ap.add_argument("-E", "--energie", type=float, required=True,
                    help="energie cinetique de translation de H2 (eV)")
    ap.add_argument("-T", "--theta", default="0",
                    help="theta : angle par rapport a la normale (deg) [0]")
    ap.add_argument("-P", "--phi", default="random",
                    help='phi : azimut (deg) ou "random" [random]')
    ap.add_argument("-o", "--out-dir", default=".",
                    help="dossier de sortie [.]")
    ap.add_argument("--height", type=float, default=5.0,
                    help="hauteur du CdM de H2 au-dessus du plus haut W (A) [5.0]")
    ap.add_argument("--hh-distance", type=float, default=R_HH_EQ,
                    help=f"distance H-H (A) [{R_HH_EQ}]")
    ap.add_argument("--orientation", default="random",
                    help='orientation H-H : random|x|y|z|"vx,vy,vz" [random]')
    ap.add_argument("--frozen-ids", default="",
                    help='IDs supplementaires a figer, ex: "35-42" [vide]')
    ap.add_argument("--seed", type=int, default=None,
                    help="seed du RNG [aleatoire]")
    ap.add_argument("--prefix", default="data_POSCAR_",
                    help='prefixe des fichiers data ["data_POSCAR_"]')
    args = ap.parse_args()

    o = args.orientation.lower()
    if o == "random":
        orientation = "random"
    elif o in ("x", "y", "z"):
        orientation = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[o]
    elif "," in o:
        orientation = tuple(float(x) for x in o.split(","))
    else:
        ap.error(f"Orientation inconnue : {args.orientation}")

    info = read_lammps_data(args.datafile)
    extra_frozen = parse_ids(args.frozen_ids)

    res = generate(
        info=info,
        out_dir=args.out_dir,
        n_traj=args.numgen,
        E_eV=args.energie,
        theta=args.theta,
        phi=args.phi,
        height_offset=args.height,
        prefix=args.prefix,
        seed=args.seed,
        hh_distance=args.hh_distance,
        hh_orientation=orientation,
        extra_frozen=extra_frozen,
    )

    print(f"[OK] {res['n_traj']} data files generes dans {args.out_dir}/")
    print(f"     |v_CdM| = {res['v_mag']:.4f} A/ps   "
          f"(E = {args.energie} eV, m_H2 = {2*M_H:.5f} amu)")
    print(f"     theta = {args.theta} deg, phi = {args.phi}")
    print(f"     r_HH = {args.hh_distance} A, orientation = {args.orientation}")
    print(f"     z(top W)       = {res['z_top_surface']:.3f} A  ->  "
          f"z(H2 CdM) = {res['z_H2']:.3f} A")
    print(f"     atomes fixes  : {res['n_fixed']}")
    print(f"     atomes mobiles: {res['n_mobile']}")
    print(f"     fichier de groupes : {Path(args.out_dir) / 'groups.lmp'}")


if __name__ == "__main__":
    main()
