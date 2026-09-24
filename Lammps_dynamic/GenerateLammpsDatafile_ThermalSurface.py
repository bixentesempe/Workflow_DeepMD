#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerateLammpsDatafile_ThermalSurface.py
=========================================
Genere N data files LAMMPS pour des trajectoires d'incidence H2, en piochant
ALEATOIREMENT, pour CHAQUE trajectoire, une surface thermalisee differente
dans un pool de POSCAR VASP issus d'un AIMD (typiquement produits par
extract_md_configs.py) -- plutot qu'une unique surface statique, figee a
vitesse nulle, comme le font GenerateLammpsDatafile_NoZPE.py et
GenerateZPELammpsDatafile_ZPE.py.

Chaque POSCAR pioche fournit a la fois les POSITIONS et les VITESSES
reelles (thermiques) des atomes de surface mobiles -- le mouvement
vibrationnel du slab au moment de l'impact est donc physique, pas nul.

Reutilise integralement le reste du pipeline (placement/orientation/vitesse
de H2, modes NoZPE/ZPE, ecriture du data file et de groups.lmp) par import
direct de GenerateLammpsDatafile_NoZPE.py (et, en mode --zpe, de
GenerateZPELammpsDatafile_ZPE.py), pour rester au format et aux conventions
strictement identiques aux deux autres generateurs.

Convention attendue pour les POSCAR du pool (celle produite par
extract_md_configs.py) :
  - especes W (-> type LAMMPS 2) et O (-> type LAMMPS 3) uniquement, pas de H
    (H1/H2 sont ajoutes ici, IDs 1 et 2 reserves comme dans les autres
    generateurs)
  - bloc "Selective dynamics" : F F F -> atome fixe, sinon mobile
  - bloc de vitesses en Angstrom/fs (convention VASP) -> converti ici en
    Angstrom/ps (unites LAMMPS "metal")
  - tous les POSCAR du pool doivent partager EXACTEMENT le meme motif
    d'atomes fixes/mobiles (meme run source) ; verifie a l'execution.

Ne depend que de numpy (+ scipy en mode --zpe, via
GenerateZPELammpsDatafile_ZPE.py) : pas de pymatgen, pour rester dans
l'empreinte de dependances du reste de ce toolkit (cf. README).

Usage typique :
   ./GenerateLammpsDatafile_ThermalSurface.py \\
       --poscar-dir /path/to/extracted_configs/0.25ML300K/POSCAR \\
       -n 1000 -E 0.1 -T 45 -P random -o out/

   # mode ZPE :
   ./GenerateLammpsDatafile_ThermalSurface.py \\
       --poscar-dir /path/to/extracted_configs/0.25ML300K/POSCAR \\
       -n 1000 -E 0.1 --zpe -v 0 -j 0 -o out/
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Reutilise les briques communes (placement/vitesse du CdM H2, ecriture des
# data files et de groups.lmp) definies dans le generateur NoZPE, pour rester
# coherent avec les deux autres generateurs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import GenerateLammpsDatafile_NoZPE as nozpe

TYPE_OF_ELEMENT = {"H": 1, "W": 2, "O": 3}
ELEMENT_OF_TYPE = {v: k for k, v in TYPE_OF_ELEMENT.items()}
MASSES = {"H": 1.00794, "O": 15.9994, "W": 183.84}  # memes valeurs que BuildLAMMPSPoscar.py
AFS_TO_APS = 1000.0  # Angstrom/fs (VASP) -> Angstrom/ps (LAMMPS "metal")


# ---------------------------------------------------------------------------
# Lecture d'un POSCAR VASP thermalise
# ---------------------------------------------------------------------------
def read_poscar(path):
    """
    Parseur POSCAR minimal (format VASP5), sans dependance externe.
    Attend : ligne d'especes, "Selective dynamics" optionnel,
    "direct"/"cartesian", puis N lignes de positions (+ flags T/F), puis
    optionnellement une ligne vide et N lignes de vitesses (convention VASP,
    Angstrom/fs).
    """
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    symbols = [sp for sp, n in zip(species, counts) for _ in range(n)]
    natoms = len(symbols)

    idx = 7
    selective = lines[idx].strip().lower().startswith("s")
    if selective:
        idx += 1
    is_direct = lines[idx].strip().lower().startswith("d")
    idx += 1

    positions = np.zeros((natoms, 3))
    fixed = np.zeros(natoms, dtype=bool)
    for i in range(natoms):
        parts = lines[idx + i].split()
        positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if selective:
            fixed[i] = not any(p == "T" for p in parts[3:6])
    idx += natoms

    if is_direct:
        positions = positions @ lattice

    velocities = np.zeros((natoms, 3))
    rest = [ln for ln in lines[idx:] if ln.strip() != ""]
    if len(rest) >= natoms:
        for i in range(natoms):
            parts = rest[i].split()
            velocities[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return {
        "lattice": lattice,
        "symbols": symbols,
        "positions": positions,   # Cartesian, Angstrom
        "fixed": fixed,
        "velocities": velocities,  # Angstrom/fs
    }


def poscar_to_info(poscar_path, start_id=3):
    """Convertit un POSCAR thermalise en 'info' dict (meme structure que
    GenerateLammpsDatafile_NoZPE.read_lammps_data : box LAMMPS triclinique,
    Masses, liste d'atomes id/type/x/y/z/fixed) + vitesses (Angstrom/ps,
    dans le meme ordre que les atomes -- nulles pour les atomes fixes).
    """
    p = read_poscar(poscar_path)
    lat = p["lattice"]

    # Le format LAMMPS triclinique exige a1 selon x, a2 dans le plan xy.
    # Les POSCAR extraits de vasprun.xml respectent deja cette forme -- on
    # le revalide ici plutot que de le supposer silencieusement.
    if abs(lat[0, 1]) > 1e-6 or abs(lat[0, 2]) > 1e-6 or abs(lat[1, 2]) > 1e-6:
        raise ValueError(
            f"{poscar_path}: maille non compatible avec la convention "
            "triclinique LAMMPS (a1 doit etre selon x, a2 dans le plan xy, "
            "cf. https://docs.lammps.org/Howto_triclinic.html)."
        )

    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi, yhi, zhi = float(lat[0, 0]), float(lat[1, 1]), float(lat[2, 2])
    xy, xz, yz = float(lat[1, 0]), float(lat[2, 0]), float(lat[2, 1])

    atoms, velocities = [], []
    for i, sym in enumerate(p["symbols"]):
        if sym not in TYPE_OF_ELEMENT or sym == "H":
            raise ValueError(
                f"{poscar_path}: element de surface inattendu '{sym}' "
                "(seuls W et O sont attendus ; H est reserve au projectile)."
            )
        atoms.append({
            "id": start_id + i,
            "type": TYPE_OF_ELEMENT[sym],
            "x": float(p["positions"][i, 0]),
            "y": float(p["positions"][i, 1]),
            "z": float(p["positions"][i, 2]),
            "fixed": bool(p["fixed"][i]),
        })
        v = p["velocities"][i] if not p["fixed"][i] else np.zeros(3)
        velocities.append(v * AFS_TO_APS)

    types_present = sorted({a["type"] for a in atoms})
    masses = {t: MASSES[ELEMENT_OF_TYPE[t]] for t in types_present}

    info = {
        "n_atoms_total": len(atoms),
        "n_types": 3,  # H(1)/W(2)/O(3), meme si H est ajoute a part
        "box": (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz),
        "masses": masses,
        "atoms": atoms,
    }
    return info, np.array(velocities)


def find_poscar_pool(poscar_dir):
    pool = sorted(Path(poscar_dir).glob("POSCAR-*"))
    if not pool:
        raise FileNotFoundError(f"Aucun fichier 'POSCAR-*' trouve dans {poscar_dir}")
    return pool


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(poscar_pool, out_dir, n_traj, E_eV, theta, phi, height_offset,
             mode="nozpe", hh_distance=nozpe.R_HH_EQ, hh_orientation="random",
             v_level=0, jrot=0, init_cond_dir=None,
             prefix="data_POSCAR_", seed=None, extra_frozen=None):
    rng = np.random.RandomState(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    internal = None
    if mode == "zpe":
        import GenerateZPELammpsDatafile_ZPE as zpe
        D_eV, a_bohrinv, re_bohr, fit_rms = zpe.fit_morse(init_cond_dir / "morse_potasym.dat")
        vmax = zpe.morse_max_vib_level(D_eV, a_bohrinv)
        if v_level > vmax:
            raise ValueError(
                f"Niveau vibrationnel v={v_level} au-dessus du puits de Morse "
                f"(D={D_eV:.4f} eV, v_max={vmax})."
            )
        evirot_ha = zpe.morse_vibrational_energy_ha(D_eV, a_bohrinv, v_level)
        print(
            f"[Morse] D = {D_eV:.4f} eV, a = {a_bohrinv:.4f} /bohr, "
            f"re = {re_bohr * zpe.ANGUA:.4f} A  (fit rms = {fit_rms:.2e} eV, v_max = {vmax})"
        )
        print(f"[ZPE]   v = {v_level}, J = {jrot}  ->  E_vib = {evirot_ha * zpe.EVUA:.5f} eV")
        internal = zpe.run_internal_sampler(n_traj, evirot_ha, jrot, init_cond_dir, seed=seed)

    m_H2 = 2.0 * nozpe.M_H
    v_mag = nozpe.velocity_magnitude(E_eV, m_H2)

    reference_fixed_pattern = None
    info = surface_atoms = None  # derniere surface tiree, reutilisee pour groups.lmp

    for i in range(1, n_traj + 1):
        poscar_path = poscar_pool[rng.randint(len(poscar_pool))]
        info, surf_vel = poscar_to_info(poscar_path)

        if extra_frozen:
            eset = set(extra_frozen)
            for a in info["atoms"]:
                if a["id"] in eset:
                    a["fixed"] = True

        fixed_pattern = tuple(a["fixed"] for a in info["atoms"])
        if reference_fixed_pattern is None:
            reference_fixed_pattern = fixed_pattern
        elif fixed_pattern != reference_fixed_pattern:
            raise ValueError(
                f"{poscar_path}: motif d'atomes fixes different des autres POSCAR "
                "deja tires -- --poscar-dir doit contenir uniquement des "
                "snapshots d'UN seul run/surface (meme especes, meme ordre, "
                "meme couches figees)."
            )

        surface_atoms = info["atoms"]
        info["n_atoms_total"] = len(surface_atoms) + 2  # + H1 + H2
        if 1 not in info["masses"]:
            info["masses"][1] = nozpe.M_H

        xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = info["box"]
        w_atoms = [a for a in surface_atoms if a["type"] == 2]
        if not w_atoms:
            raise ValueError(f"{poscar_path}: aucun atome W (type 2) trouve.")
        z_mid = 0.5 * (zhi - zlo)
        w_slab = [a for a in w_atoms if a["z"] < z_mid]
        if not w_slab:
            raise ValueError(f"{poscar_path}: aucun W sous z_mid (verifie la boite).")

        z_top_surface = max(a["z"] for a in w_slab)
        z_H2 = z_top_surface + float(height_offset)

        a1_xy = np.array([xhi - xlo, 0.0])
        a2_xy = np.array([xy, yhi - ylo])
        if zhi - z_H2 < 1.0:
            print(
                f"ATTENTION: traj {i}: H2 a z = {z_H2:.2f} A, proche du toit de la "
                f"boite (zhi = {zhi:.2f}). Augmente --height ou le vide du POSCAR source.",
                file=sys.stderr,
            )
        s3 = (z_H2 - zlo) / (zhi - zlo)
        origin_xy = np.array([xlo + xz * s3, ylo + yz * s3])
        xy_cm = nozpe.random_xy_in_cell((a1_xy, a2_xy), rng) + origin_xy
        cm = np.array([xy_cm[0], xy_cm[1], z_H2])

        v_cm = nozpe.velocity_vector(E_eV, m_H2, theta, phi, rng)

        if mode == "zpe":
            za, xa, ya, zb, xb, yb, vza, vxa, vya, vzb, vxb, vyb = internal[i - 1]
            pH1 = cm + np.array([xa, ya, za])
            pH2 = cm + np.array([xb, yb, zb])
            vH1 = v_cm + np.array([vxa, vya, vza])
            vH2 = v_cm + np.array([vxb, vyb, vzb])
            r_hh = float(np.linalg.norm(pH2 - pH1))
        else:
            if hh_orientation == "random":
                d = nozpe.random_unit_vector(rng)
            else:
                d = np.asarray(hh_orientation, dtype=float)
                d = d / np.linalg.norm(d)
            pH1 = cm - 0.5 * hh_distance * d
            pH2 = cm + 0.5 * hh_distance * d
            vH1 = vH2 = v_cm
            r_hh = hh_distance

        H1 = {"id": 1, "type": 1, "x": pH1[0], "y": pH1[1], "z": pH1[2], "fixed": False}
        H2 = {"id": 2, "type": 1, "x": pH2[0], "y": pH2[1], "z": pH2[2], "fixed": False}
        all_atoms = [H1, H2] + surface_atoms
        velocities = [vH1, vH2] + list(surf_vel)

        header = (
            f"# data file surface thermalisee, traj {i}/{n_traj}\n"
            f"# surface = {poscar_path.name}\n"
            f"# E = {E_eV} eV, theta = {theta} deg, phi = {phi}"
            + (f", v = {v_level}, J = {jrot}" if mode == "zpe" else "") + "\n"
            f"# v_CdM = ({v_cm[0]:.6f}, {v_cm[1]:.6f}, {v_cm[2]:.6f}) A/ps,  |v| = {v_mag:.6f} A/ps\n"
            f"# r_HH = {r_hh:.4f} A, z_H2_CdM = {z_H2:.4f} A"
        )

        out_path = out_dir / f"{prefix}{i}.lammps"
        nozpe.write_data_file(out_path, info, all_atoms, velocities, header)

    template_atoms = [
        {"id": 1, "type": 1, "fixed": False},
        {"id": 2, "type": 1, "fixed": False},
    ] + surface_atoms
    nozpe.write_groups_lmp(out_dir / "groups.lmp", template_atoms, info["n_types"])

    n_fixed = sum(reference_fixed_pattern)
    n_mobile = len(reference_fixed_pattern) - n_fixed
    return {
        "n_traj": n_traj,
        "v_mag": v_mag,
        "n_fixed": n_fixed,
        "n_mobile": n_mobile,
        "n_pool": len(poscar_pool),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--poscar-dir", required=True,
                    help="dossier contenant les POSCAR-* thermalises (une seule "
                         "surface/run ; ex. extract_md_configs.py's POSCAR/)")
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

    # Mode NoZPE (par defaut)
    ap.add_argument("--hh-distance", type=float, default=nozpe.R_HH_EQ,
                    help=f"[NoZPE] distance H-H (A) [{nozpe.R_HH_EQ}]")
    ap.add_argument("--orientation", default="random",
                    help='[NoZPE] orientation H-H : random|x|y|z|"vx,vy,vz" [random]')

    # Mode ZPE
    ap.add_argument("--zpe", action="store_true",
                    help="active le mode ZPE (H2 = oscillateur de Morse) au lieu "
                         "du rotor rigide (NoZPE, par defaut)")
    ap.add_argument("-v", "--vib-level", type=int, default=0,
                    help="[ZPE] niveau vibrationnel v [0 = ZPE]")
    ap.add_argument("-j", "--jrot", type=int, default=0,
                    help="[ZPE] nombre quantique rotationnel J [0]")
    ap.add_argument("--init-cond-dir", default=None,
                    help="[ZPE] dossier exe/intrep.dat/morse_potasym.dat "
                         "[<dossier du script>/init_cond]")

    ap.add_argument("--frozen-ids", default="",
                    help='IDs supplementaires a figer (IDs LAMMPS, surface '
                         'commence a 3), ex: "35-42" [vide]')
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

    init_cond_dir = (
        Path(args.init_cond_dir) if args.init_cond_dir
        else Path(__file__).resolve().parent / "init_cond"
    )

    poscar_pool = find_poscar_pool(args.poscar_dir)
    extra_frozen = nozpe.parse_ids(args.frozen_ids)

    res = generate(
        poscar_pool=poscar_pool,
        out_dir=args.out_dir,
        n_traj=args.numgen,
        E_eV=args.energie,
        theta=args.theta,
        phi=args.phi,
        height_offset=args.height,
        mode="zpe" if args.zpe else "nozpe",
        hh_distance=args.hh_distance,
        hh_orientation=orientation,
        v_level=args.vib_level,
        jrot=args.jrot,
        init_cond_dir=init_cond_dir,
        prefix=args.prefix,
        seed=args.seed,
        extra_frozen=extra_frozen,
    )

    print(f"[OK] {res['n_traj']} data files generes dans {args.out_dir}/ "
          f"(mode {'ZPE' if args.zpe else 'NoZPE'})")
    print(f"     surfaces piochees dans un pool de {res['n_pool']} POSCAR thermalises")
    print(f"     |v_CdM| = {res['v_mag']:.4f} A/ps   (E = {args.energie} eV, m_H2 = {2*nozpe.M_H:.5f} amu)")
    print(f"     theta = {args.theta} deg, phi = {args.phi}")
    print(f"     atomes fixes (par surface)  : {res['n_fixed']}")
    print(f"     atomes mobiles (par surface): {res['n_mobile']}")
    print(f"     fichier de groupes : {Path(args.out_dir) / 'groups.lmp'}")


if __name__ == "__main__":
    main()
