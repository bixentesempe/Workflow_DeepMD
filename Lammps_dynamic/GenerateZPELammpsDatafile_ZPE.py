#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerateZPELammpsDatafile_ZPE.py
=============================
Genere N data files LAMMPS pour des trajectoires d'incidence H2 sur surface,
AVEC vibration/rotation interne de H2 echantillonnee dans le potentiel de
Morse universel (r_HH, isole de la surface) donne par init_cond/morse_potasym.dat.

Contrairement a l'ancienne version, ce script ne depend plus d'un
INIT-AIMD.res pre-genere (par systeme/energie/angle) : il appelle lui-meme,
a la volee, le programme Fortran init_cond/exe pour echantillonner l'etat
interne (v, J) demande - independamment de l'energie de translation - puis
combine cet etat avec la translation du centre de masse (E, theta, phi),
exactement comme GenerateLammpsDatafile_NoZPE.py le fait pour le mode NoZPE.

Le fichier init_cond/morse_potasym.dat est le seul fichier de potentiel
necessaire : ses parametres de Morse (D, a, re) sont retrouves par un fit
automatique (moindres carres), puis utilises pour calculer analytiquement
l'energie du niveau vibrationnel v demande (formule de Morse exacte).

Usage typique :
   ./GenerateZPELammpsDatafile_ZPE.py -p LAMMPS_POSCAR -n 1000 -E 0.1 -T 45 -P random
   ./GenerateZPELammpsDatafile_ZPE.py -p LAMMPS_POSCAR -n 1000 -E 0.1 -v 0 -j 0   # ZPE pure
"""
import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

# Reutilise les briques communes (lecture LAMMPS_POSCAR, placement/vitesse du
# CdM, ecriture des data files et de groups.lmp) definies dans le generateur
# NoZPE, pour rester coherent entre les deux modes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import GenerateLammpsDatafile_NoZPE as nozpe

# ---------------------------------------------------------------------------
# Constantes (memes conventions que init_cond/init_cond.f)
# ---------------------------------------------------------------------------
EVUA = 27.2            # 1 Hartree = 27.2 eV
ANGUA = 0.529177249    # 1 bohr = 0.529177249 A
PTMASS = 1836.0        # masse du proton (unites de masse electronique)


# ---------------------------------------------------------------------------
# Potentiel de Morse : fit automatique + energie du niveau vibrationnel v
# ---------------------------------------------------------------------------
def fit_morse(morse_file):
    """
    Lit un fichier au format potasym.dat (titre, N, puis N lignes 'r V')
    et ajuste V(r) = D*(1-exp(-a*(r-re)))**2 par moindres carres.

    ATTENTION: r est en BOHR dans potasym.dat/morse_potasym.dat (unite
    attendue par init_cond.f), D est en eV. a et re sont donc retournes en
    unites atomiques (1/bohr, bohr) - pas de conversion supplementaire a
    faire avant de les injecter dans les formules de init_cond.f.
    """
    if curve_fit is None:
        raise RuntimeError(
            "scipy est requis pour ajuster le potentiel de Morse "
            "(pip install scipy)."
        )
    with open(morse_file) as f:
        lines = f.readlines()
    n = int(lines[1])
    r, v = [], []
    for line in lines[2:2 + n]:
        a, b = line.split()
        r.append(float(a))
        v.append(float(b))
    r = np.array(r)
    v = np.array(v)

    def morse(r, D, a, re):
        return D * (1.0 - np.exp(-a * (r - re))) ** 2

    p0 = [max(v), 1.0, r[np.argmin(v)]]
    popt, _ = curve_fit(morse, r, v, p0=p0, maxfev=20000)
    D_eV, a_bohrinv, re_bohr = popt
    rms = float(np.sqrt(np.mean((v - morse(r, *popt)) ** 2)))
    return float(D_eV), float(a_bohrinv), float(re_bohr), rms


def morse_omega_ha(D_eV, a_bohrinv):
    """Frequence harmonique au fond du puits (Hartree), mu = 0.5*PTMASS (H2)."""
    D_ha = D_eV / EVUA
    mu_au = 0.5 * PTMASS
    return a_bohrinv * math.sqrt(2.0 * D_ha / mu_au)


def morse_vibrational_energy_ha(D_eV, a_bohrinv, v_level):
    """Energie exacte du niveau vibrationnel v d'un oscillateur de Morse (Ha)."""
    D_ha = D_eV / EVUA
    omega = morse_omega_ha(D_eV, a_bohrinv)
    n = v_level + 0.5
    return omega * n - (omega ** 2 / (4.0 * D_ha)) * n ** 2


def morse_max_vib_level(D_eV, a_bohrinv):
    """Plus grand v tel que E_v reste sous le sommet de la parabole (< D)."""
    D_ha = D_eV / EVUA
    omega = morse_omega_ha(D_eV, a_bohrinv)
    n_star = 2.0 * D_ha / omega  # sommet de E_v(n), n=v+1/2
    return int(math.floor(n_star - 0.5))


# ---------------------------------------------------------------------------
# Echantillonnage des etats internes (r_HH, orientation, moments) via le
# sampler Fortran init_cond/exe, translation nulle (le CdM reste en (0,0,0)).
# ---------------------------------------------------------------------------
def run_internal_sampler(n, evirot_ha, jrot, init_cond_dir, seed=None):
    exe = init_cond_dir / "exe"
    intrep = init_cond_dir / "intrep.dat"
    morse_file = init_cond_dir / "morse_potasym.dat"
    for p in (exe, intrep, morse_file):
        if not p.exists():
            raise FileNotFoundError(f"Fichier requis introuvable : {p}")

    idum = int(seed) if seed else random.randint(1, 999_999_999)
    if idum == 0:
        idum = 1
    evirot_str = f"{evirot_ha:.10E}".replace("E", "D")

    with tempfile.TemporaryDirectory(prefix="zpe_sampler_") as wd:
        wd = Path(wd)
        shutil.copy(exe, wd / "exe")
        os.chmod(wd / "exe", 0o755)
        shutil.copy(intrep, wd / "intrep.dat")
        shutil.copy(morse_file, wd / "potasym.dat")

        init_aimd = (
            "1        <-- NSTART\n"
            f"{n}       <-- Number of trajectories\n"
            "0.0D0    <-- Initial energy (eV) [translation geree en Python]\n"
            "3        <-- NVPARTIR (incidence theta/phi ; ici energie=0 -> sans effet)\n"
            "0.D0     <-- thetav\n"
            "0.D0     <-- parinput (phi)\n"
            "0.D0     <-- vy\n"
            f"{evirot_str} <-- evirot (Hartree), niveau vibrationnel demande\n"
            f"{jrot}        <-- jrot\n"
            "1        <-- NXYTIR (CdM fixe)\n"
            "0.D0     <-- xzer\n"
            "0.D0     <-- yzer\n"
            "2        <-- IFIFIX (orientation azimutale aleatoire)\n"
            "0.D0     <-- fifix\n"
            "6        <-- ndim\n"
            "1.D0     <-- ma (masse H, unite proton)\n"
            "1.D0     <-- mb (masse H, unite proton)\n"
            "0.D0     <-- zin (CdM fixe a l'origine)\n"
            "1.D-6    <-- epst\n"
            "7        <-- hin\n"
            "1.D-6    <-- precit\n"
            "1.D0     <-- xelem\n"
            "1.4142D0 <-- yelem\n"
            f"{idum}    <-- idum\n"
        )
        (wd / "init_aimd.dat").write_text(init_aimd)

        result = subprocess.run(
            ["./exe"], cwd=wd, capture_output=True, text=True
        )
        res_file = wd / "INIT-AIMD.res"
        if result.returncode != 0 or not res_file.exists():
            log = (result.stdout or "") + (result.stderr or "")
            raise RuntimeError(
                f"init_cond/exe a echoue (code {result.returncode}) :\n"
                f"{log[-2000:]}"
            )

        rows = []
        with open(res_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 13:
                    continue
                rows.append([float(x) for x in parts[1:13]])

    if len(rows) < n:
        raise RuntimeError(
            f"Seulement {len(rows)}/{n} etats internes generes par init_cond/exe."
        )
    return np.array(rows[:n])


# ---------------------------------------------------------------------------
# Generation des data files LAMMPS
# ---------------------------------------------------------------------------
def generate(info, out_dir, n_traj, E_eV, theta, phi, height_offset,
             v_level, jrot, init_cond_dir, prefix="data_POSCAR_",
             seed=None, extra_frozen=None):
    rng = np.random.RandomState(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if extra_frozen:
        eset = set(extra_frozen)
        for a in info["atoms"]:
            if a["id"] in eset:
                a["fixed"] = True

    surface_atoms = info["atoms"]

    used_ids = {a["id"] for a in surface_atoms}
    if 1 in used_ids or 2 in used_ids:
        raise ValueError(
            "Le LAMMPS_POSCAR contient deja des atomes avec id 1 ou 2. "
            "Reserve les IDs 1 et 2 pour H1 et H2 (commence la surface a id 3)."
        )

    n_expected = len(surface_atoms) + 2
    if info["n_atoms_total"] != n_expected:
        print(
            f"ATTENTION: header annonce {info['n_atoms_total']} atomes, "
            f"mais surface = {len(surface_atoms)} + 2 H = {n_expected}. "
            f"On reecrira le header a {n_expected}.",
            file=sys.stderr,
        )
        info["n_atoms_total"] = n_expected

    if 1 not in info["masses"]:
        info["masses"][1] = nozpe.M_H
        print(f"NOTE: type 1 (H) absent de Masses, ajout automatique m_H = {nozpe.M_H}",
              file=sys.stderr)

    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = info["box"]

    w_atoms = [a for a in surface_atoms if a["type"] == 2]
    if not w_atoms:
        raise ValueError("Aucun atome de type W (type 2) trouve dans le data file.")
    z_mid = 0.5 * (zhi - zlo)
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
    origin_xy = np.array([xlo + xz * s3, ylo + yz * s3])

    m_H2 = 2.0 * nozpe.M_H
    v_mag = nozpe.velocity_magnitude(E_eV, m_H2)

    # --- Potentiel de Morse : fit + energie du niveau vibrationnel demande ---
    D_eV, a_bohrinv, re_bohr, fit_rms = fit_morse(init_cond_dir / "morse_potasym.dat")
    vmax = morse_max_vib_level(D_eV, a_bohrinv)
    if v_level > vmax:
        raise ValueError(
            f"Niveau vibrationnel v={v_level} au-dessus du puits de Morse "
            f"(D={D_eV:.4f} eV, v_max={vmax})."
        )
    evirot_ha = morse_vibrational_energy_ha(D_eV, a_bohrinv, v_level)
    print(
        f"[Morse] D = {D_eV:.4f} eV, a = {a_bohrinv:.4f} /bohr, "
        f"re = {re_bohr * ANGUA:.4f} A  (fit rms = {fit_rms:.2e} eV, v_max = {vmax})"
    )
    print(
        f"[ZPE]   v = {v_level}, J = {jrot}  ->  E_vib = {evirot_ha * EVUA:.5f} eV"
    )

    # --- Etats internes (r_HH, orientation, vitesses internes) ---
    internal = run_internal_sampler(n_traj, evirot_ha, jrot, init_cond_dir, seed=seed)

    n_fixed = sum(1 for a in surface_atoms if a["fixed"])
    n_mobile = len(surface_atoms) - n_fixed

    for i in range(1, n_traj + 1):
        za, xa, ya, zb, xb, yb, vza, vxa, vya, vzb, vxb, vyb = internal[i - 1]

        xy_cm = nozpe.random_xy_in_cell((a1_xy, a2_xy), rng) + origin_xy
        cm = np.array([xy_cm[0], xy_cm[1], z_H2])
        v_cm = nozpe.velocity_vector(E_eV, m_H2, theta, phi, rng)

        pH1 = cm + np.array([xa, ya, za])
        pH2 = cm + np.array([xb, yb, zb])
        vH1 = v_cm + np.array([vxa, vya, vza])
        vH2 = v_cm + np.array([vxb, vyb, vzb])

        H1 = {"id": 1, "type": 1, "x": pH1[0], "y": pH1[1], "z": pH1[2], "fixed": False}
        H2 = {"id": 2, "type": 1, "x": pH2[0], "y": pH2[1], "z": pH2[2], "fixed": False}
        all_atoms = [H1, H2] + surface_atoms
        velocities = [vH1, vH2] + [np.zeros(3)] * len(surface_atoms)

        r_hh = float(np.linalg.norm(pH2 - pH1))
        header = (
            f"# data file ZPE (Morse), traj {i}/{n_traj}\n"
            f"# E = {E_eV} eV, theta = {theta} deg, phi = {phi}, v = {v_level}, J = {jrot}\n"
            f"# v_CdM = ({v_cm[0]:.6f}, {v_cm[1]:.6f}, {v_cm[2]:.6f}) A/ps,  |v| = {v_mag:.6f} A/ps\n"
            f"# r_HH = {r_hh:.4f} A (echantillonne), z_H2_CdM = {z_H2:.4f} A"
        )

        out_path = out_dir / f"{prefix}{i}.lammps"
        nozpe.write_data_file(out_path, info, all_atoms, velocities, header)

    template_atoms = [
        {"id": 1, "type": 1, "fixed": False},
        {"id": 2, "type": 1, "fixed": False},
    ] + surface_atoms
    nozpe.write_groups_lmp(out_dir / "groups.lmp", template_atoms, info["n_types"])

    return {
        "n_traj": n_traj,
        "v_mag": v_mag,
        "z_top_surface": z_top_surface,
        "z_H2": z_H2,
        "n_fixed": n_fixed,
        "n_mobile": n_mobile,
        "D_eV": D_eV,
        "evirot_eV": evirot_ha * EVUA,
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
    ap.add_argument("-v", "--vib-level", type=int, default=0,
                    help="niveau vibrationnel v de H2 dans le potentiel de Morse [0 = ZPE]")
    ap.add_argument("-j", "--jrot", type=int, default=0,
                    help="nombre quantique rotationnel J [0]")
    ap.add_argument("--frozen-ids", default="",
                    help='IDs supplementaires a figer, ex: "35-42" [vide]')
    ap.add_argument("--init-cond-dir", default=None,
                    help="dossier contenant exe/intrep.dat/morse_potasym.dat "
                         "[<dossier du script>/init_cond]")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed du RNG [aleatoire]")
    ap.add_argument("--prefix", default="data_POSCAR_",
                    help='prefixe des fichiers data ["data_POSCAR_"]')
    args = ap.parse_args()

    init_cond_dir = (
        Path(args.init_cond_dir) if args.init_cond_dir
        else Path(__file__).resolve().parent / "init_cond"
    )

    info = nozpe.read_lammps_data(args.datafile)
    extra_frozen = nozpe.parse_ids(args.frozen_ids)

    res = generate(
        info=info,
        out_dir=args.out_dir,
        n_traj=args.numgen,
        E_eV=args.energie,
        theta=args.theta,
        phi=args.phi,
        height_offset=args.height,
        v_level=args.vib_level,
        jrot=args.jrot,
        init_cond_dir=init_cond_dir,
        prefix=args.prefix,
        seed=args.seed,
        extra_frozen=extra_frozen,
    )

    print(f"[OK] {res['n_traj']} data files ZPE generes dans {args.out_dir}/")
    print(f"     |v_CdM| = {res['v_mag']:.4f} A/ps   (E = {args.energie} eV, m_H2 = {2*nozpe.M_H:.5f} amu)")
    print(f"     theta = {args.theta} deg, phi = {args.phi}")
    print(f"     v = {args.vib_level}, J = {args.jrot}  ->  E_vib = {res['evirot_eV']:.4f} eV "
          f"(puits de Morse D = {res['D_eV']:.4f} eV)")
    print(f"     z(top W)       = {res['z_top_surface']:.3f} A  ->  "
          f"z(H2 CdM) = {res['z_H2']:.3f} A")
    print(f"     atomes fixes  : {res['n_fixed']}")
    print(f"     atomes mobiles: {res['n_mobile']}")
    print(f"     fichier de groupes : {Path(args.out_dir) / 'groups.lmp'}")


if __name__ == "__main__":
    main()
