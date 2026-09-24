#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerateAIMDPoscars_ThermalSurface.py
=======================================
Genere N POSCAR VASP pour lancer des AIMD H2/surface, en piochant
ALEATOIREMENT, pour CHAQUE structure, une surface thermalisee dans un pool
de POSCAR-* (typiquement produits par un script d'extraction AIMD a partir
de vasprun.xml, ex. extract_md_configs.py), puis en y ajoutant une molecule
H2 :

  - position XY : aleatoire uniforme dans la cellule
  - hauteur z   : fixe (--height), au-dessus de la MOYENNE de la premiere
    couche -- le groupe d'au moins --min-layer-size atomes le plus haut
    dans la moitie basse de la cellule (cf. surface_reference_height) ;
    pas juste l'atome le plus haut, qui peut etre un adsorbat isole
    au-dessus du vrai plan de surface
  - orientation H-H : (--internal-theta/--internal-phi), meme convention
    spherique que l'incidence (theta = angle / normale, phi = azimut) ;
    'random' (defaut pour les deux) => isotrope, une valeur fixe => orientation
    deterministe
  - distance H-H     : fixe (--hh-distance)
  - vitesse de translation du CdM : imposee par (-e/--theta/--phi), comme
    dans le cas LAMMPS -- -e est OBLIGATOIRE (pas de defaut silencieux),
    mets -e 0 explicitement si tu veux H2 au repos

Chaque surface piochee garde ses positions ET vitesses reelles (thermiques)
pour les atomes mobiles -- exactement comme pour la version LAMMPS de ce
pool (voir ../Lammps_dynamic/GenerateLammpsDatafile_ThermalSurface.py).

Chaque structure est ecrite dans son PROPRE dossier
<out-dir>/POSCAR_<idx>/POSCAR -- pret a etre repris tel quel par
setup_aimd_calculations.py (voir ci-contre, copie/adaptee de
../Single_Points_Calculations/setup_sp_calculations.py), qui cherche
recursivement des fichiers nommes "POSCAR" et cree un dossier de calcul
(POSCAR+INCAR+KPOINTS+POTCAR) par structure trouvee.

Ordre des especes (--order) : VASP assigne les pseudopotentiels du POTCAR
dans l'ordre ou les especes apparaissent dans le POSCAR. Le pool
thermalise a son propre ordre (ex. "O W" pour ce projet), et H est
toujours ajoute en dernier -- il faut donc reordonner pour matcher le
POTCAR concatene que tu utilises (ex. POTCAR_HOW = H+O+W dans
Single_Points_Calculations/). Par defaut : H,O,W.

Ne depend que de numpy (pas de pymatgen) : parseur/ecrivain POSCAR maison,
meme principe que Lammps_dynamic/GenerateLammpsDatafile_ThermalSurface.py::
read_poscar(), pour ne rien avoir a installer de special sur le cluster.
Verifie bit-a-bit equivalent a une version pymatgen de reference (memes
fonctions de tirage aleatoire, meme ordre d'appels RNG, meme formatage de
sortie) -- voir le commit/discussion associe si besoin de re-verifier.

Usage :
    ./GenerateAIMDPoscars_ThermalSurface.py \\
        --poscar-dir /path/extracted_configs/0.25ML300K/POSCAR \\
        -n 200 --height 3.0 --hh-distance 0.7414 \\
        --internal-theta random --internal-phi random \\
        -e 0.3 -T 45 -P random \\
        -o aimd_poscars/

    # Puis, pour preparer et soumettre les calculs VASP :
    ./setup_aimd_calculations.py \\
        -i aimd_poscars/ -o AIMD_calculations \\
        --incar-aimd ./INCAR_AIMD --kpoints ./KPOINTS --potcar ./POTCAR_HOW \\
        --vdw-kernel ./vdw_kernel.bindat
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

H_SYMBOL = "H"
M_H = 1.00794  # amu (m_H2 = 2*M_H), meme valeur que Lammps_dynamic
R_HH_EQ = 0.7414  # Angstrom, distance d'equilibre H2

# v[A/fs] = sqrt(2 E[eV] / m[amu]) * EV_AMU_TO_AFS. Vitesses VASP en A/fs
# (contrairement a LAMMPS "metal units" qui utilise A/ps -- pas de x1000 ici).
EV_AMU_TO_AFS = math.sqrt(1.602176634e-19 / 1.66053906660e-27) * 1e-5


def velocity_magnitude(E_eV: float, m_amu: float) -> float:
    return math.sqrt(2.0 * E_eV / m_amu) * EV_AMU_TO_AFS


def velocity_vector(E_eV: float, m_amu: float, theta_deg, phi_deg, rng: np.random.Generator) -> np.ndarray:
    """v=(vx,vy,vz) en A/fs. theta = angle / normale (deg), phi = azimut (deg
    ou 'random'). E_eV=0 -> vecteur nul (H2 part du repos), meme convention
    que Lammps_dynamic/GenerateLammpsDatafile_NoZPE.py::velocity_vector."""
    if E_eV == 0:
        return np.zeros(3)
    v = velocity_magnitude(E_eV, m_amu)
    th = math.radians(float(theta_deg))
    if phi_deg == "random" or phi_deg is None:
        ph = rng.uniform(0.0, 2 * math.pi)
    else:
        ph = math.radians(float(phi_deg))
    return np.array([
        v * math.sin(th) * math.cos(ph),
        v * math.sin(th) * math.sin(ph),
        -v * math.cos(th),  # vers la surface
    ])


# ---------------------------------------------------------------------------
# Lecture d'un POSCAR VASP thermalise (meme parseur maison que
# Lammps_dynamic/GenerateLammpsDatafile_ThermalSurface.py::read_poscar,
# avec un ajout : velocities=None (pas un tableau de zeros) si le bloc
# "Velocities" est absent du fichier, pour distinguer "pas de vitesses"
# de "vitesses nulles" -- important pour needs_velocities dans add_h2().
# ---------------------------------------------------------------------------
def read_poscar(path: Path) -> dict:
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
    fixed = np.zeros(natoms, dtype=bool) if selective else None
    for i in range(natoms):
        parts = lines[idx + i].split()
        positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if selective:
            fixed[i] = not any(p == "T" for p in parts[3:6])
    idx += natoms

    if is_direct:
        positions = positions @ lattice

    rest = [ln for ln in lines[idx:] if ln.strip() != ""]
    velocities = None
    if len(rest) >= natoms:
        velocities = np.zeros((natoms, 3))
        for i in range(natoms):
            parts = rest[i].split()
            velocities[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return {
        "lattice": lattice,
        "symbols": symbols,
        "positions": positions,   # Cartesian, Angstrom
        "fixed": fixed,            # None si pas de "Selective dynamics"
        "velocities": velocities,  # None si pas de bloc "Velocities"
    }


def find_poscar_pool(poscar_dir: Path) -> list[Path]:
    pool = sorted(Path(poscar_dir).glob("POSCAR-*"))
    if not pool:
        raise FileNotFoundError(f"Aucun fichier 'POSCAR-*' trouve dans {poscar_dir}")
    return pool


# ---------------------------------------------------------------------------
# Placement de H2
# ---------------------------------------------------------------------------
def orientation_vector(theta_deg, phi_deg, rng: np.random.Generator) -> np.ndarray:
    """Vecteur unitaire de l'orientation H-H, depuis (theta, phi) -- meme
    convention que velocity_vector (theta = angle / normale z, phi =
    azimut), mais sans le signe "vers la surface" : c'est un axe, pas une
    vitesse, donc theta=0 pointe vers +z (H-H vertical).

    theta='random' echantillonne cos(theta) uniformement dans [-1,1] (pas
    theta lui-meme, qui biaiserait vers les poles) -- isotrope. phi='random'
    est uniforme sur [0, 2pi). theta et phi fixes -> orientation fixe et
    deterministe (aucun aleatoire).
    """
    if theta_deg == "random" or theta_deg is None:
        th = math.acos(rng.uniform(-1.0, 1.0))
    else:
        th = math.radians(float(theta_deg))
    if phi_deg == "random" or phi_deg is None:
        ph = rng.uniform(0.0, 2 * math.pi)
    else:
        ph = math.radians(float(phi_deg))
    return np.array([
        math.sin(th) * math.cos(ph),
        math.sin(th) * math.sin(ph),
        math.cos(th),
    ])


def random_xy_in_cell(lattice_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Position XY (cartesienne) uniforme dans le parallelogramme (a1_xy, a2_xy)."""
    if abs(lattice_matrix[0, 2]) > 1e-6 or abs(lattice_matrix[1, 2]) > 1e-6:
        raise ValueError(
            "Maille non compatible : a1/a2 doivent etre dans le plan xy "
            "(convention dalle + vide selon c). Verifie le POSCAR source."
        )
    a1_xy = lattice_matrix[0, :2]
    a2_xy = lattice_matrix[1, :2]
    u, v = rng.uniform(0.0, 1.0, size=2)
    return u * a1_xy + v * a2_xy


# ---------------------------------------------------------------------------
# Hauteur de reference : moyenne de la premiere couche (pas juste l'atome le
# plus haut, qui peut etre un adsorbat isole -- cf. surface_reference_height)
# ---------------------------------------------------------------------------
def detect_layers(z_values: np.ndarray, tol: float) -> list[list[int]]:
    """Regroupe des altitudes z en couches (atomes a moins de `tol` A les uns
    des autres) -- meme algorithme que lammps2poscar.py::detect_layers.
    Retourne les groupes (listes d'indices dans `z_values`), du plus bas au
    plus haut.
    """
    order = np.argsort(z_values)
    layers: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        if z_values[idx] - z_values[layers[-1][-1]] <= tol:
            layers[-1].append(idx)
        else:
            layers.append([idx])
    return layers


def surface_reference_height(surface: dict, layer_tol: float, min_layer_size: int) -> float:
    """Hauteur de reference pour placer H2 : moyenne des z de la PREMIERE
    COUCHE, definie comme le groupe d'au moins `min_layer_size` atomes le
    plus haut dans la moitie basse de la cellule.

    Les deux filtres comptent :
      - "moitie basse" (z < z_mid, avec z_mid = moitie de la hauteur de
        boite selon c) exclut tout ce qui flotte au-dessus de la vraie
        surface (ex. un adsorbat pres du vide).
      - "au moins min_layer_size atomes" exclut un adsorbat isole qui
        serait par ailleurs dans la moitie basse (ex. un O seul plus haut
        que la derniere couche de substrat, mais pas assez haut pour
        depasser z_mid).
    """
    lattice = surface["lattice"]
    if abs(lattice[2, 0]) > 1e-6 or abs(lattice[2, 1]) > 1e-6:
        raise ValueError(
            "Maille non compatible : c doit etre selon z (convention dalle + "
            "vide). Verifie le POSCAR source."
        )
    z = surface["positions"][:, 2]
    z_mid = 0.5 * lattice[2, 2]
    layers = detect_layers(z, layer_tol)
    candidates = [layer for layer in layers if len(layer) >= min_layer_size and np.mean(z[layer]) < z_mid]
    if not candidates:
        raise ValueError(
            f"Aucune couche d'au moins {min_layer_size} atomes trouvee sous "
            f"z_mid={z_mid:.3f} A -- essaie --min-layer-size plus petit ou "
            "--layer-tol different."
        )
    top_layer = max(candidates, key=lambda layer: np.mean(z[layer]))
    return float(np.mean(z[top_layer]))


def add_h2(
    surface: dict,
    height: float,
    hh_distance: float,
    internal_theta_deg,
    internal_phi_deg,
    layer_tol: float,
    min_layer_size: int,
    E_eV: float,
    incidence_theta_deg,
    incidence_phi_deg,
    rng: np.random.Generator,
) -> dict:
    """Copie `surface` avec H1/H2 ajoutes au-dessus de la premiere couche
    (cf. surface_reference_height), avec :
      - une orientation H-H imposee par (internal_theta_deg, internal_phi_deg)
        ('random' par defaut pour chacun => isotrope)
      - une vitesse de translation imposee par (E_eV, incidence_theta_deg,
        incidence_phi_deg) -- E_eV=0 => H2 au repos (valeur obligatoire au
        niveau CLI, pas de defaut silencieux).
    """
    z_ref = surface_reference_height(surface, layer_tol, min_layer_size)
    xy_cm = random_xy_in_cell(surface["lattice"], rng)
    cm = np.array([xy_cm[0], xy_cm[1], z_ref + height])

    d = orientation_vector(internal_theta_deg, internal_phi_deg, rng)

    p1 = cm - 0.5 * hh_distance * d
    p2 = cm + 0.5 * hh_distance * d
    v_cm = velocity_vector(E_eV, 2.0 * M_H, incidence_theta_deg, incidence_phi_deg, rng)

    n_old = len(surface["symbols"])
    # Le bloc "Velocities" du POSCAR est tout-ou-rien (une ligne par atome) :
    # s'il n'existe pas deja (pool sans vitesses thermiques) mais qu'une
    # vitesse incidente est demandee, on cree le bloc avec des zeros pour
    # les atomes de surface avant d'ajouter H2.
    needs_velocities = surface["velocities"] is not None or bool(np.any(v_cm != 0.0))
    velocities_old = surface["velocities"] if surface["velocities"] is not None else np.zeros((n_old, 3))

    fixed_old = surface["fixed"] if surface["fixed"] is not None else np.zeros(n_old, dtype=bool)

    return {
        "lattice": surface["lattice"],
        "symbols": surface["symbols"] + [H_SYMBOL, H_SYMBOL],
        "positions": np.vstack([surface["positions"], p1, p2]),
        "fixed": np.concatenate([fixed_old, [False, False]]),  # H2 toujours mobile
        "velocities": (np.vstack([velocities_old, v_cm, v_cm]) if needs_velocities else None),
    }


# ---------------------------------------------------------------------------
# Ordre des especes (pour matcher le POTCAR concatene)
# ---------------------------------------------------------------------------
def reorder_species(surface: dict, order: list[str]) -> dict:
    symbols = surface["symbols"]
    present = set(symbols)
    missing = present - set(order)
    if missing:
        raise ValueError(
            f"Elements presents mais absents de --order : {sorted(missing)} "
            f"(--order actuel : {','.join(order)})"
        )
    indices = [i for el in order for i, sym in enumerate(symbols) if sym == el]
    return {
        "lattice": surface["lattice"],
        "symbols": [symbols[i] for i in indices],
        "positions": surface["positions"][indices],
        "fixed": surface["fixed"][indices] if surface["fixed"] is not None else None,
        "velocities": surface["velocities"][indices] if surface["velocities"] is not None else None,
    }


# ---------------------------------------------------------------------------
# Ecriture POSCAR -- format VASP5 standard, sans le symbole chimique en fin
# de ligne de position (juste ne pas l'ecrire, plus besoin de l'astuce
# "ecrire puis retirer" utilisee quand on passait par un writer externe).
# Meme precision (16 decimales) et meme structure de sections qu'une sortie
# pymatgen standard, verifie par comparaison directe.
# ---------------------------------------------------------------------------
def write_poscar(surface: dict, path: Path, comment: str) -> None:
    lattice = surface["lattice"]
    symbols = surface["symbols"]
    positions = surface["positions"]
    fixed = surface["fixed"]
    velocities = surface["velocities"]
    n = len(symbols)

    species_order = list(dict.fromkeys(symbols))  # ordre d'apparition (deja impose par reorder_species)
    counts = [symbols.count(sp) for sp in species_order]

    frac = positions @ np.linalg.inv(lattice)

    fmt = "{:21.16f}".format
    lines = [comment, "1.0"]
    lines += [" ".join(fmt(x) for x in row) for row in lattice]
    lines.append(" ".join(species_order))
    lines.append(" ".join(str(c) for c in counts))
    if fixed is not None:
        lines.append("Selective dynamics")
    lines.append("direct")
    for i in range(n):
        line = " ".join(fmt(x) for x in frac[i])
        if fixed is not None:
            line += " F F F" if fixed[i] else " T T T"
        lines.append(line)
    if velocities is not None:
        lines.append("")
        lines += [" ".join(fmt(x) for x in velocities[i]) for i in range(n)]

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
    poscar_pool: list[Path],
    out_dir: Path,
    n_gen: int,
    height: float,
    hh_distance: float,
    internal_theta: str,
    internal_phi: str,
    layer_tol: float,
    min_layer_size: int,
    E_eV: float,
    theta_deg,
    phi_deg,
    order: list[str],
    seed: int | None,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, n_gen + 1):
        poscar_path = poscar_pool[rng.integers(len(poscar_pool))]
        surface = read_poscar(poscar_path)

        structure = add_h2(
            surface, height, hh_distance, internal_theta, internal_phi,
            layer_tol, min_layer_size, E_eV, theta_deg, phi_deg, rng
        )
        structure = reorder_species(structure, order)

        calc_dir = out_dir / f"POSCAR_{i:05d}"
        calc_dir.mkdir(parents=True, exist_ok=True)
        write_poscar(
            structure,
            calc_dir / "POSCAR",
            comment=f"AIMD H2 incidence, surface={poscar_path.name}, "
            f"height={height}A, hh={hh_distance}A, "
            f"internal_theta={internal_theta}, internal_phi={internal_phi}, "
            f"E={E_eV}eV, theta={theta_deg}, phi={phi_deg}",
        )

    v_mag = velocity_magnitude(E_eV, 2.0 * M_H) if E_eV else 0.0
    print(f"[OK] {n_gen} structures generees dans {out_dir}/ (une par sous-dossier POSCAR_XXXXX/POSCAR)")
    print(f"     surfaces piochees dans un pool de {len(poscar_pool)} POSCAR thermalises")
    print(f"     hauteur H2 : {height} A au-dessus de la moyenne de la 1ere couche "
          f"(tol={layer_tol}A, min {min_layer_size} atomes/couche)")
    print(f"     vitesse H2 : E={E_eV} eV -> |v_CdM|={v_mag:.5f} A/fs (theta={theta_deg}, phi={phi_deg})")
    print(f"     ordre des especes (POTCAR attendu dans cet ordre) : {','.join(order)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--poscar-dir", required=True, help="dossier des POSCAR-* thermalises (une seule surface/run)")
    parser.add_argument("-n", "--numgen", type=int, default=100, help="nombre de structures a generer [100]")
    parser.add_argument("--height", type=float, default=3.0,
                        help="hauteur du CdM de H2 au-dessus de la moyenne de la 1ere couche (A) [3.0]")
    parser.add_argument("--layer-tol", type=float, default=0.5,
                        help="tolerance (A) pour regrouper des atomes en une couche [0.5]")
    parser.add_argument("--min-layer-size", type=int, default=4,
                        help="nb min. d'atomes pour qu'un groupe compte comme une couche "
                             "(ignore un adsorbat isole) [4]")
    parser.add_argument("--hh-distance", type=float, default=R_HH_EQ, help=f"distance H-H (A) [{R_HH_EQ}]")
    parser.add_argument("--internal-theta", default="random",
                        help="orientation H-H, theta : angle / normale (deg) ou 'random' [random]")
    parser.add_argument("--internal-phi", default="random",
                        help="orientation H-H, phi : azimut (deg) ou 'random' [random]")
    parser.add_argument("-e", "--energie", type=float, required=True,
                        help="energie cinetique de translation de H2 (eV) -- obligatoire, "
                             "pas de defaut silencieux (mets 0 explicitement pour H2 au repos)")
    parser.add_argument("-T", "--theta", default="0", help="theta : angle par rapport a la normale (deg) [0]")
    parser.add_argument("-P", "--phi", default="random", help='phi : azimut (deg) ou "random" [random]')
    parser.add_argument("--order", default="H,O,W", help="ordre des especes, doit matcher le POTCAR concatene [H,O,W]")
    parser.add_argument("--seed", type=int, default=None, help="seed du RNG [aleatoire]")
    parser.add_argument("-o", "--out-dir", default="aimd_poscars", help="dossier de sortie [aimd_poscars]")
    args = parser.parse_args()

    generate(
        poscar_pool=find_poscar_pool(Path(args.poscar_dir)),
        out_dir=Path(args.out_dir),
        n_gen=args.numgen,
        height=args.height,
        hh_distance=args.hh_distance,
        internal_theta=args.internal_theta,
        internal_phi=args.internal_phi,
        layer_tol=args.layer_tol,
        min_layer_size=args.min_layer_size,
        E_eV=args.energie,
        theta_deg=args.theta,
        phi_deg=args.phi,
        order=args.order.split(","),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
