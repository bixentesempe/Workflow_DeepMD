#!/usr/bin/env python3
"""
contcar2lammps.py — convertit un CONTCAR VASP (surface relaxee) en LAMMPS data file.

Format identique a BuildLAMMPSPoscar.py :
  - H (type 1) declares dans header + Masses mais NON ecrits dans Atoms
  - IDs Atoms commencent a 3 (1 et 2 reserves H)
  - Ordre Atoms : O d'abord, puis W
  - Ordre Masses : 1=H, 3=O, 2=W
  - Separation par tabulations
  - Atomes figes : par defaut AUCUN (relaxation complete). Si le CONTCAR
    porte une contrainte Selective Dynamics (ex. figee via lammps2poscar.py
    en amont), elle est reprise telle quelle. --fix-z force une altitude,
    --fix-center un repli manuel sur la couche W centrale, --no-fix force
    l'absence totale de figeage.

CORRECTION (triclinique) :
  La cellule LAMMPS est lower-triangular (a le long de x, b dans le plan xy...).
  Si la cellule VASP ne l'est pas deja (ex. a = (1.414, -1, 0) pour W(110)),
  passer en lower-triangular impose une ROTATION. Il faut appliquer cette meme
  rotation AUX POSITIONS, sinon atomes et boite se retrouvent dans deux reperes
  differents -> O et W deplaces (bug de l'ancienne version).
  On utilise ASE Prism, qui tourne cellule ET positions de facon coherente.
  Pour une cellule deja orthogonale / axis-aligned, la rotation est l'identite
  et le resultat est rigoureusement identique a l'ancien code.

Exemples :
  python contcar2lammps.py CONTCAR            LAMMPS_POSCAR_relaxed
  python contcar2lammps.py CONTCAR_1O         LAMMPS_POSCAR-1O_relaxed
  python contcar2lammps.py CONTCAR --fix-z 4.447  LAMMPS_POSCAR-1O_relaxed
"""
import argparse
import numpy as np
from ase.io import read
from ase.calculators.lammps import Prism

MASSES = {
    "H": 1.0079400000000,
    "O": 15.9994000000000,
    "W": 183.8400000000000,
}


def detect_fix_z(z_W, tol):
    """z moyen de la couche W centrale."""
    zs = np.sort(z_W)
    layers, current = [], [zs[0]]
    for zi in zs[1:]:
        if zi - current[-1] <= tol:
            current.append(zi)
        else:
            layers.append(np.mean(current))
            current = [zi]
    layers.append(np.mean(current))
    return layers[len(layers) // 2]


def main():
    ap = argparse.ArgumentParser(
        description="CONTCAR VASP -> LAMMPS data file (format BuildLAMMPSPoscar)")
    ap.add_argument("infile",  help="CONTCAR VASP (surface relaxee, sans H)")
    ap.add_argument("outfile", help="Fichier LAMMPS data en sortie")
    ap.add_argument("--fix-z", type=float, default=None,
                    help="z (A) de la couche W a marquer # fixed "
                         "(force ce choix, prioritaire sur tout le reste)")
    ap.add_argument("--fix-tol", type=float, default=0.5,
                    help="Tolerance (A) pour regrouper une couche (defaut : 0.5)")
    ap.add_argument("--fix-center", action="store_true",
                    help="figer la couche W centrale (repli manuel, utilise "
                         "seulement si demande explicitement et si le CONTCAR "
                         "ne porte pas de contrainte Selective Dynamics)")
    ap.add_argument("--no-fix", action="store_true",
                    help="ne fige rien, meme si le CONTCAR porte une "
                         "contrainte Selective Dynamics (force le desaccord)")
    ap.add_argument("--wrap", action="store_true",
                    help="replie les atomes dans la maille (utile si des couches "
                         "sont wrappees en haut de boite, ex. z~zhi au lieu de ~0)")
    args = ap.parse_args()

    # ---- lecture ----
    atoms = read(args.infile, format="vasp")
    atoms.pbc = True
    # Indices fixes tels qu'ecrits dans le CONTCAR (Selective Dynamics),
    # AVANT tout wrap/rotation qui ne change pas l'ordre des atomes.
    constrained_indices = set()
    for c in atoms.constraints:
        constrained_indices.update(int(i) for i in getattr(c, "index", []))
    if args.wrap:
        atoms.wrap()

    # ---- conversion cellule + positions coherente (lower-triangular LAMMPS) ----
    # Prism tourne la cellule ET les positions dans le meme repere. Pour une
    # cellule deja axis-aligned c'est l'identite (=> ancien comportement).
    prism = Prism(atoms.cell)
    xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
    pos = np.asarray(prism.vector_to_lammps(atoms.get_positions()))

    is_triclinic = abs(xy) > 1e-8 or abs(xz) > 1e-8 or abs(yz) > 1e-8

    sym = np.array(atoms.get_chemical_symbols())

    # ---- atomes figes. Par defaut : RIEN de fige (relaxation complete),
    # sauf si le CONTCAR porte deja une contrainte Selective Dynamics
    # (reprise du choix fait a la construction / a l'etape lammps2poscar.py).
    # Priorite : --no-fix > --fix-z > contrainte du CONTCAR > --fix-center > rien.
    fix_z = None
    fixed_mask = None  # si non None : set d'indices (repere ASE, avant rotation)
    if args.no_fix:
        fixed_mask = set()
    elif args.fix_z is not None:
        fix_z = args.fix_z
    elif constrained_indices:
        fixed_mask = constrained_indices
    elif args.fix_center:
        z_W = pos[sym == "W", 2]
        fix_z = detect_fix_z(z_W, args.fix_tol)
    else:
        fixed_mask = set()

    # ---- indices O et W ----
    idx_O = np.where(sym == "O")[0]
    idx_W = np.where(sym == "W")[0]
    n_O, n_W = len(idx_O), len(idx_W)
    has_O = n_O > 0

    # ---- header ----
    n_atoms_header = 2 + n_O + n_W   # +2 : H reserves mais non ecrits
    n_types = 3 if has_O else 2

    lines = []
    lines.append(f"(relaxed from {args.infile})")
    lines.append("")
    lines.append(f"{n_atoms_header} atoms")
    lines.append(f"{n_types} atom types")
    lines.append("")
    lines.append(f"0.0      {xhi:.10f}  xlo xhi")
    lines.append(f"0.0      {yhi:.10f}  ylo yhi")
    lines.append(f"0.0      {zhi:.10f}  zlo zhi")
    if is_triclinic:
        lines.append(f"     {xy:.10f}   {xz:.10f}   {yz:.10f}  xy xz yz")
    else:
        lines.append("0.0 0.0 0.0  xy xz yz")

    # ---- Masses (ordre BuildLAMMPSPoscar : 1=H, 3=O, 2=W) ----
    lines.append("")
    lines.append("Masses")
    lines.append("")
    lines.append(f"    1      {MASSES['H']:.13f} # H")
    if has_O:
        lines.append(f"    3      {MASSES['O']:.13f} # O")
    lines.append(f"    2      {MASSES['W']:.13f} # W")

    # ---- Atoms (O d'abord, puis W ; IDs commencent a 3) ----
    lines.append("")
    lines.append("Atoms # atomic")
    lines.append("")

    aid = 3   # 1 et 2 reserves aux H

    def is_fixed(atom_idx, z):
        if fixed_mask is not None:
            return atom_idx in fixed_mask
        return abs(z - fix_z) < args.fix_tol

    n_fix = 0
    for i in idx_O:
        x, y, z = pos[i]
        tag = " # fixed" if is_fixed(i, z) else ""
        n_fix += 1 if tag else 0
        lines.append(f"{aid}\t3\t{x:.6f}\t{y:.6f}\t{z:.6f}{tag}")
        aid += 1

    for i in idx_W:
        x, y, z = pos[i]
        tag = " # fixed" if is_fixed(i, z) else ""
        n_fix += 1 if tag else 0
        lines.append(f"{aid}\t2\t{x:.6f}\t{y:.6f}\t{z:.6f}{tag}")
        aid += 1

    with open(args.outfile, "w") as f:
        f.write("\n".join(lines) + "\n")

    if fixed_mask is not None:
        fix_info = "aucun (relaxation complete)" if not fixed_mask \
            else f"contrainte du CONTCAR ({len(fixed_mask)} atomes)"
    else:
        fix_info = f"--fix-center z={fix_z:.3f}" if args.fix_center else f"z={fix_z:.3f}"
    print(f"{args.infile} -> {args.outfile}")
    print(f"  header : {n_atoms_header} atoms ({n_O} O + {n_W} W + 2 H reserves), "
          f"{n_types} types")
    print(f"  atoms ecrits : {n_O + n_W}  (ids 3-{aid-1})")
    print(f"  figes : {fix_info} ({n_fix} atomes)")
    print(f"  cellule : {'triclinique' if is_triclinic else 'orthorhombique'} "
          f"(xy={xy:.4f}, xz={xz:.4f}, yz={yz:.4f})")


if __name__ == "__main__":
    main()
