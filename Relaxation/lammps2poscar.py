#!/usr/bin/env python3
"""
lammps2poscar.py — convertit un LAMMPS data file (style atomic) en POSCAR VASP
prêt pour une relaxation de surface.

- mappe les types LAMMPS -> éléments (par défaut 1=H, 2=W, 3=O)
- retire optionnellement des espèces (par défaut H = projectile H2)
- fige une couche via 'Selective dynamics' : par défaut, REUTILISE les
  commentaires "# fixed"/"# frozen" deja presents dans le LAMMPS data file
  (ceux ecrits par BuildLAMMPSPoscar.py, ex. --freeze-bottom) ; si le
  fichier n'en contient aucun, on retombe sur la couche W centrale (ou une
  altitude z explicite avec --fix-z). --fix-z / --no-fix restent
  prioritaires sur les tags du fichier si tu veux forcer un autre choix.
- ordonne les espèces selon --order (DOIT correspondre à l'ordre du POTCAR)

Exemples :
  python lammps2poscar.py LAMMPS_POSCAR        POSCAR_clean
  python lammps2poscar.py LAMMPS_POSCAR-1O     POSCAR_1O
  python lammps2poscar.py LAMMPS_POSCAR-2O     POSCAR_2O
  # triclinique relaxé, fixer une couche à z connu :
  python lammps2poscar.py LAMMPS-POSCAR-HW1O   POSCAR_tri1O --fix-z 4.447
"""
import argparse
import re
import numpy as np
from ase.io import read


def parse_fixed_ids(path):
    """
    Relit le fichier en texte brut pour recuperer les IDs marques comme
    fixes (commentaire "# fixed"/"# frozen"/"# F" en fin de ligne dans la
    section Atoms), comme le fait GenerateLammpsDatafile_NoZPE.py. ASE ne
    conserve pas ces commentaires, d'ou cette relecture legere.
    Retourne un set() (vide si aucun tag trouve dans le fichier).
    """
    fixed_ids = set()
    section = None
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("Masses"):
                section = "masses"; continue
            if s.startswith("Atoms"):
                section = "atoms"; continue
            if s.startswith("Velocities") or s.startswith("Bonds") or s.startswith("Angles"):
                section = "skip"; continue
            if section != "atoms":
                continue
            data, _, comment = raw.partition("#")
            p = data.split()
            if len(p) < 5:
                continue
            c = comment.strip().lower()
            if re.search(r"\b(fixed|frozen)\b", c) or c == "f" or c.startswith("f ") or c.endswith(" f"):
                fixed_ids.add(int(p[0]))
    return fixed_ids


def detect_layers(z, tol):
    """Regroupe des z en couches ; renvoie les z moyens triés (bas -> haut)."""
    zs = np.sort(z)
    layers, current = [], [zs[0]]
    for zi in zs[1:]:
        if zi - current[-1] <= tol:
            current.append(zi)
        else:
            layers.append(np.mean(current))
            current = [zi]
    layers.append(np.mean(current))
    return np.array(layers)


def main():
    ap = argparse.ArgumentParser(description="LAMMPS data -> POSCAR (relaxation)")
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--types", default="1:H,2:W,3:O",
                    help="mapping type LAMMPS -> élément (def: 1:H,2:W,3:O)")
    ap.add_argument("--drop", default="H",
                    help="espèces à retirer, séparées par virgule ('' = rien)")
    ap.add_argument("--order", default="W,O",
                    help="ordre des espèces dans le POSCAR = ordre du POTCAR")
    ap.add_argument("--fix-z", type=float, default=None,
                    help="altitude (Å) de la couche à figer ; force ce choix "
                         "meme si le fichier contient des tags '# fixed'")
    ap.add_argument("--fix-tol", type=float, default=0.5,
                    help="tolérance (Å) pour regrouper une couche (--fix-z / --fix-center)")
    ap.add_argument("--fix-center", action="store_true",
                    help="figer la couche W centrale (repli manuel, utilise "
                         "seulement si demande explicitement et si le fichier "
                         "ne contient pas de tags '# fixed')")
    ap.add_argument("--no-fix", action="store_true",
                    help="ne fige rien, meme si le fichier contient des tags "
                         "'# fixed' (force le desaccord)")
    ap.add_argument("--comment", default=None)
    args = ap.parse_args()

    # symbole -> numéro atomique (table minimale)
    Z = {"H": 1, "O": 8, "W": 74}
    Z_of_type = {}
    for tok in args.types.split(","):
        t, el = tok.split(":")
        Z_of_type[int(t)] = Z[el]

    # 'style' renommé 'atom_style' dans ASE récent : on gère les deux
    try:
        atoms = read(args.infile, format="lammps-data", atom_style="atomic",
                     Z_of_type=Z_of_type, sort_by_id=True)
    except TypeError:
        atoms = read(args.infile, format="lammps-data", style="atomic",
                     Z_of_type=Z_of_type, sort_by_id=True)
    atoms.pbc = True

    # retirer le projectile (H par défaut)
    drop = [s for s in args.drop.split(",") if s]
    if drop:
        keep = [i for i, a in enumerate(atoms) if a.symbol not in drop]
        atoms = atoms[keep]

    sym = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    z = pos[:, 2]
    ids = atoms.get_array("id")

    # Couche a figer. Par defaut : RIEN de fige (la plupart des relaxations
    # sont completes, sans contrainte). Priorite :
    #   --no-fix        -> force rien de fige, meme si le fichier a des tags
    #   --fix-z Z       -> force la couche a l'altitude Z
    #   tags "# fixed"  -> si le fichier en contient (choix fait a la
    #                      construction, ex. --freeze-bottom), on les reutilise
    #   --fix-center    -> repli manuel, couche W centrale (seulement si
    #                      explicitement demande et si pas de tags dans le fichier)
    #   (rien de tout ca) -> aucune contrainte (comportement par defaut)
    fix_z = None
    fixed_source = None
    file_fixed_ids = set() if args.no_fix or args.fix_z is not None else parse_fixed_ids(args.infile)

    if args.no_fix:
        fixed = np.zeros(len(atoms), bool)
    elif args.fix_z is not None:
        fix_z = args.fix_z
        fixed = np.abs(z - fix_z) < args.fix_tol
        fixed_source = f"--fix-z {fix_z:.3f}"
    elif file_fixed_ids:
        fixed = np.array([int(i) in file_fixed_ids for i in ids])
        fixed_source = f"tags '# fixed' du fichier ({len(file_fixed_ids)} IDs)"
    elif args.fix_center:
        zw = z[sym == "W"]
        layers = detect_layers(zw, args.fix_tol)
        fix_z = layers[len(layers) // 2]
        fixed = np.abs(z - fix_z) < args.fix_tol
        fixed_source = f"--fix-center : couche centrale z={fix_z:.3f}"
    else:
        fixed = np.zeros(len(atoms), bool)

    # ordre des espèces (celles présentes seulement), + sécurité si oubli
    order = [s for s in args.order.split(",") if s]
    present = [s for s in order if (sym == s).any()]
    for s in dict.fromkeys(sym):
        if s not in present:
            present.append(s)

    cell = atoms.cell.array
    if args.comment:
        comment = args.comment
    else:
        zinfo = "no-fix" if fixed_source is None else fixed_source
        comment = f"{args.infile} -> relax ({zinfo})"

    lines = [comment, "1.0"]
    for v in cell:
        lines.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}")
    counts = [int((sym == s).sum()) for s in present]
    lines.append("   " + "   ".join(present))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Selective dynamics")
    lines.append("Cartesian")
    for s in present:                                  # écrire espèce par espèce
        for i in np.where(sym == s)[0]:
            flag = "F F F" if fixed[i] else "T T T"
            c = pos[i]
            lines.append(f"  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}   {flag}")
    with open(args.outfile, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"{args.infile} -> {args.outfile}")
    print(f"  espèces (ordre POTCAR): {dict(zip(present, counts))}"
          + ("" if fixed_source is None else f" | figes: {fixed_source} ({int(fixed.sum())} atomes)"))


if __name__ == "__main__":
    main()
