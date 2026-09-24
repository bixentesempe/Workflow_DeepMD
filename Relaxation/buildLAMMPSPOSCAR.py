#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BuildLAMMPSPoscar.py
=====================
Construit un fichier LAMMPS_POSCAR (data file) pour une surface BCC
(W typiquement) sur laquelle on pourra ensuite faire incider H2 via
GenerateLammpsDatafile_NoZPE.py.

Caracteristiques :
   - Reseau BCC, surface (110) ou (100), parametre de maille reglable
   - Supercellule nx * ny en surface, nlayers couches
   - Vide ajoute en haut (z > z_top_W) pour laisser place a H2
   - Ajout optionnel d'oxygenes en positions cartesiennes
   - Figeage des N couches du bas (couches centrales -> bulk-like)
   - IDs 1 et 2 reserves pour H1, H2 (ajoutes plus tard par le generateur)
     -> les W et O commencent a l'ID 3
   - Header annonce le nombre d'atomes REELLEMENT ecrits (W + O, sans les 2 H
     reserves) : format standard, relisible tel quel par ASE/lammps2poscar.py.
     Les generateurs de Lammps_dynamic (qui ajoutent H1/H2) recalculent ce
     champ eux-memes avant d'ecrire leurs propres data files.

Comment indiquer les atomes fixes ?
   - Option --freeze-bottom N      : figer les N couches du bas
   - Option --freeze-z-below ZMAX  : figer les atomes a z < ZMAX
   (un commentaire "# fixed" est ajoute en fin de ligne dans la section Atoms,
   ce que GenerateLammpsDatafile_NoZPE.py reconnait automatiquement)

Conventions :
   - Surface BCC(110) : cellule de surface = a * (a sqrt(2)), 2 atomes par
     cellule primitive de surface (motif rectangle centre), empilement AB
     decale de (a/2, 0) entre couches successives, dz = a*sqrt(2)/2
   - Surface BCC(100) : cellule de surface = a * a, 1 atome par cellule
     primitive, empilement AB decale de (a/2, a/2), dz = a/2

Usage minimal :
   ./BuildLAMMPSPoscar.py -a 3.2398 --surface 110 --nx 2 --ny 2 --nlayers 5 \
       --vacuum 15 --freeze-bottom 1 -o LAMMPS_POSCAR

Avec des oxygenes adsorbes :
   ./BuildLAMMPSPoscar.py -a 3.2398 --surface 110 --nx 2 --ny 2 --nlayers 5 \
       --vacuum 15 --freeze-bottom 1 \
       --add-O 1.62,2.29,10.5 \
       --add-O 4.86,6.87,10.5 \
       -o LAMMPS_POSCAR
"""
import argparse
import math
import sys
from pathlib import Path

# Masses (g/mol = amu en unites metal)
MASSES = {"H": 1.00794, "O": 15.9994, "W": 183.84}


# ---------------------------------------------------------------------------
# Generation des positions des W
# ---------------------------------------------------------------------------
def build_W110(a, nx, ny, nlayers):
    """
    BCC(110) en supercellule nx*ny, nlayers couches.

    Cellule de surface conventionnelle (rectangle centre) : a * (a*sqrt(2))
    2 atomes par cellule primitive de surface :
       - couche A (layers 0, 2, 4...) : (0, 0) et (a/2, c)
       - couche B (layers 1, 3, 5...) : (a/2, 0) et (0, c)
       avec c = a*sqrt(2)/2

    Empilement avec dz = c entre 2 couches successives.

    Retourne :
       atoms_xyz : liste de (x, y, z, layer_index)
       (lx, ly)  : dimensions XY de la box
       dz        : distance entre couches
    """
    c = a * math.sqrt(2) / 2.0
    lx = nx * a
    ly = ny * 2.0 * c

    atoms = []
    for layer in range(nlayers):
        z = layer * c
        if layer % 2 == 0:
            in_cell = [(0.0, 0.0), (a / 2.0, c)]            # A
        else:
            in_cell = [(a / 2.0, 0.0), (0.0, c)]            # B
        for i in range(nx):
            for j in range(ny):
                for dx, dy in in_cell:
                    x = i * a + dx
                    y = j * 2.0 * c + dy
                    atoms.append((x, y, z, layer))

    return atoms, (lx, ly), c


def build_W100(a, nx, ny, nlayers):
    """
    BCC(100) en supercellule nx*ny, nlayers couches.

    Cellule de surface conventionnelle : a * a (carree)
    1 atome par cellule primitive :
       - couche A : (0, 0)
       - couche B : (a/2, a/2)

    Empilement avec dz = a/2.
    """
    dz = a / 2.0
    lx = nx * a
    ly = ny * a

    atoms = []
    for layer in range(nlayers):
        z = layer * dz
        if layer % 2 == 0:
            in_cell = [(0.0, 0.0)]
        else:
            in_cell = [(a / 2.0, a / 2.0)]
        for i in range(nx):
            for j in range(ny):
                for dx, dy in in_cell:
                    atoms.append((i * a + dx, j * a + dy, z, layer))

    return atoms, (lx, ly), dz


# ---------------------------------------------------------------------------
# Ecriture LAMMPS data
# ---------------------------------------------------------------------------
def write_data(out_path, lx, ly, lz, W_atoms, O_atoms, frozen_ids,
                add_oxygen, comment="(built by BuildLAMMPSPoscar.py)"):
    """
    W_atoms : liste de tuples (x, y, z, layer_idx)
    O_atoms : liste de tuples (x, y, z)
    frozen_ids : set d'IDs (parmi les W et O) a marquer "# fixed"
    add_oxygen : bool. Si True, type 2 = W, type 3 = O.

    IDs 1 et 2 reserves pour H -> on commence a id 3.
    Header annonce n_total = 2 (H) + n_W + n_O.
    """
    n_types = 3 if add_oxygen else 2
    # Header = nombre d'atomes REELLEMENT ecrits (W + O). Les IDs 1/2 restent
    # reserves pour H1/H2 (ajoutes plus tard par le generateur), mais le
    # header ne doit PAS les anticiper : un lecteur strict (ex. ASE, utilise
    # par lammps2poscar.py) lit exactement "n_atoms_total" lignes Atoms et
    # plante sinon. Les generateurs de Lammps_dynamic recalculent de toute
    # facon ce champ avant d'ajouter H1/H2, donc ce changement ne les affecte pas.
    n_atoms_total = len(W_atoms) + len(O_atoms)

    if add_oxygen:
        type_W = 2
        type_O = 3
    else:
        type_W = 2
        type_O = None

    with open(out_path, "w") as f:
        f.write(f"{comment}\n\n")
        f.write(f"{n_atoms_total} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"0.0      {lx:.10f}  xlo xhi\n")
        f.write(f"0.0      {ly:.10f}  ylo yhi\n")
        f.write(f"0.0      {lz:.10f}  zlo zhi\n")
        f.write("0.0 0.0 0.0  xy xz yz\n\n")

        f.write("Masses\n\n")
        f.write(f"    1      {MASSES['H']:.13f} # H\n")
        if add_oxygen:
            f.write(f"    3      {MASSES['O']:.13f} # O\n")
            f.write(f"    2      {MASSES['W']:.13f} # W\n")
        else:
            f.write(f"    2      {MASSES['W']:.13f} # W\n")

        f.write("\nAtoms # atomic\n\n")
        # On ecrit d'abord les O (s'il y en a), puis les W -- IDs 3, 4, ...
        atom_id = 3
        if add_oxygen and O_atoms:
            for ox, oy, oz in O_atoms:
                tag = "  # fixed" if atom_id in frozen_ids else ""
                f.write(f"{atom_id}\t{type_O}\t{ox:.6f}\t{oy:.6f}\t{oz:.6f}{tag}\n")
                atom_id += 1
        for x, y, z, layer in W_atoms:
            tag = "  # fixed" if atom_id in frozen_ids else ""
            f.write(f"{atom_id}\t{type_W}\t{x:.6f}\t{y:.6f}\t{z:.6f}{tag}\n")
            atom_id += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-a", "--lattice", type=float, required=True,
                    help="parametre de maille BCC a (A)")
    ap.add_argument("--surface", choices=["110", "100"], default="110",
                    help="orientation de surface [110]")
    ap.add_argument("--nx", type=int, default=2,
                    help="taille de la supercellule en x [2]")
    ap.add_argument("--ny", type=int, default=2,
                    help="taille de la supercellule en y [2]")
    ap.add_argument("--nlayers", type=int, default=5,
                    help="nombre de couches de W [5]")
    ap.add_argument("--vacuum", type=float, default=20.0,
                    help="vide au-dessus de la surface (A) [20]")

    # Atomes fixes
    ap.add_argument("--freeze-bottom", type=int, default=0,
                    help="nb de couches du bas a figer (0 = aucune) [0]")
    ap.add_argument("--freeze-z-below", type=float, default=None,
                    help="figer tous les atomes a z < ZMAX (en plus de --freeze-bottom)")

    # Oxygenes
    ap.add_argument("--add-O", action="append", default=[],
                    help='ajouter un O en position cartesienne "x,y,z" (en A); '
                         'option repetable')
    ap.add_argument("--O-fixed", action="store_true",
                    help="marquer les O ajoutes comme fixes (par defaut: mobiles)")

    ap.add_argument("-o", "--output", default="LAMMPS_POSCAR",
                    help="fichier de sortie [LAMMPS_POSCAR]")
    args = ap.parse_args()

    # Construction de la surface
    if args.surface == "110":
        W_atoms, (lx, ly), dz = build_W110(
            args.lattice, args.nx, args.ny, args.nlayers
        )
    else:
        W_atoms, (lx, ly), dz = build_W100(
            args.lattice, args.nx, args.ny, args.nlayers
        )

    # Hauteur de la box : z_max_W + vide
    z_max_W = max(z for _, _, z, _ in W_atoms)
    lz = z_max_W + args.vacuum

    # Oxygenes
    O_atoms = []
    for s in args.add_O:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            ap.error(f"--add-O attend 'x,y,z' (recu '{s}')")
        try:
            ox, oy, oz = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            ap.error(f"--add-O : coordonnees non numeriques dans '{s}'")
        O_atoms.append((ox, oy, oz))
    add_oxygen = len(O_atoms) > 0

    # Calcul des IDs fixes
    # Layout d'IDs: 3 = 1er atome ecrit (O en premier si presents, puis W)
    frozen_ids = set()
    n_O = len(O_atoms)

    # 1) IDs des O (si --O-fixed)
    if add_oxygen and args.O_fixed:
        for k in range(n_O):
            frozen_ids.add(3 + k)

    # 2) Couches du bas a figer (--freeze-bottom)
    if args.freeze_bottom > 0:
        bottom_layers = set(range(args.freeze_bottom))
        # IDs des W : commencent apres les O
        for idx, (_, _, _, layer) in enumerate(W_atoms):
            if layer in bottom_layers:
                w_id = 3 + n_O + idx
                frozen_ids.add(w_id)

    # 3) z < ZMAX
    if args.freeze_z_below is not None:
        # Pour les W
        for idx, (_, _, z, _) in enumerate(W_atoms):
            if z < args.freeze_z_below:
                frozen_ids.add(3 + n_O + idx)
        # Pour les O
        for k, (_, _, z) in enumerate(O_atoms):
            if z < args.freeze_z_below:
                frozen_ids.add(3 + k)

    # Avertissement si rien n'est figé (utile pour le slab)
    if not frozen_ids:
        print("ATTENTION: aucun atome fige. Considere --freeze-bottom 1 "
              "(ou plus) pour eviter que toute la dalle se translate "
              "pendant la simulation.", file=sys.stderr)

    # Ecriture
    out = Path(args.output)
    write_data(out, lx, ly, lz, W_atoms, O_atoms, frozen_ids, add_oxygen)

    # Recap
    print(f"[OK] {out} ecrit.")
    print(f"     surface BCC({args.surface}) a = {args.lattice} A")
    print(f"     supercell {args.nx} x {args.ny} x {args.nlayers} couches")
    print(f"     dz entre couches = {dz:.4f} A")
    print(f"     box : {lx:.4f} x {ly:.4f} x {lz:.4f} A "
          f"(vide = {args.vacuum} A au-dessus du W le plus haut)")
    print(f"     {len(W_atoms)} W, {len(O_atoms)} O (IDs >= 3)")
    print(f"     {len(frozen_ids)} atomes fixes  (couches du bas / z<ZMAX / O fixes)")
    print(f"     {len(W_atoms) + len(O_atoms) - len(frozen_ids)} atomes mobiles "
          f"(en plus de H1, H2 qui seront ajoutes)")
    print()
    print("Etape suivante :")
    print(f"   GenerateLammpsDatafile_NoZPE.py -p {out} -n 1000 -E 0.1 -T 45 -P random")


if __name__ == "__main__":
    main()
