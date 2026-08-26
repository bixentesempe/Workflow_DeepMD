#!/usr/bin/env python3
"""
============================================================================
 ÉTAPE 1 — PRÉPARATION DU CACHE AIMD
============================================================================

 À QUOI ÇA SERT
 --------------
 Lit des trajectoires AIMD VASP (vasprun.xml), applique une série de
 vérifications de qualité, et écrit un cache par groupe (.traj et/ou .npz).

 UTILISATION RAPIDE
 ------------------
   # Commande type (cas d'usage le plus courant : tout traiter d'un dossier
   # d'entrée vers un dossier de sortie)
   python AIMD_process.py -i chemin/AIMD/ -o chemin/de/sortie --all-groups

   # 1. Voir ce qui serait traité, sans rien lire  (À FAIRE EN PREMIER)
   python AIMD_process.py -i chemin/AIMD/ --all-groups --dry-run

   # 2. Lancer sur un groupe précis
   python AIMD_process.py -i chemin/AIMD/ -o chemin/de/sortie --groups 275meV_1O

   # 3. Lancer sur tous les groupes trouvés dans -i
   python AIMD_process.py -i chemin/AIMD/ -o chemin/de/sortie --all-groups

   # 4. Variantes courantes
   python AIMD_process.py -i chemin/AIMD/ -o chemin/de/sortie --groups 100meV 200meV --stride 5
   python AIMD_process.py -i chemin/AIMD/ -o chemin/de/sortie --groups 100meV --format both
   python AIMD_process.py -i chemin/AIMD/ -o chemin/de/sortie --groups 100meV --no-truncate

   # 5. Toutes les options
   python AIMD_process.py --help

 CE QUE FAIT UNE TRAJECTOIRE POUR ÊTRE GARDÉE : LES 4 ÉTAPES, DANS L'ORDRE
 --------------------------------------------------------------------------
 Chaque vasprun.xml passe par ces étapes, DANS CET ORDRE. Dès qu'une étape
 rejette (ou tronque), les étapes suivantes ne portent que sur ce qui reste.

   ÉTAPE 1 — Cohérence topologique
     L'ordre des symboles chimiques doit être identique à la première
     trajectoire valide du groupe (référence). Sinon : trajectoire écartée
     (mélange d'espèces ou de systèmes différents dans le même groupe).

   ÉTAPE 2 — Événement physique (dissociation / rebond / piégeage)
     Détecte le premier événement qui rend la suite de la trajectoire sans
     intérêt physique, et tronque juste avant :
       - DISSOCIATION : la molécule H2 se casse (d(H-H) trop grand).
       - REBOND       : H2 remonte au-dessus de sa hauteur de départ
                        (repart de la surface, plus rien à apprendre après).
       - PIÉGEAGE     : ni l'un ni l'autre ne se produit → H2 reste adsorbé
                        sur la surface toute la trajectoire, qui est alors
                        gardée dans son intégralité (event='none').
     Voir la section CRITÈRES DE TRONCATURE ci-dessous pour le détail.

   ÉTAPE 3 — Conservation de l'énergie, SUR LE SEGMENT GARDÉ (après étape 2)
     On ne juge pas la trajectoire entière : seules les frames retenues à
     l'étape 2 sont contrôlées. Une trajectoire peut être impeccable jusqu'à
     la dissociation puis dériver ensuite (H rapides, SCF qui décroche, H2
     qui traverse la boîte) sans que ça invalide les frames d'avant.
       --energy-fail cut (défaut) : si la conservation décroche quand même
           à l'intérieur du segment, on coupe juste avant et on garde le
           début, au lieu de tout jeter. Un segment de moins de
           --min-frames-kept frames est écarté (trop court pour apporter
           quoi que ce soit).
       --energy-fail drop : tout-ou-rien, la trajectoire entière est jetée
           si la conservation décroche n'importe où dans le segment.

   ÉTAPE 4 — Stride temporel
     Sous-échantillonnage final : 1 frame gardée sur --stride.

   Pas de pré-filtre grossier avant le parsing ASE : le parsing tourne donc
   sur toutes les trajectoires, y compris celles qui seront rejetées par la
   suite. C'est le prix à payer pour ne jamais perdre un segment initial
   physiquement correct à cause d'une dérive survenue après l'événement
   (étape 2) — seul le jugement fin de l'étape 3, sur le segment réellement
   gardé, décide du sort de chaque frame.

 CRITÈRES DE TRONCATURE [étape 2]
 ---------------------------------
   z_com  = <z(H)> - Z_SURF     (hauteur du centre de masse H2 / surface)
   z_init = z_com au 1er step

   REBOND        : z_com est passé sous z_init, PUIS repasse au-dessus.
                   Le signe de la vitesse est implicite : recroiser z_init
                   vers le haut après être descendu impose de monter.
                   --z-rebound-tol ajoute une hystérésis (défaut 0 = strict).
                   H2 au-dessus de son point de départ = parti, plus rien
                   à apprendre après.
   DISSOCIATION  : d(H-H) > --d-hh-dissoc, en minimum image convention.

   Le premier des deux qui se déclenche gagne. Aucun des deux -> traj gardée
   entière (event='none').

   GARDE-FOU : si d(H-H) > seuil dès les --n-min-frames premières frames, H2
   n'est pas lié au départ -> traj REJETÉE (event='start_dissociated').
   Sans ça, le masque anti-bruit dissoc[:n_min]=False laisserait passer des
   frames très au-dessus du seuil.

   ON COUPE AVANT LA FRAME DÉCLENCHEUSE (défaut). La dernière frame gardée
   est celle qui précède le franchissement. Comme on coupe au PREMIER des deux
   événements, toutes les frames cachées vérifient simultanément :
        z_com <= z_init   ET   d(H-H) <= --d-hh-dissoc
   --keep-event-frame rétablit l'ancien comportement (frame déclencheuse
   incluse, donc 1 frame au-dessus du seuil par traj).

 ÉNERGIES : DEUX QUANTITÉS DIFFÉRENTES, NE PAS CONFONDRE
 -------------------------------------------------------
   Étape 3 (conservation) : <i name="total"> lu par ElementTree = E_pot + E_cin,
                    la quantité conservée en NVE. JAMAIS écrite dans le cache.
                    Absente des calculs statiques -> traj comptée 'unchecked'.
   Label du cache : atoms.get_potential_energy() via ASE = E0 (extrapolée
                    sigma->0).

 HYPOTHÈSE GÉOMÉTRIQUE (Z_SURF)
 ------------------------------
   Le repli PBC se fait en coordonnées fractionnaires -> valable pour toute
   orientation de boîte, y compris triclinique, vecteur c incliné, etc.
   MAIS la règle "frac_z > 0.5 -> rabattre vers le bas" suppose que LE SLAB
   EST DANS LA MOITIÉ BASSE DE LA CELLULE. si Slab centré -> à revoir.

 SORTIES
 -------
   <cache-dir>/<groupe>.traj    liste d'Atoms ASE + SinglePointCalculator
                                lecture : ase.io.read('g.traj', index=':')
                                métadonnées : atoms.info['source_dir'|
                                              'frame_in_traj'|'group_name']
   <cache-dir>/<groupe>.npz     coords, forces, energies, cells, symbols,
                                source_dir, frame_in_traj
   <cache-dir>/infos/config.json             config EXACTE du run (à garder !)
   <cache-dir>/infos/cache_summary.csv       bilan par groupe
   <cache-dir>/infos/econs_rejected.csv      trajectoires rejetées à l'étape 3
                                              (conservation de l'énergie, sur
                                              le segment gardé) ; colonne
                                              `scope` toujours 'segment'
   <cache-dir>/infos/truncation_details.csv  détail troncature (étape 2) par trajectoire

 ARBORESCENCE ATTENDUE EN ENTRÉE
 -------------------------------
   MAIN_DIR/
   ├── 275meV_1O/                <- un "groupe"
   │   ├── H2_W_300_f001/vasprun.xml
   │   ├── H2_W_300_f002/vasprun.xml
   │   └── ...
   └── 100meV/ ...

 PIÈGES À NE PAS OUBLIER
 -----------------------
   * Par défaut AUCUN groupe n'est traité : il faut --groups ou --all-groups.
   * config.json contient les seuils du run : sans lui tu ne sauras plus
     dans 6 mois avec quel --max-etot-step un cache a été produit.
   * .npz se relit avec np.load(path)  (pas besoin d'allow_pickle).
============================================================================
"""

import os
import re
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm


# ============================================================
#              VALEURS PAR DÉFAUT (surchargées par CLI)
# ============================================================

DEF_MAIN_DIR   = "./"
DEF_CACHE_DIR  = "./AIMD_processed"
DEF_PATTERN    = "*/vasprun.xml"
DEF_FORMAT     = "traj"          # 'npz' | 'traj' | 'both'
DEF_STRIDE     = 1               # 1 = toutes les frames

# Groupes à ignorer même avec --all-groups
DEF_EXCLUDE    = [r'test', r'^copy', r'_PBE$']

# --- Étape 3 : conservation de l'énergie (eV), jugée sur le segment gardé
#     après l'étape 2 (troncature) ---
DEF_MAX_ETOT_DRIFT = 0.10        # dérive globale : max(E_tot) - min(E_tot)
DEF_MAX_ETOT_STEP  = 0.05        # saut inter-step : |E_tot[i+1] - E_tot[i]| (plus strict)

# Que faire quand la conservation décroche sur le segment gardé :
#   'cut'  = couper juste avant le décrochage et garder le début  [défaut]
#   'drop' = jeter toute la trajectoire (tout-ou-rien)
DEF_ENERGY_FAIL   = 'cut'
# Après coupe, un segment plus court que ça est écarté : quelques frames
# initiales quasi identiques n'apportent rien et sur-représentent le départ.
DEF_MIN_FRAMES_KEPT = 10

# --- Troncature physique ---
DEF_SURFACE_SYMBOL = 'W'         # atome définissant Z_SURF
DEF_LAYER_GAP      = 1.5         # gap inter-couches (Å) pour auto-détection
DEF_Z_REBOUND_TOL  = 0.0         # hystérésis (Å) autour de z_init
DEF_D_HH_DISSOC    = 2.3         # seuil dissociation H-H (Å)
DEF_N_MIN_FRAMES   = 5           # frames initiales ignorées pour la détection
DEF_KEEP_EVENT_FRAME = False     # False = couper AVANT la frame déclencheuse


# ============================================================
#                      SCAN DES GROUPES
# ============================================================

def list_groups(main_dir, include=None, exclude_patterns=None):
    """Liste les sous-dossiers (= groupes) à traiter."""
    if not os.path.isdir(main_dir):
        raise FileNotFoundError(f"MAIN_DIR introuvable : {main_dir}")

    subdirs = sorted(
        d for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    )
    if include:
        missing = set(include) - set(subdirs)
        if missing:
            print(f"  ⚠  groupes demandés introuvables : {sorted(missing)}")
        subdirs = [d for d in subdirs if d in include]
    for pat in (exclude_patterns or []):
        subdirs = [d for d in subdirs if not re.search(pat, d)]
    return subdirs


# ============================================================
#                     LECTURE DES FICHIERS
# ============================================================

def read_total_energies_xml(vasprun_path):
    """E_tot (= E_pot + E_cin) de chaque step, lu dans vasprun.xml.

    Structure XML :
      <calculation>
        <scstep> ... <energy>...</energy> </scstep>   <- SCF intermédiaires
        <energy>                                       <- ÉNERGIE FINALE
          <i name="kinetic"> ... </i>
          <i name="total">   ... </i>                  <- ce qu'on lit
        </energy>
      </calculation>

    On ne lit que le <energy> enfant DIRECT de <calculation>.
    Retourne un array (N_steps,) en eV, ou None si absent/illisible.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(vasprun_path).getroot()
    except Exception:
        return None

    e_tot = []
    for calc in root.iter('calculation'):
        blocks = calc.findall('energy')          # enfants directs seulement
        if not blocks:
            continue
        tag = blocks[-1].find('i[@name="total"]')
        if tag is not None:
            try:
                e_tot.append(float(tag.text))
            except (ValueError, TypeError):
                pass
    return np.array(e_tot) if e_tot else None


def read_vasprun_safe(path):
    """Frames ASE valides (energy + forces présentes). [] si erreur."""
    try:
        frames = read(path, index=':')
    except Exception as exc:
        tqdm.write(f"  ❌ lecture impossible {path}: {exc}")
        return []

    valid = []
    for atoms in frames:
        if atoms.calc is None:
            continue
        res = atoms.calc.results
        if 'energy' in res and 'forces' in res:
            valid.append(atoms)
    return valid


# ============================================================
#          ÉTAPE 3 — CONSERVATION DE L'ÉNERGIE (E_tot)
# ============================================================

def check_etot(e_tot, cfg, scope='segment'):
    """Verdict tout-ou-rien de l'étape 3 sur un profil E_tot.

    `e_tot` est le segment gardé après troncature (étape 2). `scope` n'est
    qu'une étiquette reportée dans le CSV ('segment' toujours, en pratique).

    Retourne (verdict, detail) :
      verdict : 'ok' | 'reject_global' | 'reject_step' | 'unchecked'
      detail  : dict pour le CSV, ou None
    """
    if e_tot is None or len(e_tot) < 2:
        return 'unchecked', None

    base = {
        'scope': scope,
        'n_steps': len(e_tot),
        'E_first': float(e_tot[0]),
        'E_last':  float(e_tot[-1]),
        'E_min':   float(e_tot.min()),
        'E_max':   float(e_tot.max()),
    }

    # --- Dérive globale : max(E_tot) - min(E_tot) ---
    if cfg['check_global']:
        drift = float(e_tot.max() - e_tot.min())
        if drift > cfg['max_drift']:
            return 'reject_global', {
                **base, 'check': 'global',
                'drift_eV': drift, 'worst_step': -1,
            }

    # --- Saut inter-step : |E_tot[i+1] - E_tot[i]| ---
    if cfg['check_step']:
        diffs = np.abs(np.diff(e_tot))
        worst = float(diffs.max())
        if worst > cfg['max_step']:
            return 'reject_step', {
                **base, 'check': 'per_step',
                'drift_eV': worst, 'worst_step': int(diffs.argmax()),
            }

    return 'ok', None


def find_energy_break(e_tot, cfg):
    """Première frame à partir de laquelle la conservation décroche.

    Retourne (n_keep, cause) :
      n_keep = nombre de frames à garder depuis le début (None = tout bon)
      cause  = 'per_step' | 'global' | None

    per_step : |E[i+1]-E[i]| > seuil  -> la frame i+1 est suspecte (SCF non
               convergé : forces fausses), on garde [0..i].
    global   : l'amplitude cumulée max-min sur [0..j] dépasse le seuil -> on
               garde [0..j-1]. Critère glissant, pas global a posteriori :
               c'est ce qui permet de dater le décrochage.
    """
    n = len(e_tot)
    if n < 2:
        return None, None

    cuts = []
    if cfg['check_step']:
        bad = np.where(np.abs(np.diff(e_tot)) > cfg['max_step'])[0]
        if len(bad):
            cuts.append((int(bad[0]) + 1, 'per_step'))
    if cfg['check_global']:
        cmax = np.maximum.accumulate(e_tot)
        cmin = np.minimum.accumulate(e_tot)
        bad = np.where((cmax - cmin) > cfg['max_drift'])[0]
        if len(bad):
            cuts.append((int(bad[0]), 'global'))

    if not cuts:
        return None, None
    n_keep, cause = min(cuts)
    return max(n_keep, 0), cause


# ============================================================
#     ÉTAPE 2 — ÉVÉNEMENT PHYSIQUE (dissociation / rebond / piégeage)
# ============================================================

def _compute_zsurf(coords_frame, symbols, cell,
                   surface_symbol='W', layer_gap=1.5):
    """Z cartésien moyen de la couche de surface, pour 1 frame.

    Le repli PBC se fait en coordonnées FRACTIONNAIRES : aucune hypothèse
    sur l'orientation de la boîte (triclinique OK, vecteur c incliné OK,
    convention non triangulaire OK).

    Hypothèse restante, irréductible : le slab occupe la moitié basse de la
    cellule (règle frac_z > 0.5 -> rabattre d'une période vers le bas).
    """
    surf_list = [surface_symbol] if isinstance(surface_symbol, str) else surface_symbol
    mask = np.isin(symbols, surf_list)
    if not mask.any():
        raise ValueError(f"aucun atome {surface_symbol} trouvé")

    cell = np.asarray(cell, dtype=np.float64)
    pos  = coords_frame[mask].astype(np.float64)

    # cartésien -> fractionnaire :  pos = frac @ cell  <=>  cell.T @ frac.T = pos.T
    frac = np.linalg.solve(cell.T, pos.T).T
    frac[:, 2] -= np.floor(frac[:, 2])       # wrap dans [0, 1)
    frac[frac[:, 2] > 0.5, 2] -= 1.0         # rabat le haut vers le bas
    z = (frac @ cell)[:, 2]

    z_sorted = np.sort(z)
    big = np.where(np.diff(z_sorted) > layer_gap)[0]
    top = z_sorted[big[-1] + 1:] if len(big) else z_sorted
    if len(top) < 3:                         # auto-détection ratée -> repli
        top = z_sorted[-min(8, len(z_sorted)):]
    return float(top.mean())


def _d_HH_mic(pos_H, cells):
    """Distance H-H avec minimum image convention, sur toute la trajectoire.

    Une seule inversion de matrice si la cellule est constante (cas NVE).
    """
    delta = (pos_H[:, 1, :] - pos_H[:, 0, :]).astype(np.float64)
    cells64 = cells.astype(np.float64)

    if np.allclose(cells64, cells64[0]):
        inv = np.linalg.inv(cells64[0])
        frac = delta @ inv
        frac -= np.round(frac)
        delta_mic = frac @ cells64[0]
    else:
        inv = np.linalg.inv(cells64)
        frac = np.einsum('ni,nij->nj', delta, inv)
        frac -= np.round(frac)
        delta_mic = np.einsum('ni,nij->nj', frac, cells64)

    return np.linalg.norm(delta_mic, axis=1)


def detect_truncation_index(frames, symbols, cfg):
    """Premier événement physique sur UNE trajectoire (étape 2).

    REBOND       : z_com passe sous z_init, puis repasse au-dessus.
    DISSOCIATION : d(H-H) > cfg['d_dissoc'].
    PIÉGEAGE     : ni l'un ni l'autre ne se produit -> H2 reste adsorbé sur
                   la surface toute la trajectoire (event_type='none'),
                   qui est alors gardée dans son intégralité.

    Retourne (keep_until, event_type, t_event, z_init) :
      keep_until : dernier indice à garder
                   None = pas de coupe (garder tout, cas piégeage)
                   -1   = ne rien garder (traj rejetée)
      event_type : 'rebond' | 'dissociation' | 'none' (piégeage) |
                   'start_dissociated'
      t_event    : indice de l'événement (-1 si aucun)
      z_init     : hauteur initiale du COM H2 au-dessus de la surface (Å)
    """
    n_min = cfg['n_min_frames']
    if len(frames) <= n_min:
        return None, 'none', -1, None

    H_idx = np.where(symbols == 'H')[0]
    if len(H_idx) != 2:
        return None, 'none', -1, None

    coords = np.array([a.get_positions() for a in frames], dtype=np.float64)
    cells  = np.array([np.array(a.get_cell()) for a in frames], dtype=np.float64)

    z_surf = _compute_zsurf(coords[0], symbols, cells[0],
                            surface_symbol=cfg['surface_symbol'],
                            layer_gap=cfg['layer_gap'])

    pos_H  = coords[:, H_idx, :]
    z_com  = pos_H[:, :, 2].mean(axis=1) - z_surf
    z_init = float(z_com[0])
    d_HH   = _d_HH_mic(pos_H, cells)

    # --- GARDE-FOU : H2 doit démarrer LIÉ ---
    # dissoc[:n_min] = False masque les 1res frames contre le bruit. Si H2 y
    # est déjà au-delà du seuil, ce masque laisserait passer des frames très
    # au-dessus de d_dissoc (H2 déjà cassé au départ = traj non exploitable).
    if d_HH[:max(n_min, 1)].max() > cfg['d_dissoc']:
        return -1, 'start_dissociated', 0, z_init

    # --- Rebond : descendu sous z_init, PUIS repassé au-dessus ---
    # Le signe de la vitesse est implicite : recroiser z_init vers le haut
    # après être descendu impose dz/dt > 0. Pas besoin de dériver z(t).
    tol       = cfg['z_rebound_tol']
    descended = np.minimum.accumulate(z_com) < (z_init - tol)
    rebound   = descended & (z_com > z_init + tol)
    rebound[:n_min] = False

    # --- Dissociation ---
    dissoc = d_HH > cfg['d_dissoc']
    dissoc[:n_min] = False

    t_reb = int(rebound.argmax()) if rebound.any() else None
    t_dis = int(dissoc.argmax())  if dissoc.any()  else None

    if t_reb is None and t_dis is None:
        return None, 'none', -1, z_init
    if t_dis is None or (t_reb is not None and t_reb <= t_dis):
        t_event, event_type = t_reb, 'rebond'
    else:
        t_event, event_type = t_dis, 'dissociation'

    keep_until = t_event if cfg['keep_event_frame'] else t_event - 1
    return keep_until, event_type, t_event, z_init


# ============================================================
#                  TRAITEMENT D'UN GROUPE
# ============================================================

def _make_atoms(atoms, energy, forces, source_dir, frame_in_traj, group_name):
    """Copie légère d'`atoms` avec SinglePointCalculator + métadonnées."""
    new = atoms.copy()
    new.calc = SinglePointCalculator(new, energy=float(energy),
                                     forces=np.asarray(forces))
    new.info['source_dir']    = str(source_dir)
    new.info['frame_in_traj'] = int(frame_in_traj)
    new.info['group_name']    = str(group_name)
    return new


def process_group(group_name, group_dir, cfg):
    """Traite un groupe complet. Écrit le .traj en streaming si demandé.

    Retourne un dict de résultats, ou None si rien de valide.
    """
    files = sorted(glob.glob(os.path.join(group_dir, cfg['pattern'])))
    print(f"\n[{group_name}] {len(files)} fichiers vasprun.xml trouvés")
    if not files:
        return None

    want_npz  = cfg['format'] in ('npz', 'both')
    want_traj = cfg['format'] in ('traj', 'both')

    traj_path   = os.path.join(cfg['cache_dir'], f"{group_name}.traj")
    traj_writer = Trajectory(traj_path, 'w') if want_traj else None

    coords_all, forces_all, energies_all, cells_all = [], [], [], []
    source_all, frame_all = [], []
    e_min, e_max, e_sum, e_sq, n_kept = np.inf, -np.inf, 0.0, 0.0, 0

    symbols_ref = None
    n_raw = n_trunc = n_topo = 0
    n_rej_global = n_rej_step = n_unchecked = n_mismatch = 0
    n_frames_cut_energy = 0
    trunc_stats = {'rebond': 0, 'dissociation': 0, 'none': 0,
                   'start_dissociated': 0}
    trunc_details, econs_details = [], []

    try:
        for vfile in tqdm(files, desc=f'  {group_name}', unit='traj'):
            traj_dir = os.path.basename(os.path.dirname(vfile))

            # ── E_tot : lecture XML légère, AVANT le parsing ASE ──
            # Simplement mise de côté pour l'étape 3 (jugée sur le segment
            # gardé, après troncature à l'étape 2) : aucun rejet ici, pour ne
            # jamais perdre un segment initial correct à cause d'une dérive
            # survenue plus tard dans la trajectoire.
            e_tot = read_total_energies_xml(vfile)
            if e_tot is None or len(e_tot) < 2:
                n_unchecked += 1
                tqdm.write(f"  ⚠  {traj_dir}: E_tot illisible → non contrôlée")

            # ── Parsing ASE ──
            frames = read_vasprun_safe(vfile)
            if not frames:
                continue
            n_raw += len(frames)

            if e_tot is not None and len(e_tot) != len(frames):
                n_mismatch += 1
                tqdm.write(f"  ⚠  {traj_dir}: {len(e_tot)} E_tot vs "
                           f"{len(frames)} frames ASE (steps incomplets)")

            # ── ÉTAPE 1 : cohérence topologique (ordre des symboles) ──
            sym = list(frames[0].get_chemical_symbols())
            if symbols_ref is None:
                symbols_ref = sym
            elif sym != symbols_ref:
                n_topo += len(frames)
                tqdm.write(f"  ⚠  {traj_dir}: ordre des symboles ≠ référence → skip")
                continue

            # ── ÉTAPE 2 : événement physique (dissociation / rebond / piégeage) ──
            n_before = len(frames)
            if cfg['truncate']:
                sym_arr = np.array(sym, dtype='<U2')
                keep_until, event, t_event, z_init = detect_truncation_index(
                    frames, sym_arr, cfg)

                # keep_until : None = tout garder | -1 = tout jeter | n = [:n+1]
                if keep_until is not None:
                    frames = frames[:keep_until + 1] if keep_until >= 0 else []
                    n_trunc += n_before - len(frames)
                    if cfg['verbose_trunc']:
                        tqdm.write(f"    {traj_dir}: {event} à t={t_event}, "
                                   f"gardé {len(frames)}/{n_before}")

                trunc_stats[event] = trunc_stats.get(event, 0) + 1
                trunc_details.append({
                    'traj': traj_dir, 'event': event, 't_event': t_event,
                    'z_init': z_init, 'n_kept': len(frames), 'n_total': n_before,
                })

                if event == 'start_dissociated':
                    tqdm.write(f"  ⚠  {traj_dir}: d(H-H) > {cfg['d_dissoc']} Å "
                               f"dès le départ → traj REJETÉE")
                if not frames:
                    continue

            # ── ÉTAPE 3 : conservation de l'énergie, sur le segment gardé (après étape 2) ──
            # Une trajectoire peut être physiquement impeccable jusqu'à la
            # dissociation puis dériver ensuite (H rapides, SCF qui décroche,
            # H2 qui traverse la boîte). Juger la traj entière jetterait des
            # frames parfaitement exploitables.
            if e_tot is not None:
                n_seg = len(frames)
                if len(e_tot) >= n_seg:
                    e_seg = e_tot[:n_seg]
                else:
                    e_seg = e_tot          # désalignement déjà signalé plus haut
                if cfg['energy_fail'] == 'cut':
                    # --- COUPE au décrochage, on garde l'amont ---
                    n_keep, cause = find_energy_break(e_seg, cfg)
                    if n_keep is not None:
                        n_dropped = n_seg - n_keep
                        too_short = n_keep < cfg['min_frames_kept']
                        n_frames_cut_energy += n_dropped if not too_short else n_seg
                        if cause == 'global':
                            n_rej_global += n_dropped if not too_short else n_seg
                        else:
                            n_rej_step += n_dropped if not too_short else n_seg
                        econs_details.append({
                            'traj': traj_dir, 'scope': 'segment',
                            'event': event if cfg['truncate'] else '',
                            'check': cause, 'action': 'too_short' if too_short else 'cut',
                            'n_steps': n_seg, 'n_frames_kept': 0 if too_short else n_keep,
                            'drift_eV': float(np.ptp(e_seg)),
                            'worst_step': int(n_keep),
                            'E_first': float(e_seg[0]), 'E_last': float(e_seg[-1]),
                            'E_min': float(e_seg.min()), 'E_max': float(e_seg.max()),
                        })
                        if trunc_details and trunc_details[-1]['traj'] == traj_dir:
                            trunc_details[-1]['n_kept'] = 0 if too_short else n_keep
                            trunc_details[-1]['energy_break'] = cause
                        if cfg['verbose_etot']:
                            tqdm.write(f"    {traj_dir}: {cause} à la frame {n_keep} "
                                       f"→ {'segment trop court, REJET'
                                            if too_short else
                                            f'gardé {n_keep}/{n_seg}'}")
                        if too_short:
                            continue
                        frames = frames[:n_keep]
                else:
                    # --- tout-ou-rien ---
                    v2, d2 = check_etot(e_seg, cfg, scope='segment')
                    if v2.startswith('reject'):
                        if v2 == 'reject_global':
                            n_rej_global += n_seg
                        else:
                            n_rej_step += n_seg
                        econs_details.append({'traj': traj_dir, 'action': 'dropped',
                                              'event': event if cfg['truncate'] else '',
                                              **d2})
                        if trunc_details and trunc_details[-1]['traj'] == traj_dir:
                            trunc_details[-1]['n_kept'] = 0
                            trunc_details[-1]['dropped_energy'] = True
                        if cfg['verbose_etot']:
                            tqdm.write(f"    {traj_dir}: {d2['check']} = "
                                       f"{d2['drift_eV']:.4f} eV sur le segment gardé "
                                       f"({n_seg} frames) → REJET")
                        continue

            # ── ÉTAPE 4 : stride ──
            for li in range(0, len(frames), cfg['stride']):
                atoms = frames[li]
                E = atoms.get_potential_energy()
                F = atoms.get_forces(apply_constraint=False)

                if want_traj:
                    traj_writer.write(_make_atoms(atoms, E, F, traj_dir,
                                                  li, group_name))
                if want_npz:
                    coords_all.append(atoms.get_positions())
                    forces_all.append(F)
                    energies_all.append(E)
                    cells_all.append(np.array(atoms.get_cell()))
                    source_all.append(traj_dir)
                    frame_all.append(li)

                n_kept += 1
                e_min = min(e_min, E)
                e_max = max(e_max, E)
                e_sum += E
                e_sq  += E * E
    finally:
        if traj_writer is not None:
            traj_writer.close()

    if n_kept == 0:
        print(f"  [{group_name}] aucune frame valide après filtres")
        if want_traj and os.path.exists(traj_path):
            os.remove(traj_path)
        return None

    e_mean = e_sum / n_kept
    e_std  = float(np.sqrt(max(e_sq / n_kept - e_mean ** 2, 0.0)))

    # ── Écriture .npz ──
    if want_npz:
        npz_path = os.path.join(cfg['cache_dir'], f"{group_name}.npz")
        np.savez(
            npz_path,
            symbols       = np.array(symbols_ref, dtype='<U2'),
            coords        = np.array(coords_all,   dtype=np.float32),
            forces        = np.array(forces_all,   dtype=np.float32),
            energies      = np.array(energies_all, dtype=np.float64),
            cells         = np.array(cells_all,    dtype=np.float32),
            # '<U64' et pas dtype=object : sinon np.load exige allow_pickle
            source_dir    = np.array(source_all,   dtype='<U64'),
            frame_in_traj = np.array(frame_all,    dtype=np.int32),
            n_atoms       = len(symbols_ref),
            group_name    = group_name,
        )
        print(f"  → .npz  : {npz_path}  ({os.path.getsize(npz_path)/1e6:.1f} MB)")

    if want_traj:
        print(f"  → .traj : {traj_path}  ({os.path.getsize(traj_path)/1e6:.1f} MB)")

    # ── Résumé console ──
    print(f"  [{group_name}] conservé {n_kept}/{n_raw}  "
          f"(trunc={n_trunc}, topo={n_topo}, "
          f"econs_global={n_rej_global}, econs_step={n_rej_step}, "
          f"frames_coupees_energie={n_frames_cut_energy})  "
          f"E ∈ [{e_min:.3f}, {e_max:.3f}] eV  N_atoms={len(symbols_ref)}")
    if n_unchecked:
        print(f"  [{group_name}] ⚠  {n_unchecked} traj sans E_tot lisible "
              f"(passées SANS contrôle de conservation)")
    if n_mismatch:
        print(f"  [{group_name}] ⚠  {n_mismatch} traj avec désalignement "
              f"E_tot/frames")
    if cfg['truncate']:
        ts = trunc_stats
        print(f"  [{group_name}] troncatures : rebond={ts['rebond']}, "
              f"dissociation={ts['dissociation']}, none={ts['none']}, "
              f"start_dissoc={ts['start_dissociated']} "
              f"(sur {sum(ts.values())} traj)")

    return {
        'summary': {
            'group': group_name,
            'n_raw': n_raw, 'n_kept': n_kept,
            'n_econs_global': n_rej_global, 'n_econs_step': n_rej_step,
            'n_econs': n_rej_global + n_rej_step,
            'n_econs_unchecked': n_unchecked,
            'n_etot_mismatch': n_mismatch,
            'n_trunc': n_trunc, 'n_topo': n_topo,
            'n_frames_cut_energy': n_frames_cut_energy,
            'trunc_rebond': trunc_stats['rebond'],
            'trunc_dissoc': trunc_stats['dissociation'],
            'trunc_none':   trunc_stats['none'],
            'trunc_start_dissoc': trunc_stats['start_dissociated'],
            'n_atoms': len(symbols_ref),
            'symbols': ''.join(sorted(set(symbols_ref))),
            'E_min': e_min, 'E_max': e_max,
            'E_mean': e_mean, 'E_std': e_std,
        },
        'trunc_details': trunc_details,
        'econs_details': econs_details,
    }


# ============================================================
#                            CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Étape 1 — prépare le cache AIMD (.traj / .npz) "
                    "à partir de vasprun.xml VASP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Exemple : python AIMD_process.py --groups 275meV_1O --format both",
    )

    g = p.add_argument_group('chemins')
    g.add_argument('-i', '--main-dir',  default=DEF_MAIN_DIR,
                   help='racine contenant les dossiers de groupes')
    g.add_argument('-o', '--cache-dir', default=DEF_CACHE_DIR,
                   help='dossier de sortie du cache')
    g.add_argument('--pattern',   default=DEF_PATTERN,
                   help='glob des vasprun.xml dans un groupe')

    g = p.add_argument_group('sélection des groupes')
    sel = g.add_mutually_exclusive_group()
    sel.add_argument('--groups', nargs='+', metavar='G',
                     help='groupes à traiter (ex : 100meV 275meV_1O)')
    sel.add_argument('--all-groups', action='store_true',
                     help='traiter tous les groupes de --main-dir')
    g.add_argument('--exclude', nargs='+', default=DEF_EXCLUDE, metavar='RE',
                   help='regex de groupes à exclure')

    g = p.add_argument_group('sortie')
    g.add_argument('--format', choices=['npz', 'traj', 'both'], default=DEF_FORMAT)
    g.add_argument('--stride', type=int, default=DEF_STRIDE,
                   help='[étape 4] garder 1 frame sur N')

    g = p.add_argument_group("étape 3 — conservation de l'énergie (sur le segment gardé)")
    g.add_argument('--max-etot-drift', type=float, default=DEF_MAX_ETOT_DRIFT,
                   metavar='eV', help='seuil max(E_tot)-min(E_tot)')
    g.add_argument('--max-etot-step', type=float, default=DEF_MAX_ETOT_STEP,
                   metavar='eV', help='seuil |E_tot[i]-E_tot[i-1]|')
    g.add_argument('--energy-fail', choices=['cut', 'drop'],
                   default=DEF_ENERGY_FAIL,
                   help="que faire si l'énergie décroche : 'cut' = couper "
                        "juste avant et garder le début (défaut) ; 'drop' = "
                        "jeter toute la trajectoire")
    g.add_argument('--min-frames-kept', type=int, default=DEF_MIN_FRAMES_KEPT,
                   help='après coupe, un segment plus court que ça est écarté')
    g.add_argument('--no-etot-global', action='store_true',
                   help='désactiver le contrôle de dérive globale')
    g.add_argument('--no-etot-step', action='store_true',
                   help='désactiver le contrôle de saut inter-step')

    g = p.add_argument_group('étape 2 — événement physique (troncature)')
    g.add_argument('--no-truncate', action='store_true',
                   help='désactiver la troncature')
    g.add_argument('--surface-symbol', default=DEF_SURFACE_SYMBOL,
                   help='atome définissant Z_SURF')
    g.add_argument('--layer-gap', type=float, default=DEF_LAYER_GAP,
                   metavar='A', help='gap inter-couches pour auto-détection')
    g.add_argument('--z-rebound-tol', type=float, default=DEF_Z_REBOUND_TOL,
                   metavar='A',
                   help='hystérésis autour de z_init (0 = franchissement strict)')
    g.add_argument('--d-hh-dissoc', type=float, default=DEF_D_HH_DISSOC,
                   metavar='A', help='seuil de dissociation H-H')
    g.add_argument('--n-min-frames', type=int, default=DEF_N_MIN_FRAMES,
                   help='frames initiales ignorées par la détection')
    g.add_argument('--keep-event-frame', action='store_true',
                   default=DEF_KEEP_EVENT_FRAME,
                   help="INCLURE la frame déclencheuse (par défaut on coupe "
                        "juste avant : toutes les frames gardées respectent "
                        "z_com <= z_init ET d_HH <= seuil)")

    g = p.add_argument_group('divers')
    g.add_argument('--dry-run', action='store_true',
                   help='lister groupes et nb de fichiers, puis sortir')
    g.add_argument('--verbose-etot',  action='store_true',
                   help='1 ligne par traj rejetée pour E_tot')
    g.add_argument('--verbose-trunc', action='store_true',
                   help='1 ligne par traj tronquée')

    return p


def args_to_cfg(a):
    """dict de config plat, aussi dumpé dans infos/config.json."""
    return {
        'main_dir':       a.main_dir,
        'cache_dir':      a.cache_dir,
        'pattern':        a.pattern,
        'groups':         a.groups,
        'all_groups':     a.all_groups,
        'exclude':        a.exclude,
        'format':         a.format,
        'stride':         a.stride,
        'check_global':   not a.no_etot_global,
        'check_step':     not a.no_etot_step,
        'max_drift':      a.max_etot_drift,
        'max_step':       a.max_etot_step,
        'energy_fail':     a.energy_fail,
        'min_frames_kept': a.min_frames_kept,
        'truncate':       not a.no_truncate,
        'surface_symbol': a.surface_symbol,
        'layer_gap':      a.layer_gap,
        'z_rebound_tol':  a.z_rebound_tol,
        'd_dissoc':       a.d_hh_dissoc,
        'n_min_frames':   a.n_min_frames,
        'keep_event_frame': a.keep_event_frame,
        'verbose_etot':   a.verbose_etot,
        'verbose_trunc':  a.verbose_trunc,
    }


# ============================================================
#                            MAIN
# ============================================================

def main():
    args = build_parser().parse_args()
    if args.stride < 1:
        raise SystemExit("--stride doit être >= 1")
    cfg = args_to_cfg(args)
    t0  = time.time()
    sep = '=' * 70

    include = None if args.all_groups else args.groups
    if include is None and not args.all_groups:
        raise SystemExit(
            "Aucun groupe sélectionné. Utilise --groups G1 G2  ou  --all-groups.\n"
            "Astuce : commence par  python AIMD_process.py --dry-run --all-groups"
        )

    groups = list_groups(args.main_dir, include=include,
                         exclude_patterns=args.exclude)

    # ── En-tête ──
    print(sep)
    print(f"Racine                 : {args.main_dir}")
    print(f"Cache                  : {args.cache_dir}   [format : {args.format}]")
    print(f"Groupes ({len(groups)})            : {', '.join(groups) if groups else '—'}")
    print(f"Étape 4 — stride       : {args.stride}")
    print(f"Étape 3 — E_tot global : "
          f"{'≤ %.3f eV' % cfg['max_drift'] if cfg['check_global'] else 'désactivé'}")
    print(f"Étape 3 — E_tot/step   : "
          f"{'≤ %.3f eV' % cfg['max_step'] if cfg['check_step'] else 'désactivé'}")
    print(f"Action si décrochage   : "
          f"{'coupe au décrochage (min %d frames)' % cfg['min_frames_kept']
             if cfg['energy_fail'] == 'cut' else 'rejet de la trajectoire'}")
    print(f"Portée contrôle E_tot  : SEGMENT GARDÉ (après étape 2)")
    trunc_msg = ('rebond (retour au-dessus de z_init, tol=%.2f Å) + '
                 'dissoc (d_HH > %.1f Å)' % (cfg['z_rebound_tol'], cfg['d_dissoc'])
                 ) if cfg['truncate'] else 'désactivée'
    print(f"Étape 2 — troncature   : {trunc_msg}")
    print(sep)

    if not groups:
        raise SystemExit("Aucun groupe à traiter — vérifie --groups / --exclude.")

    # ── Dry-run ──
    if args.dry_run:
        print("\nDRY-RUN — aucun fichier ne sera lu ni écrit.\n")
        total = 0
        for g in groups:
            n = len(glob.glob(os.path.join(args.main_dir, g, args.pattern)))
            total += n
            print(f"  {g:24s} : {n:5d} vasprun.xml")
        print(f"\n  {'TOTAL':24s} : {total:5d} vasprun.xml")
        print("\nRelance sans --dry-run pour traiter.")
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    infos_dir = os.path.join(args.cache_dir, 'infos')
    os.makedirs(infos_dir, exist_ok=True)

    # ── Dump de la config (traçabilité) ──
    cfg_path = os.path.join(infos_dir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'groups_resolved': groups, **cfg}, f, indent=2)

    # ── Traitement ──
    summary, all_trunc, all_econs = [], [], []
    for gname in groups:
        res = process_group(gname, os.path.join(args.main_dir, gname), cfg)
        if res is None:
            continue
        summary.append(res['summary'])
        all_trunc += [{'group': gname, **d} for d in res['trunc_details']]
        all_econs += [{'group': gname, **d} for d in res['econs_details']]

    # ── Récap ──
    print(f"\n{sep}\nRÉSUMÉ CACHE\n{sep}")
    df = pd.DataFrame(summary)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(os.path.join(infos_dir, 'cache_summary.csv'), index=False)
    n_total = int(df['n_kept'].sum()) if not df.empty else 0
    print(f"\nTotal frames cachées : {n_total}")
    print(f"Temps total          : {time.time() - t0:.1f}s")
    print(f"Config du run        : {cfg_path}")
    if not df.empty:
        print(f"Résumé CSV           : {os.path.join(infos_dir, 'cache_summary.csv')}")

    if all_econs:
        df_ec = pd.DataFrame(all_econs)
        p_ec  = os.path.join(infos_dir, 'econs_rejected.csv')
        df_ec.to_csv(p_ec, index=False)
        print(f"\nTrajectoires rejetées (conservation E_tot) :")
        for ck, label in [('global', 'global (max-min)'), ('per_step', 'saut inter-step ')]:
            sub = df_ec[df_ec['check'] == ck]
            if len(sub):
                extra = f"  pire step : {int(sub['worst_step'].max())}" if ck == 'per_step' else ""
                print(f"  {label} : {len(sub):4d} traj   "
                      f"moy={sub['drift_eV'].mean():.4f} eV   "
                      f"max={sub['drift_eV'].max():.4f} eV{extra}")
        print(f"  Détails : {p_ec}")

    if all_trunc:
        df_tr = pd.DataFrame(all_trunc)
        p_tr  = os.path.join(infos_dir, 'truncation_details.csv')
        df_tr.to_csv(p_tr, index=False)
        print(f"\nBilan troncature global :")
        for ev in ['rebond', 'dissociation', 'none', 'start_dissociated']:
            print(f"  {ev:15s} : {int((df_tr['event'] == ev).sum())} trajectoires")
        cut = df_tr[~df_tr['event'].isin(['none', 'start_dissociated'])]
        if len(cut):
            mk, mt = cut['n_kept'].mean(), cut['n_total'].mean()
            print(f"  tronquées      : {len(cut)}, frames gardées en moyenne "
                  f"{mk:.0f}/{mt:.0f} ({100*mk/mt:.1f}%)")
        print(f"  Détails : {p_tr}")


if __name__ == "__main__":
    main()
