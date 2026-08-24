#!/usr/bin/env python3
"""
============================================================================
 STEP 1 — AIMD CACHE PREPARATION
============================================================================

 WHAT THIS IS FOR
 ----------------
 Reads AIMD VASP trajectories (vasprun.xml), applies a series of quality
 checks, and writes a per-group cache (.traj and/or .npz).

 QUICK USAGE
 -----------
   # Typical command (most common use case: process everything from an
   # input folder to an output folder)
   python AIMD_process.py -i path/to/AIMD/ -o path/to/output --all-groups

   # 1. See what would be processed, without reading anything (DO THIS FIRST)
   python AIMD_process.py -i path/to/AIMD/ --all-groups --dry-run

   # 2. Run on one specific group
   python AIMD_process.py -i path/to/AIMD/ -o path/to/output --groups 275meV_1O

   # 3. Run on every group found under -i
   python AIMD_process.py -i path/to/AIMD/ -o path/to/output --all-groups

   # 4. Common variants
   python AIMD_process.py -i path/to/AIMD/ -o path/to/output --groups 100meV 200meV --stride 5
   python AIMD_process.py -i path/to/AIMD/ -o path/to/output --groups 100meV --format both
   python AIMD_process.py -i path/to/AIMD/ -o path/to/output --groups 100meV --no-truncate

   # 5. All options
   python AIMD_process.py --help

 WHAT A TRAJECTORY GOES THROUGH TO BE KEPT: THE 4 STEPS, IN ORDER
 --------------------------------------------------------------------------
 Every vasprun.xml goes through these steps, IN THIS ORDER. As soon as a
 step rejects (or truncates), the following steps only operate on what
 remains.

   STEP 1 — Topological consistency
     The order of chemical symbols must match the first valid trajectory of
     the group (the reference). Otherwise: trajectory discarded (mix of
     species or of different systems within the same group).

   STEP 2 — Physical event (dissociation / rebound / trapping)
     Detects the first event that makes the rest of the trajectory
     physically uninteresting, and truncates right before it:
       - DISSOCIATION : the H2 molecule breaks apart (d(H-H) too large).
       - REBOUND      : H2 rises back above its starting height (leaves the
                        surface, nothing left to learn afterwards).
       - TRAPPING     : neither happens -> H2 stays adsorbed on the surface
                        for the whole trajectory, which is then kept in
                        full (event='none').
     See the TRUNCATION CRITERIA section below for the details.

   STEP 3 — Energy conservation, ON THE KEPT SEGMENT (after step 2)
     The whole trajectory is not judged: only the frames retained at step 2
     are checked. A trajectory can be flawless up to the dissociation and
     then drift afterwards (fast H atoms, SCF failing to converge, H2
     crossing the box) without that invalidating the earlier frames.
       --energy-fail cut (default): if conservation still breaks down
           inside the segment, cut right before it and keep the beginning,
           instead of discarding everything. A segment shorter than
           --min-frames-kept is discarded (too short to be worth anything).
       --energy-fail drop: all-or-nothing, the entire trajectory is
           discarded if conservation breaks down anywhere in the segment.

   STEP 4 — Time stride
     Final subsampling: keep 1 frame every --stride.

   No coarse pre-filter before ASE parsing: parsing therefore runs on every
   trajectory, including the ones that will end up rejected. That's the
   price paid to never lose a physically correct initial segment because of
   a drift that happened after the event (step 2) — only the fine-grained
   judgment of step 3, on the segment actually kept, decides each frame's
   fate.

 TRUNCATION CRITERIA [step 2]
 -----------------------------
   z_com  = <z(H)> - Z_SURF     (height of the H2 center of mass above the
                                 surface)
   z_init = z_com at the 1st step

   REBOUND       : z_com drops below z_init, THEN rises back above it.
                   The sign of the velocity is implicit: crossing back above
                   z_init after having descended requires an upward motion.
                   --z-rebound-tol adds a hysteresis (default 0 = strict).
                   H2 above its starting point = gone, nothing left to learn
                   afterwards.
   DISSOCIATION  : d(H-H) > --d-hh-dissoc, using the minimum image
                   convention.

   Whichever of the two triggers first wins. Neither -> trajectory kept in
   full (event='none').

   SAFEGUARD: if d(H-H) > threshold already within the first --n-min-frames
   frames, H2 was not bonded to begin with -> trajectory REJECTED
   (event='start_dissociated'). Without this, the noise-guard mask
   dissoc[:n_min]=False would let through frames well above the threshold.

   CUT BEFORE THE TRIGGERING FRAME (default). The last frame kept is the
   one right before the crossing. Since the cut happens at the FIRST of the
   two events, every frame that is kept simultaneously satisfies:
        z_com <= z_init   AND   d(H-H) <= --d-hh-dissoc
   --keep-event-frame restores the old behaviour (triggering frame
   included, so 1 frame above threshold per trajectory).

 ENERGIES: TWO DIFFERENT QUANTITIES, DO NOT CONFUSE THEM
 ---------------------------------------------------------
   Step 3 (conservation): <i name="total"> read via ElementTree = E_pot +
                    E_kin, the quantity conserved in NVE. NEVER written to
                    the cache. Absent from static calculations -> trajectory
                    counted as 'unchecked'.
   Cache label     : atoms.get_potential_energy() via ASE = E0 (extrapolated
                    to sigma->0).

 GEOMETRIC ASSUMPTION (Z_SURF)
 ------------------------------
   PBC wrapping is done in fractional coordinates -> valid for any box
   orientation, including triclinic, tilted c vector, etc.
   BUT the rule "frac_z > 0.5 -> fold back down" assumes THE SLAB SITS IN
   THE LOWER HALF OF THE CELL. If the slab is centered -> needs revisiting.

 OUTPUTS
 -------
   <cache-dir>/<group>.traj    list of ASE Atoms + SinglePointCalculator
                                read with: ase.io.read('g.traj', index=':')
                                metadata: atoms.info['source_dir'|
                                          'frame_in_traj'|'group_name']
   <cache-dir>/<group>.npz     coords, forces, energies, cells, symbols,
                                source_dir, frame_in_traj
   <cache-dir>/infos/config.json             exact config of the run (keep it!)
   <cache-dir>/infos/cache_summary.csv       per-group summary
   <cache-dir>/infos/econs_rejected.csv      trajectories rejected at step 3
                                              (energy conservation, on the
                                              kept segment); `scope` column
                                              is always 'segment'
   <cache-dir>/infos/truncation_details.csv  per-trajectory truncation detail
                                              (step 2)

 EXPECTED INPUT LAYOUT
 ----------------------
   MAIN_DIR/
   ├── 275meV_1O/                <- a "group"
   │   ├── H2_W_300_f001/vasprun.xml
   │   ├── H2_W_300_f002/vasprun.xml
   │   └── ...
   └── 100meV/ ...

 PITFALLS TO REMEMBER
 ---------------------
   * By default NO group is processed: you need --groups or --all-groups.
   * config.json holds the run's thresholds: without it you won't know in
     6 months which --max-etot-step a given cache was produced with.
   * .npz is read back with np.load(path) (no need for allow_pickle).
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
#                DEFAULT VALUES (overridden via CLI)
# ============================================================

DEF_MAIN_DIR   = "./"
DEF_CACHE_DIR  = "./AIMD_processed"
DEF_PATTERN    = "*/vasprun.xml"
DEF_FORMAT     = "traj"          # 'npz' | 'traj' | 'both'
DEF_STRIDE     = 1               # 1 = keep every frame

# Groups to ignore even with --all-groups
DEF_EXCLUDE    = [r'test', r'^copy', r'_PBE$']

# --- Step 3: energy conservation (eV), judged on the segment kept
#     after step 2 (truncation) ---
DEF_MAX_ETOT_DRIFT = 0.10        # global drift: max(E_tot) - min(E_tot)
DEF_MAX_ETOT_STEP  = 0.05        # inter-step jump: |E_tot[i+1] - E_tot[i]| (stricter)

# What to do when conservation breaks down on the kept segment:
#   'cut'  = cut right before the break and keep the beginning  [default]
#   'drop' = discard the whole trajectory (all-or-nothing)
DEF_ENERGY_FAIL   = 'cut'
# After cutting, a segment shorter than this is discarded: a few nearly
# identical initial frames add nothing and over-represent the start.
DEF_MIN_FRAMES_KEPT = 10

# --- Physical truncation ---
DEF_SURFACE_SYMBOL = 'W'         # atom defining Z_SURF
DEF_LAYER_GAP      = 1.5         # inter-layer gap (Å) for auto-detection
DEF_Z_REBOUND_TOL  = 0.0         # hysteresis (Å) around z_init
DEF_D_HH_DISSOC    = 2.3         # H-H dissociation threshold (Å)
DEF_N_MIN_FRAMES   = 5           # initial frames ignored for detection
DEF_KEEP_EVENT_FRAME = False     # False = cut BEFORE the triggering frame


# ============================================================
#                       GROUP SCANNING
# ============================================================

def list_groups(main_dir, include=None, exclude_patterns=None):
    """List the subdirectories (= groups) to process."""
    if not os.path.isdir(main_dir):
        raise FileNotFoundError(f"MAIN_DIR not found: {main_dir}")

    subdirs = sorted(
        d for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    )
    if include:
        missing = set(include) - set(subdirs)
        if missing:
            print(f"  ⚠  requested groups not found: {sorted(missing)}")
        subdirs = [d for d in subdirs if d in include]
    for pat in (exclude_patterns or []):
        subdirs = [d for d in subdirs if not re.search(pat, d)]
    return subdirs


# ============================================================
#                       FILE READING
# ============================================================

def read_total_energies_xml(vasprun_path):
    """E_tot (= E_pot + E_kin) of each step, read from vasprun.xml.

    XML structure:
      <calculation>
        <scstep> ... <energy>...</energy> </scstep>   <- intermediate SCF steps
        <energy>                                       <- FINAL ENERGY
          <i name="kinetic"> ... </i>
          <i name="total">   ... </i>                  <- what we read
        </energy>
      </calculation>

    Only the <energy> that is a DIRECT child of <calculation> is read.
    Returns an array (N_steps,) in eV, or None if absent/unreadable.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(vasprun_path).getroot()
    except Exception:
        return None

    e_tot = []
    for calc in root.iter('calculation'):
        blocks = calc.findall('energy')          # direct children only
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
    """Valid ASE frames (energy + forces present). [] on error."""
    try:
        frames = read(path, index=':')
    except Exception as exc:
        tqdm.write(f"  ❌ could not read {path}: {exc}")
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
#          STEP 3 — ENERGY CONSERVATION (E_tot)
# ============================================================

def check_etot(e_tot, cfg, scope='segment'):
    """All-or-nothing verdict of step 3 on an E_tot profile.

    `e_tot` is the segment kept after truncation (step 2). `scope` is just
    a label reported in the CSV ('segment', always, in practice).

    Returns (verdict, detail):
      verdict : 'ok' | 'reject_global' | 'reject_step' | 'unchecked'
      detail  : dict for the CSV, or None
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

    # --- Global drift: max(E_tot) - min(E_tot) ---
    if cfg['check_global']:
        drift = float(e_tot.max() - e_tot.min())
        if drift > cfg['max_drift']:
            return 'reject_global', {
                **base, 'check': 'global',
                'drift_eV': drift, 'worst_step': -1,
            }

    # --- Inter-step jump: |E_tot[i+1] - E_tot[i]| ---
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
    """First frame from which conservation breaks down.

    Returns (n_keep, cause):
      n_keep = number of frames to keep from the start (None = all good)
      cause  = 'per_step' | 'global' | None

    per_step: |E[i+1]-E[i]| > threshold -> frame i+1 is suspect (SCF did not
              converge: forces are wrong), keep [0..i].
    global  : the cumulative max-min amplitude on [0..j] exceeds the
              threshold -> keep [0..j-1]. A rolling criterion, not a
              posteriori global one: this is what lets us date the break.
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
#     STEP 2 — PHYSICAL EVENT (dissociation / rebound / trapping)
# ============================================================

def _compute_zsurf(coords_frame, symbols, cell,
                   surface_symbol='W', layer_gap=1.5):
    """Mean cartesian Z of the surface layer, for 1 frame.

    PBC wrapping is done in FRACTIONAL coordinates: no assumption on box
    orientation (triclinic OK, tilted c vector OK, non-triangular
    convention OK).

    Remaining, irreducible assumption: the slab occupies the lower half of
    the cell (rule frac_z > 0.5 -> fold back down by one period).
    """
    surf_list = [surface_symbol] if isinstance(surface_symbol, str) else surface_symbol
    mask = np.isin(symbols, surf_list)
    if not mask.any():
        raise ValueError(f"no {surface_symbol} atom found")

    cell = np.asarray(cell, dtype=np.float64)
    pos  = coords_frame[mask].astype(np.float64)

    # cartesian -> fractional:  pos = frac @ cell  <=>  cell.T @ frac.T = pos.T
    frac = np.linalg.solve(cell.T, pos.T).T
    frac[:, 2] -= np.floor(frac[:, 2])       # wrap into [0, 1)
    frac[frac[:, 2] > 0.5, 2] -= 1.0         # fold the top back down
    z = (frac @ cell)[:, 2]

    z_sorted = np.sort(z)
    big = np.where(np.diff(z_sorted) > layer_gap)[0]
    top = z_sorted[big[-1] + 1:] if len(big) else z_sorted
    if len(top) < 3:                         # auto-detection failed -> fallback
        top = z_sorted[-min(8, len(z_sorted)):]
    return float(top.mean())


def _d_HH_mic(pos_H, cells):
    """H-H distance using the minimum image convention, over the whole
    trajectory.

    A single matrix inversion if the cell is constant (NVE case).
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
    """First physical event on ONE trajectory (step 2).

    REBOUND      : z_com drops below z_init, then rises back above it.
    DISSOCIATION : d(H-H) > cfg['d_dissoc'].
    TRAPPING     : neither happens -> H2 stays adsorbed on the surface for
                   the whole trajectory (event_type='none'), which is then
                   kept in full.

    Returns (keep_until, event_type, t_event, z_init):
      keep_until : last index to keep
                   None = no cut (keep everything, trapping case)
                   -1   = keep nothing (trajectory rejected)
      event_type : 'rebound' | 'dissociation' | 'none' (trapping) |
                   'start_dissociated'
      t_event    : index of the event (-1 if none)
      z_init     : initial height of the H2 center of mass above the
                   surface (Å)
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

    # --- SAFEGUARD: H2 must start BONDED ---
    # dissoc[:n_min] = False masks the first frames against noise. If H2 is
    # already past the threshold there, this mask would let through frames
    # well above d_dissoc (H2 already broken at the start = trajectory not
    # usable).
    if d_HH[:max(n_min, 1)].max() > cfg['d_dissoc']:
        return -1, 'start_dissociated', 0, z_init

    # --- Rebound: dropped below z_init, THEN rose back above it ---
    # The sign of the velocity is implicit: crossing back above z_init after
    # having descended requires dz/dt > 0. No need to differentiate z(t).
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
        t_event, event_type = t_reb, 'rebound'
    else:
        t_event, event_type = t_dis, 'dissociation'

    keep_until = t_event if cfg['keep_event_frame'] else t_event - 1
    return keep_until, event_type, t_event, z_init


# ============================================================
#                    PROCESSING ONE GROUP
# ============================================================

def _make_atoms(atoms, energy, forces, source_dir, frame_in_traj, group_name):
    """Lightweight copy of `atoms` with a SinglePointCalculator + metadata."""
    new = atoms.copy()
    new.calc = SinglePointCalculator(new, energy=float(energy),
                                     forces=np.asarray(forces))
    new.info['source_dir']    = str(source_dir)
    new.info['frame_in_traj'] = int(frame_in_traj)
    new.info['group_name']    = str(group_name)
    return new


def process_group(group_name, group_dir, cfg):
    """Process one whole group. Streams the .traj to disk if requested.

    Returns a results dict, or None if nothing valid was produced.
    """
    files = sorted(glob.glob(os.path.join(group_dir, cfg['pattern'])))
    print(f"\n[{group_name}] {len(files)} vasprun.xml files found")
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
    trunc_stats = {'rebound': 0, 'dissociation': 0, 'none': 0,
                   'start_dissociated': 0}
    trunc_details, econs_details = [], []

    try:
        for vfile in tqdm(files, desc=f'  {group_name}', unit='traj'):
            traj_dir = os.path.basename(os.path.dirname(vfile))

            # ── E_tot: lightweight XML read, BEFORE ASE parsing ──
            # Simply set aside for step 3 (judged on the kept segment, after
            # truncation at step 2): no rejection here, so we never lose a
            # correct initial segment because of a drift that happens later
            # in the trajectory.
            e_tot = read_total_energies_xml(vfile)
            if e_tot is None or len(e_tot) < 2:
                n_unchecked += 1
                tqdm.write(f"  ⚠  {traj_dir}: E_tot unreadable → not checked")

            # ── ASE parsing ──
            frames = read_vasprun_safe(vfile)
            if not frames:
                continue
            n_raw += len(frames)

            if e_tot is not None and len(e_tot) != len(frames):
                n_mismatch += 1
                tqdm.write(f"  ⚠  {traj_dir}: {len(e_tot)} E_tot vs "
                           f"{len(frames)} ASE frames (incomplete steps)")

            # ── STEP 1: topological consistency (order of symbols) ──
            sym = list(frames[0].get_chemical_symbols())
            if symbols_ref is None:
                symbols_ref = sym
            elif sym != symbols_ref:
                n_topo += len(frames)
                tqdm.write(f"  ⚠  {traj_dir}: symbol order ≠ reference → skip")
                continue

            # ── STEP 2: physical event (dissociation / rebound / trapping) ──
            n_before = len(frames)
            if cfg['truncate']:
                sym_arr = np.array(sym, dtype='<U2')
                keep_until, event, t_event, z_init = detect_truncation_index(
                    frames, sym_arr, cfg)

                # keep_until: None = keep all | -1 = discard all | n = [:n+1]
                if keep_until is not None:
                    frames = frames[:keep_until + 1] if keep_until >= 0 else []
                    n_trunc += n_before - len(frames)
                    if cfg['verbose_trunc']:
                        tqdm.write(f"    {traj_dir}: {event} at t={t_event}, "
                                   f"kept {len(frames)}/{n_before}")

                trunc_stats[event] = trunc_stats.get(event, 0) + 1
                trunc_details.append({
                    'traj': traj_dir, 'event': event, 't_event': t_event,
                    'z_init': z_init, 'n_kept': len(frames), 'n_total': n_before,
                })

                if event == 'start_dissociated':
                    tqdm.write(f"  ⚠  {traj_dir}: d(H-H) > {cfg['d_dissoc']} Å "
                               f"from the start → trajectory REJECTED")
                if not frames:
                    continue

            # ── STEP 3: energy conservation, on the kept segment (after step 2) ──
            # A trajectory can be physically flawless up to the dissociation
            # and then drift afterwards (fast H atoms, SCF failing to
            # converge, H2 crossing the box). Judging the whole trajectory
            # would discard perfectly usable frames.
            if e_tot is not None:
                n_seg = len(frames)
                if len(e_tot) >= n_seg:
                    e_seg = e_tot[:n_seg]
                else:
                    e_seg = e_tot          # mismatch already reported above
                if cfg['energy_fail'] == 'cut':
                    # --- CUT at the break, keep what came before ---
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
                            outcome_msg = ('segment too short, REJECTED' if too_short
                                          else f'kept {n_keep}/{n_seg}')
                            tqdm.write(f"    {traj_dir}: {cause} at frame {n_keep} "
                                       f"→ {outcome_msg}")
                        if too_short:
                            continue
                        frames = frames[:n_keep]
                else:
                    # --- all-or-nothing ---
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
                                       f"{d2['drift_eV']:.4f} eV on the kept segment "
                                       f"({n_seg} frames) → REJECTED")
                        continue

            # ── STEP 4: stride ──
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
        print(f"  [{group_name}] no valid frame after filtering")
        if want_traj and os.path.exists(traj_path):
            os.remove(traj_path)
        return None

    e_mean = e_sum / n_kept
    e_std  = float(np.sqrt(max(e_sq / n_kept - e_mean ** 2, 0.0)))

    # ── Write .npz ──
    if want_npz:
        npz_path = os.path.join(cfg['cache_dir'], f"{group_name}.npz")
        np.savez(
            npz_path,
            symbols       = np.array(symbols_ref, dtype='<U2'),
            coords        = np.array(coords_all,   dtype=np.float32),
            forces        = np.array(forces_all,   dtype=np.float32),
            energies      = np.array(energies_all, dtype=np.float64),
            cells         = np.array(cells_all,    dtype=np.float32),
            # '<U64' instead of dtype=object: otherwise np.load requires allow_pickle
            source_dir    = np.array(source_all,   dtype='<U64'),
            frame_in_traj = np.array(frame_all,    dtype=np.int32),
            n_atoms       = len(symbols_ref),
            group_name    = group_name,
        )
        print(f"  → .npz  : {npz_path}  ({os.path.getsize(npz_path)/1e6:.1f} MB)")

    if want_traj:
        print(f"  → .traj : {traj_path}  ({os.path.getsize(traj_path)/1e6:.1f} MB)")

    # ── Console summary ──
    print(f"  [{group_name}] kept {n_kept}/{n_raw}  "
          f"(trunc={n_trunc}, topo={n_topo}, "
          f"econs_global={n_rej_global}, econs_step={n_rej_step}, "
          f"frames_cut_energy={n_frames_cut_energy})  "
          f"E ∈ [{e_min:.3f}, {e_max:.3f}] eV  N_atoms={len(symbols_ref)}")
    if n_unchecked:
        print(f"  [{group_name}] ⚠  {n_unchecked} traj with unreadable E_tot "
              f"(passed through WITHOUT conservation check)")
    if n_mismatch:
        print(f"  [{group_name}] ⚠  {n_mismatch} traj with E_tot/frames "
              f"mismatch")
    if cfg['truncate']:
        ts = trunc_stats
        print(f"  [{group_name}] truncations: rebound={ts['rebound']}, "
              f"dissociation={ts['dissociation']}, none={ts['none']}, "
              f"start_dissoc={ts['start_dissociated']} "
              f"(out of {sum(ts.values())} traj)")

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
            'trunc_rebound': trunc_stats['rebound'],
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
        description="Step 1 — prepares the AIMD cache (.traj / .npz) "
                    "from VASP vasprun.xml files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python AIMD_process.py --groups 275meV_1O --format both",
    )

    g = p.add_argument_group('paths')
    g.add_argument('-i', '--main-dir',  default=DEF_MAIN_DIR,
                   help='root directory containing the group folders')
    g.add_argument('-o', '--cache-dir', default=DEF_CACHE_DIR,
                   help='output cache directory')
    g.add_argument('--pattern',   default=DEF_PATTERN,
                   help='glob for vasprun.xml files within a group')

    g = p.add_argument_group('group selection')
    sel = g.add_mutually_exclusive_group()
    sel.add_argument('--groups', nargs='+', metavar='G',
                     help='groups to process (e.g.: 100meV 275meV_1O)')
    sel.add_argument('--all-groups', action='store_true',
                     help='process every group found under --main-dir')
    g.add_argument('--exclude', nargs='+', default=DEF_EXCLUDE, metavar='RE',
                   help='regex patterns of groups to exclude')

    g = p.add_argument_group('output')
    g.add_argument('--format', choices=['npz', 'traj', 'both'], default=DEF_FORMAT)
    g.add_argument('--stride', type=int, default=DEF_STRIDE,
                   help='[step 4] keep 1 frame every N')

    g = p.add_argument_group("step 3 — energy conservation (on the kept segment)")
    g.add_argument('--max-etot-drift', type=float, default=DEF_MAX_ETOT_DRIFT,
                   metavar='eV', help='threshold for max(E_tot)-min(E_tot)')
    g.add_argument('--max-etot-step', type=float, default=DEF_MAX_ETOT_STEP,
                   metavar='eV', help='threshold for |E_tot[i]-E_tot[i-1]|')
    g.add_argument('--energy-fail', choices=['cut', 'drop'],
                   default=DEF_ENERGY_FAIL,
                   help="what to do if energy breaks down: 'cut' = cut "
                        "right before it and keep the beginning (default); "
                        "'drop' = discard the whole trajectory")
    g.add_argument('--min-frames-kept', type=int, default=DEF_MIN_FRAMES_KEPT,
                   help='after a cut, a segment shorter than this is discarded')
    g.add_argument('--no-etot-global', action='store_true',
                   help='disable the global-drift check')
    g.add_argument('--no-etot-step', action='store_true',
                   help='disable the inter-step-jump check')

    g = p.add_argument_group('step 2 — physical event (truncation)')
    g.add_argument('--no-truncate', action='store_true',
                   help='disable truncation')
    g.add_argument('--surface-symbol', default=DEF_SURFACE_SYMBOL,
                   help='atom defining Z_SURF')
    g.add_argument('--layer-gap', type=float, default=DEF_LAYER_GAP,
                   metavar='A', help='inter-layer gap for auto-detection')
    g.add_argument('--z-rebound-tol', type=float, default=DEF_Z_REBOUND_TOL,
                   metavar='A',
                   help='hysteresis around z_init (0 = strict crossing)')
    g.add_argument('--d-hh-dissoc', type=float, default=DEF_D_HH_DISSOC,
                   metavar='A', help='H-H dissociation threshold')
    g.add_argument('--n-min-frames', type=int, default=DEF_N_MIN_FRAMES,
                   help='initial frames ignored by the detection')
    g.add_argument('--keep-event-frame', action='store_true',
                   default=DEF_KEEP_EVENT_FRAME,
                   help="INCLUDE the triggering frame (by default the cut "
                        "happens right before it: every frame kept "
                        "satisfies z_com <= z_init AND d_HH <= threshold)")

    g = p.add_argument_group('misc')
    g.add_argument('--dry-run', action='store_true',
                   help='list groups and file counts, then exit')
    g.add_argument('--verbose-etot',  action='store_true',
                   help='1 line per trajectory rejected on E_tot')
    g.add_argument('--verbose-trunc', action='store_true',
                   help='1 line per truncated trajectory')

    return p


def args_to_cfg(a):
    """Flat config dict, also dumped to infos/config.json."""
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
        raise SystemExit("--stride must be >= 1")
    cfg = args_to_cfg(args)
    t0  = time.time()
    sep = '=' * 70

    include = None if args.all_groups else args.groups
    if include is None and not args.all_groups:
        raise SystemExit(
            "No group selected. Use --groups G1 G2  or  --all-groups.\n"
            "Tip: start with  python AIMD_process.py --dry-run --all-groups"
        )

    groups = list_groups(args.main_dir, include=include,
                         exclude_patterns=args.exclude)

    # ── Header ──
    print(sep)
    print(f"Root                   : {args.main_dir}")
    print(f"Cache                  : {args.cache_dir}   [format: {args.format}]")
    print(f"Groups ({len(groups)})            : {', '.join(groups) if groups else '—'}")
    print(f"Step 4 — stride        : {args.stride}")
    print(f"Step 3 — E_tot global  : "
          f"{'≤ %.3f eV' % cfg['max_drift'] if cfg['check_global'] else 'disabled'}")
    print(f"Step 3 — E_tot/step    : "
          f"{'≤ %.3f eV' % cfg['max_step'] if cfg['check_step'] else 'disabled'}")
    action_msg = ('cut at breakdown (min %d frames)' % cfg['min_frames_kept']
                  if cfg['energy_fail'] == 'cut' else 'reject the trajectory')
    print(f"Action on breakdown    : {action_msg}")
    print(f"E_tot check scope      : KEPT SEGMENT (after step 2)")
    trunc_msg = ('rebound (back above z_init, tol=%.2f Å) + '
                 'dissoc (d_HH > %.1f Å)' % (cfg['z_rebound_tol'], cfg['d_dissoc'])
                 ) if cfg['truncate'] else 'disabled'
    print(f"Step 2 — truncation    : {trunc_msg}")
    print(sep)

    if not groups:
        raise SystemExit("No group to process — check --groups / --exclude.")

    # ── Dry-run ──
    if args.dry_run:
        print("\nDRY-RUN — no file will be read or written.\n")
        total = 0
        for g in groups:
            n = len(glob.glob(os.path.join(args.main_dir, g, args.pattern)))
            total += n
            print(f"  {g:24s} : {n:5d} vasprun.xml")
        print(f"\n  {'TOTAL':24s} : {total:5d} vasprun.xml")
        print("\nRe-run without --dry-run to process.")
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    infos_dir = os.path.join(args.cache_dir, 'infos')
    os.makedirs(infos_dir, exist_ok=True)

    # ── Dump the config (traceability) ──
    cfg_path = os.path.join(infos_dir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'groups_resolved': groups, **cfg}, f, indent=2)

    # ── Processing ──
    summary, all_trunc, all_econs = [], [], []
    for gname in groups:
        res = process_group(gname, os.path.join(args.main_dir, gname), cfg)
        if res is None:
            continue
        summary.append(res['summary'])
        all_trunc += [{'group': gname, **d} for d in res['trunc_details']]
        all_econs += [{'group': gname, **d} for d in res['econs_details']]

    # ── Recap ──
    print(f"\n{sep}\nCACHE SUMMARY\n{sep}")
    df = pd.DataFrame(summary)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(os.path.join(infos_dir, 'cache_summary.csv'), index=False)
    n_total = int(df['n_kept'].sum()) if not df.empty else 0
    print(f"\nTotal cached frames   : {n_total}")
    print(f"Total time            : {time.time() - t0:.1f}s")
    print(f"Run config            : {cfg_path}")
    if not df.empty:
        print(f"Summary CSV           : {os.path.join(infos_dir, 'cache_summary.csv')}")

    if all_econs:
        df_ec = pd.DataFrame(all_econs)
        p_ec  = os.path.join(infos_dir, 'econs_rejected.csv')
        df_ec.to_csv(p_ec, index=False)
        print(f"\nTrajectories rejected (E_tot conservation):")
        for ck, label in [('global', 'global (max-min)'), ('per_step', 'inter-step jump ')]:
            sub = df_ec[df_ec['check'] == ck]
            if len(sub):
                extra = f"  worst step: {int(sub['worst_step'].max())}" if ck == 'per_step' else ""
                print(f"  {label} : {len(sub):4d} traj   "
                      f"mean={sub['drift_eV'].mean():.4f} eV   "
                      f"max={sub['drift_eV'].max():.4f} eV{extra}")
        print(f"  Details: {p_ec}")

    if all_trunc:
        df_tr = pd.DataFrame(all_trunc)
        p_tr  = os.path.join(infos_dir, 'truncation_details.csv')
        df_tr.to_csv(p_tr, index=False)
        print(f"\nGlobal truncation summary:")
        for ev in ['rebound', 'dissociation', 'none', 'start_dissociated']:
            print(f"  {ev:15s} : {int((df_tr['event'] == ev).sum())} trajectories")
        cut = df_tr[~df_tr['event'].isin(['none', 'start_dissociated'])]
        if len(cut):
            mk, mt = cut['n_kept'].mean(), cut['n_total'].mean()
            print(f"  truncated      : {len(cut)}, frames kept on average "
                  f"{mk:.0f}/{mt:.0f} ({100*mk/mt:.1f}%)")
        print(f"  Details: {p_tr}")


if __name__ == "__main__":
    main()
