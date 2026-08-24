"""
step06_test_set.py
------------------
Generate a held-out test set by drawing from structures not already
used in the training / validation sets.

WHERE THE TEST SET COMES FROM — READ THIS BEFORE TRUSTING TEST METRICS
--------------------------------------------------------------------------
The candidate pool ("universe") is EVERY structure step 01 ever discovered
(the full raw dataset, from its provenance table) — NOT the step-03 sampled
subset. From that universe, this step excludes only what's already claimed
by an existing `*_selected_manifest.csv` under `selected/` (i.e. step 03's
sample, then split into train/val by step 05). So the test set is drawn from
whatever step 03 did NOT pick.
That is deliberate — it tests generalisation on data the model never got a
chance to be biased toward via coverage sampling — but it also means the
test set has a different character than train/val: it wasn't selected for
descriptor-space diversity, just left over. Don't be surprised if it looks
"easier" (more redundant/typical) than train/val, especially with
`stratified: false`.

Two sampling modes:
  - random (default): uniform random draw from the leftover pool.
  - stratified: for this H-on-surface system, classify leftover structures
    by how far the gas molecule sits above the surface (z_mean, Å) and draw
    a chosen fraction from "close" (near-surface, chemically interesting —
    adsorption/reaction) vs "far" (free gas) so the test set isn't
    accidentally dominated by whichever regime is more common in the
    leftovers. Classifying requires reading each candidate structure from
    disk, so only a `pre_sample_factor * n_test`-sized pool is classified,
    not the whole leftover set — a speed/cost trade-off.

Outputs (in selected/testset/)
------------------------------
  TEST_selected_manifest.csv
  TEST_selected.traj
  TEST_selected.xyz

Usage
-----
    python step06_test_set.py
    python step06_test_set.py --config my_cfg.yaml
    python step06_test_set.py --override test.n_test=500
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write
from tqdm import tqdm

from pipeline_utils import (
    apply_overrides,
    base_parser,
    load_config,
    resolve_work_dir,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------

def _z_gas_above_surface(
    atoms,
    gas_species: list[str],
    surface_species: list[str],
) -> float:
    """
    Compute z_mean of gas atoms above the surface.

    z_surf = max z of surface atoms in the lower half of the cell
             (avoids periodic images at the top).
    Returns z_mean(gas) - z_surf in Angstroms.
    """
    symbols  = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell_z   = float(atoms.cell[2, 2])

    # Surface atoms in lower half only
    surf_z = [
        positions[i, 2]
        for i, s in enumerate(symbols)
        if s in surface_species and positions[i, 2] < cell_z / 2
    ]
    if not surf_z:
        raise ValueError(
            f"No surface atoms ({surface_species}) found in lower half of cell."
        )
    z_surf = max(surf_z)

    # Gas atoms
    gas_z = [positions[i, 2] for i, s in enumerate(symbols) if s in gas_species]
    if not gas_z:
        raise ValueError(f"No gas atoms ({gas_species}) found in structure.")

    return float(np.mean(gas_z)) - z_surf


def _stratified_sample(
    remaining: pd.DataFrame,
    n_test: int,
    gas_species: list[str],
    surface_species: list[str],
    z_threshold: float,
    close_fraction: float,
    pre_sample_factor: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Pre-sample a pool, classify close/far, then draw stratified test set.
    """
    # Classifying reads each structure from disk (ase.io.read) just to get
    # z_gas — too slow to do on the full `remaining` pool when it's much
    # bigger than n_test. Pre-sample a pool `pre_sample_factor`x larger than
    # n_test instead: enough slack to satisfy close_fraction even if one
    # class is rarer in the leftovers, without classifying everything.
    pool_size = min(pre_sample_factor * n_test, len(remaining))
    pool_idx  = rng.choice(remaining.index, size=pool_size, replace=False)
    pool      = remaining.loc[pool_idx].dropna(subset=["file_path"]).reset_index(drop=True)
    logger.info("Pre-sampled pool : %d structures", len(pool))

    # Classify
    close_idx, far_idx = [], []
    for i, row in tqdm(pool.iterrows(), total=len(pool), desc="Classifying z_gas"):
        try:
            atoms = read(str(row["file_path"]), index=int(row["struct_id"]))
            z = _z_gas_above_surface(atoms, gas_species, surface_species)
            if z <= z_threshold:
                close_idx.append(i)
            else:
                far_idx.append(i)
        except Exception as exc:
            logger.debug("Skipping %s frame %d : %s", row["file_path"], row["struct_id"], exc)

    logger.info("Pool classified  : %d close (z≤%.1f Å)  /  %d far",
                len(close_idx), z_threshold, len(far_idx))

    # Target counts
    n_close = int(round(close_fraction * n_test))
    n_far   = n_test - n_close

    if len(close_idx) < n_close:
        logger.warning(
            "Only %d close structures available, requested %d. Using all.",
            len(close_idx), n_close,
        )
        n_close = len(close_idx)
        n_far   = min(n_far, len(far_idx))

    if len(far_idx) < n_far:
        logger.warning(
            "Only %d far structures available, requested %d. Using all.",
            len(far_idx), n_far,
        )
        n_far = len(far_idx)

    sel_close = rng.choice(close_idx, size=n_close, replace=False)
    sel_far   = rng.choice(far_idx,   size=n_far,   replace=False)
    sel       = np.concatenate([sel_close, sel_far])

    logger.info("Stratified test set : %d close + %d far = %d total",
                n_close, n_far, len(sel))

    return pool.loc[sel].sort_values(["file_path", "struct_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(cfg: dict, logger: logging.Logger) -> dict:
    """Execute the test-set generation step."""
    tcfg = cfg["test"]
    work_dir = resolve_work_dir(cfg)

    desc_dir = work_dir / tcfg.get("desc_dir", "desc")
    selected_root = work_dir / tcfg.get("selected_root", "selected")
    test_dir = selected_root / "testset"
    test_dir.mkdir(parents=True, exist_ok=True)

    n_test = int(tcfg.get("n_test", 2000))
    seed   = int(tcfg.get("seed", 12345))

    # Stratification config
    stratified        = bool(tcfg.get("stratified", False))
    gas_species       = list(tcfg.get("gas_species", ["H"]))
    surface_species   = list(tcfg.get("surface_species", ["W"]))
    z_threshold       = float(tcfg.get("z_threshold", 3.0))
    close_fraction    = float(tcfg.get("close_fraction", 0.75))
    pre_sample_factor = int(tcfg.get("pre_sample_factor", 10))

    # ------------------------------------------------------------------
    # 1. Load provenance + filemap (universe of available structures)
    # ------------------------------------------------------------------
    prov_candidates = (
        list(desc_dir.glob("*_provenance*.parquet"))
        + list(desc_dir.glob("*_provenance*.csv"))
    )
    if not prov_candidates:
        raise FileNotFoundError(
            f"No provenance table in {desc_dir}. Run step01 first."
        )
    prov_path = max(prov_candidates, key=lambda p: p.stat().st_mtime)
    meta = pd.read_parquet(prov_path) if prov_path.suffix == ".parquet" else pd.read_csv(prov_path)
    logger.info("Loaded provenance from %s  rows=%d", prov_path.name, len(meta))

    filemap_candidates = list(desc_dir.glob("*_filemap*.json"))
    if not filemap_candidates:
        raise FileNotFoundError(f"No filemap in {desc_dir}. Run step01 first.")
    filemap_path = max(filemap_candidates, key=lambda p: p.stat().st_mtime)
    with filemap_path.open() as fh:
        fmap = {int(k): v for k, v in json.load(fh).items()}

    struct_id_col = "source_struct_id" if "source_struct_id" in meta.columns else "struct_id"
    universe = (
        meta.loc[:, ["file_id", struct_id_col]]
        .drop_duplicates()
        .rename(columns={struct_id_col: "struct_id"})
        .sort_values(["file_id", "struct_id"])
    )
    universe["file_path"] = universe["file_id"].map(fmap)
    logger.info("Universe: %d unique structures", len(universe))

    # ------------------------------------------------------------------
    # 2. Exclude structures already used in training / validation
    # ------------------------------------------------------------------
    manifests = sorted(
        m for m in selected_root.rglob("*_selected_manifest.csv")
        if m.name != "TEST_selected_manifest.csv" and "testset" not in m.parts
    )
    logger.info("Found %d existing split manifests.", len(manifests))

    used_frames: list[pd.DataFrame] = []
    for m in manifests:
        df = pd.read_csv(m, usecols=["file_path", "struct_id"])
        used_frames.append(df.assign(struct_id=df["struct_id"].astype(int)))

    if used_frames:
        used_df = pd.concat(used_frames, ignore_index=True).drop_duplicates()
        remaining = universe.merge(
            used_df, on=["file_path", "struct_id"], how="left", indicator=True
        )
        remaining = (
            remaining.loc[remaining["_merge"] == "left_only", universe.columns]
            .reset_index(drop=True)
        )
        logger.info(
            "Already used: %d structures. Remaining candidates: %d",
            len(used_df), len(remaining),
        )
    else:
        remaining = universe.reset_index(drop=True)
        logger.info("No existing manifests. All %d structures are candidates.", len(remaining))

    # ------------------------------------------------------------------
    # 3. Draw test set — random or stratified
    # ------------------------------------------------------------------
    n_test = min(n_test, len(remaining))
    if n_test == 0:
        raise ValueError("No remaining structures available to build a test set.")

    rng = np.random.default_rng(seed)

    if stratified:
        logger.info(
            "Stratified sampling : gas=%s  surface=%s  z_threshold=%.1f Å  "
            "close_fraction=%.0f%%  pre_sample_factor=%d",
            gas_species, surface_species, z_threshold,
            close_fraction * 100, pre_sample_factor,
        )
        test_df = _stratified_sample(
            remaining, n_test,
            gas_species, surface_species,
            z_threshold, close_fraction, pre_sample_factor,
            rng, logger,
        )
    else:
        logger.info("Random sampling : n_test=%d", n_test)
        sel_idx = rng.choice(remaining.index, size=n_test, replace=False)
        test_df = (
            remaining.loc[sel_idx]
            .sort_values(["file_path", "struct_id"])
            .reset_index(drop=True)
            .dropna(subset=["file_path"])
            .reset_index(drop=True)
        )

    if len(test_df) == 0:
        raise ValueError("Test set is empty after dropping missing file paths.")

    # Always 0 here: this column means "how many atoms of this structure were
    # individually hit by the sampler" in step03's manifest (see
    # atoms_to_structures in src/sampler_utils.py) — kept only for schema
    # parity with that CSV, since test-set structures are drawn whole, not
    # via atom-level hit-counting.
    test_df["n_atoms_hit"] = 0
    logger.info("Test set: %d structures selected (seed=%d)", len(test_df), seed)

    # ------------------------------------------------------------------
    # 4. Save manifest
    # ------------------------------------------------------------------
    manifest_path = test_dir / "TEST_selected_manifest.csv"
    test_df.to_csv(manifest_path, index=False)
    logger.info("Manifest saved → %s", manifest_path)

    # ------------------------------------------------------------------
    # 5. Load and export structures
    # ------------------------------------------------------------------
    images = []
    for fp, sid in tqdm(
        zip(test_df["file_path"], test_df["struct_id"]),
        total=len(test_df),
        desc="Loading TEST structures",
    ):
        images.append(read(fp, index=int(sid)))

    traj_path = test_dir / "TEST_selected.traj"
    xyz_path  = test_dir / "TEST_selected.xyz"
    write(str(traj_path), images)
    write(str(xyz_path), images)

    logger.info("Wrote %d frames:", len(images))
    logger.info("  .traj → %s", traj_path)
    logger.info("  .xyz  → %s", xyz_path)

    return {
        "manifest": manifest_path,
        "traj":     traj_path,
        "xyz":      xyz_path,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = base_parser("step06_test_set", description=__doc__)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    logger = setup_logging(
        log_file=work_dir / "logs" / "step06_test_set.log",
        level=getattr(logging, args.log_level),
        step_name="step06",
    )

    logger.info("=== Step 06 — Test Set Generation ===")
    logger.info("n_test=%d  seed=%d",
                cfg["test"].get("n_test", 2000),
                cfg["test"].get("seed", 12345))

    try:
        run(cfg, logger)
        logger.info("=== Step 06 completed successfully ===")
    except Exception:
        logger.exception("Step 06 failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
