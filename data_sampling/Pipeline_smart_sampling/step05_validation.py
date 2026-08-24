"""
step05_validation.py
--------------------
Split the sampled trajectory into a validation set and a training set.

The split is a plain UNIFORM RANDOM draw, not another coverage-based
sampling pass — the diversity work was already done by step 03, so at this
point every frame in the sampled set is considered equally worth keeping,
and a random hold-out is enough to get an unbiased validation subset.

Outputs (in selected/valset/ and selected/trainset/)
------------------------------------------------------
  *_valset_{seed}_{pct}.traj / .xyz / .txt
  *_trainset_{seed}_{100-pct}.traj / .xyz / .txt

Usage
-----
    python step05_validation.py
    python step05_validation.py --config my_cfg.yaml
    python step05_validation.py --override validation.percent=10 validation.seed=0
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import write
from ase.io.trajectory import Trajectory

from pipeline_utils import (
    apply_overrides,
    base_parser,
    load_config,
    resolve_work_dir,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(cfg: dict, logger: logging.Logger) -> dict:
    """Execute the validation-set extraction step."""
    vcfg = cfg["validation"]
    work_dir = resolve_work_dir(cfg)

    selected_dir = work_dir / vcfg.get("selected_dir", "selected")
    method = vcfg.get("method", "FPS")
    percent = int(vcfg.get("percent", 20))
    seed = vcfg.get("seed", 42)

    assert 0 < percent <= 100, "validation.percent must be in (0, 100]."

    # ------------------------------------------------------------------
    # 1. Locate source trajectory
    # ------------------------------------------------------------------
    src_traj = selected_dir / f"{method}_selected.traj"
    if not src_traj.exists():
        candidates = sorted(selected_dir.glob("*_selected.traj"))
        if len(candidates) == 1:
            src_traj = candidates[0]
            logger.info("Source trajectory not found for method '%s'. Using: %s", method, src_traj)
        else:
            raise FileNotFoundError(
                f"Missing source trajectory: {src_traj}. "
                f"Available: {candidates}"
            )

    traj = Trajectory(str(src_traj), mode="r")
    n_frames = len(traj)
    logger.info("Source: %s  (%d frames)", src_traj, n_frames)

    if n_frames == 0:
        raise ValueError(f"Source trajectory is empty: {src_traj}")

    # ------------------------------------------------------------------
    # 2. Random split
    # ------------------------------------------------------------------
    k_val = max(1, int(round(percent * n_frames / 100.0)))
    rng = np.random.default_rng(seed)
    val_indices = np.sort(rng.choice(n_frames, size=k_val, replace=False))
    train_indices = np.setdiff1d(np.arange(n_frames), val_indices)

    logger.info(
        "Split: %d validation (%.0f%%) + %d training (%.0f%%)",
        len(val_indices), percent, len(train_indices), 100 - percent,
    )

    # ------------------------------------------------------------------
    # 3. Write validation set
    # ------------------------------------------------------------------
    val_dir = selected_dir / "valset"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_prefix = val_dir / f"{method}_selected_valset_{seed}_{percent}"
    val_traj_path = val_prefix.with_suffix(".traj")
    val_xyz_path = val_prefix.with_suffix(".xyz")
    val_txt_path = val_prefix.with_suffix(".txt")

    _write_split(traj, val_indices, val_traj_path, val_xyz_path)
    _write_manifest(
        traj, val_indices,
        path=val_txt_path,
        title="Validation set manifest",
        src=src_traj,
        n_frames=n_frames,
        percent=percent,
        seed=seed,
        label="Selected frames",
    )
    logger.info("Validation set → %s  (%d frames)", val_dir, len(val_indices))

    # ------------------------------------------------------------------
    # 4. Write training set (complement)
    # ------------------------------------------------------------------
    train_dir = selected_dir / "trainset"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_prefix = train_dir / f"{method}_selected_trainset_{seed}_{100 - percent}"
    train_traj_path = train_prefix.with_suffix(".traj")
    train_xyz_path = train_prefix.with_suffix(".xyz")
    train_txt_path = train_prefix.with_suffix(".txt")

    _write_split(traj, train_indices, train_traj_path, train_xyz_path)
    _write_manifest(
        traj, train_indices,
        path=train_txt_path,
        title="Training set manifest",
        src=src_traj,
        n_frames=n_frames,
        percent=100 - percent,
        seed=seed,
        label="Training frames",
    )
    logger.info("Training set → %s  (%d frames)", train_dir, len(train_indices))

    return {
        "val_traj": val_traj_path,
        "val_xyz": val_xyz_path,
        "val_manifest": val_txt_path,
        "train_traj": train_traj_path,
        "train_xyz": train_xyz_path,
        "train_manifest": train_txt_path,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_split(
    traj: "Trajectory",
    indices: np.ndarray,
    traj_path: Path,
    xyz_path: Path,
) -> None:
    if traj_path.exists():
        traj_path.unlink()
    with Trajectory(str(traj_path), mode="w") as T:
        for i in indices:
            T.write(traj[i])

    if xyz_path.exists():
        xyz_path.unlink()
    first = True
    for i in indices:
        write(str(xyz_path), traj[i], append=not first, format="extxyz")
        first = False


def _write_manifest(
    traj: "Trajectory",
    indices: np.ndarray,
    path: Path,
    title: str,
    src: Path,
    n_frames: int,
    percent: int,
    seed: int | None,
    label: str,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n")
        fh.write(f"# Source: {src.resolve()}\n")
        fh.write(f"# Created: {ts}\n")
        fh.write(f"# Total frames: {n_frames}\n")
        fh.write(f"# Percent: {percent}\n")
        fh.write(f"# Seed: {seed}\n")
        fh.write(f"# {label}: {len(indices)}\n\n")
        fh.write("index\tnatoms\tcell_a\tcell_b\tcell_c\tpbc\tenergy\n")
        for i in indices:
            atoms = traj[i]
            cell = atoms.cell.lengths()
            pbc = tuple(bool(x) for x in atoms.pbc)
            energy = atoms.get_potential_energy()
            fh.write(
                f"{i}\t{len(atoms)}\t"
                f"{cell[0]:.6f}\t{cell[1]:.6f}\t{cell[2]:.6f}\t"
                f"{pbc}\t{energy}\n"
            )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = base_parser("step05_validation", description=__doc__)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    logger = setup_logging(
        log_file=work_dir / "logs" / "step05_validation.log",
        level=getattr(logging, args.log_level),
        step_name="step05",
    )

    logger.info("=== Step 05 — Validation Set Extraction ===")
    logger.info("method=%s  percent=%d  seed=%s",
                cfg["validation"].get("method", "FPS"),
                cfg["validation"].get("percent", 20),
                cfg["validation"].get("seed", 42))

    try:
        run(cfg, logger)
        logger.info("=== Step 05 completed successfully ===")
    except Exception:
        logger.exception("Step 05 failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
