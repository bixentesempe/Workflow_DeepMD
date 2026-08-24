"""
step03_sampler.py
-----------------
Sample structures from the embedded descriptor space using coverage-based
strategies (FPS, grid, kpp, kmedoids, density_fps, hdbscan).

WHAT "COVERAGE" MEANS HERE
---------------------------
Each principal component (PC) of the step-02 embedding is cut into bins
(`coverage_mode`/`target_width`/`min_bins`/`max_bins`). "Coverage" of a PC is
the fraction of its bins touched by the current sample; `mean_cov` averages
that across all PCs. High coverage means the selection spans the full spread
of the dataset along every axis, not just its densest region — that's the
whole point of sampling instead of taking a random subset.

TWO WAYS TO REACH THE COVERAGE TARGET (`sampling.auto_strategy`)
-------------------------------------------------------------------
  "seeded" (default, recommended) — grow the sample INCREMENTALLY: draw a
      batch of `batch` new points from the still-unsampled data, merge with
      what's already selected, recompute coverage, repeat until
      `target_mean_cov` is reached (or the plateau/`k_max_seeded` stop
      kicks in). Previously selected points are never thrown away.
  "resample" — NOT incremental: restart from scratch each round with a
      larger k (`k_start`, then × `growth` each round, or +`check_every` if
      set), and keep only the final draw. Simpler, but wasteful for large k
      since every round re-does the sampling of all previous points too.

If `sampling.auto_sampling` is false, none of the above runs: a single
fixed-size draw of `fixed_k` points is taken instead, no coverage target
involved.

Usage
-----
    python step03_sampler.py
    python step03_sampler.py --config my_cfg.yaml
    python step03_sampler.py --override sampling.method=kpp sampling.fixed_k=30000
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import (
    apply_overrides,
    base_parser,
    load_config,
    resolve_work_dir,
    setup_logging,
)

from src.sampler_utils import (
    atoms_to_structures,
    load_reduced,
    pc_coverage_bins_auto,
    sample,
    sample_to_coverage,
    sample_to_coverage_seeded_batches,
)
from src.plots_exports_utils import plot_sampling_pca, export_selected_structures
from src.workflow_config import ensure_directory


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(cfg: dict, logger: logging.Logger) -> dict:
    """
    Execute the sampling step.

    Returns
    -------
    dict with paths to saved artefacts.
    """
    scfg = cfg["sampling"]
    work_dir = resolve_work_dir(cfg)

    desc_dir = work_dir / scfg.get("input_dir", "desc")
    emb_dir = work_dir / scfg.get("embedding_dir", "embedding")
    out_dir = work_dir / scfg.get("output_dir", "selected")
    ensure_directory(str(out_dir))

    # ------------------------------------------------------------------
    # 1. Load embedding or raw descriptors
    # ------------------------------------------------------------------
    use_embedding = scfg.get("use_embedding", True)

    if use_embedding:
        npy_files = sorted(emb_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file in {emb_dir}. Run step02 first.")
        if len(npy_files) > 1:
            raise RuntimeError(f"Multiple .npy files in {emb_dir}: {npy_files}. Keep only one.")
        npy_file = npy_files[0]
        logger.info("Loading embedding from %s", npy_file)
    else:
        npy_files = sorted(desc_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy descriptor file in {desc_dir}. Run step01 first.")
        npy_file = npy_files[0]
        logger.info("Loading raw descriptors from %s", npy_file)

    X = load_reduced(npy_file)
    logger.info("Loaded matrix  shape=%s", X.shape)

    # ------------------------------------------------------------------
    # 2. Load provenance and filemap
    # ------------------------------------------------------------------
    prov_candidates = list(desc_dir.glob("*.parquet")) + list(desc_dir.glob("*_provenance*.csv"))
    if not prov_candidates:
        raise FileNotFoundError(f"No provenance table in {desc_dir}. Run step01 first.")
    prov_candidates = sorted(prov_candidates)
    prov_file = next((p for p in prov_candidates if p.suffix == ".parquet"), prov_candidates[0])

    import pandas as pd
    meta = pd.read_parquet(prov_file) if prov_file.suffix == ".parquet" else pd.read_csv(prov_file)
    logger.info("Loaded provenance from %s  rows=%d", prov_file.name, len(meta))

    json_files = [p for p in desc_dir.glob("*.json") if "filemap" in p.name]
    if not json_files:
        raise FileNotFoundError(f"No filemap JSON in {desc_dir}. Run step01 first.")
    with open(json_files[0]) as fh:
        filemap = json.load(fh)

    # ------------------------------------------------------------------
    # 3. Whiten
    # ------------------------------------------------------------------
    # Z-score each column so no single PC dominates distances/coverage bins
    # just because it happens to have a larger numeric range.
    X = np.asarray(X, dtype=np.float64, order="C")
    if not (np.isnan(X).any() or np.isinf(X).any()):
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        X = (X - mu) / sd
    else:
        logger.warning("NaN/Inf detected in embedding — skipping whitening.")
    logger.info("Whitened X  shape=%s  abs_max=%.4f", X.shape, float(np.nanmax(np.abs(X))))

    # ------------------------------------------------------------------
    # 4. Sampling
    # ------------------------------------------------------------------
    method = scfg.get("method", "fps").lower()
    seed = int(scfg.get("seed", 0))
    progress = bool(scfg.get("progress", True))
    auto_sampling = bool(scfg.get("auto_sampling", True))
    auto_strategy = scfg.get("auto_strategy", "seeded")

    coverage_kwargs = dict(
        coverage_mode=scfg.get("coverage_mode", "width"),
        target_width=float(scfg.get("target_width", 0.10)),
        min_bins=int(scfg.get("min_bins", 20)),
        max_bins=int(scfg.get("max_bins", 10_000)),
    )

    if auto_sampling:
        target_mean_cov = float(scfg.get("target_mean_cov", 0.99))
        # Plateau stop: once mean_cov is already above cov_thresh, stop early
        # if it improves by less than delta_cov over the last plateau_window
        # rounds — avoids grinding toward target_mean_cov with diminishing
        # returns once most of the space is already covered.
        enable_plateau = bool(scfg.get("enable_plateau_stop", True))
        cov_thresh = float(scfg.get("coverage_threshold_min", 0.95))
        plateau_window = int(scfg.get("plateau_window", 5))
        delta_cov = float(scfg.get("delta_cov", 0.002))

        # See module docstring for "seeded" vs "resample" — seeded is the
        # recommended default (incremental, keeps prior picks).
        if auto_strategy == "seeded":
            logger.info("[%s] Auto-sampling with seeded-batches strategy", method.upper())
            idx_atoms, info = sample_to_coverage_seeded_batches(
                X, method=method,
                target_mean_cov=target_mean_cov,
                enable_plateau_stop=enable_plateau,
                coverage_threshold_min=cov_thresh,
                plateau_window=plateau_window,
                delta_cov=delta_cov,
                k_start=int(scfg.get("k_start", 100)),
                batch=int(scfg.get("batch", 100)),
                k_max=int(scfg.get("k_max_seeded", 200_000)),
                seed=seed, progress=progress,
                **coverage_kwargs,
            )
        else:
            logger.info("[%s] Auto-sampling with resample strategy", method.upper())
            idx_atoms, info = sample_to_coverage(
                X, method=method,
                target_mean_cov=target_mean_cov,
                enable_plateau_stop=enable_plateau,
                coverage_threshold_min=cov_thresh,
                plateau_window=plateau_window,
                delta_cov=delta_cov,
                k_start=int(scfg.get("k_start", 100)),
                k_max=int(scfg.get("k_max", 200_000)),
                growth=float(scfg.get("growth", 1.25)),
                check_every=scfg.get("check_every"),
                seed=seed, progress=progress,
                **coverage_kwargs,
            )

        per_pc = info["per_pc"]
        mean_cov = info["mean_cov"]
        bins_used = info["bins_used"]
        coverage_history = info["history"]
        logger.info(
            "[%s] Auto-stop at k=%d  mean_cov=%.4f  reason=%s",
            method.upper(), info["k"], mean_cov, info.get("stop_reason", "unknown"),
        )

    else:
        fixed_k = int(scfg.get("fixed_k", 50_000))
        logger.info("[%s] Fixed-k sampling  k=%d", method.upper(), fixed_k)
        idx_atoms = sample(X, method=method, k=fixed_k, seed=seed, progress=progress)
        per_pc, mean_cov, bins_used = pc_coverage_bins_auto(X, idx_atoms, **coverage_kwargs)
        coverage_history = [{"k": int(len(idx_atoms)), "mean_cov": float(mean_cov)}]
        logger.info("[%s] mean_cov=%.4f", method.upper(), mean_cov)

    logger.info("Sampled %d atoms", len(idx_atoms))

    # ------------------------------------------------------------------
    # 5. Lift atom indices → structure list
    # ------------------------------------------------------------------
    chosen_structs = atoms_to_structures(idx_atoms, meta, filemap, choose="all")

    # ------------------------------------------------------------------
    # 6. Save raw index files
    # ------------------------------------------------------------------
    np.save(out_dir / f"sampled_atom_indices_{method}.npy", idx_atoms)
    chosen_structs.to_csv(out_dir / f"sampled_structures_{method}.csv", index=False)
    logger.info("Saved atom indices and structure CSV.")

    # ------------------------------------------------------------------
    # 7. Coverage-evolution plot
    # ------------------------------------------------------------------
    steps = [pt["k"] for pt in coverage_history]
    mean_coverages = [pt["mean_cov"] for pt in coverage_history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, mean_coverages, marker="o", linewidth=2)
    ax.set_xlabel("Sampling step (number of sampled atoms)")
    ax.set_ylabel("Mean coverage")
    ax.set_title(f"Coverage evolution — {method.upper()}")
    ax.grid(True, alpha=0.3)
    if auto_sampling:
        ax.axhline(scfg.get("target_mean_cov", 0.99), color="tab:red",
                   linestyle="--", linewidth=1, label=f"target = {scfg.get('target_mean_cov', 0.99):.2f}")
        if scfg.get("enable_plateau_stop", True):
            ax.axhline(scfg.get("coverage_threshold_min", 0.95), color="tab:orange",
                       linestyle=":", linewidth=1, label=f"plateau check = {scfg.get('coverage_threshold_min', 0.95):.2f}")
        ax.legend(frameon=False)
    fig.tight_layout()
    cov_plot_path = out_dir / f"{method}_coverage_evolution.png"
    fig.savefig(cov_plot_path, dpi=300)
    plt.close(fig)
    logger.info("Coverage-evolution plot saved → %s", cov_plot_path)

    # ------------------------------------------------------------------
    # 8. PCA scatter plot of sampled atoms
    # ------------------------------------------------------------------
    try:
        plot_sampling_pca(X, idx_atoms, method=method, per_pc=per_pc,
                          npy_file=npy_file, outdir=out_dir)
    except Exception as exc:
        logger.warning("Could not produce PCA scatter plot: %s", exc)

    # ------------------------------------------------------------------
    # 9. Export .traj / .xyz if requested
    # ------------------------------------------------------------------
    artifacts: dict[str, Path] = {
        "atom_indices": out_dir / f"sampled_atom_indices_{method}.npy",
        "structures_csv": out_dir / f"sampled_structures_{method}.csv",
        "coverage_plot": cov_plot_path,
    }

    if scfg.get("save_files", True):
        paths = export_selected_structures(
            chosen_structs,
            method_name=method.upper(),
            outdir=out_dir,
            extra_formats=("xyz",),
            load_mode="auto",
        )
        artifacts.update(paths)
        logger.info("Exported structures: %s", paths)

    return artifacts


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = base_parser("step03_sampler", description=__doc__)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    logger = setup_logging(
        log_file=work_dir / "logs" / "step03_sampler.log",
        level=getattr(logging, args.log_level),
        step_name="step03",
    )

    logger.info("=== Step 03 — Sampler ===")
    logger.info("Config: %s  method: %s", args.config, cfg["sampling"].get("method", "fps"))

    try:
        run(cfg, logger)
        logger.info("=== Step 03 completed successfully ===")
    except Exception:
        logger.exception("Step 03 failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
