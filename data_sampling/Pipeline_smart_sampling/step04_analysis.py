"""
step04_analysis.py
------------------
Energy and force-norm distribution comparison between the full dataset
and the sampled structures. Saves histograms and a summary report.

WHY THIS STEP EXISTS, ON TOP OF STEP 03's "COVERAGE"
-------------------------------------------------------
Step 03 optimises coverage in SOAP/PCA (descriptor) space — it doesn't know
what energy or force values each structure has. This step is the sanity
check on the physical properties that actually matter for training: does the
sampled subset still span the full dataset's energy and per-species force
range, or did the sampler accidentally leave out (e.g.) the highest-energy
tail? A structurally diverse sample is not automatically a physically
representative one.

NOTE — "coverage" here means something DIFFERENT from step 03. There it was
the fraction of PCA bins touched (distribution shape). Here `_coverage_minmax`
is just the overlap of [min, max] ranges between sample and full dataset —
a much cruder "did we lose the extremes" check, not a shape comparison (the
saved histograms are what let you eyeball the shape).

Usage
-----
    python step04_analysis.py
    python step04_analysis.py --config my_cfg.yaml
    python step04_analysis.py --override analysis.bin_width_energy=0.05
"""

from __future__ import annotations

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

from src.workflow_config import discover_structure_files
from src.sampling_analysis_utils import energies, force_norms_by_species, load_traj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coverage_minmax(full_vals: np.ndarray, samp_vals: np.ndarray) -> float:
    f = full_vals[~np.isnan(full_vals)]
    s = samp_vals[~np.isnan(samp_vals)]
    if len(f) == 0 or len(s) == 0:
        return float("nan")
    fmin, fmax = float(f.min()), float(f.max())
    smin, smax = float(s.min()), float(s.max())
    frange = fmax - fmin
    if frange <= 0:
        return float("nan")
    covered = max(0.0, min(smax, fmax) - max(smin, fmin))
    return covered / frange


def _stats(vals: np.ndarray, name: str, logger: logging.Logger) -> dict:
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        logger.info("%s: no data", name)
        return {}
    qs = np.quantile(v, [0.05, 0.5, 0.95])
    stats = {
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "std": float(v.std()),
        "q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2]),
    }
    logger.info(
        "%s: min=%.4g  max=%.4g  mean=%.4g  std=%.4g  q05=%.4g  q50=%.4g  q95=%.4g",
        name, stats["min"], stats["max"], stats["mean"], stats["std"],
        stats["q05"], stats["q50"], stats["q95"],
    )
    return stats


def _histogram(
    full_vals: np.ndarray,
    samp_vals: np.ndarray,
    bin_width: float,
    xlabel: str,
    title: str,
    save_path: Path,
    full_label: str = "FULL",
    samp_label: str = "SAMPLED",
) -> None:
    arr = np.concatenate([full_vals[~np.isnan(full_vals)], samp_vals[~np.isnan(samp_vals)]])
    if len(arr) == 0:
        return
    lo, hi = arr.min(), arr.max()
    n_bins = max(1, int(np.ceil((hi - lo) / bin_width)))
    edges = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots()
    if len(full_vals[~np.isnan(full_vals)]):
        ax.hist(full_vals[~np.isnan(full_vals)], bins=edges, alpha=0.5, label=full_label, density=True)
    if len(samp_vals[~np.isnan(samp_vals)]):
        ax.hist(samp_vals[~np.isnan(samp_vals)], bins=edges, alpha=0.5, label=samp_label, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(cfg: dict, logger: logging.Logger) -> dict:
    """Execute the sampling analysis step."""
    acfg = cfg["analysis"]
    work_dir = resolve_work_dir(cfg)

    selected_dir = work_dir / acfg.get("selected_dir", "selected")
    out_dir = work_dir / acfg.get("output_dir", "selected")
    plots_dir = out_dir / "analysis_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load full dataset
    # ------------------------------------------------------------------
    analysis_roots = [work_dir / Path(r) for r in acfg.get("input_roots", ["data"])]
    patterns = acfg.get("patterns", ["vasprun*.xml", "XDATCAR*", "*.xyz", "*.extxyz"])
    vasp_files = discover_structure_files(
        [str(p) for p in analysis_roots], patterns=patterns, recursive=True
    )
    logger.info("Found %d input files for full dataset.", len(vasp_files))
    full = load_traj(vasp_files)
    logger.info("Full dataset: %d structures", len(full))

    # ------------------------------------------------------------------
    # 2. Load sampled dataset — try FPS first, then Random, then any .traj
    # ------------------------------------------------------------------
    # step03 names its export after the sampling method (e.g. FPS_selected.traj,
    # KPP_selected.traj — see export_selected_structures in step03). FPS/Random
    # are just checked first as the most common cases; any other method still
    # gets picked up by the "*_selected.traj" fallback glob below.
    method_trajs = {
        "FPS": selected_dir / "FPS_selected.traj",
        "Random": selected_dir / "Random_selected.traj",
    }
    sampl_path = None
    for name, path in method_trajs.items():
        if path.exists():
            sampl_path = path
            logger.info("Using sampled trajectory: %s (%s)", path, name)
            break

    if sampl_path is None:
        # Fallback: any .traj in selected_dir
        trajs = sorted(selected_dir.glob("*_selected.traj"))
        if trajs:
            sampl_path = trajs[0]
            logger.info("Using sampled trajectory (fallback): %s", sampl_path)
        else:
            raise FileNotFoundError(
                f"No sampled trajectory found in {selected_dir}. Run step03 first."
            )

    sampl = load_traj(sampl_path)
    logger.info("Sampled dataset: %d structures", len(sampl))

    # ------------------------------------------------------------------
    # 3. Energy analysis
    # ------------------------------------------------------------------
    E_full = energies(full)
    E_samp = energies(sampl)
    bin_width_en = float(acfg.get("bin_width_energy", 0.02))

    cov_E = _coverage_minmax(E_full, E_samp)
    logger.info("Energy range coverage (sample vs full): %.2f%%", cov_E * 100)
    _stats(E_full, "E_full", logger)
    _stats(E_samp, "E_samp", logger)

    _histogram(
        E_full, E_samp,
        bin_width=bin_width_en,
        xlabel="Energy (eV)",
        title="Energy distribution",
        save_path=plots_dir / "energy_distribution.png",
    )
    logger.info("Energy histogram saved.")

    # ------------------------------------------------------------------
    # 4. Force analysis by species
    # ------------------------------------------------------------------
    F_full = force_norms_by_species(full)
    F_samp = force_norms_by_species(sampl)
    bin_width_f = float(acfg.get("bin_width_force", 0.05))

    for sp in sorted(set(F_full) | set(F_samp)):
        f_full = F_full.get(sp, np.array([], dtype=float))
        f_samp = F_samp.get(sp, np.array([], dtype=float))
        cov_F = _coverage_minmax(f_full, f_samp)
        logger.info("Force coverage [%s]: %.2f%%", sp, cov_F * 100)
        _stats(f_full, f"F_full[{sp}]", logger)
        _stats(f_samp, f"F_samp[{sp}]", logger)

        _histogram(
            f_full, f_samp,
            bin_width=bin_width_f,
            xlabel="|F| (eV/Å)",
            title=f"Force distribution: {sp}",
            save_path=plots_dir / f"force_distribution_{sp}.png",
            full_label=f"{sp} FULL",
            samp_label=f"{sp} SAMPLED",
        )
    logger.info("Force histograms saved.")

    # ------------------------------------------------------------------
    # 5. Summary report
    # ------------------------------------------------------------------
    n_full, n_samp = len(full), len(sampl)
    pct = 100.0 * n_samp / n_full if n_full else 0.0
    report_path = out_dir / "sampling_analysis_report.txt"
    with report_path.open("w") as fh:
        fh.write(f"Sampling Analysis Report\n{'='*40}\n")
        fh.write(f"Full dataset:    {n_full} structures\n")
        fh.write(f"Sampled dataset: {n_samp} structures ({pct:.2f}%)\n")
        fh.write(f"Energy coverage: {cov_E:.2%}\n\n")
        fh.write("Force coverage by species:\n")
        for sp in sorted(set(F_full) | set(F_samp)):
            f_full = F_full.get(sp, np.array([], dtype=float))
            f_samp = F_samp.get(sp, np.array([], dtype=float))
            cov_F = _coverage_minmax(f_full, f_samp)
            fh.write(f"  {sp}: {cov_F:.2%}\n")
    logger.info("Analysis report saved → %s", report_path)

    return {"report": report_path, "plots_dir": plots_dir}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = base_parser("step04_analysis", description=__doc__)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    logger = setup_logging(
        log_file=work_dir / "logs" / "step04_analysis.log",
        level=getattr(logging, args.log_level),
        step_name="step04",
    )

    logger.info("=== Step 04 — Sampling Analysis ===")

    try:
        run(cfg, logger)
        logger.info("=== Step 04 completed successfully ===")
    except Exception:
        logger.exception("Step 04 failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
