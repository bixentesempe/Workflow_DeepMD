"""
run_pipeline.py
---------------
Orchestrator: run the full pipeline or selected steps in sequence.

Usage
-----
    # Run the entire pipeline
    python run_pipeline.py

    # Run only steps 1, 2 and 3
    python run_pipeline.py --steps 1 2 3

    # Run with a custom config and override one parameter
    python run_pipeline.py --config my_config.yaml --override sampling.method=kpp

    # Dry-run: print what would be executed without running anything
    python run_pipeline.py --dry-run

    # Skip a step even if it is in the list (useful for testing)
    python run_pipeline.py --skip 4

Available steps
---------------
    1  Descriptor computation  (step01_descriptors.py)
    2  Dimensionality reduction (step02_dim_reduction.py)
    3  Sampling                (step03_sampler.py)
    4  Sampling analysis       (step04_analysis.py)
    5  Validation extraction   (step05_validation.py)
    6  Test-set generation     (step06_test_set.py)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable

from pipeline_utils import (
    apply_overrides,
    load_config,
    resolve_work_dir,
    setup_logging,
)

# Import each step's run() function
import step01_descriptors
import step02_dim_reduction
import step03_sampler
import step04_analysis
import step05_validation
import step06_test_set


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

# Single source of truth mapping a step number to its display name and its
# run(cfg, logger) entry point. Both the orchestrator loop below and the CLI
# (--steps / --skip choices) read from this dict, so adding/removing a step
# only means editing it here.
STEPS: dict[int, tuple[str, Callable]] = {
    1: ("SOAP Descriptor Computation", step01_descriptors.run),
    2: ("Dimensionality Reduction",    step02_dim_reduction.run),
    3: ("Sampling",                    step03_sampler.run),
    4: ("Sampling Analysis",           step04_analysis.run),
    5: ("Validation Set Extraction",   step05_validation.run),
    6: ("Test Set Generation",         step06_test_set.run),
}

ALL_STEPS = list(STEPS.keys())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: dict,
    steps: list[int],
    skip: list[int],
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """
    Execute the selected pipeline steps in order.

    Returns True if all steps succeeded, False otherwise.
    """
    to_run = [s for s in sorted(steps) if s not in skip]

    logger.info("Pipeline steps to run: %s", to_run)
    if skip:
        logger.info("Steps skipped: %s", skip)

    separator = "─" * 60
    all_ok = True
    t_pipeline_start = time.perf_counter()

    for step_id in to_run:
        name, run_fn = STEPS[step_id]
        logger.info(separator)
        logger.info("▶  Step %d — %s", step_id, name)
        logger.info(separator)

        if dry_run:
            logger.info("  [DRY-RUN] would call %s.run(cfg, logger)", run_fn.__module__)
            continue

        t_start = time.perf_counter()
        try:
            run_fn(cfg, logger)
            elapsed = time.perf_counter() - t_start
            logger.info("✔  Step %d completed in %.1f s", step_id, elapsed)
        except Exception:
            elapsed = time.perf_counter() - t_start
            logger.exception("✘  Step %d FAILED after %.1f s", step_id, elapsed)
            all_ok = False
            # Abort remaining steps — a later step almost certainly depends on this one
            remaining = [s for s in to_run if s > step_id]
            if remaining:
                logger.error(
                    "Aborting pipeline. Remaining steps %s will not run.", remaining
                )
            break

    total = time.perf_counter() - t_pipeline_start
    logger.info(separator)
    if all_ok and not dry_run:
        logger.info("Pipeline finished successfully in %.1f s", total)
    elif dry_run:
        logger.info("Dry-run complete. No steps were actually executed.")
    else:
        logger.error("Pipeline finished with errors after %.1f s", total)

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=ALL_STEPS,
        choices=ALL_STEPS,
        metavar="N",
        help=f"Steps to run (default: all = {ALL_STEPS}). Space-separated list of integers.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        type=int,
        default=[],
        choices=ALL_STEPS,
        metavar="N",
        help="Steps to skip even if they appear in --steps.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config values using dot-notation, e.g. "
            "--override descriptors.soap.r_cut=8.0 sampling.seed=1"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which steps would run without executing them.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    logger = setup_logging(
        log_file=work_dir / "logs" / "pipeline.log",
        level=getattr(logging, args.log_level),
        step_name="pipeline",
    )

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║           Data Sampling Pipeline             ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Config: %s", args.config)
    logger.info("Work dir: %s", work_dir)

    if args.override:
        logger.info("Active overrides: %s", args.override)

    ok = run_pipeline(
        cfg=cfg,
        steps=args.steps,
        skip=args.skip or [],
        logger=logger,
        dry_run=args.dry_run,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
