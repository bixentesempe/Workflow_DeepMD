"""
pipeline_utils.py
-----------------
Shared utilities: config loading, CLI override merging, and logging setup.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return it as a nested dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """Set a nested dict value using dot-notation key (e.g. 'soap.r_cut')."""
    parts = key_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    # Try to cast to numeric/bool if possible
    try:
        value = int(value)
    except (ValueError, TypeError):
        try:
            value = float(value)
        except (ValueError, TypeError):
            if isinstance(value, str) and value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif isinstance(value, str) and value.lower() == "null":
                value = None
    d[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply CLI overrides of the form ``key.subkey=value`` to the config dict.

    Example
    -------
    ``--override descriptors.soap.r_cut=8.0 sampling.seed=1``
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be in the form 'key.subkey=value', got: {item!r}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), value.strip())
    return cfg


def resolve_work_dir(cfg: dict) -> Path:
    """Return the absolute work_dir from config, creating it if needed."""
    work_dir = Path(cfg.get("work_dir", ".")).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    step_name: str = "",
) -> logging.Logger:
    """
    Configure a logger that writes to both stdout and an optional log file.

    Parameters
    ----------
    log_file:   Path to the log file. If None, only stdout is used.
    level:      Logging level (default INFO).
    step_name:  Name used as logger name and in the file stem.
    """
    logger = logging.getLogger(step_name or "pipeline")
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Common CLI base parser
# ---------------------------------------------------------------------------

def base_parser(step_name: str, description: str) -> argparse.ArgumentParser:
    """
    Return an ArgumentParser pre-loaded with the common --config and --override
    arguments shared by every step script.
    """
    parser = argparse.ArgumentParser(
        prog=step_name,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one or more config values using dot-notation, e.g. "
            "--override descriptors.soap.r_cut=8.0 sampling.seed=1"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser
