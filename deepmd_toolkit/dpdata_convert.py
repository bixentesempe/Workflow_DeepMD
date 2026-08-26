    #!/usr/bin/env python3
"""
dpdata_convert.py
-----------------
Convert ASE .traj files to DeepMD npy format.

On the sampled files apply this script (you must have the files selected with *.traj dataset)

- "*train*" → BASE_DIR/trainset/set.000/
- "*val*"   → BASE_DIR/valset/set.000/

A global TYPE_MAP is enforced on both datasets.

Usage
-----
    python dpdata_convert.py
    python dpdata_convert.py --input-dir selected --base-dir deepmd_data
    python dpdata_convert.py --type-map H W O
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


# ============================================================
#  CONFIGURATION
# ============================================================

INPUT_DIR = Path("selected")     # root directory to search for .traj files (recursive)
BASE_DIR  = Path("deepmd_data")  # trainset/ and valset/ will be created here
TYPE_MAP: list[str] | None = ["H", "W", "O"]   # None = keep dpdata default


# ============================================================
#  Helpers
# ============================================================

def route(stem: str, base: Path) -> Path:
    """Return trainset/ or valset/ based on filename."""
    name = stem.lower()
    if "train" in name:
        return base / "trainset"
    elif "val" in name:
        return base / "valset"
    else:
        raise ValueError(
            f"Cannot route '{stem}': filename must contain 'train' or 'val'."
        )


def enforce_type_map(out_dir: Path, global_map: list[str], logger: logging.Logger) -> None:
    """Overwrite type_map.raw and remap type.raw if needed."""
    type_map_path = out_dir / "type_map.raw"
    type_raw_path = out_dir / "type.raw"

    if not type_map_path.exists():
        logger.warning("  type_map.raw not found in %s — skipping.", out_dir)
        return

    current_map = type_map_path.read_text().strip().splitlines()
    logger.info("  dpdata type_map : %s", current_map)
    logger.info("  global type_map : %s", global_map)

    missing = [sp for sp in current_map if sp not in global_map]
    if missing:
        raise ValueError(
            f"Species {missing} present in trajectory but missing from "
            f"global TYPE_MAP {global_map}. Add them to TYPE_MAP."
        )

    remap = {i: global_map.index(sp) for i, sp in enumerate(current_map)}
    needs_remap = any(old != new for old, new in remap.items())

    if needs_remap and type_raw_path.exists():
        types = np.loadtxt(type_raw_path, dtype=int)
        new_types = np.vectorize(remap.get)(types)
        np.savetxt(type_raw_path, new_types, fmt="%d")
        logger.info("  type.raw remapped : %s", remap)

    type_map_path.write_text("\n".join(global_map) + "\n")
    logger.info("  type_map.raw → %s", global_map)


# ============================================================
#  Core conversion
# ============================================================

def convert(
    traj_file: Path,
    base_dir: Path,
    type_map: list[str] | None,
    logger: logging.Logger,
) -> Path:
    """Convert a single .traj to DeepMD npy. Returns the output directory."""
    import dpdata
    import ase.io

    out_dir = route(traj_file.stem, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("─" * 50)
    logger.info("Input  : %s", traj_file)
    logger.info("Output : %s", out_dir)

    frames   = ase.io.read(str(traj_file), index=":")
    n_frames = len(frames)
    logger.info("Frames : %d", n_frames)

    system = dpdata.LabeledSystem(
        str(traj_file),
        fmt="ase/traj",
        set_size=n_frames,
    )

    system.to("deepmd/npy", str(out_dir))
    logger.info("DeepMD npy written → %s", out_dir)

    if type_map is not None:
        enforce_type_map(out_dir, type_map, logger)

    return out_dir


# ============================================================
#  CLI
# ============================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dpdata_convert",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir",  default=str(INPUT_DIR))
    parser.add_argument("-o", "--base-dir",   default=str(BASE_DIR))
    parser.add_argument("--type-map",   nargs="*", default=TYPE_MAP, metavar="SPECIES")
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
        stream=sys.stdout,
    )
    logger = logging.getLogger("dpdata_convert")

    input_dir = Path(args.input_dir)
    base_dir  = Path(args.base_dir)

    traj_files = sorted(input_dir.rglob("*.traj"))
    traj_files = [f for f in traj_files
                  if "train" in f.stem.lower() or "val" in f.stem.lower()]

    if not traj_files:
        logger.error("No train/val .traj files found under %s", input_dir)
        sys.exit(1)

    logger.info("Found %d file(s):", len(traj_files))
    for f in traj_files:
        logger.info("  %s", f)

    failed = []
    for traj_file in traj_files:
        try:
            convert(traj_file, base_dir, args.type_map, logger)
        except Exception:
            logger.exception("FAILED : %s", traj_file.name)
            failed.append(traj_file)

    logger.info("─" * 50)
    logger.info("Done : %d ok, %d failed.", len(traj_files) - len(failed), len(failed))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
