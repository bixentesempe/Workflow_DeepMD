#!/usr/bin/env python3

"""Fail-fast validation for the QBC active-learning runtime environment."""

from __future__ import annotations

import glob
import importlib
import os
import sys
import tempfile
from pathlib import Path


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "ase": "ase",
    "deepmd": "deepmd-kit",
}

SUPPORTED_PYTHON = ((3, 9), (3, 13))


def _read_input_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "#" in raw:
            raw = raw.split("#", 1)[0].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def _resolve(base: Path, value: str | Path) -> Path:
    """
    Resolve a path found in input.in.

    Anchor is the directory holding input.in -- NOT the code directory.
    A value that is already absolute is returned untouched.
    """
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def main(config_path: str) -> int:
    problems: list[str] = []
    mpl_dir = Path(tempfile.gettempdir()) / "qbc-active-learning-mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    py_min, py_max = SUPPORTED_PYTHON
    py_now = sys.version_info[:2]
    print("[check] validating Python version")
    if not (py_min <= py_now < py_max):
        problems.append(
            "unsupported Python version: "
            f"{sys.version.split()[0]} (supported range is >= {py_min[0]}.{py_min[1]}, "
            f"< {py_max[0]}.{py_max[1]})"
        )
    else:
        print(f"[ok] Python {sys.version.split()[0]}")

    print("[check] validating Python imports")
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            problems.append(f"missing import: {module_name} (install package `{package_name}`) -> {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"[ok] {module_name} {version}")

    input_path = Path(config_path).resolve()
    base = input_path.parent

    if not input_path.exists():
        problems.append(f"missing input.in: {input_path}")
    else:
        print(f"[check] validating input.in references : {input_path}")
        cfg = _read_input_config(input_path)

        model_value = cfg.get("MODELS", "")
        model_paths = [_resolve(base, p.strip()) for p in model_value.split(",") if p.strip()]
        if len(model_paths) < 2:
            problems.append("MODELS must contain at least two model paths")
        else:
            missing_models = [str(path) for path in model_paths if not path.exists()]
            if missing_models:
                problems.append(f"missing model files: {', '.join(missing_models)}")
            else:
                print(f"[ok] found {len(model_paths)} committee model files")
                try:
                    from deepmd.infer import DeepPot

                    DeepPot(str(model_paths[0]))
                except Exception as exc:
                    problems.append(
                        "failed to load first DeePMD model with its runtime backend "
                        f"({model_paths[0]}) -> {exc}"
                    )
                else:
                    print(f"[ok] DeePMD backend can load {model_paths[0]}")

        pool_pattern = cfg.get("POOL", "")
        if not pool_pattern:
            problems.append("POOL is not set in input.in")
        else:
            pool_glob = pool_pattern if os.path.isabs(pool_pattern) else str(base / pool_pattern)
            matches = glob.glob(pool_glob, recursive=True)
            if not matches:
                problems.append(f"POOL pattern matched no files: {pool_glob}")
            else:
                print(f"[ok] POOL matched {len(matches)} trajectory file(s) from {pool_glob}")

        type_map_value = cfg.get("TYPE_MAP", "")
        if not type_map_value:
            problems.append("TYPE_MAP is not set in input.in")
        elif "," not in type_map_value:
            type_map_path = _resolve(base, type_map_value)
            if not type_map_path.exists():
                problems.append(f"TYPE_MAP file does not exist: {type_map_path}")
            else:
                print(f"[ok] TYPE_MAP file exists: {type_map_path}")
        else:
            type_names = [item.strip() for item in type_map_value.split(",") if item.strip()]
            if not type_names:
                problems.append("TYPE_MAP inline list is empty")
            else:
                print(f"[ok] TYPE_MAP inline list has {len(type_names)} entries")

    if problems:
        print("[fail] environment validation found problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("[ok] environment looks ready for run_qbc.py")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: validate_environment.py <path/to/input.in>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
