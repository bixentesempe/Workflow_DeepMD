from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence
import json

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.io import read
from tqdm.auto import tqdm

from src.desc_comp_utils import fixed_mask


def select_structure_files(
    paths: Sequence[str | Path],
    mode: str = "all",
    random_count: int | None = None,
    file_stride: int | None = None,
    random_seed: int = 0,
) -> List[Path]:
    """
    Select a subset of structure files from an already-discovered path list.

    Parameters
    ----------
    paths
        Input file paths, typically already sorted by discovery.
    mode
        Selection mode: "all", "random", or "stride".
    random_count
        Number of files to sample when mode="random".
    file_stride
        Keep every Nth file when mode="stride" (indices 0, N, 2N, ...).
    random_seed
        Seed used for reproducible random selection.
    """
    selected = [Path(path).expanduser().resolve() for path in paths]
    if mode == "all":
        return selected
    if mode == "random":
        if random_count is None:
            raise ValueError("random_count must be provided when mode='random'.")
        if random_count <= 0:
            raise ValueError("random_count must be >= 1 when mode='random'.")
        if random_count > len(selected):
            raise ValueError(
                f"random_count ({random_count}) cannot exceed the number of files ({len(selected)})."
            )
        rng = np.random.default_rng(random_seed)
        picked = rng.choice(len(selected), size=random_count, replace=False)
        return [selected[i] for i in sorted(picked.tolist())]
    if mode == "stride":
        if file_stride is None:
            raise ValueError("file_stride must be provided when mode='stride'.")
        if file_stride <= 0:
            raise ValueError("file_stride must be >= 1 when mode='stride'.")
        return selected[::file_stride]
    raise ValueError(
        f"Unsupported file selection mode: {mode!r}. "
        "Expected one of {'all', 'random', 'stride'}."
    )


def load_structure_sets(
    paths: Sequence[str | Path],
    frame_stride: int = 1,
    return_frame_indices: bool = False,
    show_progress: bool = False,
) -> List[List[Atoms]] | tuple[List[List[Atoms]], List[List[int]]]:
    """
    Load each file in `paths` as a list of ASE Atoms objects.

    Parameters
    ----------
    frame_stride
        Keep every Nth frame from each file (0, N, 2N, ...). Use 1 to keep all.
    return_frame_indices
        If True, also return the original frame indices kept from each file.
    show_progress
        If True, show a per-file progress bar while loading structures.
    """
    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1.")

    out: List[List[Atoms]] = []
    frame_ids: List[List[int]] = []
    progress = tqdm(paths, desc="Loading structures", unit="file") if show_progress else paths
    for path in progress:
        path_obj = Path(path).expanduser().resolve()
        if show_progress:
            progress.set_description(f"Loading structure from {path_obj.name}")
        frames = read(str(path_obj), index=":")
        if isinstance(frames, Atoms):
            full_structures = [frames]
        else:
            full_structures = list(frames)
        kept_ids = list(range(0, len(full_structures), frame_stride))
        out.append([full_structures[i] for i in kept_ids])
        frame_ids.append(kept_ids)
    if return_frame_indices:
        return out, frame_ids
    return out


def unique_species(structure_sets: Sequence[Sequence[Atoms]]) -> List[str]:
    symbols = {
        symbol
        for struct_list in structure_sets
        for atoms in struct_list
        for symbol in atoms.get_chemical_symbols()
    }
    return sorted(symbols)


def build_filemap(paths: Sequence[str | Path]) -> Dict[int, str]:
    return {i: str(Path(path).expanduser().resolve()) for i, path in enumerate(paths)}


def build_provenance_table(
    structure_sets: Sequence[Sequence[Atoms]],
    paths: Sequence[str | Path],
    fixed_mask_fn: Callable[[Atoms], np.ndarray] = fixed_mask,
    source_struct_ids: Sequence[Sequence[int]] | None = None,
) -> pd.DataFrame:
    if len(structure_sets) != len(paths):
        raise ValueError(
            f"structure_sets and paths must have the same length: "
            f"{len(structure_sets)} != {len(paths)}"
        )
    if source_struct_ids is not None and len(source_struct_ids) != len(structure_sets):
        raise ValueError(
            f"source_struct_ids and structure_sets must have the same length: "
            f"{len(source_struct_ids)} != {len(structure_sets)}"
        )
    rows: List[pd.DataFrame] = []
    for file_id, struct_list in enumerate(structure_sets):
        source_ids = None if source_struct_ids is None else source_struct_ids[file_id]
        if source_ids is not None and len(source_ids) != len(struct_list):
            raise ValueError(
                f"source_struct_ids[{file_id}] and structure_sets[{file_id}] must have the same length: "
                f"{len(source_ids)} != {len(struct_list)}"
            )
        for struct_id, atoms in enumerate(struct_list):
            n_atoms = len(atoms)
            row = {
                "file_id": file_id,
                "struct_id": struct_id,
                "atom_id": np.arange(n_atoms, dtype=np.int32),
                "symbol": atoms.get_chemical_symbols(),
                "is_fixed": fixed_mask_fn(atoms),
            }
            if source_ids is not None:
                row["source_struct_id"] = np.full(n_atoms, source_ids[struct_id], dtype=np.int32)
            rows.append(pd.DataFrame(row))
    if not rows:
        columns = ["file_id", "struct_id", "atom_id", "symbol", "is_fixed"]
        if source_struct_ids is not None:
            columns.append("source_struct_id")
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def serialize_config(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def clear_directory_outputs(
    outdir: str | Path,
    patterns: Sequence[str],
) -> List[str]:
    """
    Remove previously generated files matching the provided glob patterns.

    Returns the list of removed file paths as strings.
    """
    outdir = Path(outdir).expanduser().resolve()
    removed: List[str] = []
    if not outdir.exists():
        return removed

    seen: set[Path] = set()
    for pattern in patterns:
        for path in outdir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            path.unlink()
            seen.add(path)
            removed.append(str(path))
    return removed


def save_descriptor_run(
    outdir: str | Path,
    descriptors: np.ndarray,
    provenance: pd.DataFrame,
    filemap: Dict[int, str],
    run_config,
    base_name: str,
    timestamp: str | None = None,
    clear_existing: bool = False,
) -> Dict[str, str]:
    """
    Save the descriptor matrix, provenance table, file map, and a config log.
    """
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_directory_outputs(
            outdir,
            patterns=(
                "*.npy",
                "*_provenance.parquet",
                "*_provenance.csv",
                "*_filemap.json",
                "*_params.txt",
                "*_config.json",
            ),
        )
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{base_name}_{timestamp}"

    npy_path = outdir / f"{prefix}.npy"
    parquet_path = outdir / f"{prefix}_provenance.parquet"
    csv_path = outdir / f"{prefix}_provenance.csv"
    filemap_path = outdir / f"{prefix}_filemap.json"
    log_path = outdir / f"{prefix}_params.txt"
    config_path = outdir / f"{prefix}_config.json"

    np.save(npy_path, descriptors)
    try:
        provenance.to_parquet(parquet_path, index=False)
        provenance_path = parquet_path
    except Exception:
        provenance.to_csv(csv_path, index=False)
        provenance_path = csv_path
    with filemap_path.open("w") as fh:
        json.dump(filemap, fh, indent=2)
    with log_path.open("w") as fh:
        fh.write(f"Run: {base_name}\n")
        fh.write(f"Timestamp: {timestamp}\n")
        fh.write(f"Descriptor rows: {descriptors.shape[0]}\n")
        fh.write(f"Descriptor cols: {descriptors.shape[1]}\n")
        fh.write("\nConfig:\n")
        fh.write(json.dumps(serialize_config(run_config), indent=2, sort_keys=True))
    with config_path.open("w") as fh:
        json.dump(serialize_config(run_config), fh, indent=2, sort_keys=True)

    return {
        "npy": str(npy_path),
        "provenance": str(provenance_path),
        "filemap": str(filemap_path),
        "log": str(log_path),
        "config": str(config_path),
    }
