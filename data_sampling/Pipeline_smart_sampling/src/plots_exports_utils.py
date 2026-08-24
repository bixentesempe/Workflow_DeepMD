from __future__ import annotations
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.io import read, write
from tqdm import tqdm

### plot utils.py

def plot_sampling_pca(
    X,
    idx_atoms,
    method,
    per_pc=None,
    npy_file=None,
    max_bar=4,
    outdir: str | Path = "selected",
):
    """
    Plot PCA coverage (bar chart) and PCA scatter with sampled points overlay.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        PCA embedding matrix. Must come from PCA (d >= 2).
    idx_atoms : ndarray, shape (k,)
        Indices of sampled atoms in X.
    method : str
        Sampling method name (e.g., "fps", "adaptive_kmedoids").
    per_pc : array-like or None
        Coverage per principal component, e.g. `info["per_pc"]` returned by
        `pc_coverage_bins_auto` / `sample_to_coverage*` in sampler_utils.py.
        Optional.
    npy_file : str or None
        File name of embedding. If given, checked for "pca" in name.
    max_bar : int
        Maximum number of PCs to show in coverage bar plot.

    Raises
    ------
    RuntimeError
        If X does not look like a PCA embedding (d < 2 or file name does not contain "pca").

    Returns
    -------
    None
        Plots are shown directly.
    """
    # Guard: must be PCA
    is_pca = (X.shape[1] >= 2)
    if npy_file is not None:
        is_pca = is_pca and ("pca" in os.path.basename(npy_file).lower())
    if not is_pca:
        raise RuntimeError("Plotting requires a 2D+ PCA embedding. Load a PCA .npy first.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Coverage bar plot ---
    if per_pc is not None:
        m = min(max_bar, len(per_pc), X.shape[1])
        plt.figure(figsize=(7, 3))
        plt.bar(range(1, m + 1), per_pc[:m])
        plt.xlabel("PC", fontsize=14)
        plt.ylabel("Coverage", fontsize=14)
        plt.title("Bin coverage", fontsize=14)
        plt.tight_layout()
        plt.savefig(outdir / f"{method}_pca_coverage.png", dpi=300)

    # --- PCA scatter with sampled points ---
    fig, ax = plt.subplots(figsize=(8, 6))
    N = X.shape[0]
    if N > 2_000_000:
        hb = ax.hexbin(X[:, 0], X[:, 1], gridsize=500, bins="log")
    else:
        ax.scatter(X[:, 0], X[:, 1], s=1, alpha=0.05)

    ax.scatter(
        X[idx_atoms, 0], X[idx_atoms, 1],
        s=2, alpha=0.9, c="r", marker="o", label=f"sample ({len(idx_atoms)})"
    )

    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.set_title(f"PCA space with {len(idx_atoms)} sampled atoms ({method})", fontsize=14)
    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=False, loc="best", fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / f"{method}_pca_scatter.png", dpi=300)



## export_utils.py


#for single structure loading 

def _frame_index_col(df: pd.DataFrame) -> str:
    """Canonical frame index column for reloading source structures."""
    return "source_struct_id" if "source_struct_id" in df.columns else "struct_id"

def _load_struct(path: str, frame_idx: int) -> Atoms:
    """Read a single structure index from a vasprun.xml-like file."""
    return read(path, index=int(frame_idx))

def _collect_images(chosen_structs: pd.DataFrame) -> List[Atoms]:

    """
    chosen_structs must have columns: ['file_path','struct_id'] or
    ['file_path','source_struct_id'].
    Returns list of Atoms in row order.
    """
    frame_col = _frame_index_col(chosen_structs)
    req_cols = {"file_path", frame_col}
    missing = req_cols - set(chosen_structs.columns)
    if missing:
        raise KeyError(f"chosen_structs missing columns: {missing}")


    imgs: List[Atoms] = []
    for _, row in tqdm(chosen_structs.iterrows(), total=len(chosen_structs), desc="Loading structures"):
        imgs.append(_load_struct(row["file_path"], int(row[frame_col])))
    return imgs

def _can_write_single_xdatcar(images: List[Atoms]) -> bool:
    """XDATCAR needs constant atom count and species order across frames."""
    if not images:
        return False
    n0 = len(images[0])
    syms0 = images[0].get_chemical_symbols()
    for a in images[1:]:
        if len(a) != n0 or a.get_chemical_symbols() != syms0:
            return False
    return True

def _species_sequence(atoms: Atoms) -> Tuple[str, ...]:
    """Ordered species labels for one structure."""
    return tuple(atoms.get_chemical_symbols())

def _validate_images_for_export(images: List[Atoms]) -> None:
    """
    Basic export-time sanity checks.

    Ensures each loaded structure has a consistent atom count and symbol list.
    This does not enforce the same composition across frames; it only guarantees
    that we export the exact structure as loaded, without silent reinterpretation
    of atom types.
    """
    for i, atoms in enumerate(images):
        symbols = atoms.get_chemical_symbols()
        if len(symbols) != len(atoms):
            raise ValueError(
                f"Loaded structure {i} has inconsistent atom metadata: "
                f"{len(atoms)} atoms but {len(symbols)} symbols."
            )

def _build_export_manifest(chosen_structs: pd.DataFrame, images: List[Atoms]) -> pd.DataFrame:
    """
    Add structural identity fields to the export manifest so downstream pipelines
    can verify atom count and ordered species sequence for each exported frame.
    """
    if len(chosen_structs) != len(images):
        raise ValueError(
            f"Manifest/image length mismatch: {len(chosen_structs)} != {len(images)}"
        )

    manifest = chosen_structs.copy()
    manifest["n_atoms"] = [len(a) for a in images]
    manifest["formula"] = [a.get_chemical_formula() for a in images]
    manifest["species_sequence"] = [" ".join(_species_sequence(a)) for a in images]
    return manifest

#for multiple structure loading

def _load_structs(path: str, frame_idx_list: List[int]) -> Dict[int, Atoms]:
    """
    Read several structures at once from a file using a single slice read.
    Works with readers that don't accept list indices (e.g., VASP XML).
    Returns {frame_idx: Atoms}. Keeps calculators.
    """
    idxs = sorted({int(i) for i in frame_idx_list})
    i_min, i_max = idxs[0], idxs[-1]

    # one-shot load via slice [i_min : i_max+1]
    frames = read(path, index=slice(i_min, i_max + 1))
    if isinstance(frames, Atoms):
        frames = [frames]

    # sanity: expected count
    expected = i_max - i_min + 1
    if len(frames) != expected:
        raise RuntimeError(f"{path}: expected {expected} frames in slice, got {len(frames)}")

    # map requested absolute indices -> Atoms from the slice
    return {k: frames[k - i_min] for k in idxs}

def _group_indices(chosen_structs: pd.DataFrame) -> Dict[str, List[int]]:
    """Map file_path -> sorted unique struct indices to load."""
    frame_col = _frame_index_col(chosen_structs)
    REQ_COLS = {"file_path", frame_col}
    missing = REQ_COLS - set(chosen_structs.columns)
    if missing:
        raise KeyError(f"chosen_structs missing columns: {missing}")
    groups = defaultdict(set)
    for _, row in chosen_structs.iterrows():
        groups[row["file_path"]].add(int(row[frame_col]))
    return {fp: sorted(idxs) for fp, idxs in groups.items()}

def _collect_images_grouped(chosen_structs: pd.DataFrame) -> List[Atoms]:
    """
    Open each file once, extract requested frames, and return Atoms
    in the original row order. Duplicates return independent copies.
    """
    frame_col = _frame_index_col(chosen_structs)
    REQ_COLS = {"file_path", frame_col}
    missing = REQ_COLS - set(chosen_structs.columns)
    if missing:
        raise KeyError(f"chosen_structs missing columns: {missing}")

    by_file = _group_indices(chosen_structs)

    # Cache per-file frame dicts
    cache: Dict[str, Dict[int, Atoms]] = {}
    for fp in tqdm(by_file.keys(), desc="Reading files"):
        cache[fp] = _load_structs(fp, by_file[fp])

    # Handle duplicates by copying
    key_counts = Counter((row["file_path"], int(row[frame_col])) 
                         for _, row in chosen_structs.iterrows())

    imgs: List[Atoms] = []
    for _, row in tqdm(chosen_structs.iterrows(), total=len(chosen_structs), desc="Assembling"):
        fp = row["file_path"]
        idx = int(row[frame_col])
        try:
            a = cache[fp][idx]
        except KeyError:
            raise KeyError(f"Frame {idx} not found in {fp}")
        imgs.append(a.copy() if key_counts[(fp, idx)] > 1 else a)
    return imgs

## export selected structures

def _choose_load_mode(chosen_structs: pd.DataFrame) -> str:
    # group if any file has multiple frames
    frame_col = _frame_index_col(chosen_structs)
    counts = chosen_structs.groupby("file_path")[frame_col].nunique()
    return "grouped" if (counts > 1).any() else "single"

def export_selected_structures(
    chosen_structs: pd.DataFrame,
    method_name: str,
    outdir: str = "selected",
    extra_formats: Tuple[str, ...] = ("xyz", "xdatcar"),
    manifest: bool = True,
    write_traj: bool = True,
    load_mode: str = "auto",  # "single" | "grouped" | "auto"
) -> Dict[str, str]:
    """
    Build a merged trajectory for one method and export viewer formats.
    load_mode:
      - "single": uses _load_struct / _collect_images
      - "grouped": uses _load_structs / _collect_images_grouped
      - "auto": picks grouped if any file has >1 requested frame
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    mode = _choose_load_mode(chosen_structs) if load_mode == "auto" else load_mode
    if mode not in {"single", "grouped"}:
        raise ValueError("load_mode must be 'single', 'grouped', or 'auto'")

    images: List[Atoms] = (
        _collect_images_grouped(chosen_structs) if mode == "grouped"
        else _collect_images(chosen_structs)
    )
    _validate_images_for_export(images)
    manifest_df = _build_export_manifest(chosen_structs, images)

    out: Dict[str, str] = {}

    if write_traj:
        traj_path = os.path.join(outdir, f"{method_name}_selected.traj")
        write(traj_path, images)
        out["traj"] = traj_path

    if manifest:
        manifest_df.to_csv(
            os.path.join(outdir, f"{method_name}_selected_manifest.csv"), index=False
        )

    if "xyz" in extra_formats:
        xyz_path = os.path.join(outdir, f"{method_name}_selected.xyz")
        write(xyz_path, images)
        out["xyz"] = xyz_path

    if "xdatcar" in extra_formats:
        if _can_write_single_xdatcar(images):
            xd_path = os.path.join(outdir, f"{method_name}_selected_XDATCAR")
            write(xd_path, images, format="vasp-xdatcar")
            out["xdatcar"] = xd_path
        else:
            out["xdatcar"] = "multiple"
            # respect chosen load strategy per file as well
            for fpath, group in chosen_structs.groupby("file_path", sort=False):
                frame_col = _frame_index_col(group)
                idxs = [int(i) for i in group[frame_col].tolist()]
                if mode == "grouped":
                    fmap = _load_structs(fpath, idxs)     # one-shot
                    imgs = [fmap[i] for i in idxs]
                else:
                    imgs = [_load_struct(fpath, i) for i in idxs]
                if not _can_write_single_xdatcar(imgs):
                    continue
                stem = Path(fpath).stem.replace(".xml", "")
                xd_path = os.path.join(outdir, f"{method_name}_{stem}_XDATCAR")
                write(xd_path, imgs, format="vasp-xdatcar")
                out[f"xdatcar::{stem}"] = xd_path

    return out
