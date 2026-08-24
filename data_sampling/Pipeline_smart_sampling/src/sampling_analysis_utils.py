# sampling_analysis_utils.py
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from ase.io import read

# ---------- IO ----------

def load_traj(paths: Union[str, Sequence[str]]):
    """
    Read one or many trajectory files and FLATTEN into List[Atoms].
    Accepts a single path or a list/tuple of paths.
    """
    if isinstance(paths, (list, tuple)):
        images = []
        for p in paths:
            imgs = read(p, ":")
            # imgs may be a single Atoms or a list
            if isinstance(imgs, Atoms):
                images.append(imgs)
            else:
                images.extend(imgs)
        return images
    else:
        imgs = read(paths, ":")
        return [imgs] if isinstance(imgs, Atoms) else imgs

# ---------- ENERGY / FORCES ----------

def energies(images: List[Atoms]) -> np.ndarray:
    """
    Return per-structure potential energies if available.
    Missing values become NaN.
    """
    out = []
    for a in images:
        try:
            out.append(a.get_potential_energy())
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def force_components(images: List[Atoms]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate forces and symbols across all frames.
    Returns (F_all, symbols_all), where:
      F_all shape = (sum_i n_i, 3), symbols_all shape = (sum_i n_i,)
    Missing forces become NaN rows.
    """
    Fs, syms = [], []
    for a in images:
        try:
            F = a.get_forces(apply_constraint=False)
        except Exception:
            F = np.full((len(a), 3), np.nan)
        Fs.append(F)
        syms += a.get_chemical_symbols()
    return np.vstack(Fs), np.array(syms, dtype=object)

def force_norms_by_species(images: List[Atoms]) -> Dict[str, np.ndarray]:
    """
    Return {species: |F| vector} across all frames.
    """
    F, sym = force_components(images)
    norms = np.linalg.norm(F, axis=1)
    out: Dict[str, List[float]] = {}
    for s, f in zip(sym, norms):
        out.setdefault(s, []).append(f)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
