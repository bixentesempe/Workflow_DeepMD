from __future__ import annotations

# %% [markdown]
# # HPC version — QBC selector for DeePMD → POSCAR exporter
# ## **Purpose**: score MD frames with a DeePMD committee, pick uncertain ones, export as POSCAR + selection.csv.
# ### **No training. No job launch.**

# %%
# %%
# --- Imports

import os, glob, shutil, math, json, tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

np = None
pd = None
read = None
write = None
symbols2numbers = None
Atoms = None
DeepPot = None


def load_runtime_dependencies():
    """Import third-party runtime dependencies with a clear failure message."""
    global np, pd, read, write, symbols2numbers, Atoms, DeepPot
    try:
        import numpy as np
        import pandas as pd
        from ase.io import read, write
        from ase.symbols import symbols2numbers
        from ase import Atoms
        from deepmd.infer import DeepPot
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Run `python -m qbc_runtime.validate_environment` "
            "and install the required packages before launching the selector."
        ) from exc
    return np, pd, read, write, symbols2numbers, Atoms, DeepPot

# %%
# ----------------------------
# Config parsing
# ----------------------------
def _coerce_scalar(v: str):
    v = v.strip()
    if v.lower() == "none":
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except Exception:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v

def _coerce_list(v: str):
    return [s.strip() for s in v.split(",") if s.strip()]

def read_input(path: str) -> Dict[str, Any]:
    print(f"[INFO] Reading input configuration from: {path}")
    cfg: Dict[str, Any] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            # remove inline comments
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            if "=" not in raw:
                continue
            k, v = raw.split("=", 1)
            k, v = k.strip(), v.strip()
            if not k:
                continue
            if k == "MODELS":
                cfg[k] = [s.strip() for s in v.split(",") if s.strip()]
            elif k in {"POOL"}:
                cfg[k] = v
            else:
                cfg[k] = _coerce_scalar(v)
            print(f"[CFG] {k:<15} = {cfg[k]}")

    # defaults
    defaults = {
        "ASE_FORMAT": "lammps-dump-text",
        "METRIC": "score",
        "TOP_K": None,
        "W_FORCE": 1.0,
        "W_ENE": 0.0,
        "W_VIR": 0.0,
        "FORCE_REDUCER": "max",
        "SKIP": 0,
        "STRIDE": 1,
        "LIMIT": None,
        "OUT_DIR": "./qbc_poscars",
        "OUT_DIR_XYZ": "./qbc_XYZ",
        "OUT_DIR_RESULTS": "./qbc_outputs",   # new outputs directory
        "TAG": "",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # sanity checks
    if cfg.get("THRESH_LOW") is None and cfg.get("THRESH_HIGH") is None and cfg.get("TOP_K") is None:
        raise ValueError("Provide THRESH_LOW/THRESH_HIGH or TOP_K in input.in")
    if cfg.get("THRESH_LOW") is not None and cfg.get("THRESH_HIGH") is not None:
        if float(cfg["THRESH_LOW"]) > float(cfg["THRESH_HIGH"]):
            raise ValueError("THRESH_LOW must be <= THRESH_HIGH")

    print(f"[INFO] Input file parsed successfully.")
    print(f"[INFO] Threshold window: low={cfg.get('THRESH_LOW')} high={cfg.get('THRESH_HIGH')}")
    print(f"[INFO] Metric: {cfg['METRIC']}")
    print(f"[INFO] Models: {len(cfg['MODELS'])} file(s)")
    print(f"[INFO] Pool pattern: {cfg['POOL']}")
    print(f"[INFO] Output dirs: {cfg['OUT_DIR']}, {cfg['OUT_DIR_XYZ']}, {cfg['OUT_DIR_RESULTS']}")
    print(f"[INFO] --- End of configuration ---")
    return cfg

# %%
# ----------------------------
# Utilities
# ----------------------------
def _parse_type_map(type_map) -> List[str]:
    if isinstance(type_map, (list, tuple)):
        return [str(s) for s in type_map]
    p = Path(str(type_map))
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [s.strip() for s in str(type_map).split(",") if s.strip()]

def ase_to_deepmd_arrays(atoms, type_map: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = atoms.get_positions().astype(np.float64)
    cell = atoms.cell.array.astype(np.float64)
    allowed = symbols2numbers(type_map)
    Z = atoms.get_atomic_numbers()
    lut = {Zsym: i for i, Zsym in enumerate(allowed)}
    try:
        atype = np.array([lut[z] for z in Z], dtype=np.int32)
    except KeyError:
        missing = sorted(set(Z) - set(allowed))
        raise ValueError(f"Atomic numbers not in type-map: {missing}")
    return coords, cell, atype

def committee_predict(models, coords: np.ndarray, cell: np.ndarray, atype: np.ndarray):
    """
    Evaluate a committee of DeePMD models on the same structure.
    Returns energies [eV], forces [eV/Å], virials [eV].
    """
    E, F, V = [], [], []
    for m in models:
        e, f, v = m.eval(coords, cell, atype)
        # ensure scalar energy to avoid DeprecationWarning
        e_scalar = float(np.asarray(e).item())
        E.append(e_scalar)
        F.append(np.array(f, dtype=np.float64))
        V.append(np.array(v, dtype=np.float64))
    return np.array(E), np.stack(F, axis=0), np.stack(V, axis=0)

# def disagreement_components(E: np.ndarray, F: np.ndarray, V: np.ndarray, reducer: str = "max"):
#     fnorm = np.linalg.norm(F, axis=-1)  # (M,N)
#     std_f = np.std(fnorm, axis=0)       # (N,)
#     dF_max = float(np.max(std_f))
#     dF_mean = float(np.mean(std_f))
#     dF = dF_max if reducer == "max" else dF_mean
#     dE = float(np.std(E))
#     vnorm = np.linalg.norm(V.reshape(V.shape[0], -1), axis=1)
#     dV = float(np.std(vnorm))
#     return dF, dF_max, dE, dV, std_f

def disagreement_components(E: np.ndarray, F: np.ndarray, V: np.ndarray,
                            reducer: str = "max",
                            force_std_mode: str = "mag"):
    """
    force_std_mode:
      - 'mag'       : std over models of |F| per atom           (your current)
      - 'comp_norm' : L2-norm of component-wise std per atom    (DP-GEN-like)
      - 'comp_max'  : max component-wise std per atom
    """
    # F shape: (M, N, 3)
    if force_std_mode == "mag":
        # std of |F| across models
        fnorm = np.linalg.norm(F, axis=-1)         # (M,N)
        std_f = np.std(fnorm, axis=0)              # (N,)
    elif force_std_mode == "comp_norm":
        # std per component, then L2 across x,y,z
        std_comp = np.std(F, axis=0)               # (N,3)
        std_f = np.linalg.norm(std_comp, axis=1)   # (N,)
    elif force_std_mode == "comp_max":
        std_comp = np.std(F, axis=0)               # (N,3)
        std_f = np.max(std_comp, axis=1)           # (N,)
    else:
        raise ValueError(f"Unknown FORCE_STD_MODE={force_std_mode}")

    dF_max = float(np.max(std_f))
    dF_mean = float(np.mean(std_f))
    dF = dF_max if reducer == "max" else dF_mean

    dE = float(np.std(E))
    if V is None:
        dV = 0.0
    else:
        vnorm = np.linalg.norm(V.reshape(V.shape[0], -1), axis=1)
        dV = float(np.std(vnorm))

    return dF, dF_max, dE, dV, std_f

def score_from_components(dF: float, dE: float, dV: float, wF: float, wE: float, wV: float) -> float:
    return wF * dF + wE * dE + wV * dV

def ensure_clean_dir(path: Path):
    if path.exists():
        for f in path.iterdir():
            if f.is_file() or f.is_symlink():
                try:
                    f.unlink()
                except PermissionError:
                    try:
                        os.chmod(f, 0o700)
                        f.unlink()
                    except PermissionError as exc:
                        raise PermissionError(
                            f"Cannot clean output file {f}. Check write permissions for {path}."
                        ) from exc
            elif f.is_dir():
                try:
                    shutil.rmtree(f, onerror=_rmtree_force_writable)
                except PermissionError as exc:
                    raise PermissionError(
                        f"Cannot clean output directory {f}. Check write permissions for {path}."
                    ) from exc
    else:
        path.mkdir(parents=True)


def _rmtree_force_writable(func, path, exc_info):
    os.chmod(path, 0o700)
    func(path)


# %%
# ----------------------------
# Main
# ----------------------------
def main():
    print("[INFO] ===== QBC ACTIVE LEARNING SELECTION START =====")
    load_runtime_dependencies()

    cfg = read_input("input.in")

    TYPE_MAP = _parse_type_map(cfg["TYPE_MAP"])
    MODELS = cfg["MODELS"]
    if MODELS is None or len(MODELS) < 2:
        raise ValueError("Need at least 2 models for a committee")

    FILES = sorted(glob.glob(cfg["POOL"], recursive=True))
    if not FILES:
        raise FileNotFoundError(f"No input files matched POOL={cfg['POOL']}")
    print(f"[INFO] Found {len(FILES)} trajectory file(s).")

    OUT_DIR = Path(str(cfg["OUT_DIR"]))
    OUT_DIR_XYZ = Path(str(cfg["OUT_DIR_XYZ"]))
    OUT_DIR_RESULTS = Path(str(cfg["OUT_DIR_RESULTS"]))
    ensure_clean_dir(OUT_DIR_RESULTS)
    ensure_clean_dir(OUT_DIR)
    ensure_clean_dir(OUT_DIR_XYZ)

    print(f"[INFO] Output directories ready:")
    print(f"        POSCARs → {OUT_DIR}")
    print(f"        XYZ     → {OUT_DIR_XYZ}")
    print(f"        Results → {OUT_DIR_RESULTS}")

    with open(OUT_DIR / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Output directories prepared and configuration saved.")

    # --- Load committee models ---
    print(f"[INFO] Loading {len(MODELS)} DeePMD models ...")
    print("")
    COMMITTEE = [DeepPot(p) for p in MODELS]
    print("[INFO] All models loaded successfully.")

    # --- Scanning trajectories ---
    print("[INFO] Starting scoring of trajectories ...")
    SKIP, STRIDE = int(cfg["SKIP"]), int(cfg["STRIDE"])
    LIMIT = None if cfg["LIMIT"] is None else int(cfg["LIMIT"])
    ASE_FORMAT = None if cfg["ASE_FORMAT"] in (None, "None") else str(cfg["ASE_FORMAT"])
    FORCE_REDUCER = str(cfg["FORCE_REDUCER"]).lower()
    W_FORCE, W_ENE, W_VIR = float(cfg["W_FORCE"]), float(cfg["W_ENE"]), float(cfg["W_VIR"])

    #rows = []
    #atoms_refs = []
    #gidx = 0

    rows = []
    # old: atoms_refs = []
    atoms_bank = []          # aligned with rows
    stdF_bank = []           # per-atom std_f arrays
    localScore_bank = []     # per-atom composite proxy arrays
    gidx = 0

    # scan files
    for f_id, fpath in enumerate(FILES):
        print(f"[INFO] ({f_id}/{len(FILES)}) Reading: {fpath}")
        #seq = read(fpath, format=ASE_FORMAT, index=":", specorder=TYPE_MAP) if ASE_FORMAT else read(fpath, index=":")
        _FORMATS_NO_SPECORDER = {"traj", "db", "json", "cif"}

        if ASE_FORMAT:
            read_kwargs = {"format": ASE_FORMAT, "index": ":"}
            if ASE_FORMAT not in _FORMATS_NO_SPECORDER:
                read_kwargs["specorder"] = TYPE_MAP
            seq = read(fpath, **read_kwargs)
        else:
            seq = read(fpath, index=":")
        if not isinstance(seq, (list, tuple)):
            seq = [seq]
        seq = seq[SKIP::STRIDE]

        for i, at in enumerate(seq):
            if LIMIT is not None and gidx >= LIMIT:
                break
            N = len(at)
            coords, cell, atype = ase_to_deepmd_arrays(at, TYPE_MAP)
            E, F, V = committee_predict(COMMITTEE, coords, cell, atype)
            force_std_mode = str(cfg.get("FORCE_STD_MODE", "mag"))
            #dF, dF_max, dE, dV, std_f = disagreement_components(E, F, V, reducer=FORCE_REDUCER)
            dF, dF_max, dE, dV, std_f = disagreement_components(E, F, V,
                                                    reducer=FORCE_REDUCER,
                                                    force_std_mode=force_std_mode)
            score = score_from_components(dF, dE, dV, W_FORCE, W_ENE, W_VIR)

                # per-atom composite proxy (same as notebook)
            shared = (W_ENE * dE + W_VIR * dV) / np.sqrt(max(1, N))
            local_score = W_FORCE * std_f + shared

            rows.append({
                "global_idx": gidx,
                "seq_idx": i,
                "file": fpath,
                "frame_abs": i * STRIDE + SKIP,
                "n_atoms": N,
                "score": score,
                "dF": dF,
                "dF_max": dF_max,
                "dE": dE,
                "dV": dV
            })
            # keep a small cache for std_f only if potentially needed later to avoid recompute
            #atoms_refs.append((fpath, i, std_f.astype(float)))

            atoms_bank.append(at)                      # cache Atoms
            stdF_bank.append(std_f.astype(float))      # cache per-atom std
            localScore_bank.append(local_score.astype(float))
            gidx += 1

        if LIMIT is not None and gidx >= LIMIT:
            break

    if not rows:
        raise RuntimeError("No frames scanned")

    print(f"[INFO] Total frames scored: {len(rows)}")


    df = pd.DataFrame(rows)

    # sort by METRIC
    METRIC = str(cfg["METRIC"])
    order = np.argsort(df[METRIC].to_numpy())[::-1]
    df = df.iloc[order].reset_index(drop=True)
    #atoms_refs = [atoms_refs[j] for j in order]
    atoms_bank      = [atoms_bank[j] for j in order]
    stdF_bank       = [stdF_bank[j] for j in order]
    localScore_bank = [localScore_bank[j] for j in order]

    # write core tables for later plotting
    scores_csv = OUT_DIR_RESULTS / "scores.csv"
    df.to_csv(scores_csv, index=False)

    S = df[METRIC].to_numpy().astype(float)
    np.save(OUT_DIR_RESULTS / f"{METRIC}_values.npy", S)
    counts, edges = np.histogram(S, bins=100)
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}) \
      .to_csv(OUT_DIR_RESULTS / f"{METRIC}_histogram.csv", index=False)

    # selection by user-specified thresholds
    low = cfg.get("THRESH_LOW")
    high = cfg.get("THRESH_HIGH")
    topk = cfg.get("TOP_K")

    if low is not None and high is not None:
        sel_idx = np.where((S >= float(low)) & (S <= float(high)))[0]
    elif low is not None and high is None:
        sel_idx = np.where(S >= float(low))[0]
    elif low is None and high is not None:
        sel_idx = np.where(S <= float(high))[0]
    else:
        k = min(int(topk), len(df))
        sel_idx = np.arange(k)

    sel_idx_sorted = np.sort(sel_idx)

    # write selection.csv
    df_sel = df.iloc[sel_idx_sorted].copy()
    df_sel.insert(0, "rank", np.arange(len(df_sel)))
    sel_csv = OUT_DIR_RESULTS / "selection.csv"
    df_sel.to_csv(sel_csv, index=False)

    # export POSCARs and marked selection XYZ
    TAG = str(cfg["TAG"]).strip()
    suffix = f"_{TAG}" if TAG else ""
    WRITE_XYZ_TRAJ = True
    XYZ_TRAJ_PATH = OUT_DIR_XYZ / "marked_selection.xyz"
    if WRITE_XYZ_TRAJ and XYZ_TRAJ_PATH.exists():
        XYZ_TRAJ_PATH.unlink()

    print(f"[INFO] Exporting POSCARs and EXTXYZ for {len(sel_idx_sorted)} selected frames ...")

    for r, j in enumerate(sel_idx_sorted):
        at = atoms_bank[j]  # no reload
        N  = len(at)

        subdir_name = f"POSCAR_rank{r:05d}_g{int(df.iloc[j]['global_idx']):07d}{suffix}"
        subdir_path = OUT_DIR / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)
        write(subdir_path / "POSCAR", at, format="vasp", sort=False, vasp5=True, direct=True)

        if WRITE_XYZ_TRAJ:
            std_f = np.asarray(stdF_bank[j]).reshape(-1)
            local_score = np.asarray(localScore_bank[j]).reshape(-1)

            if std_f.size != N or local_score.size != N:
                # rare shape mismatch → recompute once
                coords, cell, atype = ase_to_deepmd_arrays(at, TYPE_MAP)
                E, F, V = committee_predict(COMMITTEE, coords, cell, atype)
                fn = np.linalg.norm(np.asarray(F), axis=-1)
                std_f = np.std(fn, axis=0)
                dE = float(df.iloc[j]["dE"]); dV = float(df.iloc[j]["dV"])
                shared = (W_ENE * dE + W_VIR * dV) / np.sqrt(max(1, N))
                local_score = W_FORCE * std_f + shared

            i_mark = int(np.argmax(local_score))
            at2 = at.copy()
            at2.new_array("std_f", std_f.astype(float))
            at2.new_array("local_score", local_score.astype(float))
            syms = at2.get_chemical_symbols(); syms[i_mark] = "He"; at2.set_chemical_symbols(syms)

            comment = (
                f"rank={r}; gidx={int(df.iloc[j]['global_idx'])}; "
                f"marked0={i_mark}; score={float(df.iloc[j]['score']):.6g}; "
                f"dFmax={float(df.iloc[j]['dF_max']):.6g}"
            )
            write(XYZ_TRAJ_PATH, at2, format="extxyz", append=True, comment=comment)

        if (r + 1) % 20 == 0:
            print(f"        Exported {r + 1} structures ...")


    summary = {
        "n_files": len(FILES),
        "n_frames": len(df),
        "metric": METRIC,
        "threshold_low": low,
        "threshold_high": high,
        "selected_count": int(len(sel_idx_sorted)),
        "outputs": {
            "scores_csv": str(scores_csv),
            "histogram_csv": str(OUT_DIR_RESULTS / f"{METRIC}_histogram.csv"),
            "metric_values_npy": str(OUT_DIR_RESULTS / f"{METRIC}_values.npy"),
            "selection_csv": str(sel_csv),
            "poscar_dir": str(OUT_DIR),
            "marked_xyz": str(XYZ_TRAJ_PATH) if WRITE_XYZ_TRAJ else None,
            "results_dir": str(OUT_DIR_RESULTS)
        }
    }
    with open(OUT_DIR_RESULTS / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Scored {len(df)} frames. Selected {len(sel_idx_sorted)}.")
    print(f"Scores: {scores_csv}")
    print(f"Histogram: {OUT_DIR_RESULTS / f'{METRIC}_histogram.csv'}")
    print(f"Selection: {sel_csv}")
    print(f"POSCAR dir: {OUT_DIR}")
    if WRITE_XYZ_TRAJ:
        print(f"Marked EXTXYZ: {XYZ_TRAJ_PATH}")

if __name__ == "__main__":
    main()
