# samplers.py
"""
Sampling utilities for large descriptor matrices (SOAP or reduced embeddings).
Provides multiple diversity-oriented samplers and helpers to lift selected
atoms back to structures using provenance tables.

Typical workflow:
  1) X = PCA/UMAP embedding (float32, shape: [n_atoms, d]) or standardized SOAP.
  2) idx = sample(X, method="fps", k=50000)
  3) chosen_structs = atoms_to_structures(idx, metadata_df, filemap, choose="top", top_k=200)
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _pbar(iterable, total=None, desc=None, progress=True):
    return tqdm(iterable, total=total, desc=desc) if progress else iterable

# ---------- core samplers ----------

def random_sample(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Uniform random sampling without replacement.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Input matrix (descriptors or embedding).
    k : int
        Number of rows to sample (clipped to N).
    seed : int
        RNG seed.

    Returns
    -------
    idx : ndarray, shape (k,)
        Sorted indices of selected rows.

    Notes
    -----
    Complexity: O(k). Baseline for comparisons.
    """
    print("Using random sampling ...")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return np.sort(idx)

def stratified_grid(
    X: np.ndarray,
    k: int,
    dims: Tuple[int, int] = (0, 1),
    bins: Tuple[int, int] = (200, 200),
    per_bin: int = 5,
    seed: int = 0,
    progress=True,
) -> np.ndarray:
    """
    Grid-based stratified sampling on a 2D projection to cover space uniformly.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Input matrix.
    k : int
        Target sample size.
    dims : (int, int)
        Two column indices used as 2D projection (e.g., (PC1, PC2)).
    bins : (int, int)
        Number of bins along each axis.
    per_bin : int
        Max samples per non-empty bin.
    seed : int
        RNG seed.

    Returns
    -------
    idx : ndarray, shape (~k,)
        Sorted indices. If bins are sparse, fallback fills with random picks.

    Notes
    -----
    Complexity: O(N + #bins). Good for quick uniform coverage.
    """
    print("Using stratified grid sampling ...")
    rng = np.random.default_rng(seed)
    x = X[:, dims[0]]; y = X[:, dims[1]]
    xb = np.linspace(x.min(), x.max(), bins[0]+1)
    yb = np.linspace(y.min(), y.max(), bins[1]+1)
    idx_out = []
    outer = _pbar(range(bins[0]), total=bins[0], desc="Grid-x", progress=progress)
    for i in outer:
        xi = (x >= xb[i]) & (x < xb[i+1])
        inner = _pbar(range(bins[1]), total=bins[1], desc="Grid-y", progress=False if not progress else False)
        for j in inner:
            m = xi & (y >= yb[j]) & (y < yb[j+1])
            cand = np.nonzero(m)[0]
            if cand.size == 0: 
                continue
            take = min(per_bin, cand.size)
            idx_out.extend(rng.choice(cand, size=take, replace=False))
            if len(idx_out) >= k:
                return np.sort(np.array(idx_out[:k], dtype=int))
    if len(idx_out) < k:
        left = np.setdiff1d(np.arange(X.shape[0]), np.array(idx_out, dtype=int), assume_unique=False)
        fill = rng.choice(left, size=min(k - len(idx_out), left.size), replace=False)
        idx_out.extend(fill.tolist())
    return np.sort(np.array(idx_out[:k], dtype=int))

def fps(X: np.ndarray, k: int, seed: int = 0, chunk: int = 200_000, progress=True) -> np.ndarray:
    """
    Farthest-Point Sampling (k-center greedy) in Euclidean space.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Input matrix (use PCA ~20D for speed).
    k : int
        Number of points to select.
    seed : int
        RNG seed for first pivot.
    chunk : int
        Chunk size for distance updates to limit memory.

    Returns
    -------
    idx : ndarray, shape (k,)
        Sorted indices of selected rows.

    Notes
    -----
    Objective: maximize the minimum distance to the selected set.
    Complexity: O(N * k) distance ops; chunked to control memory.
    Very effective coverage for large N.
    """
    print("Using Farthest-Point Sampling (FPS) ...")
    rng = np.random.default_rng(seed)
    n = X.shape[0]; k = min(k, n)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = rng.integers(n)
    d2 = np.full(n, np.inf, dtype=np.float32)

    for t in _pbar(range(1, k), total=k-1, desc="FPS", progress=progress):
        x = X[sel[t-1]]
        for i in range(0, n, chunk):
            B = X[i:i+chunk]
            dd = np.sum((B - x)**2, axis=1)
            d2[i:i+chunk] = np.minimum(d2[i:i+chunk], dd)
        sel[t] = int(np.argmax(d2))
    return np.sort(sel)


def kmeans_pp(X: np.ndarray, k: int, seed: int = 0, progress=True) -> np.ndarray:
    """
    k-means++ seeding only. Cheap diversity similar to FPS.

    Parameters
    ----------
    X : ndarray, shape (N, d)
    k : int
        Centers to pick.
    seed : int

    Returns
    -------
    idx : ndarray, shape (k,)
        Sorted indices of chosen centers from the dataset.

    Notes
    -----
    Complexity: O(N * k).
    Good first-pass sampler when FPS is too slow.
    """
    print("Using k-means++ sampling ...")
    rng = np.random.default_rng(seed)
    n = X.shape[0]; k = min(k, n)
    centers = []
    idx = rng.integers(n); centers.append(idx)
    d2 = np.sum((X - X[idx])**2, axis=1)

    for _ in _pbar(range(1, k), total=k-1, desc="k-means++", progress=progress):
        probs = d2 / d2.sum()
        idx = int(rng.choice(n, p=probs))
        centers.append(idx)
        d2 = np.minimum(d2, np.sum((X - X[idx])**2, axis=1))
    return np.sort(np.array(centers, dtype=int))


def kmeans_medoids(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    subsample: int = 200_000,
    progress=True
) -> np.ndarray:
    """
    k-means on a subset, then return nearest real points (medoids) in full X.

    Parameters
    ----------
    X : ndarray, shape (N, d)
    k : int
        Number of representatives to return.
    seed : int
    subsample : int
        If N > subsample, fit k-means on a random subset for speed.

    Returns
    -------
    idx : ndarray, shape (k,)
        Sorted indices of medoids in X.

    Notes
    -----
    Steps: fit k-means → find nearest neighbor in X to each centroid.
    Balances speed and representativeness. Requires scikit-learn.
    """
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    print("Using k-means medoids sampling ...")
    rng = np.random.default_rng(seed)
    if X.shape[0] > subsample:
        take = rng.choice(X.shape[0], size=subsample, replace=False)
        Xs = X[take]
    else:
        take = None; Xs = X
    # Fit (no native progress)
    km = KMeans(n_clusters=min(k, Xs.shape[0]), n_init=10, random_state=seed).fit(Xs)
    # Nearest real points (batched progress)
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    centers = km.cluster_centers_
    # batch knn for progress feedback
    batch = 1024
    idx_list = []
    for i in _pbar(range(0, centers.shape[0], batch), total=(centers.shape[0]+batch-1)//batch,
                   desc="Medoids KNN", progress=progress):
        _, idx = nn.kneighbors(centers[i:i+batch])
        idx_list.append(idx.ravel())
    return np.sort(np.concatenate(idx_list))


def density_aware_fps(
    X: np.ndarray,
    k: int,
    dims: Tuple[int, int] = (0, 1),
    bins: int = 256,
    seed: int = 0,
    progress=True
) -> np.ndarray:
    """
    Density-aware farthest-point sampling using a 2D density reweighting.

    Parameters
    ----------
    X : ndarray, shape (N, d)
    k : int
        Number of points to select.
    dims : (int, int)
        Two columns used to estimate density (typically PC1, PC2).
    bins : int
        Histogram bins per axis.
    seed : int

    Returns
    -------
    idx : ndarray, shape (k,)
        Sorted indices.

    Notes
    -----
    Computes a 2D histogram on (dims), sets per-point weight w=1/sqrt(density),
    rescales X by w, then runs FPS in the weighted space.
    Mitigates oversampling dense regions.
    """
    print("Using density-aware FPS ...")
    u = X[:, dims[0]]
    v = X[:, dims[1]]
    H, ux, vx = np.histogram2d(u, v, bins=bins)
    H += 1e-12  # avoid div by zero
    ui = np.clip(np.digitize(u, ux) - 1, 0, bins - 1)
    vi = np.clip(np.digitize(v, vx) - 1, 0, bins - 1)
    w = 1.0 / np.sqrt(H[ui, vi])
    Xw = (X.T * w).T
    return fps(Xw, k, seed=seed, progress=progress)


def hdbscan_medoids(
    X: np.ndarray,
    min_cluster_size: int = 50,
    per_cluster: int = 1,
    seed: int = 0,
    progress=True
) -> np.ndarray:
    """
    Medoid selection per HDBSCAN cluster (+ some noise points).

    Parameters
    ----------
    X : ndarray, shape (N, d)
    min_cluster_size : int
        HDBSCAN parameter controlling cluster granularity.
    per_cluster : int
        Number of representatives per cluster (≥1).
    seed : int

    Returns
    -------
    idx : ndarray
        Sorted indices of chosen medoids and some noise points.

    Notes
    -----
    For each cluster: choose point nearest to cluster centroid (approximate medoid).
    Adds a small sample from noise label -1. Requires `hdbscan`.
    """
    try:
        import hdbscan
    except Exception as e:
        raise RuntimeError("Install hdbscan to use hdbscan_medoids") from e
    print("Using HDBSCAN medoids sampling ...")
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    idx_out = []
    rng = np.random.default_rng(seed)
    clabs = np.unique(labels[labels >= 0])
    for lab in _pbar(clabs, total=len(clabs), desc="HDBSCAN medoids", progress=progress):
        idx = np.where(labels == lab)[0]
        if idx.size == 0: 
            continue
        c = X[idx].mean(axis=0)
        j = idx[np.argmin(np.sum((X[idx] - c)**2, axis=1))]
        idx_out.append(j)
        if per_cluster > 1 and idx.size > 1:
            rest = np.setdiff1d(idx, np.array([j]))
            take = min(per_cluster - 1, rest.size)
            idx_out.extend(rng.choice(rest, size=take, replace=False))
    noise = np.where(labels == -1)[0]
    if noise.size:
        take = min(len(idx_out)//4 + 1, noise.size)
        idx_out.extend(rng.choice(noise, size=take, replace=False))
    return np.sort(np.array(idx_out, dtype=int))

#### --------- adaptive k selection ----------

def _sanitize_X(X, clip=None):
    X = np.asarray(X, dtype=np.float32, order="C")
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if clip is not None:
        np.clip(X, -clip, clip, out=X)
    return X

def _take_subset(X, n, seed):
    rng = np.random.default_rng(seed)
    if X.shape[0] > n:
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx]
    return X

def _tss_on(Xs):
    mu = Xs.mean(axis=0, dtype=np.float64)
    return float(((Xs - mu) ** 2).sum())

def _kmeans_inertia_centers(Xs, k, seed):
    import warnings
    Xs64 = np.asarray(Xs, dtype=np.float64, order="C")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*matmul.*")
        km = KMeans(n_clusters=min(k, Xs64.shape[0]),
                    n_init=10, random_state=seed, algorithm="elkan").fit(Xs64)
    return float(km.inertia_), km.cluster_centers_.astype(np.float32, copy=False)

def _nearest_medoids_full(X, centers):
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    _, idx = nn.kneighbors(centers)
    return idx.ravel()

def adaptive_kmeans_medoids(
    X, target_explained=0.95, k_start=16, k_max=100_000,
    growth=1.5, seed=0, subsample=200_000, progress=True, standardize=True
):
    """
    Auto-select k: increase k until 1 - inertia/TSS >= target_explained.
    Returns medoid indices in X and diagnostics in info.
    """
    X = _sanitize_X(X, clip=1e6)
    if standardize:
        mu = X.mean(0, keepdims=True)
        std = X.std(0, keepdims=True) + 1e-12
        X = (X - mu) / std

    Xs = _take_subset(X, subsample, seed)
    if Xs.shape[0] < 2:
        raise ValueError("Not enough valid rows for clustering.")
    tss = _tss_on(Xs)

    k = max(2, int(k_start))
    history = []
    bar = tqdm(total=None, desc="Adaptive k-means", disable=not progress)

    while True:
        inertia, centers = _kmeans_inertia_centers(Xs, k, seed)
        explained = float(np.clip(1.0 - inertia / tss, 0.0, 1.0))
        history.append((k, explained))
        bar.set_postfix({"k": k, "expl": f"{explained:.3f}"})

        if explained >= target_explained or k >= k_max:
            bar.close()
            medoid_idx = np.sort(_nearest_medoids_full(X, centers))
            info = {
                "k": int(k),
                "explained": explained,
                "inertia_subset": float(inertia),
                "tss_subset": float(tss),
                "subset_size": int(Xs.shape[0]),
                "history": history,
            }
            return medoid_idx, info

        k = int(min(k_max, max(k + 1, np.ceil(k * growth))))

# ---------- dispatcher ----------

def sample(X: np.ndarray, method: str, k: int, **kwargs) -> np.ndarray:
    """
    Dispatch to a sampling strategy by name.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Input matrix (prefer PCA-reduced for speed).
    method : {"random","grid","fps","kpp","kmedoids","density_fps","hdbscan"}
        Sampling strategy.
    k : int
        Target number of selected rows (some methods may return ≤k if N is small).
    **kwargs :
        Extra parameters per method:
          - seed:int, chunk:int, dims:(int,int), bins:int or (int,int),
            per_bin:int, subsample:int, min_cluster_size:int, per_cluster:int

    Returns
    -------
    idx : ndarray
        Sorted indices into rows of X.
    """

    m = method.lower()
    progress = kwargs.get("progress", True)
    if m == "random":
        return random_sample(X, k, seed=kwargs.get("seed", 0))
    if m == "grid":
        return stratified_grid(X, k,
                               dims=kwargs.get("dims", (0,1)),
                               bins=kwargs.get("bins", (200,200)),
                               per_bin=kwargs.get("per_bin", 5),
                               seed=kwargs.get("seed", 0),
                               progress=progress)
    if m == "fps":
        return fps(X, k, seed=kwargs.get("seed", 0),
                   chunk=kwargs.get("chunk", 200_000),
                   progress=progress)
    if m == "kpp":
        return kmeans_pp(X, k, seed=kwargs.get("seed", 0), progress=progress)
    if m == "kmedoids":
        return kmeans_medoids(X, k, seed=kwargs.get("seed", 0),
                              subsample=kwargs.get("subsample", 200_000),
                              progress=progress)
    if m == "density_fps":
        return density_aware_fps(X, k,
                                 dims=kwargs.get("dims", (0,1)),
                                 bins=kwargs.get("bins", 256),
                                 seed=kwargs.get("seed", 0),
                                 progress=progress)
    if m == "hdbscan":
        return hdbscan_medoids(X,
                               min_cluster_size=kwargs.get("min_cluster_size", 50),
                               per_cluster=kwargs.get("per_cluster", 1),
                               seed=kwargs.get("seed", 0),
                               progress=progress)
    
    if m == "adaptive_kmedoids":
        idx, info = adaptive_kmeans_medoids(
            X,
            target_explained=kwargs.get("target_explained", 0.95),
            k_start=kwargs.get("k_start", 16),
            k_max=kwargs.get("k_max", 100000),
            growth=kwargs.get("growth", 1.5),
            seed=kwargs.get("seed", 0),
            subsample=kwargs.get("subsample", 200000),
            progress=progress,
            standardize=kwargs.get("standardize", True),
        )
        # you can return idx only, or (idx, info); keeping idx for API consistency
        return idx
    
    raise ValueError(f"Unknown method: {method}")

def sample_to_coverage(
    X,
    method="fps",
    target_mean_cov=0.90,          # e.g., 90% mean PCA-bin coverage
    enable_plateau_stop=False,
    coverage_threshold_min=None,
    plateau_window=5,
    delta_cov=0.002,
    k_start=5_000,
    k_max=200_000,
    growth=1.3,                    # multiplicative growth of k each round
    check_every=None,              # if set (e.g., 5_000), use additive growth instead
    seed=0,
    coverage_mode="width",         # passed to pc_coverage_bins_auto
    target_width=0.10,
    min_bins=20,
    max_bins=10_000,
    progress=True,
    **kwargs,                      # forwarded to sample()
):
    """
    Iteratively increase k, resample, and stop when mean PCA coverage >= target_mean_cov.
    Works with any sampling method in `sample(...)`.
    This function IS NOT INCREMENTAL - each iteration resamples from scratch with larger k.
    """
    if enable_plateau_stop:
        if coverage_threshold_min is None:
            raise ValueError("coverage_threshold_min must be provided when enable_plateau_stop=True.")
        if coverage_threshold_min > target_mean_cov:
            raise ValueError("coverage_threshold_min must be <= target_mean_cov.")
        if plateau_window < 2:
            raise ValueError("plateau_window must be >= 2 when enable_plateau_stop=True.")
        if delta_cov < 0:
            raise ValueError("delta_cov must be >= 0 when enable_plateau_stop=True.")

    def coverage(X, idx):
        per_pc, mean_cov, bins_used = pc_coverage_bins_auto(
            X, idx, mode=coverage_mode, target_width=target_width,
            min_bins=min_bins, max_bins=max_bins
        )
        return per_pc, mean_cov, bins_used

    def plateau_reached(history):
        if not enable_plateau_stop or len(history) < plateau_window:
            return False
        window = history[-plateau_window:]
        current_cov = window[-1]["mean_cov"]
        if current_cov < coverage_threshold_min:
            return False
        cov_values = [point["mean_cov"] for point in window]
        return (max(cov_values) - min(cov_values)) <= delta_cov

    k = int(k_start)
    best = {"k": 0, "idx": None, "per_pc": None, "mean_cov": 0.0, "bins_used": None}
    history = []

    while k <= k_max:
        idx = sample(X, method=method, k=k, seed=seed, progress=progress, **kwargs)
        per_pc, mean_cov, bins_used = coverage(X, idx)
        history.append({"k": int(k), "mean_cov": float(mean_cov)})

        if progress:
            print(f"[{method}] k={k:,}  mean_cov={mean_cov:.3f}  bins_used={bins_used[:8]}")

        if mean_cov >= target_mean_cov:
            return idx, {
                "k": k,
                "per_pc": per_pc,
                "mean_cov": mean_cov,
                "bins_used": bins_used,
                "history": history,
                "stop_reason": "target_reached",
            }

        if mean_cov > best["mean_cov"]:
            best = {"k": k, "idx": idx, "per_pc": per_pc, "mean_cov": mean_cov, "bins_used": bins_used}

        if plateau_reached(history):
            if progress:
                print(
                    f"[{method}] plateau stop at k={k:,}: "
                    f"mean_cov window span <= {delta_cov:.4f} after reaching {coverage_threshold_min:.3f}"
                )
            return best["idx"], {
                "k": best["k"],
                "per_pc": best["per_pc"],
                "mean_cov": best["mean_cov"],
                "bins_used": best["bins_used"],
                "history": history,
                "stop_reason": "plateau",
            }

        # grow k
        if check_every is not None and check_every > 0:
            k += int(check_every)
        else:
            k = int(min(k_max, max(k + 1, round(k * growth))))

    # not reached target; return best so far
    if progress:
        print(f"Target not reached. Best mean_cov={best['mean_cov']:.3f} at k={best['k']:,}.")
    return best["idx"], {
        "k": best["k"],
        "per_pc": best["per_pc"],
        "mean_cov": best["mean_cov"],
        "bins_used": best["bins_used"],
        "history": history,
        "stop_reason": "k_max",
    }

def sample_to_coverage_seeded_batches(
    X,
    method="fps",
    target_mean_cov=0.90,
    enable_plateau_stop=False,
    coverage_threshold_min=None,
    plateau_window=5,
    delta_cov=0.002,
    k_start=10_000,
    batch=5_000,
    k_max=200_000,
    seed=0,
    progress=True,
    coverage_mode="width",
    target_width=0.10,
    min_bins=20,
    max_bins=10_000,
    **kwargs
):
    """
    Iteratively sample a dataset in batches until a target PCA coverage is reached.

    Each iteration draws new points from the remaining (unsampled) data, 
    using a different random seed to ensure diversity. The new points are 
    merged with previously sampled ones, and coverage is recalculated.
    The process stops when the mean PCA coverage ≥ target_mean_cov or 
    when the total number of sampled points reaches k_max.

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Input data matrix, typically PCA-reduced descriptors.
    method : str
        Sampling strategy passed to `sample(...)` 
        (e.g., "fps", "kpp", "kmedoids", "density_fps", "hdbscan").
    target_mean_cov : float, default=0.90
        Target average PCA-bin coverage to reach before stopping.
    k_start : int, default=10_000
        Initial number of points to sample.
    batch : int, default=5_000
        Number of new points to add per iteration if coverage is not reached.
    k_max : int, default=200_000
        Maximum total number of sampled points (stop condition).
    seed : int, default=0
        Base random seed; each batch increments the seed by +1 to ensure diversity.
    progress : bool, default=True
        If True, print coverage progress at each iteration.
    coverage_mode : {"width","bins"}, default="width"
        Mode passed to `pc_coverage_bins_auto` to define coverage calculation.
    target_width : float, default=0.10
        Bin width in PCA units (used if coverage_mode="width").
    min_bins, max_bins : int, default=(20, 10000)
        Lower and upper limits on number of bins per PCA component.
    **kwargs :
        Extra arguments passed to `sample(...)` (e.g., chunk size, subsample, etc.).

    Returns
    -------
    idx : ndarray of shape (k,)
        Indices of the final sampled points in X.
    info : dict
        Dictionary with keys:
          - "k": total number of sampled points
          - "mean_cov": final mean PCA coverage
          - "per_pc": list of per-component coverages
          - "bins_used": number of bins used per PC

    Notes
    -----
    • Each batch samples only from points not already selected.
    • Each batch uses a new seed (seed + iteration) for diversity.
    • Sampling method can be FPS, k-means++, k-medoids, etc.
    • This approach is simpler than a fully incremental FPS: 
      it avoids recomputing pairwise distances and works generically.
    """
    if enable_plateau_stop:
        if coverage_threshold_min is None:
            raise ValueError("coverage_threshold_min must be provided when enable_plateau_stop=True.")
        if coverage_threshold_min > target_mean_cov:
            raise ValueError("coverage_threshold_min must be <= target_mean_cov.")
        if plateau_window < 2:
            raise ValueError("plateau_window must be >= 2 when enable_plateau_stop=True.")
        if delta_cov < 0:
            raise ValueError("delta_cov must be >= 0 when enable_plateau_stop=True.")

    def coverage(idx):
        per_pc, mean_cov, bins_used = pc_coverage_bins_auto(
            X, idx, mode=coverage_mode, target_width=target_width,
            min_bins=min_bins, max_bins=max_bins
        )
        return per_pc, mean_cov, bins_used

    def plateau_reached(history):
        if not enable_plateau_stop or len(history) < plateau_window:
            return False
        window = history[-plateau_window:]
        current_cov = window[-1]["mean_cov"]
        if current_cov < coverage_threshold_min:
            return False
        cov_values = [point["mean_cov"] for point in window]
        return (max(cov_values) - min(cov_values)) <= delta_cov

    N = X.shape[0]
    k0 = min(k_start, k_max, N)

    # initial draw
    idx = sample(X, method=method, k=k0, seed=seed, progress=progress, **kwargs)
    sel_mask = np.zeros(N, dtype=bool)
    sel_mask[idx] = True

    per_pc, mean_cov, bins_used = coverage(idx)
    history = [{"k": int(idx.size), "mean_cov": float(mean_cov)}]
    if progress:
        print(f"[{method}-seeded] k={idx.size:,} mean_cov={mean_cov:.3f}")

    # loop: add batches from remaining pool with new seeds
    round_id = 1
    while mean_cov < target_mean_cov and idx.size < min(k_max, N):
        need = min(batch, k_max - idx.size, N - idx.size)
        if need <= 0:
            break

        remain = np.where(~sel_mask)[0]
        if remain.size == 0:
            break

        # sample only from remaining, then map back
        k_try = min(need, remain.size)
        sub = X[remain]
        cand_sub = sample(sub, method=method, k=k_try,
                          seed=seed + round_id, progress=progress, **kwargs)
        cand = remain[cand_sub]

        # add uniques (remain already ensures uniqueness)
        sel_mask[cand] = True
        idx = np.where(sel_mask)[0]

        per_pc, mean_cov, bins_used = coverage(idx)
        history.append({"k": int(idx.size), "mean_cov": float(mean_cov)})
        if progress:
            print(f"[{method}-seeded] +{k_try:,} → k={idx.size:,} mean_cov={mean_cov:.3f}")
        if plateau_reached(history):
            if progress:
                print(
                    f"[{method}-seeded] plateau stop at k={idx.size:,}: "
                    f"mean_cov window span <= {delta_cov:.4f} after reaching {coverage_threshold_min:.3f}"
                )
            return idx, {
                "k": idx.size,
                "per_pc": per_pc,
                "mean_cov": mean_cov,
                "bins_used": bins_used,
                "history": history,
                "stop_reason": "plateau",
            }
        round_id += 1

    return idx, {
        "k": idx.size,
        "per_pc": per_pc,
        "mean_cov": mean_cov,
        "bins_used": bins_used,
        "history": history,
        "stop_reason": "target_reached" if mean_cov >= target_mean_cov else "k_max",
    }

# ---------- lifting to structures ----------

def atoms_to_structures(
    sel_idx: np.ndarray,
    metadata_df,
    filemap: Dict,
    choose: str = "top",
    top_k: Optional[int] = None,
    min_atoms_per_struct: int = 1,
    file_id_col: str = "file_id",
    struct_id_col: str = "struct_id",
):
    """
    Aggregate selected atom rows to whole structures via provenance.

    Parameters
    ----------
    sel_idx : ndarray, shape (k,)
        Row indices into X (and into metadata_df) for selected atoms.
    metadata_df : pandas.DataFrame
        Provenance table with columns at least ["file_id","struct_id"].
        If "source_struct_id" is present, it is used as the canonical frame
        index for structure-level aggregation.
    filemap : dict
        Maps file_id (int or str) -> absolute file path.
    choose : {"all","top","threshold"}
        - "all": return all hit structures with their hit counts.
        - "top": return top_k structures by number of selected atoms.
        - "threshold": keep structures with >= min_atoms_per_struct hits.
    top_k : int or None
        Used when choose="top". If None, returns all, sorted by hits.
    min_atoms_per_struct : int
        Used when choose="threshold".

    Returns
    -------
    chosen : pandas.DataFrame
        Columns: ["file_id","struct_id","n_atoms_hit","file_path"], sorted.

    Notes
    -----
    Use this to move from atom-level sampling to a list of whole structures
    to extract and build an initial training set.
    """
    effective_struct_id_col = (
        "source_struct_id"
        if struct_id_col == "struct_id" and "source_struct_id" in metadata_df.columns
        else struct_id_col
    )
    hit = (
        metadata_df.iloc[sel_idx]
        .groupby([file_id_col, effective_struct_id_col], as_index=False)
        .size()
        .rename(columns={"size": "n_atoms_hit"})
    )
    if effective_struct_id_col != struct_id_col:
        hit = hit.rename(columns={effective_struct_id_col: struct_id_col})
    if choose == "all":
        chosen = hit
    elif choose == "threshold":
        chosen = hit[hit["n_atoms_hit"] >= max(1, int(min_atoms_per_struct))]
    else:  # "top"
        if top_k is None:
            top_k = len(hit)
        chosen = hit.sort_values("n_atoms_hit", ascending=False).head(top_k)

    # normalize file_id keys
    fmap = {int(k): v for k, v in filemap.items()}
    chosen["file_path"] = chosen[file_id_col].map(fmap)
    
    n_structs = len(chosen)
    print(f"Selected {n_structs} unique structures.")
    return chosen.sort_values(["file_path", struct_id_col]).reset_index(drop=True)

# ---------- convenience IO ----------

def load_reduced(path_npy: str) -> np.ndarray:
    """
    Load a reduced matrix from .npy (e.g., PCA/UMAP embedding).

    Parameters
    ----------
    path_npy : str
        Path to .npy file.

    Returns
    -------
    X : ndarray
        Loaded matrix.
    """
    return np.load(path_npy)


def pc_coverage_bins_auto(
    Z_all: np.ndarray,
    idx: np.ndarray,
    mode: str = "width",
    target_width: float = 1.0,
    min_bins: int = 10,
    max_bins: int = 200,
    eps: float = 1e-12,
):
    """
    Compute per-PC bin coverage of a sampled subset with automatic bin selection.

    Parameters
    ----------
    Z_all : ndarray, shape (N, d)
        PCA scores of all points.
    idx : ndarray, shape (k,)
        Indices of sampled points in Z_all.
    mode : {"width", "fd", "scott"}, default="width"
        Strategy to choose bin width:
        - "width": use fixed target width.
        - "fd": Freedman–Diaconis rule (robust to outliers).
        - "scott": Scott’s rule (assumes Gaussian).
    target_width : float
        Bin width when mode="width".
    min_bins : int
        Minimum number of bins.
    max_bins : int
        Maximum number of bins.
    eps : float
        Numerical safeguard for zero ranges.

    Returns
    -------
    per_pc : list of float
        Coverage fraction [0,1] per principal component.
    mean_cov : float
        Mean coverage across PCs.
    bins_used : list of int
        Number of bins used per PC.
    """
    Z_all = np.asarray(Z_all)
    Zs = Z_all[idx]
    N, d = Z_all.shape

    per_pc, bins_used = [], []

    for j in range(d):
        x = Z_all[:, j]
        s = Zs[:, j]
        xmin, xmax = np.min(x), np.max(x)
        span = max(xmax - xmin, eps)

        # --- choose bin width
        if mode == "fd":
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            width = max(2.0 * max(iqr, eps) * (N ** (-1.0 / 3.0)), eps)
        elif mode == "scott":
            std = max(np.std(x), eps)
            width = max(3.5 * std * (N ** (-1.0 / 3.0)), eps)
        else:  # "width"
            width = max(target_width, eps)

        nb = int(np.clip(np.ceil(span / width), min_bins, max_bins))
        bins_used.append(nb)

        edges = np.linspace(xmin, xmax, nb + 1)

        # bins occupied by full dataset
        b_all = np.digitize(x, edges) - 1
        m_all = (b_all >= 0) & (b_all < nb)
        support = np.unique(b_all[m_all])
        denom = max(len(support), 1)

        # bins covered by subset
        b_s = np.digitize(s, edges) - 1
        m_s = (b_s >= 0) & (b_s < nb)
        covered = np.intersect1d(np.unique(b_s[m_s]), support).size

        per_pc.append(covered / denom)

    mean_cov = float(np.mean(per_pc)) if d else 0.0
    return per_pc, mean_cov, bins_used
