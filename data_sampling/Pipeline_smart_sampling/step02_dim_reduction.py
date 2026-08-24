"""
step02_dim_reduction.py
-----------------------
Dimensionality reduction (PCA / UMAP / t-SNE) on SOAP descriptor matrices.

SOAP vectors from step 01 have hundreds of dimensions — too many for the
sampler (step 03) to reason about distances in efficiently, and impossible to
plot. This step standardises them and projects them down to a handful of
components (`dim_reduction.<method>.n_components`) that still preserve most
of the structural variance, and saves a 2-D scatter plot of the result so you
can eyeball whether the dataset covers the space you expect.

HOW THIS STEP FINDS STEP 01's OUTPUT
-------------------------------------
There is no explicit "path to step01" config key. Instead, `_load_artifacts`
picks the MOST RECENTLY MODIFIED `*.npy` / `*.parquet` / `*filemap*.json` /
`*_config.json` files inside `dim_reduction.input_dir` (default: "desc").
This works well when step 01 always runs with `clean_output_dir: true`
(the default — old files get removed first, so at most one candidate set
exists). If you ever disable that and let descriptor runs accumulate in the
same folder, double-check `input_dir` only contains the run you intend —
"most recent" will silently pick up whichever finished last, right or wrong.

Usage
-----
    python step02_dim_reduction.py
    python step02_dim_reduction.py --config my_cfg.yaml
    python step02_dim_reduction.py --override dim_reduction.method=umap
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    apply_overrides,
    base_parser,
    load_config,
    resolve_work_dir,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(directory: Path, pattern: str) -> Path:
    """Return the most recently modified file matching *pattern* in *directory*."""
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return candidates[-1]


def _load_artifacts(desc_dir: Path, logger: logging.Logger) -> tuple:
    """Load descriptors, provenance, filemap and run-config from the desc directory."""
    npy_file = _latest(desc_dir, "*.npy")
    all_descriptors = np.load(npy_file)
    logger.info("Loaded descriptors from %s  shape=%s", npy_file.name, all_descriptors.shape)

    # Provenance (parquet preferred)
    prov_candidates = list(desc_dir.glob("*.parquet")) + list(desc_dir.glob("*_provenance*.csv"))
    if not prov_candidates:
        raise FileNotFoundError(f"No provenance table in {desc_dir}")
    prov_path = next((p for p in prov_candidates if p.suffix == ".parquet"), prov_candidates[0])
    import pandas as pd
    metadata_df = pd.read_parquet(prov_path) if prov_path.suffix == ".parquet" else pd.read_csv(prov_path)
    logger.info("Loaded provenance from %s", prov_path.name)

    # Filemap
    filemap_files = [p for p in desc_dir.glob("*.json") if "filemap" in p.name]
    if not filemap_files:
        raise FileNotFoundError(f"No filemap JSON in {desc_dir}")
    with filemap_files[0].open() as fh:
        filemap = json.load(fh)
    logger.info("Loaded filemap from %s", filemap_files[0].name)

    # Run config (optional)
    config_files = list(desc_dir.glob("*_config.json"))
    run_config: dict = {}
    if config_files:
        config_path = max(config_files, key=lambda p: p.stat().st_mtime)
        with config_path.open() as fh:
            run_config = json.load(fh)
        logger.info("Loaded run config from %s", config_path.name)

    return all_descriptors, metadata_df, filemap, run_config


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(cfg: dict, logger: logging.Logger) -> dict:
    """
    Execute the dimensionality reduction step.

    Returns
    -------
    dict with the paths of saved outputs.
    """
    drcfg = cfg["dim_reduction"]
    work_dir = resolve_work_dir(cfg)

    desc_dir = work_dir / drcfg.get("input_dir", "desc")
    out_dir = work_dir / drcfg.get("output_dir", "embedding")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load descriptor artifacts
    # ------------------------------------------------------------------
    all_descriptors, metadata_df, filemap, run_config = _load_artifacts(desc_dir, logger)

    # step01 may have appended force columns after the SOAP ones (see its
    # `include_forces` option). use_forces=false (default) here drops them
    # back off before reduction, so the embedding reflects pure geometry;
    # set dim_reduction.use_forces=true to let force diversity also shape
    # the projection.
    soap_dim = run_config.get("soap_dim")
    force_dim = run_config.get("force_dim", 0)
    include_forces = run_config.get("include_forces", False)
    use_forces = drcfg.get("use_forces", False)

    if include_forces and soap_dim and force_dim and not use_forces:
        all_descriptors = all_descriptors[:, :soap_dim]
        logger.info("Using SOAP-only columns: shape=%s", all_descriptors.shape)

    # ------------------------------------------------------------------
    # 2. Standardise
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(all_descriptors)
    logger.info("Descriptors standardised. shape=%s", X.shape)

    # ------------------------------------------------------------------
    # 3. Dimensionality reduction
    # ------------------------------------------------------------------
    method = drcfg.get("method", "pca").lower()
    random_state = 42
    embedding = None
    model = None
    artifacts: dict[str, Path] = {}

    if method == "pca":
        pcfg = drcfg.get("pca", {})
        random_state = int(pcfg.get("random_state", 42))
        n_components = pcfg.get("n_components", 0.95)
        # YAML loads 0.95 as float, int values as int — both are valid for PCA
        logger.info("Running PCA  n_components=%s", n_components)
        model = PCA(n_components=n_components, random_state=random_state)
        embedding = model.fit_transform(X)
        logger.info("PCA embedding shape: %s", embedding.shape)
        logger.info(
            "Explained variance: %.4f (first component=%.4f)",
            model.explained_variance_ratio_.sum(),
            model.explained_variance_ratio_[0],
        )

        if pcfg.get("save_model", True):
            model_path = out_dir / "pca_model.json"
            model_data = {
                "components": model.components_.tolist(),
                "explained_variance": model.explained_variance_.tolist(),
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                "mean": model.mean_.tolist(),
                "n_components": int(getattr(model, "n_components_", model.n_components)),
                "n_features": int(model.n_features_in_),
            }
            with model_path.open("w") as fh:
                json.dump(model_data, fh, indent=2)
            artifacts["pca_model"] = model_path
            logger.info("PCA model saved → %s", model_path)

        if pcfg.get("save_matrix", True):
            emb_path = out_dir / "pca_embedding.npy"
            np.save(emb_path, embedding)
            artifacts["embedding"] = emb_path
            logger.info("PCA embedding saved → %s", emb_path)

    elif method == "umap":
        try:
            import umap.umap_ as umap_lib
        except ImportError:
            raise RuntimeError("UMAP not installed. Run: pip install umap-learn")

        ucfg = drcfg.get("umap", {})
        n_components = int(ucfg.get("n_components", 10))
        # UMAP/t-SNE scale poorly with input dimensionality — pre-reducing
        # the (already standardised) descriptors with a quick PCA pass first
        # is a common speed/stability trick. Off by default (null); set to an
        # int to enable.
        pca_pre = ucfg.get("pca_prereduce_dim")
        random_state = int(ucfg.get("random_state", 42))

        X_in = X
        if pca_pre:
            logger.info("Pre-reducing to %d components with PCA before UMAP", pca_pre)
            X_in = PCA(n_components=min(int(pca_pre), X.shape[1]), random_state=random_state).fit_transform(X)

        logger.info("Running UMAP  n_components=%d", n_components)
        model = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=int(ucfg.get("n_neighbors", 15)),
            min_dist=float(ucfg.get("min_dist", 0.0)),
            metric=ucfg.get("metric", "euclidean"),
            random_state=random_state,
            verbose=True,
        )
        embedding = model.fit_transform(X_in)
        emb_path = out_dir / "umap_embedding.npy"
        np.save(emb_path, embedding)
        artifacts["embedding"] = emb_path
        logger.info("UMAP embedding saved → %s  shape=%s", emb_path, embedding.shape)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        tcfg = drcfg.get("tsne", {})
        n_components = int(tcfg.get("n_components", 10))
        pca_pre = tcfg.get("pca_prereduce_dim")
        random_state = int(tcfg.get("random_state", 42))

        X_in = X
        if pca_pre:
            logger.info("Pre-reducing to %d components with PCA before t-SNE", pca_pre)
            X_in = PCA(n_components=min(int(pca_pre), X.shape[1]), random_state=random_state).fit_transform(X)

        logger.info("Running t-SNE  n_components=%d  shape=%s", n_components, X_in.shape)
        model = TSNE(
            n_components=n_components,
            perplexity=float(tcfg.get("perplexity", 30)),
            n_iter=int(tcfg.get("n_iter", 1000)),
            learning_rate="auto",
            init="pca",
            random_state=random_state,
            verbose=1,
            method="barnes_hut" if X_in.shape[0] < 50_000 else "exact",
        )
        embedding = model.fit_transform(X_in)
        emb_path = out_dir / "tsne_embedding.npy"
        np.save(emb_path, embedding)
        artifacts["embedding"] = emb_path
        logger.info("t-SNE embedding saved → %s  shape=%s", emb_path, embedding.shape)

    else:
        raise ValueError(f"Unknown dim_reduction.method: {method!r}. Choose pca | umap | tsne.")

    # ------------------------------------------------------------------
    # 4. Save 2-D scatter plot (first two components)
    # ------------------------------------------------------------------
    if embedding is not None and embedding.shape[1] >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend — safe for HPC
            import matplotlib.pyplot as plt

            if "symbol" in metadata_df.columns:
                labels = metadata_df["symbol"].astype(str).astype("category")
                cats = list(labels.cat.categories)
            else:
                labels = None
                cats = []

            fig, ax = plt.subplots(figsize=(8, 6))
            if cats:
                for lab in cats:
                    idx = (labels == lab).to_numpy()
                    ax.scatter(embedding[idx, 0], embedding[idx, 1], s=1, alpha=0.1, label=lab)
                ax.legend(markerscale=3, frameon=False, loc="best", title="symbol", fontsize=14)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.1)

            ax.set_xlabel("Component 1", fontsize=16)
            ax.set_ylabel("Component 2", fontsize=16)
            ax.set_title(f"{method.upper()} Embedding of SOAP Descriptors", fontsize=16)
            ax.grid(True, linewidth=0.2)
            fig.tight_layout()
            plot_path = out_dir / f"{method}_embedding.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            artifacts["plot"] = plot_path
            logger.info("Scatter plot saved → %s", plot_path)
        except Exception as exc:
            logger.warning("Could not save embedding plot: %s", exc)

    return artifacts


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = base_parser("step02_dim_reduction", description=__doc__)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    logger = setup_logging(
        log_file=work_dir / "logs" / "step02_dim_reduction.log",
        level=getattr(logging, args.log_level),
        step_name="step02",
    )

    logger.info("=== Step 02 — Dimensionality Reduction ===")
    logger.info("Config: %s  method: %s", args.config, cfg["dim_reduction"].get("method", "pca"))

    try:
        run(cfg, logger)
        logger.info("=== Step 02 completed successfully ===")
    except Exception:
        logger.exception("Step 02 failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
