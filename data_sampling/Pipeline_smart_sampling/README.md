# Data Sampling Pipeline

## How it works, in one paragraph per step

1. **Descriptors** (SOAP) turn every atomic structure into a fixed-length
   vector so "how similar are these two structures" becomes a distance
   between two vectors, not a manual comparison of coordinates.
2. **Dimensionality reduction** (PCA/UMAP/t-SNE) compresses those vectors to
   a handful of components — cheap to compute distances on, and plottable.
3. **Sampling** picks a subset of structures that spans that reduced space
   well ("coverage": are all regions of the space represented, not just the
   densest ones), instead of a plain random subset that would just mirror
   whatever is most common in the raw data.
4. **Analysis** is a sanity check on top of step 3: does the sampled subset
   also span the full dataset's *physical* properties (energy, per-species
   force) — a structurally diverse sample isn't automatically a physically
   representative one.
5. **Validation split** is a plain random hold-out of the already-sampled
   set — the diversity work is done, every frame is equally worth keeping.
6. **Test set** is drawn from whatever step 3 did **not** select (the
   leftovers from the full step-01 dataset), so it tests generalisation on
   data the model was never biased toward via coverage sampling. Its
   character is therefore different from train/val — more "typical", not
   selected for diversity. Optionally stratified by gas-molecule height
   above the surface (close/far) for H-on-surface systems.

## File structure

```
pipeline/
├── config.yaml              # ← All parameters live here
├── run_pipeline.py          # ← Orchestrator (run everything, or selected steps)
├── pipeline_utils.py        # ← Shared: config loading, logging, CLI base
│
├── step01_descriptors.py    # Compute SOAP descriptors
├── step02_dim_reduction.py  # PCA / UMAP / t-SNE
├── step03_sampler.py        # FPS / grid / kpp / ... sampling
├── step04_analysis.py       # Energy & force distribution analysis
├── step05_validation.py     # Split sampled set → train / val
├── step06_test_set.py       # Draw held-out test set
│
└── slurm_template.sh        # HPC submission script
```

## Quick start

### 1. Edit `config.yaml`

At minimum, set your input data path:
```yaml
descriptors:
  input_roots: ["./data"]
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

### 3. Run selected steps

```bash
# Only steps 1, 2 and 3
python run_pipeline.py --steps 1 2 3

# Skip the analysis step
python run_pipeline.py --skip 4
```

### 4. Run a single step independently

```bash
python step01_descriptors.py
python step02_dim_reduction.py
python step03_sampler.py
python step04_analysis.py
python step05_validation.py
python step06_test_set.py
```

### 5. Override parameters on the fly (no need to edit config.yaml)

```bash
# Change SOAP cutoff and sampling method
python run_pipeline.py --override descriptors.soap.r_cut=8.0 sampling.method=kpp

# Use UMAP instead of PCA
python step02_dim_reduction.py --override dim_reduction.method=umap

# Change validation split percentage
python step05_validation.py --override validation.percent=10
```

### 6. Dry-run (see what would execute without running anything)

```bash
python run_pipeline.py --dry-run
```

## Output directory layout

```
./
├── desc/                    # Step 01 outputs
│   ├── SOAP_H-W_<ts>.npy
│   ├── SOAP_H-W_<ts>_provenance.parquet
│   ├── SOAP_H-W_<ts>_filemap.json
│   └── SOAP_H-W_<ts>_config.json
│
├── embedding/               # Step 02 outputs
│   ├── pca_embedding.npy
│   └── pca_model.json
│
├── selected/                # Steps 03–06 outputs
│   ├── FPS_selected.traj
│   ├── FPS_selected.xyz
│   ├── FPS_coverage_evolution.png
│   ├── analysis_plots/
│   │   ├── energy_distribution.png
│   │   └── force_distribution_*.png
│   ├── valset/
│   │   ├── FPS_selected_valset_42_20.traj
│   │   └── FPS_selected_valset_42_20.txt
│   ├── trainset/
│   │   └── FPS_selected_trainset_42_80.traj
│   └── testset/
│       ├── TEST_selected_manifest.csv
│       ├── TEST_selected.traj
│       └── TEST_selected.xyz
│
└── logs/                    # One log file per step + global pipeline.log
    ├── pipeline.log
    ├── step01_descriptors.log
    ├── step02_dim_reduction.log
    └── ...
```

## HPC usage

`slurm_template.sh` is a thin wrapper: it just calls `run_pipeline.py` with
the same CLI you'd use locally, after loading the environment and checking
that the code, the config, and the dependencies (`dscribe`, `ase`, `sklearn`,
`yaml`) are actually there — so a broken environment fails in seconds
instead of after 20 minutes of SOAP computation.

Submit from the directory that holds YOUR `config.yaml` (the run happens in
`$SLURM_SUBMIT_DIR`, not in the code's directory):
```bash
cd /work/you/runs/my_run          # contains config.yaml
sbatch /path/to/Pipeline_smart_sampling/slurm_template.sh
```

Override without editing the script:
```bash
sbatch --export=ALL,CONFIG=/path/to/other.yaml   slurm_template.sh
sbatch --export=ALL,STEPS="1 2 3"                slurm_template.sh
sbatch --export=ALL,PIPELINE_DIR=/path/to/code   slurm_template.sh
```

Edit the `#SBATCH` header for resources, and `PIPELINE_DIR`'s default plus
the `module load` / venv activation lines for your own environment.

## Key config options

| Section | Key | Default | Description |
|---|---|---|---|
| `descriptors` | `input_roots` | `["./data"]` | Folders with structure files |
| `descriptors` | `file_selection_mode` | `"all"` | `all` / `random` / `stride` |
| `descriptors` | `frame_stride` | `1` | Keep 1 in N frames |
| `descriptors.soap` | `r_cut` | `5.0` | SOAP cutoff radius (Å) |
| `dim_reduction` | `method` | `"pca"` | `pca` / `umap` / `tsne` |
| `sampling` | `method` | `"fps"` | `fps` / `grid` / `kpp` / `kmedoids` / `density_fps` / `hdbscan` / `adaptive_kmedoids` |
| `sampling` | `auto_sampling` | `true` | Stop at coverage target |
| `sampling` | `auto_strategy` | `"seeded"` | `seeded` (incremental, recommended) / `resample` (restarts from scratch each round) |
| `sampling` | `target_mean_cov` | `0.99` | Coverage target |
| `validation` | `percent` | `20` | % of sampled set → validation |
| `test` | `n_test` | `2000` | Max structures in test set |
| `test` | `stratified` | `false` | Split leftovers by gas-height-above-surface (close/far) instead of plain random |
