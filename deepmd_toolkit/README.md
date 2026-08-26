# deepmd_toolkit

Tools for turning sampled AIMD structures into a trained DeepMD potential
and checking how good it is:

```
 .traj files (from data_sampling)
        │
        ▼
 1. dpdata_convert.py   → DeepMD npy training/validation sets
        │
        ▼
 2. dp train (input.json + train_modelDPMD.slurm)  → graph.pb + lcurve.out
        │
        ▼
 3. dp_model_analysis.py   → training curves, DFT-vs-DP inference, error plots
```

## Environment

Dependencies are listed in `environment.txt`. Name the environment after the
folder (`deepmd_toolkit`), and use **Python below 3.12** — `deepmd-kit` (the
version this toolkit targets) is not compatible with newer Python yet.

```bash
python -m venv deepmd_toolkit
source deepmd_toolkit/bin/activate
pip install -r environment.txt
```

(Conda works too: `conda create -n deepmd_toolkit python=3.11 && conda activate deepmd_toolkit`.)

This environment is only needed for `dpdata_convert.py` and
`dp_model_analysis.py` (data prep + analysis, run locally). Training itself
(`dp train`) runs on the cluster through the module system — see step 2.

## 1. Convert sampled trajectories — `dpdata_convert.py`

Takes the `.traj` files produced by the sampling step (`data_sampling/`) and
converts them to the `deepmd/npy` format `dp train` expects, using
[`dpdata`](https://github.com/deepmodeling/dpdata) under the hood.

**Routing:** each input file is sent to `trainset/` or `valset/` based on its
*filename* — it must contain `train` or `val` (case-insensitive) somewhere in
the name (e.g. `HW_train_01.traj`, `dataset_val.traj`). Anything else is
skipped.

```bash
python dpdata_convert.py
python dpdata_convert.py --input-dir selected --base-dir deepmd_data
python dpdata_convert.py -i selected -o deepmd_data
python dpdata_convert.py --type-map H W O
```

| Option | Default | Meaning |
|---|---|---|
| `-i`, `--input-dir` | `selected` | Searched **recursively** for `*.traj` files |
| `-o`, `--base-dir` | `deepmd_data` | Where `trainset/` and `valset/` are written |
| `--type-map` | `H W O` | Global element ordering enforced on the output (see below) |

Output layout:

```
deepmd_data/
  trainset/
    set.000/
  valset/
    set.000/
```

### Why `--type-map` matters

DeepMD doesn't store element symbols in the training data — it stores
**integer type indices**, and `type_map.raw` says what each index means.
`dpdata` assigns those indices per file, in the order it first encounters
each element, so two `.traj` files can end up with *different* index → element
mappings even if the elements are the same.

`dpdata_convert.py` fixes this: after converting each file, it reads the
file's own `type_map.raw`, remaps `type.raw` so every index matches the
**global** `--type-map` order you gave it, and rewrites `type_map.raw`
accordingly (see `enforce_type_map()`). If a file contains a species that
isn't in `--type-map`, conversion fails loudly rather than silently
mislabeling atoms.

**This global order is the one you must reuse everywhere downstream** — most
importantly in `input.json`'s `model.type_map` (step 2). If the two don't
match, training will silently learn the wrong element for each type index.

## 2. Train a model — `input.json` + `train_modelDPMD.slurm`

### `input.json`

The DeepMD training configuration (`dp train` config). The parts you'll
actually touch between runs:

- **`model.type_map`** — must be **exactly** the same list, in the same
  order, as the `--type-map` used in step 1 (default here: `["H", "W", "O"]`).
  This is what turns type indices back into elements.
- **`model.descriptor.sel`** — max number of neighbors of each type inside
  the cutoff, **in `type_map` order**. In the shipped config
  (`[10, 30, 10]` for `H, W, O`) that means up to 10 H, 30 W, and 10 O
  neighbors per atom. Pick these generously enough for your densest
  structure, or `dp train` will complain / skip the check with
  `--skip-neighbor-stat` (used in the Slurm script — meaning under-sizing
  `sel` will *not* be caught automatically, so check it yourself if you
  change the dataset).
- **`training.training_data.systems` / `validation_data.systems`** —
  relative paths to the `trainset/` and `valset/` folders from step 1
  (default `./trainset`, `./valset`).
- **`training.numb_steps`**, **`learning_rate`**, `descriptor`/`fitting_net`
  architecture — the usual DeepMD hyperparameters, tune as needed per run.

### `train_modelDPMD.slurm`

A Slurm batch script that trains one or several models in one job, each in
its own subfolder of the submit directory.

Expected layout **before submitting**, one folder per model you want to
train:

```
<submit_dir>/
  graph_1/
    input.json
    trainset/
    valset/
  graph_2/
    input.json
    trainset/
    valset/
  ...
```

Submit it with the model folder names as arguments:

```bash
sbatch train_modelDPMD.slurm graph_1 graph_2 graph_3
```

For each folder given, the script:

1. copies `input.json`, `trainset*` and `valset*` to the compute node's
   local scratch (`$TMPDIR`),
2. runs `dp train input.json --skip-neighbor-stat -l log.log`,
3. freezes the result with `dp freeze -c checkpoint_dir/ -o graph.pb`,
4. `rsync`s everything (checkpoints, `graph.pb`, `lcurve.out`, `log.log`)
   back to the submit directory, and removes the local scratch copy only if
   the sync succeeded.

Options:

```bash
sbatch train_modelDPMD.slurm -r graph_1/checkpoint_dir/model.ckpt graph_1   # resume from a checkpoint
sbatch train_modelDPMD.slurm --debug graph_1                                # bash -x tracing
sbatch train_modelDPMD.slurm -h                                             # help
```

Naming your model folders `<name>_1`, `<name>_2`, ... (e.g. `graph_1`,
`graph_2`, matching the descriptor.sel sweeps or seeds you want to compare)
is not required by the training script, but **is** required by
`dp_model_analysis.py` below — so it's worth doing consistently from the
start.

## 3. Analyze trained models — `dp_model_analysis.py`

Compares one or more trained models (loss curves + accuracy against a
reference trajectory) and writes plots/reports. Runs locally, using the
`deepmd_toolkit` environment from above.

**Expected folder layout** (`--base-dir`, default: current directory):

```
BASE_DIR/
  graph_1/
    graph.pb        ← frozen model from step 2
    lcurve.out       ← training log from step 2
  graph_2/
    graph.pb
    lcurve.out
  ...
```

Only folders that match **exactly** `<model_name>_<integer>` are picked up
(with `--model-name graph`: `graph_1`, `graph_2`, `graph_12` — but not
`graph_AL_1` or `graph_bis`), and only if they contain a `*.pb` file.

The pipeline has three steps, run all together or individually with
`--steps`:

| Step | What it does | Requires |
|---|---|---|
| **1** | Reads `lcurve.out` from every model folder, overlays learning-rate and energy/force RMSE curves across all models, writes a summary CSV ranked by validation force RMSE | `lcurve.out` in each folder |
| **2** | Runs each frozen model (`graph.pb`) on a reference trajectory via ASE + DeepMD's `DeepPot`/calculator, writes per-frame and per-atom energy/force comparisons | `graph.pb` + `--trajectory` (a labeled `.traj`, e.g. a held-out test set) |
| **3** | Builds parity plots (energy, energy/atom, force components, force magnitude, force-vector angle) and per-model error-statistics reports from step 2's output | Step 2 output already present in `--output-dir` |

```bash
# Full pipeline
python dp_model_analysis.py --model-name graph --trajectory ./test.traj

# Different base directory
python dp_model_analysis.py --model-name graph --base-dir /data/runs \
                             --trajectory ./test.traj

# Only the training curves (no inference — fast, no test set needed)
python dp_model_analysis.py --model-name graph --steps 1

# Re-do only the plots after inference has already run
python dp_model_analysis.py --model-name graph --steps 3

# Wider window for the final RMSE summary table (default: last 1000 steps)
python dp_model_analysis.py --model-name graph --trajectory ./test.traj \
                             --summary-window 500
```

Output (`--output-dir`, default `<model_name>_analysis/`):

```
<model_name>_analysis/
  training_curves/
    training_summary.csv
    lr_schedule.png
    energy_rmse.png
    force_rmse.png
  graph_1/ graph_2/ ...          ← step 2 raw inference output per model
    energies_forces.txt
    energies_forces_atoms.txt
  inference_summary.csv
  correlation_reports/
    graph_1_stats.txt ...
  correlation_plots/
    cross_model_rmse.png
    energy_parity.png
    energy_per_atom_parity.png
    energy_per_atom_error_hist.png
    force_x_parity.png / force_y_parity.png / force_z_parity.png
    force_magnitude_parity.png
    force_angle_hist.png
```

> Note: the script's own `--help` text and log messages are in French; the
> behavior described above is accurate to the current code regardless of
> that.

## Typical end-to-end run

```bash
# 0. environment (once)
python3.11 -m venv deepmd_toolkit && source deepmd_toolkit/bin/activate
pip install -r environment.txt

# 1. convert sampled .traj files, using a fixed element order
python dpdata_convert.py --input-dir selected --base-dir graph_1 --type-map H W O

# 2. edit graph_1/input.json (type_map must match the --type-map above),
#    then train on the cluster
sbatch train_modelDPMD.slurm graph_1

# 3. once training is done, analyze it against a held-out test trajectory
python dp_model_analysis.py --model-name graph --trajectory selected/test.traj
```
