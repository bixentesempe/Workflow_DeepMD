# data_sampling

Tools for turning raw AIMD trajectories into DeepMD-ready training data:
first clean and cache the trajectories, then pick a diverse, representative
subset of structures out of that cache.

## Environment

Dependencies are listed in `environment.txt`. Name the environment after the
folder (`data_sampling`) — simplest way to know which env goes with which
project.

```bash
conda create -n data_sampling python=3.11
conda activate data_sampling
pip install -r environment.txt
```

(A plain `python -m venv data_sampling && source data_sampling/bin/activate`
works too if you don't use conda.)

## AIMD_process.py

Reads raw AIMD VASP trajectories (`vasprun.xml`), applies quality checks
(topology consistency, physical event detection — dissociation / rebound /
trapping, energy conservation on the kept segment), and writes a per-group
cache (`.traj` and/or `.npz`) ready to sample from.

```bash
python AIMD_process.py -i path/to/AIMD/ -o path/to/output --all-groups
python AIMD_process.py --help
```

Full details (step-by-step logic, all options, output layout) are in the
script's own docstring. `AIMD_process_fr.py` is an archived French copy kept
for reference — not maintained, don't run it, `AIMD_process.py` is the one
to use.

## Pipeline_smart_sampling/

Takes a cache of structures (e.g. `AIMD_process.py`'s output) and picks a
diverse, representative subset out of it for training: SOAP descriptors →
dimensionality reduction → coverage-based sampling → distribution sanity
checks → train/validation split → held-out test set. Runs locally or via
Slurm.

```bash
cd Pipeline_smart_sampling
python run_pipeline.py --dry-run
```

See `Pipeline_smart_sampling/README.md` for the full picture (config
options, output layout, HPC usage).
