#!/bin/bash
# ==============================================================
#  SLURM — data_sampling pipeline
#
#  RUNNABLE FROM ANYWHERE:
#      cd /work/bsempe/runs/my_run          # contains config.yaml and data/
#      sbatch /home/bsempe/Scripts/Workflow_DeepMD/New/data_sampling/slurm_template.sh
#
#  The code is located via PIPELINE_DIR (absolute, below).
#  The run happens in the SUBMISSION directory ($SLURM_SUBMIT_DIR).
#
#  OVERRIDES ON THE FLY:
#      sbatch --export=ALL,CONFIG=/path/other.yaml   slurm_template.sh
#      sbatch --export=ALL,PIPELINE_DIR=/other/code  slurm_template.sh
#      sbatch --export=ALL,STEPS="1 2 3"             slurm_template.sh
# ==============================================================
#SBATCH --job-name=data_sampling
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# --- Resources ---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# --- GPU for UMAP/t-SNE (uncomment if needed) ---
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu
set -euo pipefail
# ==============================================================
#  Environment
# ==============================================================
VENV_DIR="$HOME/Environnement/data_sampling"
PY="$VENV_DIR/bin/python3"
[[ -x "$PY" ]] || { echo "ERROR: interpreteur introuvable : $PY" >&2; exit 1; }
# ==============================================================
#  Paths
# ==============================================================
PIPELINE_DIR="${PIPELINE_DIR:-/home/bsempe/github/Workflow_DeepMD/data_sampling/Pipeline_smart_sampling}"
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG="${CONFIG:-$RUN_DIR/config.yaml}"
STEPS="${STEPS:-}"
cd "$RUN_DIR"
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# ==============================================================
#  Guardrails — fail in 2 s rather than after 20 min of SOAP
# ==============================================================
[[ -f "$PIPELINE_DIR/run_pipeline.py" ]] || {
    echo "ERROR: run_pipeline.py not found in $PIPELINE_DIR" >&2; exit 1; }
[[ -f "$CONFIG" ]] || {
    echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
"$PY" -c "import dscribe, ase, sklearn, yaml" || {
    echo "ERROR: missing dependencies in the environment" >&2; exit 1; }
# ==============================================================
#  Run
# ==============================================================
echo "=============================================="
echo "Job       : ${SLURM_JOB_ID:-interactive}"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "Code      : $PIPELINE_DIR"
echo "Run dir   : $RUN_DIR"
echo "Config    : $CONFIG"
echo "Python    : $PY"
echo "CPUs      : ${SLURM_CPUS_PER_TASK:-?}"
echo "=============================================="
if [[ -n "$STEPS" ]]; then
    "$PY" "$PIPELINE_DIR/run_pipeline.py" --config "$CONFIG" --steps $STEPS
else
    "$PY" "$PIPELINE_DIR/run_pipeline.py" --config "$CONFIG"
fi
echo "=============================================="
echo "End       : $(date)"
echo "=============================================="
