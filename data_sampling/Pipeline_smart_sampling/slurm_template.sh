#!/bin/bash
# ==============================================================
#  SLURM — data_sampling pipeline
#
#  RUNNABLE FROM ANYWHERE:
#      cd /work/bsempe/runs/my_run          # contains config.yaml
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
# No subfolder here: SLURM opens these files BEFORE executing a single line
# of the script. A non-existent "logs/" = dead job with no log at all.
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

#set -euo pipefail

# ==============================================================
#  Environment
# ==============================================================
module load python/3.13.9
source /work/bsempe/env/data_sampling/.data_sampling/bin/activate

# ==============================================================
#  Paths
# ==============================================================
# Where the code lives. Absolute and fixed. Overridable via --export.
# WARNING: ${BASH_SOURCE[0]} does NOT work here — SLURM copies the script
# into its spool, you'd get /var/spool/slurmd/job*/slurm_script instead.
PIPELINE_DIR="${PIPELINE_DIR:-/home/bsempe/Scripts/Workflow_DeepMD/New/data_sampling/Pipeline_smart_sampling}"

# Where the run happens = submission directory (fallback: CWD if launched by hand)
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG="${CONFIG:-$RUN_DIR/config.yaml}"
STEPS="${STEPS:-}"

cd "$RUN_DIR"

# Redundant as long as run_pipeline.py is called by its path (Python
# automatically puts the script's directory in sys.path[0]), but necessary
# if run_pipeline launches the steps as subprocesses.
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ==============================================================
#  Guardrails — fail in 2 s rather than after 20 min of SOAP
# ==============================================================
[[ -f "$PIPELINE_DIR/run_pipeline.py" ]] || {
    echo "ERROR: run_pipeline.py not found in $PIPELINE_DIR" >&2; exit 1; }
[[ -f "$CONFIG" ]] || {
    echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
python3 -c "import dscribe, ase, sklearn, yaml" 2>/dev/null || {
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
echo "CPUs      : ${SLURM_CPUS_PER_TASK:-?}"
echo "=============================================="

if [[ -n "$STEPS" ]]; then
    python3 "$PIPELINE_DIR/run_pipeline.py" --config "$CONFIG" --steps $STEPS
else
    python3 "$PIPELINE_DIR/run_pipeline.py" --config "$CONFIG"
fi

echo "=============================================="
echo "End       : $(date)"
echo "=============================================="
