#!/bin/bash
# ==============================================================
#  SLURM — QBC active learning selector (DeePMD)
#
#  Usage :
#    sbatch slurm_qbc.sh HW2O
#    sbatch slurm_qbc.sh HW
#    sbatch slurm_qbc.sh HW1O
# ==============================================================

#SBATCH --job-name=qbc
#SBATCH --output=logs/qbc_%j.out
#SBATCH --error=logs/qbc_%j.err

# --- Ressources ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

# --- GPU ---
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# --- Notifications ---
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.com

# ==============================================================
#  Vérification de l'argument
# ==============================================================

if [ -z "$1" ]; then
    echo "[ERROR] Donner le nom du modèle en argument."
    echo "  Usage : sbatch slurm_qbc.sh HW2O"
    exit 1
fi

MODEL_NAME="$1"

# ==============================================================
#  Configuration à adapter
# ==============================================================

INPUT_DIR_NAME="HW2O-QBC1"
MODELS_DIR="$HOME/Lammps/models"
POOL="/scratch/bsempe/Lammps/New-QBC/*/dump*"       # adapter selon ton système
#POOL="/scratch/bsempe/Lammps/HW110-O-DF-2x2/CALCDIR_NoZPE_0.*HW1O_AL4_*/*/dump*"       # adapter selon ton système
ASE_FORMAT="lammps-dump-text"              # "traj" | "lammps-dump-text"
STRIDE=1                     # A adapter selon le pas de temps et la chemin de réaction
BATCH_SIZE=256
THRESH_LOW=0.05
THRESH_HIGH=0.50

# ==============================================================
#  Environnement
# ==============================================================

module load use.own
#module load python/3.10.10
#module load deepmd/3.0.3-cpu
#module load deepmd/3.1.2-gpu
module load mlip/dpmdtk

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export OMP_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=2
export PYTHONUNBUFFERED=1	

mkdir -p logs

# ==============================================================
#  Lancement
# ==============================================================

echo "Job started   : $(date)"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Model name    : ${MODEL_NAME}"

echo "===== NVIDIA ====="
nvidia-smi
echo "=================="

python /gpfs/home/bsempe/QBC_AL_DeePMD/run_qbc.py \
    --model-name   ${MODEL_NAME} \
    --models-dir   ${MODELS_DIR} \
    --pool         "${POOL}" \
    --ase-format   ${ASE_FORMAT} \
    --stride       ${STRIDE} \
    --batch-size   ${BATCH_SIZE} \
    --thresh-low   ${THRESH_LOW} \
    --thresh-high  ${THRESH_HIGH} \
    -o ${INPUT_DIR_NAME} #HW1O_AL2-stride4K-qbc

echo "Job finished  : $(date)"
