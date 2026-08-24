#!/bin/bash
# ==============================================================
#  SLURM — pipeline data_sampling
#
#  LANÇABLE DE N'IMPORTE OÙ :
#      cd /work/bsempe/runs/mon_run          # contient config.yaml
#      sbatch /home/bsempe/Scripts/Workflow_DeepMD/New/data_sampling/slurm_template.sh
#
#  Le code est trouvé via PIPELINE_DIR (absolu, ci-dessous).
#  Le run se fait dans le répertoire de SOUMISSION ($SLURM_SUBMIT_DIR).
#
#  SURCHARGES À LA VOLÉE :
#      sbatch --export=ALL,CONFIG=/chemin/autre.yaml  slurm_template.sh
#      sbatch --export=ALL,PIPELINE_DIR=/autre/code   slurm_template.sh
#      sbatch --export=ALL,STEPS="1 2 3"              slurm_template.sh
# ==============================================================
#SBATCH --job-name=data_sampling
# Pas de sous-dossier ici : SLURM ouvre ces fichiers AVANT d'exécuter la
# moindre ligne du script. Un "logs/" inexistant = job mort sans aucun log.
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# --- Ressources ---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# --- GPU pour UMAP/t-SNE (décommenter si besoin) ---
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu

#set -euo pipefail

# ==============================================================
#  Environnement
# ==============================================================
module load python/3.13.9
source /work/bsempe/env/data_sampling/.data_sampling/bin/activate

# ==============================================================
#  Chemins
# ==============================================================
# Où vit le code. Absolu et fixe. Surchargeable via --export.
# ATTENTION : ${BASH_SOURCE[0]} ne marche PAS ici — SLURM copie le script
# dans son spool, tu obtiendrais /var/spool/slurmd/job*/slurm_script.
PIPELINE_DIR="${PIPELINE_DIR:-/home/bsempe/Scripts/Workflow_DeepMD/New/data_sampling/Pipeline_smart_sampling}"

# Où le run se fait = répertoire de soumission (fallback : CWD si lancé à la main)
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG="${CONFIG:-$RUN_DIR/config.yaml}"
STEPS="${STEPS:-}"

cd "$RUN_DIR"

# Redondant tant qu'on appelle run_pipeline.py par son chemin (Python met
# automatiquement le dossier du script en sys.path[0]), mais nécessaire si
# run_pipeline lance les steps en sous-processus.
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ==============================================================
#  Garde-fous — échouer en 2 s plutôt qu'après 20 min de SOAP
# ==============================================================
[[ -f "$PIPELINE_DIR/run_pipeline.py" ]] || {
    echo "ERREUR: run_pipeline.py introuvable dans $PIPELINE_DIR" >&2; exit 1; }
[[ -f "$CONFIG" ]] || {
    echo "ERREUR: config introuvable : $CONFIG" >&2; exit 1; }
python3 -c "import dscribe, ase, sklearn, yaml" 2>/dev/null || {
    echo "ERREUR: dépendances manquantes dans l'environnement" >&2; exit 1; }

# ==============================================================
#  Run
# ==============================================================
echo "=============================================="
echo "Job       : ${SLURM_JOB_ID:-interactif}"
echo "Noeud     : $(hostname)"
echo "Début     : $(date)"
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
echo "Fin       : $(date)"
echo "=============================================="
