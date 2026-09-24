#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
# ============================================================================
# lammps_worker.slurm
# ----------------------------------------------------------------------------
# Worker d'un job array : execute sequentiellement TRAJ_PER_JOB trajectoires
# LAMMPS sur 1 seul CPU. Les trajectoires traitees sont determinees par
# SLURM_ARRAY_TASK_ID :
#
#     traj_start = (TASK_ID - 1) * TRAJ_PER_JOB + 1
#     traj_end   = min(traj_start + TRAJ_PER_JOB - 1, NTRAJ)
#
# Ce script n'est PAS lance directement : il est soumis par
# massif_lammps.sh via `sbatch --array=1-NJOBS --export=...`.
#
# Variables d'environnement attendues (passees via --export) :
#   CALCDIR        dossier de travail de ce (modele, energie) sur scratch
#   FINALDIR       dossier de depot final des resultats
#   MODEL_DIR      chemin vers le dossier contenant graph.pb
#   INPUT_SCRIPT   chemin vers in.simulation
#   NTRAJ          nombre total de trajectoires a faire pour ce couple
#   TRAJ_PER_JOB   nombre de trajectoires a enchainer par tache array
# ============================================================================

set -uo pipefail   # pas de -e : si LAMMPS foire sur une traj, on continue

echo "=== Worker SLURM ==="
echo "Job       : $SLURM_JOB_NAME ($SLURM_JOB_ID / array_task $SLURM_ARRAY_TASK_ID)"
echo "Hote      : $(hostname)"
echo "Date      : $(date)"
echo "CALCDIR   : $CALCDIR"
echo "FINALDIR  : $FINALDIR"
echo "MODEL_DIR : $MODEL_DIR"
echo "NTRAJ     : $NTRAJ"
echo "TRAJ/JOB  : $TRAJ_PER_JOB"
echo

# ----------------------------------------------------------------------------
# Environnement
# ----------------------------------------------------------------------------
. /etc/profile.d/modules.sh
module load use.own
module load deepmd/3.0.3-cpu

ulimit -s unlimited

# ----------------------------------------------------------------------------
# Calcul de l'intervalle de trajectoires a traiter
# ----------------------------------------------------------------------------
traj_start=$(( (SLURM_ARRAY_TASK_ID - 1) * TRAJ_PER_JOB + 1 ))
traj_end=$(( traj_start + TRAJ_PER_JOB - 1 ))
(( traj_end > NTRAJ )) && traj_end=$NTRAJ

echo "Trajectoires a executer : $traj_start .. $traj_end"
echo

mkdir -p "$FINALDIR"

# ----------------------------------------------------------------------------
# Boucle sequentielle sur les trajectoires de ce slot
# ----------------------------------------------------------------------------
nb_ok=0
nb_ko=0

for (( traj=traj_start; traj<=traj_end; traj++ )); do

    DATAFILE="$CALCDIR/data_POSCAR_${traj}.lammps"
    GROUPFILE="$CALCDIR/groups.lmp"

    if [[ ! -f "$DATAFILE" ]]; then
        echo "WARN: $DATAFILE introuvable, trajectoire $traj ignoree"
        nb_ko=$(( nb_ko + 1 ))
        continue
    fi

    echo "--- Trajectoire $traj ---"
    t0=$SECONDS

    lmp \
        -log    "$CALCDIR/log_${traj}.lammps" \
        -screen none \
        -var i        "$traj" \
        -var DATAFILE "$DATAFILE" \
        -var CALCDIR  "$CALCDIR" \
        -var folder   "$MODEL_DIR" \
	-var GROUPS_FILE "$GROUPFILE" \
        -in     "$INPUT_SCRIPT"
    rc=$?

    dt=$(( SECONDS - t0 ))
    echo "    terminee en ${dt}s (rc=$rc)"

    if (( rc == 0 )); then
        # Deplacement des resultats vers le dossier final (home / espace partage)
        for f in "log_${traj}.lammps" "dump_${traj}.lammpstrj" "energies_${traj}.lammpstrj" "data_POSCAR_${traj}.lammps" "HHcom_${traj}.dat"; do
            if [[ -f "$CALCDIR/$f" ]]; then
                mv -f "$CALCDIR/$f" "$FINALDIR/"
            else
                echo "    WARN: $f manquant apres LAMMPS"
            fi
        done
        # On supprime le data file pour liberer le scratch
        #rm -f "$DATAFILE"i
        nb_ok=$(( nb_ok + 1 ))
    else
        echo "    !! LAMMPS a echoue pour traj $traj (rc=$rc)"
        # On garde log + data pour debug
        [[ -f "$CALCDIR/log_${traj}.lammps" ]] && \
            mv -f "$CALCDIR/log_${traj}.lammps" "$FINALDIR/FAILED_log_${traj}.lammps"
        nb_ko=$(( nb_ko + 1 ))
    fi
done

echo
echo "=== Worker termine : $nb_ok OK / $nb_ko KO ==="
