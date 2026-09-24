#!/bin/bash
# ============================================================================
# massif_lammps.sh
# ----------------------------------------------------------------------------
# Lance "en masse" des trajectoires LAMMPS pour plusieurs couples
# (modele MLPES, energie cinetique initiale), regroupant N trajectoires par
# job SLURM (1 CPU) via un job array.
#
# DEUX MODES pour l'etat interne de H2 (orthogonal a la source de surface
# ci-dessous) :
#
#  1) Mode ZPE (par defaut) :
#       - GenerateZPELammpsDatafile_ZPE.py place H2 comme en mode NoZPE (CdM,
#         translation selon E/theta/phi) mais ajoute la vraie vibration/
#         rotation interne de H2 (r_HH, orientation, vitesses internes),
#         echantillonnee a la volee dans le potentiel de Morse universel
#         SCRIPTS_DIR/init_cond/morse_potasym.dat pour le niveau (v, J)
#         demande (options --vib-level / --jrot). Aucun fichier de
#         conditions initiales a pre-generer par energie/angle.
#
#  2) Mode NoZPE (--no-zpe) :
#       - GenerateLammpsDatafile_NoZPE.py lit le LAMMPS_POSCAR et place H2
#         avec :
#           * position XY aleatoire dans la cellule
#           * z = z_top_surface + height (option --height)
#           * orientation H-H aleatoire (par defaut)
#           * distance H-H = 0.7414 A (option --hh-distance)
#           * vitesse de translation pure du CdM selon (E, theta, phi)
#       - Pas besoin d'INIT-AIMD.res.
#
# DEUX SOURCES DE SURFACE (independantes du mode ZPE/NoZPE ci-dessus),
# choisies EXPLICITEMENT via --surface-mode (jamais devine d'apres ce qui
# est rempli, pour eviter de se tromper de surface sans s'en rendre compte) :
#
#  a) --surface-mode cold (par defaut) : un seul LAMMPS_POSCAR
#     (-o/--path2poscar), figee a vitesse nulle (0K) pour toutes les
#     trajectoires. Les atomes fixes sont lus depuis les commentaires
#     "# fixed" du LAMMPS_POSCAR (et/ou via --frozen-ids passe au generateur).
#
#  b) --surface-mode thermal (+ --thermal-surface-dir DIR) : GenerateLammpsDatafile_
#     ThermalSurface.py pioche, POUR CHAQUE trajectoire, un POSCAR-* different
#     dans DIR (positions ET vitesses reelles issues d'un AIMD a une
#     temperature et un coverage donnes, ex. produites par
#     extract_md_configs.py) -- mouvement thermique reel du slab au moment
#     de l'impact. --path2poscar/LAMMPS_POSCAR n'est pas utilise dans ce mode.
#     Tous les POSCAR-* de DIR doivent venir du meme run (meme especes, meme
#     couches figees) : verifie par le generateur, sinon erreur.
#
# Dans tous les cas, le template in.simulation inclut groups.lmp (genere par
# le script Python a cote des data files) qui definit les groupes 'frozen' et
# 'mobile'.
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Aide
# ----------------------------------------------------------------------------
phelp() {
cat <<EOF
Usage: $(basename "$0") [OPTIONS] MODEL1 [MODEL2 ...]

Lance massivement des trajectoires LAMMPS (mode ZPE ou NoZPE).

Organisation (portable) :
  - SCRIPTS_DIR : dossier contenant ce script, les generateurs Python et
    lammps_worker.slurm. Auto-detecte (a cote de ce script), surchargeable
    avec --scripts-dir.
  - CASE_DIR : dossier du "cas" a simuler, contenant LAMMPS_POSCAR et
    (optionnel) un in.simulation personnalise. Par defaut le dossier courant
    (\$PWD), surchargeable avec -d/--case-dir. Si in.simulation est absent
    de CASE_DIR, le template SCRIPTS_DIR/in.simulation.default est utilise.

Prerequis communs :
  - Un dossier 'models' contenant graph.pb par modele (option -p)
  - CASE_DIR/LAMMPS_POSCAR (ou --path2poscar pour un chemin different),
    SAUF en --surface-mode thermal (voir plus bas)
  - SCRIPTS_DIR/lammps_worker.slurm (le worker appele par sbatch)

Mode ZPE (par defaut) :
  - SCRIPTS_DIR/GenerateZPELammpsDatafile_ZPE.py
  - SCRIPTS_DIR/init_cond/{exe,intrep.dat,morse_potasym.dat} : sampler Fortran
    + potentiel de Morse universel pour H2 (r_HH). L'etat vibrationnel/
    rotationnel (v, J) est echantillonne a la volee, plus besoin de
    pre-generer INIT-AIMD.res par energie/angle.

Mode NoZPE (--no-zpe) :
  - SCRIPTS_DIR/GenerateLammpsDatafile_NoZPE.py

Surface thermalisee (--surface-mode thermal --thermal-surface-dir DIR),
combinable avec ZPE/NoZPE :
  - SCRIPTS_DIR/GenerateLammpsDatafile_ThermalSurface.py
  - DIR doit contenir des POSCAR-* (un par snapshot AIMD thermalise, meme
    run/surface -- meme temperature, meme coverage -- pour tous) ; un est
    tire au hasard par trajectoire. --path2poscar/LAMMPS_POSCAR n'est pas
    utilise dans ce mode.

Fichier de config (-c/--config FICHIER), optionnel :
  - Fichier bash "source"-e (KEY=VALUE / tableaux), memes noms de variables
    que les options ci-dessous (voir config.example.conf). Charge AVANT les
    flags CLI : un flag explicite surcharge toujours le fichier (y compris
    -e/--energie et les modeles positionnels, qui REMPLACENT -- n'ajoutent
    pas a -- la liste du fichier).
  - Utile pour ne pas retaper une commande a rallonge et versionner un run.

Options :
  -h, --help                  Afficher cette aide
  -c, --config FICHIER        Fichier de config a charger (voir plus haut)  [aucun]
      --no-zpe                Mode sans ZPE (translation pure de H2)
      --surface-mode MODE     "cold" (LAMMPS_POSCAR figee, 0K) ou
                               "thermal" (pool --thermal-surface-dir,
                               T/coverage donnes) -- explicite, ne se
                               devine pas d'apres ce qui est rempli [cold]
  -d, --case-dir DIR          Dossier du cas (LAMMPS_POSCAR,
                               in.simulation optionnel)           [\$PWD]
      --scripts-dir DIR       Dossier des scripts (generateurs,
                               worker, template in.simulation)    [auto]
  -p, --path2models DIR       Dossier des modeles                [./models]
  -o, --path2poscar FILE      (surface_mode=cold) LAMMPS_POSCAR de
                               reference              [CASE_DIR/LAMMPS_POSCAR]
      --thermal-surface-dir DIR
                              (surface_mode=thermal) pool de POSCAR-*
                              thermalises (AIMD, ex. un run a une temperature
                              et un coverage donnes) ; un tire au hasard par
                              trajectoire                          [aucun]
  -n, --name NOM              Nom du dossier de resultats        [results_<date>]
  -e, --energie E             Energie cinetique (eV), repetable
  -s, --nstep N               Nombre de pas par trajectoire      [5000]
  -t, --ntraj N               Nombre de trajectoires             [20000]
  -i, --timestep DT           Pas de temps en ps                 [0.0001]
  -j, --traj-per-job N        Trajectoires par job SLURM (1 CPU) [10]
  -a, --theta ANGLE           Angle / normale (deg)              [0]
  -z, --phi ANGLE             Azimut / plan XY (deg) ou random   [random]
      --height H              Hauteur initiale du CdM de H2
                              au-dessus du W le plus haut (A)    [9.0]
      --vib-level V           (ZPE) niveau vibrationnel de H2
                              dans le potentiel de Morse         [0]
      --jrot J                (ZPE) nombre quantique rotationnel [0]
      --hh-distance R         (NoZPE) distance H-H (A)           [0.7414]
      --orientation X         (NoZPE) random|x|y|z|vx,vy,vz       [random]
      --frozen-ids RANGE      IDs supplementaires a figer,
                              ex: "35-42" ou "35,36,37"          [vide]
      --seed S                seed du RNG                        [aleatoire]
      --debug                 Mode debug (set -x)

Exemples :
  # Mode ZPE (v=0, J=0 par defaut) :
  $(basename "$0") -e 0.1 -e 0.2 -t 20000 -j 10 model_A model_B

  # Mode ZPE, niveau vibrationnel excite (v=1) :
  $(basename "$0") -e 0.1 --vib-level 1 -t 5000 -j 10 model_A

  # Mode NoZPE, incidence a 45 deg avec azimut aleatoire :
  $(basename "$0") --no-zpe -e 0.1 -a 45 -t 1000 -j 10 model_A

  # Mode NoZPE avec couche bottom figee a la main (ids 35-42) :
  $(basename "$0") --no-zpe --frozen-ids "35-42" -e 0.1 -t 1000 -j 10 model_A

  # Surface thermalisee (AIMD, 300K/0.25ML ici), mode ZPE :
  $(basename "$0") --surface-mode thermal \\
      --thermal-surface-dir /path/extracted_configs/0.25ML300K/POSCAR \\
      -e 0.1 -t 5000 -j 10 model_A

  # Surface thermalisee + NoZPE :
  $(basename "$0") --no-zpe --surface-mode thermal --thermal-surface-dir /path/.../POSCAR \\
      -e 0.1 -t 5000 -j 10 model_A

  # Fichier de config (voir config.example.conf), avec surcharge ponctuelle :
  $(basename "$0") --config run.conf -e 0.5
EOF
}

# ----------------------------------------------------------------------------
# Valeurs par defaut
# ----------------------------------------------------------------------------
mode="zpe"                  # "zpe" ou "nozpe" (etat interne de H2)
surface_mode="cold"       # "cold" (LAMMPS_POSCAR figee) ou "thermal"
                             # (pool --thermal-surface-dir) -- explicite,
                             # ne se devine pas d'apres ce qui est rempli
case_dir="none"
scripts_dir="none"
path2models="none"
path2poscar="none"
thermal_surface_dir=""
results_name="none"
tstep=0.0001
nstep=5000
ntraj=20000
traj_per_job=10
theta="0"
phi="random"
height=9.0
vib_level=0
jrot=0
hh_distance=0.7414
orientation="random"
frozen_ids=""
seed=""
declare -a energies=()
declare -a folders=()
energies_overridden_by_cli=0

# ----------------------------------------------------------------------------
# Fichier de config (-c/--config) : optionnel, "source"-e directement (bash
# pur, meme noms de variables que ci-dessus -> aucune traduction a
# maintenir). Detecte AVANT le parsing normal des options pour que les
# flags CLI explicites (traites ensuite) le surchargent, pas l'inverse.
# Voir config.example.conf pour le format.
# ----------------------------------------------------------------------------
config_file=""
prev=""
for a in "$@"; do
    if [[ "$prev" == "-c" || "$prev" == "--config" ]]; then
        config_file="$a"
    elif [[ "$a" == --config=* ]]; then
        config_file="${a#--config=}"
    fi
    prev="$a"
done
if [[ -n "$config_file" ]]; then
    [[ -f "$config_file" ]] \
        || { echo "ERREUR: fichier de config introuvable : $config_file" >&2; exit 1; }
    echo "Chargement de la config : $config_file"
    # shellcheck disable=SC1090
    source "$config_file"
fi

# ----------------------------------------------------------------------------
# Parsing des arguments
# ----------------------------------------------------------------------------
OPTS=$(getopt \
    --options hc:d:p:o:s:t:e:i:j:n:a:z: \
    --longoptions help,config:,no-zpe,surface-mode:,case-dir:,scripts-dir:,path2models:,path2poscar:,thermal-surface-dir:,nstep:,ntraj:,energie:,timestep:,traj-per-job:,name:,theta:,phi:,height:,vib-level:,jrot:,hh-distance:,orientation:,frozen-ids:,seed:,debug \
    --name "$(basename "$0")" -- "$@")
eval set -- "$OPTS"

while true; do
    case "$1" in
        -h|--help)          phelp; exit 0 ;;
        -c|--config)        shift 2 ;;  # deja charge plus haut (avant getopt), ignore ici
        --no-zpe)           mode="nozpe";       shift   ;;
        --surface-mode)     surface_mode=$2;    shift 2 ;;
        -d|--case-dir)      case_dir=$2;        shift 2 ;;
        --scripts-dir)      scripts_dir=$2;     shift 2 ;;
        -p|--path2models)   path2models=$2;     shift 2 ;;
        -o|--path2poscar)   path2poscar=$2;     shift 2 ;;
        --thermal-surface-dir) thermal_surface_dir=$2; shift 2 ;;
        -s|--nstep)         nstep=$2;           shift 2 ;;
        -t|--ntraj)         ntraj=$2;           shift 2 ;;
        -e|--energie)
            # Remplace (pas n'ajoute pas) une liste d'energies deja fixee
            # par le fichier de config, comme les autres options scalaires.
            if (( ! energies_overridden_by_cli )); then
                energies=()
                energies_overridden_by_cli=1
            fi
            energies+=("$2"); shift 2 ;;
        -i|--timestep)      tstep=$2;           shift 2 ;;
        -j|--traj-per-job)  traj_per_job=$2;    shift 2 ;;
        -n|--name)          results_name=$2;    shift 2 ;;
        -a|--theta)         theta=$2;           shift 2 ;;
        -z|--phi)           phi=$2;             shift 2 ;;
        --height)           height=$2;          shift 2 ;;
        --vib-level)        vib_level=$2;       shift 2 ;;
        --jrot)              jrot=$2;            shift 2 ;;
        --hh-distance)      hh_distance=$2;     shift 2 ;;
        --orientation)      orientation=$2;     shift 2 ;;
        --frozen-ids)       frozen_ids=$2;      shift 2 ;;
        --seed)             seed=$2;            shift 2 ;;
        --debug)            set -x;             shift   ;;
        --) shift; break ;;
        *)  echo "Option inconnue : $1" >&2; phelp; exit 1 ;;
    esac
done

# Modeles : remplace la liste eventuellement fixee par le fichier de config
# UNIQUEMENT si des arguments positionnels sont explicitement donnes en CLI
# (sinon "$@" est vide ici et on ecraserait silencieusement le config par
# une liste vide).
[[ $# -gt 0 ]] && folders=( "$@" )

# ----------------------------------------------------------------------------
# Resolution des chemins par defaut
# ----------------------------------------------------------------------------
SUBMIT_DIR=$PWD

# SCRIPTS_DIR : dossier contenant ce script (generateurs, worker, template
# in.simulation par defaut). Auto-detecte pour que le dossier entier soit
# deplacable/clonable n'importe ou sans reconfiguration.
[[ "$scripts_dir" == "none" ]] && scripts_dir="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SCRIPTS_DIR=$(readlink -f "$scripts_dir")

# CASE_DIR : dossier du cas simule (LAMMPS_POSCAR + in.simulation optionnel).
[[ "$case_dir" == "none" ]] && case_dir="$SUBMIT_DIR"
CASE_DIR=$(readlink -f "$case_dir" 2>/dev/null || echo "$case_dir")

[[ "$path2models"   == "none" ]] && path2models="$SUBMIT_DIR/models"
[[ "$path2poscar"   == "none" ]] && path2poscar="$CASE_DIR/LAMMPS_POSCAR"

# Resolution en chemins absolus (les scripts Python sont appeles depuis $CALCDIR)
path2poscar=$(readlink -f "$path2poscar" 2>/dev/null || echo "$path2poscar")
path2models=$(readlink -f "$path2models" 2>/dev/null || echo "$path2models")
[[ -n "$thermal_surface_dir" ]] \
    && thermal_surface_dir=$(readlink -f "$thermal_surface_dir" 2>/dev/null || echo "$thermal_surface_dir")

# Template in.simulation : celui du CASE_DIR s'il existe, sinon le defaut
# livre avec les scripts.
if [[ -f "$CASE_DIR/in.simulation" ]]; then
    IN_SIMULATION_TEMPLATE="$CASE_DIR/in.simulation"
else
    IN_SIMULATION_TEMPLATE="$SCRIPTS_DIR/in.simulation.default"
fi

WORKER_SCRIPT="$SCRIPTS_DIR/lammps_worker.slurm"
GEN_ZPE_SCRIPT="$SCRIPTS_DIR/GenerateZPELammpsDatafile_ZPE.py"
GEN_NOZPE_SCRIPT="$SCRIPTS_DIR/GenerateLammpsDatafile_NoZPE.py"
GEN_THERMAL_SCRIPT="$SCRIPTS_DIR/GenerateLammpsDatafile_ThermalSurface.py"
INIT_COND_DIR="$SCRIPTS_DIR/init_cond"

# ----------------------------------------------------------------------------
# Verifications prealables
# ----------------------------------------------------------------------------
err=0
[[ ${#folders[@]}  -eq 0 ]] && { echo "ERREUR: aucun modele fourni (argument positionnel apres les options)" >&2; err=1; }
[[ ${#energies[@]} -eq 0 ]] && { echo "ERREUR: aucune energie fournie (option -e)"                          >&2; err=1; }
[[ -x "$WORKER_SCRIPT" ]] \
    || { echo "ERREUR: worker introuvable ou non executable : $WORKER_SCRIPT" >&2; err=1; }
# GEN_NOZPE_SCRIPT est une dependance commune : appele directement en mode
# NoZPE, mais aussi importe par GenerateZPELammpsDatafile_ZPE.py et par
# GenerateLammpsDatafile_ThermalSurface.py (code partage), donc toujours requis.
[[ -x "$GEN_NOZPE_SCRIPT" ]] \
    || { echo "ERREUR: generateur introuvable ou non executable : $GEN_NOZPE_SCRIPT" >&2; err=1; }
[[ -d "$path2models" ]] || { echo "ERREUR: dossier models introuvable : $path2models"        >&2; err=1; }
[[ -f "$IN_SIMULATION_TEMPLATE" ]] \
    || { echo "ERREUR: template in.simulation introuvable : $IN_SIMULATION_TEMPLATE" >&2; err=1; }
for model in "${folders[@]}"; do
    [[ -f "$path2models/$model/graph.pb" ]] \
        || { echo "ERREUR: graph.pb manquant pour le modele '$model' (cherche dans $path2models/$model/)" >&2; err=1; }
done

case "$surface_mode" in
    cold|thermal) ;;
    *) echo "ERREUR: --surface-mode invalide : '$surface_mode' (cold|thermal attendu)" >&2; err=1 ;;
esac

if [[ "$surface_mode" == "thermal" ]]; then
    [[ -n "$thermal_surface_dir" ]] \
        || { echo "ERREUR: surface_mode=thermal mais --thermal-surface-dir n'est pas defini" >&2; err=1; }
    [[ -z "$thermal_surface_dir" || -d "$thermal_surface_dir" ]] \
        || { echo "ERREUR: --thermal-surface-dir introuvable : $thermal_surface_dir" >&2; err=1; }
    if [[ -n "$thermal_surface_dir" && -d "$thermal_surface_dir" ]] \
        && ! compgen -G "$thermal_surface_dir/POSCAR-*" > /dev/null; then
        echo "ERREUR: aucun fichier POSCAR-* dans $thermal_surface_dir" >&2; err=1
    fi
    [[ -x "$GEN_THERMAL_SCRIPT" ]] \
        || { echo "ERREUR: generateur introuvable ou non executable : $GEN_THERMAL_SCRIPT" >&2; err=1; }
else
    [[ -f "$path2poscar" ]] || { echo "ERREUR: LAMMPS_POSCAR introuvable : $path2poscar" >&2; err=1; }
    [[ -n "$thermal_surface_dir" ]] && echo \
        "ATTENTION: surface_mode=cold -> --thermal-surface-dir ('$thermal_surface_dir') est defini mais IGNORE. Mets surface_mode=thermal pour l'utiliser." >&2
fi

if [[ "$mode" == "zpe" ]]; then
    [[ -x "$GEN_ZPE_SCRIPT" ]] \
        || { echo "ERREUR: generateur introuvable ou non executable : $GEN_ZPE_SCRIPT" >&2; err=1; }
    [[ -x "$INIT_COND_DIR/exe" ]] \
        || { echo "ERREUR: sampler introuvable ou non executable : $INIT_COND_DIR/exe" >&2; err=1; }
    [[ -f "$INIT_COND_DIR/intrep.dat" ]] \
        || { echo "ERREUR: fichier manquant : $INIT_COND_DIR/intrep.dat" >&2; err=1; }
    [[ -f "$INIT_COND_DIR/morse_potasym.dat" ]] \
        || { echo "ERREUR: fichier manquant : $INIT_COND_DIR/morse_potasym.dat" >&2; err=1; }
fi
(( err )) && exit 1

# ----------------------------------------------------------------------------
# Preparation du repertoire de travail sur /scratch
# ----------------------------------------------------------------------------
START=$(date '+%Y-%m-%d_%Hh%M')
WORKDIR="/scratch/$USER/massif_$START"
RESULT_DIR="/scratch/$USER/Lammps"

if [[ "$results_name" == "none" ]]; then
    RESULTS_ROOT="$RESULT_DIR/results_$START"
else
    RESULTS_ROOT="$RESULT_DIR/$results_name"
fi

#if [[ "$results_name" == "none" ]]; then
#    RESULTS_ROOT="$SUBMIT_DIR/results_$START"
#else
#    RESULTS_ROOT="$SUBMIT_DIR/$results_name"
#fi


mkdir -p "$WORKDIR/models" "$RESULTS_ROOT"

# Recapitulatif
echo "========================================"
echo "=== massif_lammps - recapitulatif    ==="
echo "========================================"
echo "MODE         : $mode"
echo "SCRIPTS_DIR  : $SCRIPTS_DIR"
echo "CASE_DIR     : $CASE_DIR"
echo "SUBMIT_DIR   : $SUBMIT_DIR"
echo "WORKDIR      : $WORKDIR"
echo "RESULTS      : $RESULTS_ROOT"
if [[ "$surface_mode" == "thermal" ]]; then
    echo "SURFACE      : thermalisee (surface_mode=thermal), pool = $thermal_surface_dir"
else
    echo "SURFACE      : figee a 0K (surface_mode=cold), LAMMPS_POSCAR = $path2poscar"
fi
echo "in.simulation: $IN_SIMULATION_TEMPLATE"
echo "height H2    : $height A au-dessus du W le plus haut"
if [[ "$mode" == "zpe" ]]; then
    echo "vib-level/J  : v=$vib_level, J=$jrot (potentiel de Morse: $INIT_COND_DIR/morse_potasym.dat)"
else
    echo "r_HH         : $hh_distance A"
    echo "orientation  : $orientation"
fi
[[ -n "$frozen_ids" ]] && echo "frozen-ids   : $frozen_ids (en plus des '# fixed' du LAMMPS_POSCAR)"
echo "Modeles      : ${folders[*]}"
echo "Energies     : ${energies[*]}"
echo "theta / phi  : $theta deg / $phi deg"
echo "ntraj        : $ntraj"
echo "traj/job     : $traj_per_job"
echo "nstep / dt   : $nstep pas / $tstep ps"
echo

# Copie des graph.pb sur scratch
for model in "${folders[@]}"; do
    mkdir -p "$WORKDIR/models/$model"
    cp "$path2models/$model/graph.pb" "$WORKDIR/models/$model/"
done

# ----------------------------------------------------------------------------
# Template in.simulation
# ----------------------------------------------------------------------------
# - Inclut groups.lmp (genere par le script Python a cote des data files)
#   qui definit H, W, [O,] frozen, mobile
# - Geler les atomes du groupe "frozen"
# - Integration NVE sur le groupe "mobile" uniquement
#
# Copie du template resolu plus haut (CASE_DIR/in.simulation si present,
# sinon SCRIPTS_DIR/in.simulation.default). ${DATAFILE}, ${MODEL_DIR}, ${i}...
# restent litteraux dans le template : ils sont resolus par LAMMPS via
# lammps_worker.slurm (-var ...), pas par ce script.
cp "$IN_SIMULATION_TEMPLATE" "$WORKDIR/in.simulation"

sed -i "s|__TSTEP__|$tstep|g; s|__NSTEP__|$nstep|g" "$WORKDIR/in.simulation"

# ----------------------------------------------------------------------------
# Generation des data files + soumission des job arrays
# ----------------------------------------------------------------------------
njobs=$(( (ntraj + traj_per_job - 1) / traj_per_job ))
echo "Nombre de taches array par (modele, energie) : $njobs"
echo "Nombre total de jobs soumis                  : $(( ${#folders[@]} * ${#energies[@]} * njobs ))"
echo

for model in "${folders[@]}"; do
    echo "=== Modele : $model ==="
    for E in "${energies[@]}"; do

        # Suffixe (incluant les angles si non defaut)
        if [[ "$theta" == "0" && "$phi" == "random" ]]; then
            suffix="${E}_${model}"
        else
            suffix="${E}_T${theta}_P${phi}_${model}"
        fi
        [[ "$mode" == "nozpe" ]] && suffix="NoZPE_${suffix}"
        [[ "$surface_mode" == "thermal" ]] && suffix="Thermal_${suffix}"

        CALCDIR="$WORKDIR/CALCDIR_${suffix}"
        FINALDIR="$RESULTS_ROOT/CALCDIR_${suffix}"
        mkdir -p "$CALCDIR" "$FINALDIR"

        if [[ "$surface_mode" == "thermal" ]]; then
            # ----- SURFACE THERMALISEE (AIMD), ZPE ou NoZPE pour H2 -----
            if [[ "$mode" == "zpe" ]]; then
                echo ">> [Thermal+ZPE] Generation de $ntraj data files pour $model @ $E eV"
                echo "           (theta=$theta, phi=$phi, height=$height, v=$vib_level, J=$jrot)..."
            else
                echo ">> [Thermal+NoZPE] Generation de $ntraj data files pour $model @ $E eV"
                echo "           (theta=$theta, phi=$phi, height=$height, r_HH=$hh_distance)..."
            fi
            seed_arg="";   [[ -n "$seed" ]]       && seed_arg="--seed $seed"
            frozen_arg=""; [[ -n "$frozen_ids" ]] && frozen_arg="--frozen-ids $frozen_ids"
            if [[ "$mode" == "zpe" ]]; then
                mode_args=(--zpe --vib-level "$vib_level" --jrot "$jrot" --init-cond-dir "$INIT_COND_DIR")
            else
                mode_args=(--hh-distance "$hh_distance" --orientation "$orientation")
            fi
            (
                cd "$CALCDIR"
                "$GEN_THERMAL_SCRIPT" \
                    --poscar-dir "$thermal_surface_dir" \
                    -n "$ntraj" \
                    -E "$E" \
                    -T "$theta" \
                    -P "$phi" \
                    --height "$height" \
                    "${mode_args[@]}" \
                    $frozen_arg \
                    $seed_arg \
                    -o .
            )
        elif [[ "$mode" == "zpe" ]]; then
            # ----- MODE ZPE (Morse, echantillonnage a la volee) -----
            echo ">> [ZPE] Generation de $ntraj data files pour $model @ $E eV"
            echo "           (theta=$theta, phi=$phi, height=$height, v=$vib_level, J=$jrot)..."
            seed_arg="";   [[ -n "$seed" ]]       && seed_arg="--seed $seed"
            frozen_arg=""; [[ -n "$frozen_ids" ]] && frozen_arg="--frozen-ids $frozen_ids"
            (
                cd "$CALCDIR"
                "$GEN_ZPE_SCRIPT" \
                    -p "$path2poscar" \
                    -n "$ntraj" \
                    -E "$E" \
                    -T "$theta" \
                    -P "$phi" \
                    --height "$height" \
                    --vib-level "$vib_level" \
                    --jrot "$jrot" \
                    --init-cond-dir "$INIT_COND_DIR" \
                    $frozen_arg \
                    $seed_arg \
                    -o .
            )
        else
            # ----- MODE NoZPE -----
            echo ">> [NoZPE] Generation de $ntraj data files pour $model @ $E eV"
            echo "           (theta=$theta, phi=$phi, height=$height, r_HH=$hh_distance)..."
            seed_arg="";       [[ -n "$seed" ]]       && seed_arg="--seed $seed"
            frozen_arg="";     [[ -n "$frozen_ids" ]] && frozen_arg="--frozen-ids $frozen_ids"
            (
                cd "$CALCDIR"
                "$GEN_NOZPE_SCRIPT" \
                    -p "$path2poscar" \
                    -n "$ntraj" \
                    -E "$E" \
                    -T "$theta" \
                    -P "$phi" \
                    --height "$height" \
                    --hh-distance "$hh_distance" \
                    --orientation "$orientation" \
                    $frozen_arg \
                    $seed_arg \
                    -o .
            )
        fi

        # --- Soumission du job array ---
        echo ">> Soumission du job array ($njobs taches) pour $model @ $E eV..."
        sbatch \
            --array=1-${njobs} \
            --job-name="lmp_${suffix}" \
            --output="$FINALDIR/slurm-%A_%a.out" \
            --export=ALL,CALCDIR="$CALCDIR",FINALDIR="$FINALDIR",MODEL_DIR="$WORKDIR/models/$model",INPUT_SCRIPT="$WORKDIR/in.simulation",GROUPS_FILE="$CALCDIR/groups.lmp",NTRAJ="$ntraj",TRAJ_PER_JOB="$traj_per_job" \
            "$WORKER_SCRIPT"
        echo
    done
done

echo "========================================"
echo "=== Soumission terminee              ==="
echo "========================================"
echo "Suivi des jobs    : squeue -u $USER"
echo "Resultats finaux  : $RESULTS_ROOT/"
echo "Scratch de travail: $WORKDIR"
echo
echo "Pour tout annuler : scancel -u $USER"
