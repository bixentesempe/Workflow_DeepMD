#!/bin/bash
#SBATCH -o deepNN.%j.out
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00

function phelp() {
    appname="to_train_NN.sh"
    printf 'Usage for %s:\n' ${appname}
    printf '%s [OPTION]... [MODELS ...]\n' ${appname}
    printf 'Train network with deepmd\n You should provide at least one model.\n'
    printf 'You need :\n'
    printf '\t- a models folder in your submit directory specify at the end of options\n'
    printf '\t\t this folder have to contain a input.json file\n\t\t with accessible training and validation data directory from compute node\n\t\t mean in home or projects\n'
    printf 'OPTION are :\n'
    printf '\t-r --restart [file] : to restart give the file from which you want to restart (checkpoint_dir/model.ckpt)\n'
    printf '\t-h --help           : display this help\n'

}
dodebug="false"
torestart="no"
f2restart="none"
cd $TMPDIR
OPTS=$(getopt --options r:,h --longoptions 'restart:,debug,help' -n to_train_NN.sh  -- $@)
eval set -- "$OPTS"
while true
do
    case "${1}"
        in
        -h | --help) phelp ; shift 1; exit 0 ;;
        -r | --restart) torestart="true";f2restart=${2} ; shift 2 ;;
        --debug) set -x ;dodebug="yes"; shift 1 ;;
        --)
            shift 1
            break
            ;;
        *)
            echo "Unrecognized option '$1'" >&2
            phelp
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))
declare -a modeles
modeles+=( $@ )

. /etc/profile.d/modules.sh
module load deepmd/3.0.3

for model in ${modeles[@]}
do
    mkdir ${TMPDIR}/${model}
    cp ${SLURM_SUBMIT_DIR}/${model}/input.json ${TMPDIR}/${model}
    cp -r ${SLURM_SUBMIT_DIR}/${model}/trainset* ${TMPDIR}/${model}
    cp -r ${SLURM_SUBMIT_DIR}/${model}/valset* ${TMPDIR}/${model}
    cd ${TMPDIR}/${model} || exit 1
    dp train input.json --skip-neighbor-stat -l log.log
    dp freeze -c checkpoint_dir/ -o graph.pb
    wait
    #we copy result to submited folder
    rsync -avz ${TMPDIR}/${model}/ ${SLURM_SUBMIT_DIR}/${model}/
    goodcopy=$?
    cd ..
    if [ $goodcopy -eq 0 ]
    then
        #if data have been copied erase them on the compute node
        rm -rf ./${model}
    fi
done
