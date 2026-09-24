#!/bin/bash
#SBATCH --job-name=relaxation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32        # Curta : 32 coeurs/noeud
#SBATCH --mem=64G            
#SBATCH -t 24:00:00
#SBATCH -o relax_%j.out
#SBATCH -e relax_%j.err

module load vasp/cowboy             # même module que tes SP

# vdW-DF : le noyau doit être présent dans le dossier de run.
# Copie-le depuis un de tes dossiers SP s'il n'y est pas :
[ -f vdw_kernel.bindat ] || echo "ATTENTION : vdw_kernel.bindat absent de $(pwd)"

# (reprise après walltime UNIQUEMENT — jamais au 1er run) :
# [ -s CONTCAR ] && { cp POSCAR POSCAR.start.bak; cp CONTCAR POSCAR; }

# communication intra-nœud (identique à tes SP)
export OMPI_MCA_btl=self,vader
export OMPI_MCA_pml=ob1

mpirun vasp_std |& tee vasp.out     # 32 ranks (alloc SLURM) ; PAS 'vasp_std' seul

# contrôle de convergence ionique
grep -q "reached required accuracy" OUTCAR \
  && echo "OK : relaxation convergée." \
  || echo "ATTENTION : convergence ionique NON atteinte — relancer depuis CONTCAR."

