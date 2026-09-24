#!/usr/bin/env python3
"""
setup_aimd_calculations.py
---------------------------
Prepare des dossiers de calcul VASP AIMD a partir de POSCARs (typiquement
generes par GenerateAIMDPoscars_ThermalSurface.py, voir ce meme dossier),
et genere les scripts SLURM necessaires pour les lancer tous en job array.

Copie/adaptee de ../Single_Points_Calculations/setup_sp_calculations.py
("sp" -> "aimd" partout) : meme mecanique, seul le contexte change (AIMD au
lieu de single-point). Toute la logique de decouverte des POSCAR, de
preparation des dossiers de calcul, de chunking SLURM et de generation des
scripts est identique et deja battle-tested sur les campagnes SP.

Ce que fait le script, etape par etape :
  1. Trouve tous les fichiers POSCAR dans --poscar-dir (recherche recursive,
     en excluant --output-dir pour ne pas re-avaler ses propres copies)
  2. Pour chaque POSCAR, cree un dossier de calcul contenant :
       POSCAR, INCAR, KPOINTS, POTCAR, vdw_kernel.bindat
     POTCAR et vdw_kernel.bindat sont des LIENS SYMBOLIQUES par defaut
     (voir --copy-inputs) : economise de l'espace disque par calcul.
  3. Ecrit dirlist.txt : une ligne = un chemin de dossier de calcul
  4. Genere run_array.slurm : chaque TACHE traite --calcs-per-task calculs
     en sequence
  5. Genere submit_all.sh : les commandes sbatch necessaires pour TOUT
     lancer, en respectant la limite MaxArraySize du cluster

Les deux limites SLURM a respecter
----------------------------------
  (a) MaxArraySize — taille max d'UN seul --array (souvent 1000/1001).
      Contournee en soumettant plusieurs sbatch avec un OFFSET different.
      Verifier : scontrol show config | grep MaxArraySize

  (b) MaxSubmitJobs / limite de jobs PENDING par utilisateur (~10 000).
      Chaque TACHE d'array compte comme un job dans la queue, meme
      decoupee en plusieurs sbatch. Beaucoup de POSCARs a 1 calcul/tache
      peut donc depasser cette limite : sbatch serait rejete.
      Contournee avec --calcs-per-task N : les calculs sont groupes par N
      et executes en sequence dans la tache.
      Verifier : sacctmgr show assoc user=$USER format=user,maxsubmit

  Regle pratique : n_taches = n_calculs / calcs_per_task, et n_taches
  doit rester nettement sous (b). Le script previent si ce n'est pas le
  cas et suggere une valeur.

Fichiers d'entree partages
--------------------------
POTCAR et vdw_kernel.bindat sont typiquement identiques d'un calcul AIMD a
l'autre pour une meme campagne. Par defaut le script en fige UNE copie dans
<output-dir>/inputs/ et place un lien symbolique RELATIF dans chaque
dossier de calcul. VASP suit les liens en lecture : le calcul est
rigoureusement identique.

La copie figee dans inputs/ evite aussi qu'une edition ulterieure du
POTCAR source ne change silencieusement tous les calculs pas encore
lances. Chaque campagne a son propre jeu d'entrees, immuable.

Le lien est relatif, donc l'arborescence reste valide apres un
deplacement ou un `rsync -a` de <output-dir>.

--copy-inputs restaure l'ancien comportement (copie integrale).

Chemins
-------
Tous les chemins sont resolus en absolu avant usage. Le script SLURM
genere utilise `#SBATCH -D <output_dir>` : les logs atterrissent dans
output_dir/logs/ quel que soit le dossier depuis lequel tu lances
submit_all.sh.

Usage
-----
    python setup_aimd_calculations.py -i ./aimd_poscars -o ./AIMD_calculations \\
                                       --incar-aimd ./INCAR_AIMD \\
                                       --kpoints ./KPOINTS --potcar ./POTCAR_HOW
    python setup_aimd_calculations.py -i ./aimd_poscars --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path


# ============================================================
#  CONFIGURATION PAR DEFAUT
#  (modifiable directement ici, ou via les arguments --xxx)
# ============================================================

AIMD_POSCAR_DIR = Path("./")                   # ou chercher les POSCAR (recursif)
OUTPUT_DIR      = Path("./AIMD_calculations")   # ou creer les dossiers de calcul
INCAR_AIMD      = Path("./INCAR")
KPOINTS         = Path("./KPOINTS")
POTCAR          = Path("./POTCAR")
VDW_KERNEL      = Path("./vdw_kernel.bindat")

MAX_ARRAY_SIZE = 1000    # limite (a) : taille d'un seul --array
MAX_PENDING    = 10000   # limite (b) : jobs en attente par utilisateur
CALCS_PER_TASK = 1       # 1 = un calcul AIMD par tache SLURM

# Fichiers d'entree identiques d'un calcul a l'autre : lies plutot que
# copies, sauf si --copy-inputs. Le POSCAR est evidemment exclu, l'INCAR
# et le KPOINTS aussi (quelques Ko, et on veut la trace des parametres
# dans chaque dossier).
SHARED_INPUTS = ("POTCAR", "vdw_kernel.bindat")


# ============================================================
#  Logging
# ============================================================

def _setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("setup_aimd")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


# ============================================================
#  Etape 1 — Decouverte des POSCAR
# ============================================================

def _is_prepared_calc_dir(d: Path) -> bool:
    """
    Un dossier de calcul deja prepare contient POSCAR + INCAR + POTCAR.
    Un POSCAR genere par GenerateAIMDPoscars_ThermalSurface.py est seul
    dans son dossier (POSCAR_XXXXX/POSCAR).
    Ce critere structurel attrape les outputs de TOUS les runs precedents,
    pas seulement celui du run courant.

    Note : Path.exists() suit les liens symboliques, donc un POTCAR lie
    (mode par defaut) est bien detecte ici.
    """
    return (d / "INCAR").exists() and (d / "POTCAR").exists()


def _find_poscars(
    poscar_dir: Path,
    output_dir: Path,
    exclude: list[str],
    logger: logging.Logger,
) -> list[Path]:
    """
    Recherche recursive des POSCAR, en excluant :
      - tout ce qui est sous output_dir
      - tout dossier de calcul deja prepare (POSCAR + INCAR + POTCAR)
      - tout chemin matchant un motif --exclude

    FIX : avec les defauts (--poscar-dir ./ et --output-dir ./AIMD_calculations),
    le rglob du deuxieme run trouvait les POSCAR que le premier run avait
    copies dans AIMD_calculations/ : chaque relance doublait silencieusement
    le nombre de calculs.

    Exclure output_dir seul ne suffit pas : lancer -o AIMD_run1 puis
    -o AIMD_run2 depuis le meme -i fait avaler AIMD_run1 par le second run.
    D'ou le critere structurel _is_prepared_calc_dir.
    """
    output_res = output_dir.resolve()
    poscar_files: list[Path] = []
    n_under_output = 0
    n_prepared     = 0
    n_excluded     = 0

    for p in sorted(poscar_dir.rglob("POSCAR")):
        p_res = p.resolve()

        if output_res == p_res or output_res in p_res.parents:
            n_under_output += 1
            continue

        if _is_prepared_calc_dir(p.parent):
            n_prepared += 1
            continue

        if any(p_res.match(pat) or pat in str(p_res) for pat in exclude):
            n_excluded += 1
            continue

        poscar_files.append(p)

    if n_under_output:
        logger.info("POSCARs ignores (sous %s)              : %d",
                    output_dir.name, n_under_output)
    if n_prepared:
        logger.info("POSCARs ignores (dossier de calcul deja prepare) : %d",
                    n_prepared)
    if n_excluded:
        logger.info("POSCARs ignores (--exclude)            : %d", n_excluded)

    if not poscar_files:
        logger.error("Aucun POSCAR trouve dans %s", poscar_dir)
        if n_prepared or n_under_output:
            logger.error("(%d ont ete ignores comme deja prepares — "
                         "verifie --poscar-dir)", n_prepared + n_under_output)
        sys.exit(1)

    logger.info("POSCARs retenus : %d", len(poscar_files))
    return poscar_files


def _calc_dir_name(poscar_path: Path, poscar_dir: Path, unique_names: bool) -> str:
    """
    Nom du dossier de calcul derive du POSCAR source.

    Par defaut : le nom du dossier parent (ex. POSCAR_00001, deja unique
    quand la source est GenerateAIMDPoscars_ThermalSurface.py).
    Avec --unique-names : le chemin relatif complet aplati, ce qui rend
    les collisions impossibles au prix de noms plus longs (utile si
    --poscar-dir regroupe plusieurs campagnes de generation differentes).
    """
    parent = poscar_path.parent
    if not unique_names:
        return parent.name

    try:
        rel = parent.resolve().relative_to(poscar_dir.resolve())
    except ValueError:
        return parent.name

    flat = str(rel).replace(os.sep, "__")
    return flat if flat not in (".", "") else parent.name


# ============================================================
#  Etape 2 — Creation des dossiers de calcul
# ============================================================

def _place(src: Path, dst: Path, link: bool) -> None:
    """
    Met src a disposition sous le nom dst : lien symbolique relatif si
    link=True, copie sinon.

    Le lien est relatif (et non absolu) pour que l'arborescence survive a
    un deplacement de output_dir ou a un `rsync -a` vers un autre cluster.

    dst est retire d'abord : sans ca, une relance sur un dossier existant
    echouerait sur FileExistsError, ou pire, symlink_to ecrirait A TRAVERS
    un lien deja en place.
    """
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if link:
        dst.symlink_to(os.path.relpath(src.resolve(), dst.parent))
    else:
        shutil.copy(src, dst)


def _freeze_shared_inputs(
    output_dir: Path,
    potcar: Path,
    vdw_kernel: Path,
    dry_run: bool,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Copie UNE fois POTCAR et vdw_kernel.bindat dans <output_dir>/inputs/,
    et retourne les chemins de ces copies : ce sont elles que les liens
    des dossiers de calcul viseront.

    Sans ce figeage, tous les liens pointeraient vers le POTCAR passe en
    --potcar. Une edition ou un deplacement ulterieur de ce fichier
    changerait silencieusement tous les calculs pas encore lances.
    """
    shared = output_dir / "inputs"
    if dry_run:
        return shared / "POTCAR", shared / "vdw_kernel.bindat"

    shared.mkdir(parents=True, exist_ok=True)
    frozen_potcar = shared / "POTCAR"
    frozen_vdw    = shared / "vdw_kernel.bindat"
    shutil.copy(potcar,     frozen_potcar)
    shutil.copy(vdw_kernel, frozen_vdw)
    logger.info("Entrees partagees figees dans : %s", shared)
    return frozen_potcar, frozen_vdw


def _prepare_calc_dirs(
    poscar_files: list[Path],
    poscar_dir: Path,
    output_dir: Path,
    incar_aimd: Path,
    kpoints: Path,
    potcar: Path,
    vdw_kernel: Path,
    unique_names: bool,
    link_inputs: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> list[Path]:
    """
    Cree un dossier de calcul par POSCAR, avec tous les fichiers d'entree
    VASP mis a disposition dedans. Retourne la liste des dossiers
    (chemins absolus).

    POSCAR, INCAR et KPOINTS sont toujours copies (quelques Ko, et on veut
    la trace des parametres dans chaque dossier). POTCAR et
    vdw_kernel.bindat sont lies, sauf si link_inputs=False.

    FIX : detection des collisions de noms. Deux POSCAR issus de campagnes
    de generation differentes peuvent avoir le meme dossier parent (ex:
    run1/POSCAR_00001/ et run2/POSCAR_00001/). Avant, le second ecrasait
    le POSCAR du premier ET dirlist.txt recevait deux lignes identiques —
    donc deux taches SLURM lancaient VASP dans le MEME dossier en
    parallele (WAVECAR corrompu, OUTCAR entrelace).
    On refuse maintenant de continuer.
    """
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if link_inputs:
        potcar, vdw_kernel = _freeze_shared_inputs(
            output_dir, potcar, vdw_kernel, dry_run, logger
        )

    seen: dict[Path, Path] = {}   # calc_dir -> POSCAR source
    calc_dirs: list[Path] = []

    for poscar_path in poscar_files:
        calc_dir = (output_dir / _calc_dir_name(poscar_path, poscar_dir, unique_names)).resolve()

        if calc_dir in seen:
            logger.error("Collision de noms : %s", calc_dir)
            logger.error("  vient de : %s", seen[calc_dir])
            logger.error("  ET de    : %s", poscar_path)
            logger.error("Deux taches VASP tourneraient dans le meme dossier.")
            logger.error("Relance avec --unique-names pour deriver le nom du "
                         "chemin relatif complet.")
            sys.exit(1)
        seen[calc_dir] = poscar_path

        if not dry_run:
            calc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(poscar_path, calc_dir / "POSCAR")
            shutil.copy(incar_aimd,  calc_dir / "INCAR")
            shutil.copy(kpoints,     calc_dir / "KPOINTS")
            _place(potcar,     calc_dir / "POTCAR",             link_inputs)
            _place(vdw_kernel, calc_dir / "vdw_kernel.bindat",  link_inputs)

        calc_dirs.append(calc_dir)
        logger.debug("  Prepare : %s", calc_dir.name)

    return calc_dirs


# ============================================================
#  Etape 3 — Ecriture de dirlist.txt
# ============================================================

def _write_dirlist(calc_dirs: list[Path], output_dir: Path, logger: logging.Logger) -> Path:
    """
    Un chemin de dossier de calcul par ligne. C'est ce fichier que le
    script SLURM lit pour savoir quels dossiers traiter dans chaque tache.
    """
    dirlist = output_dir / "dirlist.txt"
    dirlist.write_text("\n".join(str(d) for d in calc_dirs) + "\n")
    logger.info("dirlist.txt ecrit : %d entrees", len(calc_dirs))
    return dirlist


# ============================================================
#  Etape 4 — Generation du template SLURM
# ============================================================

def _write_slurm_template(
    dirlist: Path,
    output_dir: Path,
    calcs_per_task: int,
    time_limit: str,
    mem: str,
    cores: int,
    vasp_module: str,
    logger: logging.Logger,
) -> Path:
    """
    Genere le script SLURM.

    Chaque TACHE traite calcs_per_task calculs en sequence (pas en
    parallele). Avec calcs_per_task=1 chaque tache fait un seul calcul
    AIMD -- mais CE calcul est lui-meme parallelise sur `cores` coeurs MPI
    (contrairement a un simple point, un AIMD est trop lent sur 1 seul
    coeur). --cores doit correspondre a NPAR*<coeurs par groupe de bandes>
    dans l'INCAR (regle usuelle : NPAR ~ sqrt(cores), divisant cores
    exactement -- ex. NPAR=2 pour cores=4, le defaut).

    Indices :
      chunk_index = SLURM_ARRAY_TASK_ID + OFFSET   (0-indexe)
      CHUNK_START = chunk_index * calcs_per_task + 1   (1-indexe pour sed)
      CHUNK_END   = CHUNK_START + calcs_per_task - 1

    FIX : `#SBATCH -D` fixe le repertoire de travail de SLURM a
    output_dir. Avant, `-o logs/...` etait relatif au dossier depuis
    lequel on lancait submit_all.sh — si ce n'etait pas output_dir, le
    dossier logs/ n'existait pas et les logs partaient a la poubelle (ou
    le job echouait au demarrage).
    """
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=vasp_aimd
#SBATCH -D {output_dir}
#SBATCH -o logs/aimd_%A_%a.out
#SBATCH -e logs/aimd_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={cores}
#SBATCH --mem={mem}
#SBATCH -t {time_limit}

module load {vasp_module}

CALCS_PER_TASK={calcs_per_task}
DIRLIST={dirlist}

# OFFSET est passe via --export a la soumission. Valeur de repli pour
# pouvoir lancer le script a la main sans --export.
OFFSET=${{OFFSET:-0}}

# Numero de chunk traite par cette tache (0-indexe)
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))

# Plage de lignes correspondante dans dirlist.txt (1-indexee pour sed)
CHUNK_START=$((CHUNK_INDEX * CALCS_PER_TASK + 1))
CHUNK_END=$((CHUNK_START + CALCS_PER_TASK - 1))

echo "Tache $SLURM_ARRAY_TASK_ID (chunk $CHUNK_INDEX) : lignes $CHUNK_START a $CHUNK_END"

mapfile -t CALCDIRS < <(sed -n "${{CHUNK_START}},${{CHUNK_END}}p" "$DIRLIST")

if [ ${{#CALCDIRS[@]}} -eq 0 ]; then
    echo "Aucun calcul pour ce chunk (fin de dirlist.txt) — rien a faire."
    exit 0
fi

for CALCDIR in "${{CALCDIRS[@]}}"; do
    echo "  -> $CALCDIR"
    # Sous-shell : le cd ne fuit pas d'une iteration a l'autre, et un
    # calcul qui echoue ne fait pas derailler les suivants.
    (
        cd "$CALCDIR" || exit 1
        export OMPI_MCA_btl=self,vader
        export OMPI_MCA_pml=ob1
        mpirun vasp_std |& tee vasp.out   # {cores} rangs (alloc SLURM) ; PAS 'vasp_std' seul
    ) || echo "  [WARN] echec : $CALCDIR"
done

echo "Termine : chunk $CHUNK_INDEX (${{#CALCDIRS[@]}} calculs)"
"""

    slurm_path = output_dir / "run_array.slurm"
    slurm_path.write_text(slurm_content)
    slurm_path.chmod(0o755)
    (output_dir / "logs").mkdir(exist_ok=True)
    return slurm_path


# ============================================================
#  Etape 5 — Generation de submit_all.sh
# ============================================================

def _write_submit_script(
    n_chunks: int,
    slurm_path: Path,
    output_dir: Path,
    max_array_size: int,
    logger: logging.Logger,
) -> Path:
    """
    Soumet autant de job arrays que necessaire pour couvrir tous les
    chunks, chaque array restant sous max_array_size. L'OFFSET est
    exprime en CHUNKS, pas en calculs individuels.

    Exemple pour 2600 chunks et max_array_size=1000 :
        sbatch --array=0-999 --export=ALL,OFFSET=0    run_array.slurm
        sbatch --array=0-999 --export=ALL,OFFSET=1000 run_array.slurm
        sbatch --array=0-599 --export=ALL,OFFSET=2000 run_array.slurm
    """
    submit_path = output_dir / "submit_all.sh"
    lines = [
        "#!/bin/bash",
        f"# Soumet {n_chunks} taches (chunks) en job arrays de {max_array_size} max",
        "# Genere par setup_aimd_calculations.py — ne pas editer a la main.",
        "set -e",
        "",
    ]

    start = 0
    while start < n_chunks:
        end   = min(start + max_array_size - 1, n_chunks - 1)
        count = end - start   # borne haute de --array (0-indexe)
        lines.append(
            f"sbatch --array=0-{count} --export=ALL,OFFSET={start} {slurm_path}"
        )
        start = end + 1

    submit_path.write_text("\n".join(lines) + "\n")
    submit_path.chmod(0o755)
    return submit_path


# ============================================================
#  Orchestration
# ============================================================

def setup(
    poscar_dir: Path,
    output_dir: Path,
    incar_aimd: Path,
    kpoints: Path,
    potcar: Path,
    vdw_kernel: Path,
    max_array_size: int,
    max_pending: int,
    calcs_per_task: int,
    unique_names: bool,
    link_inputs: bool,
    exclude: list[str],
    time_limit: str,
    mem: str,
    cores: int,
    vasp_module: str,
    dry_run: bool,
    logger: logging.Logger,
) -> int:

    if calcs_per_task < 1:
        logger.error("--calcs-per-task doit valoir au moins 1")
        return 1

    if not poscar_dir.is_dir():
        logger.error("--poscar-dir n'est pas un dossier : %s", poscar_dir)
        return 1

    required = {
        "INCAR":            incar_aimd,
        "KPOINTS":          kpoints,
        "POTCAR":           potcar,
        "vdw_kernel.bindat": vdw_kernel,
    }
    for name, path in required.items():
        if not path.exists():
            logger.error("%s introuvable : %s", name, path)
            return 1

    logger.info("Entrees partagees : %s",
                "liees (symlink relatif)" if link_inputs else "copiees")

    poscar_files = _find_poscars(poscar_dir, output_dir, exclude, logger)
    calc_dirs = _prepare_calc_dirs(
        poscar_files, poscar_dir, output_dir, incar_aimd, kpoints, potcar,
        vdw_kernel, unique_names, link_inputs, dry_run, logger,
    )
    n_calcs  = len(calc_dirs)
    n_chunks = -(-n_calcs // calcs_per_task)   # arrondi au superieur
    n_arrays = -(-n_chunks // max_array_size)

    # Avertissement sur la limite (b) : mieux vaut le savoir maintenant
    # que de voir sbatch se faire rejeter au 11e array.
    if n_chunks > max_pending:
        suggested = -(-n_calcs // max_pending)
        logger.warning("─" * 55)
        logger.warning("%d taches > limite de jobs en attente (%d).",
                       n_chunks, max_pending)
        logger.warning("sbatch sera probablement rejete.")
        logger.warning("Suggestion : --calcs-per-task %d (soit %d taches)",
                       suggested, -(-n_calcs // suggested))
        logger.warning("─" * 55)

    if dry_run:
        saved_gb = (n_calcs * _shared_size(potcar, vdw_kernel)) / 1e9
        logger.info("─" * 55)
        logger.info("DRY-RUN : rien n'a ete cree.")
        logger.info("Calculs           : %d", n_calcs)
        logger.info("Calculs par tache : %d", calcs_per_task)
        logger.info("Taches (chunks)   : %d", n_chunks)
        logger.info("Arrays sbatch     : %d", n_arrays)
        if link_inputs:
            logger.info("Volume economise  : %.1f Go (POTCAR + vdw_kernel lies)",
                        saved_gb)
        return 0

    dirlist    = _write_dirlist(calc_dirs, output_dir, logger)
    slurm_path = _write_slurm_template(
        dirlist, output_dir, calcs_per_task, time_limit, mem, cores, vasp_module, logger
    )
    submit_path = _write_submit_script(
        n_chunks, slurm_path, output_dir, max_array_size, logger
    )

    logger.info("─" * 55)
    logger.info("Calculs prepares      : %d", n_calcs)
    logger.info("Calculs par tache     : %d", calcs_per_task)
    logger.info("Taches (chunks)       : %d   <- ce nombre compte dans la queue",
                n_chunks)
    logger.info("Arrays sbatch generes : %d (max %d taches chacun)",
                n_arrays, max_array_size)
    if link_inputs:
        saved_gb = (n_calcs * _shared_size(potcar, vdw_kernel)) / 1e9
        logger.info("Volume economise      : %.1f Go (POTCAR + vdw_kernel lies)",
                    saved_gb)
    logger.info("")
    logger.info("Pour tout lancer :")
    logger.info("  bash %s", submit_path)
    return 0


def _shared_size(potcar: Path, vdw_kernel: Path) -> int:
    """Taille cumulee des entrees partagees, en octets (0 si illisible)."""
    total = 0
    for p in (potcar, vdw_kernel):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


# ============================================================
#  CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="setup_aimd_calculations",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--poscar-dir", default=str(AIMD_POSCAR_DIR),
                        help="Dossier ou chercher les POSCAR (recursif).")
    parser.add_argument("-o", "--output-dir", default=str(OUTPUT_DIR),
                        help="Dossier ou creer les dossiers de calcul.")
    parser.add_argument("--incar-aimd", default=str(INCAR_AIMD))
    parser.add_argument("--kpoints",    default=str(KPOINTS))
    parser.add_argument("--potcar",     default=str(POTCAR))
    parser.add_argument("--vdw-kernel", default=str(VDW_KERNEL))

    parser.add_argument("--calcs-per-task", type=int, default=CALCS_PER_TASK,
                        help=f"Calculs AIMD par tache, executes en sequence "
                             f"(defaut: {CALCS_PER_TASK}). Augmenter si la limite "
                             f"de jobs en attente du cluster est depassee.")
    parser.add_argument("--max-array-size", type=int, default=MAX_ARRAY_SIZE,
                        help=f"Limite MaxArraySize du cluster (defaut: {MAX_ARRAY_SIZE}). "
                             f"scontrol show config | grep MaxArraySize")
    parser.add_argument("--max-pending", type=int, default=MAX_PENDING,
                        help=f"Limite de jobs en attente par utilisateur "
                             f"(defaut: {MAX_PENDING}). Sert uniquement a avertir.")
    parser.add_argument("--unique-names", action="store_true",
                        help="Nommer les dossiers de calcul d'apres le chemin "
                             "relatif complet (evite les collisions entre campagnes "
                             "de generation differentes).")

    parser.add_argument("--copy-inputs", action="store_true",
                        help="Copier POTCAR et vdw_kernel.bindat dans chaque "
                             "dossier au lieu de les lier (ancien comportement).")

    parser.add_argument("--exclude", action="append", default=[], metavar="PAT",
                        help="Motif de chemin a ignorer (repetable), "
                             "ex: --exclude '*/backup/*'.")

    parser.add_argument("--time",        default="24:00:00", help="Walltime SLURM.")
    parser.add_argument("--mem",         default="12G",       help="Memoire SLURM [12G, calibre sur "
                                                                    "sacct MaxRSS mesure a 4 coeurs (~1.4G/tache) "
                                                                    "+ marge].")
    parser.add_argument("--cores",       type=int, default=4,
                        help="Coeurs MPI par calcul AIMD [4 -- meilleur debit total quand le "
                             "nombre de coeurs simultanes est la contrainte, pas le nombre de jobs "
                             "(benchmark : 399 coeur.s/pas a 4c vs 439/541/640 a 8/16/32c)]. "
                             "Doit correspondre au NPAR de l'INCAR (NPAR doit diviser --cores "
                             "exactement, ex. NPAR=2 pour --cores=4).")
    parser.add_argument("--vasp-module", default="vasp/cowboy",
                        help="Module a charger dans le script SLURM.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Compter et verifier, sans rien creer.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args   = parser.parse_args()
    logger = _setup_logging(getattr(logging, args.log_level))

    # Tous les chemins figes en absolu maintenant, relativement au cwd de
    # l'appelant. Rien de relatif ne survit au-dela de ce point.
    cwd = Path.cwd()

    def _abs(p: str) -> Path:
        path = Path(os.path.expanduser(p))
        return path if path.is_absolute() else (cwd / path).resolve()

    poscar_dir = _abs(args.poscar_dir)
    output_dir = _abs(args.output_dir)

    logger.info("POSCARs AIMD : %s", poscar_dir)
    logger.info("Output       : %s", output_dir)

    return setup(
        poscar_dir     = poscar_dir,
        output_dir     = output_dir,
        incar_aimd     = _abs(args.incar_aimd),
        kpoints        = _abs(args.kpoints),
        potcar         = _abs(args.potcar),
        vdw_kernel     = _abs(args.vdw_kernel),
        max_array_size = args.max_array_size,
        max_pending    = args.max_pending,
        calcs_per_task = args.calcs_per_task,
        unique_names   = args.unique_names,
        link_inputs    = not args.copy_inputs,
        exclude        = args.exclude,
        time_limit     = args.time,
        mem            = args.mem,
        cores          = args.cores,
        vasp_module    = args.vasp_module,
        dry_run        = args.dry_run,
        logger         = logger,
    )


if __name__ == "__main__":
    sys.exit(main())
