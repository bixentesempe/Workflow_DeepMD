#!/usr/bin/env python3
"""
run_qbc.py
----------
Lanceur pour le sélecteur QBC DeePMD.

Tu donnes le dossier de tes modèles et le pattern de tes trajectoires.
Le script détecte automatiquement les membres du comité et nomme les
outputs en conséquence. Plus besoin d'éditer input.in à la main.

Convention de nommage stricte :
  models/{model_name}_1/graph.pb
  models/{model_name}_2/graph.pb  →  modèle détecté : {model_name}
  ...

Seuls les suffixes purement numériques sont acceptés (ex: _1, _2, _12).
Un dossier comme {model_name}_AL1_2 ou {model_name}_analysis est ignoré.

Chemins
-------
Ce script est *path-agnostique* : il peut être lancé depuis n'importe quel
répertoire, avec des chemins relatifs ou absolus.

  - --models-dir, --pool, --output-dir, --config-out sont TOUS résolus en
    absolu par rapport au cwd de l'appelant, AVANT tout changement de
    répertoire. Un chemin relatif signifie donc toujours "relatif à là où
    j'ai tapé la commande".

Concurrence
-----------
input.in est écrit DANS --output-dir, pas à côté du script. C'est ce qui
rend les runs parallèles sûrs : deux jobs QBC qui partagent un input.in
s'écrasent mutuellement, et le perdant score le mauvais pool avec les
mauvais modèles — sans crash, sans avertissement. Un input.in par
output_dir = aucun état partagé entre jobs.

Corollaire : deux runs simultanés DOIVENT avoir des --output-dir
différents. Le script refuse de démarrer si l'output_dir contient déjà un
input.in différent (utiliser --force pour écraser).

Le chemin de config est passé explicitement à qbc_runtime quand sa
signature l'accepte ; sinon on retombe sur un chdir vers l'output_dir,
encadré par un context manager (garanti réversible).

Usage
-----
    # Cas général
    python run_qbc.py --models-dir ./models --model-name HW2O \\
                      --pool "./chemin/vers/dump/**/*.lammpstrj"

    # Avec options
    python run_qbc.py --models-dir ~/Lammps/models --model-name HW2O \\
                      --pool "./md_pools/*.traj" \\
                      --thresh-low 0.05 --thresh-high 0.50 \\
                      --stride 10 --batch-size 128

    # Choisir le dossier de sortie
    python run_qbc.py --models-dir ./models --model-name HW2O \\
                      --pool "./md_pools/*.traj" --output-dir ./mes_resultats

    # Vérifier ce qui serait lancé, sans rien lancer
    python run_qbc.py --models-dir ./models --model-name HW2O \\
                      --pool "./md_pools/*.traj" --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import re
import sys
import tempfile
from pathlib import Path

from qbc_runtime.validate_environment import main as validate_environment


# ============================================================
#  Utilitaires chemins
# ============================================================

def abspath_keep_globs(pattern: str, base: Path) -> str:
    """
    Rend un pattern glob absolu sans toucher au système de fichiers.

    On n'utilise volontairement PAS Path.resolve() : resolve() suit les
    symlinks et se comporte de façon surprenante sur des chemins qui
    n'existent pas (les wildcards n'existent jamais en tant que fichiers).
    os.path.normpath se contente de nettoyer les './' et '../' textuellement,
    ce qui préserve '*', '**' et '?' intacts.

      "./md_pools/*.traj"  + base=/a/b  ->  "/a/b/md_pools/*.traj"
      "~/pools/**/*.traj"              ->  "/home/user/pools/**/*.traj"
      "/deja/absolu/*.traj"            ->  inchangé
    """
    p = os.path.expanduser(pattern)
    if not os.path.isabs(p):
        p = os.path.join(str(base), p)
    return os.path.normpath(p)

# ============================================================
#  Découverte des modèles
# ============================================================

def _find_pb_files(models_dir: Path, model_name: str) -> list[Path]:
    """
    Cherche les sous-dossiers {model_name}_{entier} et retourne le fichier
    *.pb de chacun, trié par numéro de suffixe croissant.

    Le suffixe doit être un entier pur — pas de underscore supplémentaire,
    pas de lettre. Exemples :
      HW2O_1, HW2O_2, HW2O_12      -> acceptés
      HW2O_AL1_2, HW2O_analysis    -> rejetés

    Chemins retournés en absolu (models_dir l'est déjà).
    """
    if not models_dir.is_dir():
        raise FileNotFoundError(f"--models-dir n'est pas un dossier : {models_dir}")

    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)$")

    matches: list[tuple[int, Path]] = []
    for d in models_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            matches.append((int(m.group(1)), d))

    if not matches:
        # Message actionnable : liste ce qui existe réellement, sinon on
        # perd dix minutes à se demander si c'est le nom ou le chemin.
        available = sorted(d.name for d in models_dir.iterdir() if d.is_dir())
        raise FileNotFoundError(
            f"Aucun dossier '{model_name}_<entier>' dans {models_dir}\n"
            f"        Sous-dossiers présents : {available or '(aucun)'}"
        )

    matches.sort(key=lambda t: t[0])

    pb_files: list[Path] = []
    for _, d in matches:
        pbs = sorted(d.glob("*.pb"))
        if not pbs:
            raise FileNotFoundError(f"Aucun *.pb dans {d}")
        if len(pbs) > 1:
            raise RuntimeError(f"Plusieurs *.pb dans {d} : {[p.name for p in pbs]}")
        pb_files.append(pbs[0].resolve())

    if len(pb_files) < 2:
        raise ValueError(
            f"Au moins 2 modèles requis pour un comité, trouvé : {len(pb_files)}"
        )

    return pb_files


# ============================================================
#  Génération de input.in
# ============================================================

def _build_config(
    model_name: str,
    pb_files: list[Path],
    pool: str,
    type_map: str,
    thresh_low: float,
    thresh_high: float | None,
    top_k: int | None,
    metric: str,
    w_force: float,
    w_ene: float,
    w_vir: float,
    force_reducer: str,
    force_std_mode: str,
    skip: int,
    stride: int,
    limit: int | None,
    batch_size: int,
    output_dir: Path,
    tag: str,
    ase_format: str,
) -> str:
    models_str = ",".join(str(p) for p in pb_files)
    out_poscars = output_dir / "poscars"
    out_xyz     = output_dir / "xyz"
    out_results = output_dir / "results"

    lines = [
        f"# Auto-généré par run_qbc.py — modèle : {model_name}",
        "# NE PAS ÉDITER À LA MAIN : ce fichier est réécrit à chaque run.",
        "",
        "# --- chemins (tous absolus) ---",
        f"MODELS={models_str}",
        f"POOL={pool}",
        f"ASE_FORMAT={ase_format}",
        f"TYPE_MAP={type_map}",
        "",
        "# --- seuils de sélection ---",
        f"THRESH_LOW={thresh_low}",
    ]
    if thresh_high is not None:
        lines.append(f"THRESH_HIGH={thresh_high}")
    if top_k is not None:
        lines.append(f"TOP_K={top_k}")
    lines += [
        f"METRIC={metric}",
        "",
        "# --- poids du score ---",
        f"W_FORCE={w_force}",
        f"W_ENE={w_ene}",
        f"W_VIR={w_vir}",
        f"FORCE_REDUCER={force_reducer}",
        f"FORCE_STD_MODE={force_std_mode}  # mag | comp_norm | comp_max",
        "",
        "# --- sous-échantillonnage ---",
        f"SKIP={skip}",
        f"STRIDE={stride}          # 1 frame sur {stride} scorée",
        f"LIMIT={'None' if limit is None else limit}",
        "",
        "# --- performance ---",
        f"BATCH_SIZE={batch_size}   # frames envoyées en une fois au GPU/CPU",
        "",
        "# --- sorties ---",
        f"OUT_DIR={out_poscars}",
        f"OUT_DIR_XYZ={out_xyz}",
        f"OUT_DIR_RESULTS={out_results}",
        f"TAG={tag}",
    ]
    return "\n".join(lines) + "\n"


# ============================================================
#  CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_qbc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Requis
    parser.add_argument(
        "--models-dir", required=True, metavar="PATH",
        help="Dossier contenant les sous-dossiers {model_name}_{entier}.",
    )
    parser.add_argument(
        "--model-name", required=True, metavar="NAME",
        help="Nom de base du modèle, ex: HW2O. Le script cherche les "
             "dossiers NAME_<entier> dans --models-dir.",
    )
    parser.add_argument(
        "--pool", required=True, metavar="PATTERN",
        help="Pattern glob vers les trajectoires, ex: \"./md_pools/*.traj\". "
             "Relatif au dossier depuis lequel tu lances la commande. "
             "Pense aux guillemets pour que le shell ne développe pas le glob.",
    )

    # Optionnels fréquents
    parser.add_argument("--type-map",    default="H,W,O", metavar="A,B,C",
                        help="Espèces atomiques séparées par virgule (défaut: H,W,O). "
                             "DOIT correspondre à l'ordre du type_map d'entraînement.")
    parser.add_argument("--ase-format",  default="traj",
                        help="Format ASE des trajectoires (défaut: traj).")
    parser.add_argument("--thresh-low",  type=float, default=0.05,
                        help="Score minimum pour sélectionner une frame (défaut: 0.05).")
    parser.add_argument("--thresh-high", type=float, default=0.50,
                        help="Score maximum pour sélectionner une frame (défaut: 0.50).")
    parser.add_argument("--top-k",       type=int, default=None,
                        help="Sélectionner les K meilleures frames (remplace thresh).")
    parser.add_argument("--stride",      type=int, default=10,
                        help="1 frame sur N scorée (défaut: 10).")
    parser.add_argument("--batch-size",  type=int, default=128,
                        help="Frames par batch GPU (défaut: 128, réduire si OOM).")
    parser.add_argument("-o", "--output-dir", default=None, metavar="PATH",
                        help="Dossier de sortie (défaut: ./<model_name>_qbc/).")
    parser.add_argument("--config-out",  default=None, metavar="PATH",
                        help="Où écrire input.in (défaut: <output-dir>/input.in). "
                             "Ne pointe jamais deux runs simultanés au même endroit.")
    parser.add_argument("--force", action="store_true",
                        help="Écraser un input.in existant et différent dans "
                             "l'output-dir (sinon le script refuse de démarrer).")

    # Optionnels avancés
    parser.add_argument("--metric",         default="score",
                        choices=["score", "dF", "dF_max", "dE", "dV"])
    parser.add_argument("--w-force",        type=float, default=1.0)
    parser.add_argument("--w-ene",          type=float, default=0.0)
    parser.add_argument("--w-vir",          type=float, default=0.0)
    parser.add_argument("--force-reducer",  default="max", choices=["max", "mean"])
    parser.add_argument("--force-std-mode", default="mag",
                        choices=["mag", "comp_norm", "comp_max"])
    parser.add_argument("--skip",  type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag",   default="")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Ne pas valider l'environnement avant de lancer.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Écrire input.in, l'afficher, et s'arrêter là.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ── Tous les chemins sont figés MAINTENANT, relativement au cwd de
    #    l'appelant. Rien de relatif ne survit au-delà de ce bloc.
    caller_cwd = Path.cwd()
    root       = Path(__file__).resolve().parent   # racine de qbc_runtime

    models_dir = Path(os.path.expanduser(args.models_dir))
    if not models_dir.is_absolute():
        models_dir = caller_cwd / models_dir
    models_dir = models_dir.resolve()

    pool = abspath_keep_globs(args.pool, caller_cwd)

    if args.output_dir:
        output_dir = Path(os.path.expanduser(args.output_dir))
        if not output_dir.is_absolute():
            output_dir = caller_cwd / output_dir
    else:
        output_dir = caller_cwd / f"{args.model_name}_qbc"
    output_dir = Path(os.path.normpath(output_dir))

    if args.config_out:
        config_path = Path(os.path.expanduser(args.config_out))
        if not config_path.is_absolute():
            config_path = caller_cwd / config_path
        config_path = Path(os.path.normpath(config_path))
    else:
        # Dans l'output_dir : un input.in par run, jamais partagé.
        config_path = output_dir / "input.in"

    # ── Découverte des modèles ────────────────────────────────────────────
    try:
        pb_files = _find_pb_files(models_dir, args.model_name)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    name = args.model_name
    print(f"[INFO] Modèle        : {name}")
    print(f"[INFO] Models dir    : {models_dir}")
    print(f"[INFO] {len(pb_files)} modèle(s) : "
          f"{[str(p.relative_to(models_dir)) for p in pb_files]}")
    print(f"[INFO] Pool          : {pool}")
    print(f"[INFO] Outputs     → {output_dir}")

    # ── Génération de input.in ────────────────────────────────────────────
    config_content = _build_config(
        model_name     = name,
        pb_files       = pb_files,
        pool           = pool,
        type_map       = args.type_map,
        thresh_low     = args.thresh_low,
        thresh_high    = args.thresh_high if args.top_k is None else None,
        top_k          = args.top_k,
        metric         = args.metric,
        w_force        = args.w_force,
        w_ene          = args.w_ene,
        w_vir          = args.w_vir,
        force_reducer  = args.force_reducer,
        force_std_mode = args.force_std_mode,
        skip           = args.skip,
        stride         = args.stride,
        limit          = args.limit,
        batch_size     = args.batch_size,
        output_dir     = output_dir,
        tag            = args.tag,
        ase_format     = args.ase_format,
    )

    # Garde anti-collision : si un input.in DIFFÉRENT est déjà là, c'est
    # soit un run concurrent qui vise le même output_dir, soit un reste
    # d'une config précédente. Dans les deux cas, écraser en silence
    # produirait des résultats faux et intraçables.
    if config_path.exists() and not args.force:
        existing = config_path.read_text()
        if existing != config_content:
            print(f"[ERROR] Un input.in différent existe déjà : {config_path}")
            print( "[ERROR] Un autre run vise-t-il le même --output-dir ?")
            print( "[ERROR] Utilise un --output-dir distinct, ou --force pour écraser.")
            return 1

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    print(f"[INFO] input.in généré → {config_path}")

    # Reste de l'ancienne version du script, qui écrivait input.in à côté
    # du code. Plus jamais lu, mais autant le signaler une fois.
    orphan = root / "input.in"
    if orphan.exists():
        print(f"[NOTE] input.in orphelin à côté du script : {orphan} — "
              f"il n'est plus lu, tu peux le supprimer.")

    if args.dry_run:
        print("\n--- input.in ---")
        print(config_content, end="")
        print("--- (dry-run : rien n'a été lancé) ---")
        return 0

    # ── Environnement ─────────────────────────────────────────────────────
    mpl_dir = Path(tempfile.gettempdir()) / "qbc-mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    # Certains runtimes lisent aussi la config depuis l'env : coût nul,
    # et ça évite un chdir si la variable est supportée.
    os.environ["QBC_INPUT"] = str(config_path)

    # ── Validation ────────────────────────────────────────────────────────
    if not args.skip_validation:
        status = validate_environment(str(config_path))
        if status != 0:
            return status

    # ── Lancement ─────────────────────────────────────────────────────────
    from qbc_runtime import app as qbc
    qbc.main(str(config_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
