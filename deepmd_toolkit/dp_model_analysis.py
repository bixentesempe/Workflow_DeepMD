#!/usr/bin/env python3
"""
dp_model_analysis.py
--------------------
Pipeline d'analyse des modèles DeepMD.

  Step 1 — Courbes d'entraînement : lit lcurve.out dans chaque dossier modèle,
            superpose toutes les courbes sur les mêmes graphes.
  Step 2 — Inférence              : fait tourner chaque modèle sur la trajectoire
            de référence et écrit les fichiers énergie/forces.
  Step 3 — Plots de corrélation   : parités, histogrammes d'erreur, angles.

Structure de dossiers attendue (avec --model-name graph) :
  BASE_DIR/
    graph_1/
      graph.pb        ← modèle DeepMD gelé
      lcurve.out      ← log d'entraînement
    graph_2/
      graph.pb
      lcurve.out
    graph_3/ ...

Note sur le matching des dossiers
----------------------------------
Seuls les dossiers correspondant EXACTEMENT au pattern
    <model_name>_<entier>
sont pris en compte (ex: HW_1, HW_2, HW_12). Un dossier comme
HW_AL_1 ou HW_bis_3 est ignoré, même s'il commence par "HW_".

Usage
-----
    # Pipeline complet
    python dp_model_analysis.py --model-name graph --trajectory ./test.traj

    # Dossier de base différent du répertoire courant
    python dp_model_analysis.py --model-name graph --base-dir /data/runs \\
                                --trajectory ./test.traj

    # Seulement les courbes d'entraînement (pas d'inférence)
    python dp_model_analysis.py --model-name graph --steps 1

    # Refaire seulement les plots après une inférence déjà faite
    python dp_model_analysis.py --model-name graph --steps 3

    # Fenêtre pour le résumé RMSE final
    python dp_model_analysis.py --model-name graph --trajectory ./test.traj \\
                                --summary-window 500
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("dp_analysis")
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Découverte des dossiers modèles
# ---------------------------------------------------------------------------

def _model_dir_pattern(model_name: str) -> re.Pattern:
    """
    Construit le pattern strict <model_name>_<entier> (ancré début/fin),
    pour éviter qu'un nom comme HW_AL_1 ne soit pris pour HW_1.
    """
    return re.compile(rf"^{re.escape(model_name)}_(\d+)$")


def _find_model_dirs(base_dir: Path, model_name: str) -> list[Path]:
    """
    Retourne la liste triée (numériquement, par le suffixe entier) des
    sous-dossiers correspondant EXACTEMENT au pattern ${model_name}_<entier>
    qui contiennent au moins un fichier .pb.

    Exemple avec model_name="HW" :
        HW_1, HW_2, HW_12   → acceptés
        HW_AL_1, HW_bis     → rejetés
    """
    pattern = _model_dir_pattern(model_name)
    matches: list[tuple[int, Path]] = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        if not list(d.glob("*.pb")):
            continue
        matches.append((int(m.group(1)), d))

    if not matches:
        raise FileNotFoundError(
            f"Aucun dossier '{model_name}_<N>' (N entier) avec un *.pb trouvé dans {base_dir}"
        )
    matches.sort(key=lambda t: t[0])
    return [d for _, d in matches]


def _find_pb(model_dir: Path) -> Path:
    pbs = list(model_dir.glob("*.pb"))
    if len(pbs) != 1:
        raise RuntimeError(
            f"Attendu exactement un *.pb dans {model_dir}, trouvé : {pbs}"
        )
    return pbs[0]


# ---------------------------------------------------------------------------
# Statistiques partagées
# ---------------------------------------------------------------------------

def _stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Retourne (MAE, RMSE, R²)."""
    diff  = y_pred - y_true
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2    = 1.0 - float(np.sum(diff ** 2)) / denom if (denom > 0 and len(y_true) > 1) else float("nan")
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Step 1 — Courbes d'entraînement (toutes superposées)
# ---------------------------------------------------------------------------

def step1_training_curves(
    base_dir: Path,
    model_name: str,
    output_dir: Path,
    summary_window: int,
    logger: logging.Logger,
) -> None:
    logger.info("=== Step 1 — Courbes d'entraînement ===")
    plots_dir = output_dir / "training_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _find_model_dirs(base_dir, model_name)
    curve_data: dict[str, dict] = {}

    for md in model_dirs:
        label       = md.name
        lcurve_path = md / "lcurve.out"
        if not lcurve_path.exists():
            logger.warning("  [%s] lcurve.out introuvable — ignoré.", label)
            continue
        data = np.genfromtxt(lcurve_path, comments="#", dtype=float)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 8:
            logger.warning("  [%s] lcurve.out a moins de 8 colonnes — ignoré.", label)
            continue
        curve_data[label] = {
            "step":          data[:, 0],
            "rmse_e_val":    data[:, 3],
            "rmse_e_trn":    data[:, 4],
            "rmse_f_val":    data[:, 5],
            "rmse_f_trn":    data[:, 6],
            "learning_rate": data[:, 7],
        }
        logger.info("  [%s]  %d pas chargés.", label, len(data))

    if not curve_data:
        logger.error("  Aucun fichier lcurve.out valide trouvé.")
        return

    # ── Tableau résumé ───────────────────────────────────────────────────────
    rows = []
    for label, v in curve_data.items():
        n = min(summary_window, len(v["step"]))
        rows.append({
            "model":                       label,
            "n_steps":                     int(len(v["step"])),
            "final_lr":                    float(v["learning_rate"][-1]),
            "energy_rmse_train_last_win":  float(v["rmse_e_trn"][-n:].mean()),
            "energy_rmse_val_last_win":    float(v["rmse_e_val"][-n:].mean()),
            "force_rmse_train_last_win":   float(v["rmse_f_trn"][-n:].mean()),
            "force_rmse_val_last_win":     float(v["rmse_f_val"][-n:].mean()),
        })
    summary_df = (
        pd.DataFrame(rows)
        .sort_values("force_rmse_val_last_win")
        .reset_index(drop=True)
    )
    csv_path = plots_dir / "training_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("  Résumé → %s", csv_path)

    # ── Learning rate (toutes courbes superposées) ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, v in curve_data.items():
        ax.plot(v["step"], v["learning_rate"], label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "lr_schedule.png", dpi=150)
    plt.close(fig)

    # ── RMSE énergie (val + train superposés) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, v in curve_data.items():
        ax.plot(v["step"], v["rmse_e_val"], label=f"{label} val")
        ax.plot(v["step"], v["rmse_e_trn"], linestyle="--", label=f"{label} trn")
    ax.set_xlabel("Step"); ax.set_ylabel("Energy RMSE (eV/atom)")
    ax.set_title("Energy RMSE — train vs validation")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_rmse.png", dpi=150)
    plt.close(fig)

    # ── RMSE forces (val + train superposés) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, v in curve_data.items():
        ax.plot(v["step"], v["rmse_f_val"], label=f"{label} val")
        ax.plot(v["step"], v["rmse_f_trn"], linestyle="--", label=f"{label} trn")
    ax.set_xlabel("Step"); ax.set_ylabel("Force RMSE (eV/Å)")
    ax.set_title("Force RMSE — train vs validation")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "force_rmse.png", dpi=150)
    plt.close(fig)

    logger.info("  Plots sauvegardés dans %s", plots_dir)
    logger.info("=== Step 1 terminé ===")


# ---------------------------------------------------------------------------
# Step 2 — Inférence
# ---------------------------------------------------------------------------

def step2_inference(
    base_dir: Path,
    model_name: str,
    trajectory_file: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    logger.info("=== Step 2 — Inférence ===")

    try:
        from ase.io import read as ase_read
        from deepmd.infer import DeepPot
        try:
            from ase.calculators.deepmd import DeepMDCalculator as DP
        except Exception:
            from deepmd.calculator import DP
    except ImportError as exc:
        logger.error("  DeepMD / ASE non disponibles : %s", exc)
        raise

    frames   = ase_read(str(trajectory_file), index=":")
    n_frames = len(frames)
    logger.info("  %d frames chargées depuis %s", n_frames, trajectory_file)

    for md in _find_model_dirs(base_dir, model_name):
        label   = md.name
        pb_file = _find_pb(md)
        out_dir = output_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("  [%s] inférence avec %s …", label, pb_file.name)

        model = DeepPot(str(pb_file))
        calc  = DP(model=str(pb_file), type_map=model.get_type_map())

        ef_path  = out_dir / "energies_forces.txt"
        efa_path = out_dir / "energies_forces_atoms.txt"

        with (
            ef_path.open("w",  encoding="utf-8") as fh_e,
            efa_path.open("w", encoding="utf-8") as fh_a,
        ):
            fh_e.write("# frame energy_DFT_eV energy_DP_eV\n")
            fh_a.write(
                "# frame atom element "
                "fx_DFT fy_DFT fz_DFT "
                "fx_DP fy_DP fz_DP "
                "energy_DFT energy_DP\n"
            )
            for idx, atoms in enumerate(frames):
                ref       = atoms.copy(); ref.calc = atoms.calc
                dp_atoms  = atoms.copy(); dp_atoms.calc = calc
                e_ref = ref.get_potential_energy()
                f_ref = ref.get_forces()
                e_dp  = dp_atoms.get_potential_energy()
                f_dp  = dp_atoms.get_forces()
                fh_e.write(f"{idx:6d} {e_ref: .10f} {e_dp: .10f}\n")
                for ai, sym in enumerate(ref.get_chemical_symbols()):
                    fxr, fyr, fzr = f_ref[ai]
                    fxd, fyd, fzd = f_dp[ai]
                    fh_a.write(
                        f"{idx:6d} {ai:6d} {sym:2s} "
                        f"{fxr: .6f} {fyr: .6f} {fzr: .6f} "
                        f"{fxd: .6f} {fyd: .6f} {fzd: .6f} "
                        f"{e_ref: .10f} {e_dp: .10f}\n"
                    )

        logger.info("  [%s] ok → %s", label, out_dir)

    logger.info("=== Step 2 terminé ===")


# ---------------------------------------------------------------------------
# Step 3 — Plots de corrélation
# ---------------------------------------------------------------------------

def _angle_diff_deg(atom_df: pd.DataFrame) -> np.ndarray:
    ref  = atom_df[["fx_DFT", "fy_DFT", "fz_DFT"]].to_numpy(float)
    dp   = atom_df[["fx_DP",  "fy_DP",  "fz_DP" ]].to_numpy(float)
    nr   = np.linalg.norm(ref, axis=1)
    nd   = np.linalg.norm(dp,  axis=1)
    mask = (nr > 1e-12) & (nd > 1e-12)
    cos  = np.clip(np.sum(ref[mask] * dp[mask], axis=1) / (nr[mask] * nd[mask]), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _make_panel_axes(n: int, fw: int = 6, fh: int = 5):
    ncols = min(2, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fw * ncols, fh * nrows),
                              squeeze=False)
    return fig, axes.ravel()


def _hide_extra(axes: np.ndarray, used: int) -> None:
    for ax in axes[used:]:
        ax.axis("off")


def step3_correlation_plots(
    model_name: str,
    output_dir: Path,
    logger: logging.Logger,
    n_atoms: int | None = None,
) -> None:
    logger.info("=== Step 3 — Plots de corrélation ===")

    reports_dir = output_dir / "correlation_reports"
    plots_dir   = output_dir / "correlation_plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement des résultats d'inférence ─────────────────────────────────
    SKIP = {"correlation_reports", "correlation_plots", "training_curves"}
    pattern = _model_dir_pattern(model_name)
    run_dirs: list[tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_dir() or p.name in SKIP:
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        run_dirs.append((int(m.group(1)), p))
    run_dirs.sort(key=lambda t: t[0])

    run_data: dict[str, dict] = {}
    for _, run_dir in run_dirs:
        ef  = run_dir / "energies_forces.txt"
        efa = run_dir / "energies_forces_atoms.txt"
        if not ef.exists() or not efa.exists():
            continue
        energy_df = pd.read_csv(
            ef, comment="#", sep=r"\s+",
            names=["frame", "energy_DFT_eV", "energy_DP_eV"],
            engine="python",
        ).dropna().reset_index(drop=True)
        atom_df = pd.read_csv(
            efa, comment="#", sep=r"\s+",
            names=["frame", "atom", "element",
                   "fx_DFT", "fy_DFT", "fz_DFT",
                   "fx_DP",  "fy_DP",  "fz_DP",
                   "energy_DFT", "energy_DP"],
            engine="python",
        ).dropna().reset_index(drop=True)
        run_data[run_dir.name] = {"energy": energy_df, "atom": atom_df}

    if not run_data:
        logger.error("  Aucun résultat d'inférence trouvé dans %s.", output_dir)
        return

    labels   = list(run_data)
    n_models = len(labels)
    logger.info("  %d modèle(s) : %s", n_models, labels)

    # ── Résumé cross-modèles ─────────────────────────────────────────────────
    summary_rows = []
    for label, tables in run_data.items():
        et  = tables["energy"]["energy_DFT_eV"].to_numpy(float)
        ep  = tables["energy"]["energy_DP_eV"].to_numpy(float)
        _,  rmse_e, _ = _stats(et, ep)
        fmr = np.sqrt(
            tables["atom"]["fx_DFT"]**2 + tables["atom"]["fy_DFT"]**2 + tables["atom"]["fz_DFT"]**2
        ).to_numpy(float)
        fmd = np.sqrt(
            tables["atom"]["fx_DP"]**2  + tables["atom"]["fy_DP"]**2  + tables["atom"]["fz_DP"]**2
        ).to_numpy(float)
        _, rmse_f, _ = _stats(fmr, fmd)
        summary_rows.append({"model": label, "n_frames": len(tables["energy"]),
                              "energy_rmse_eV": rmse_e, "force_rmse_eVA": rmse_f})

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("energy_rmse_eV")
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_dir / "inference_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(summary_df["model"], summary_df["energy_rmse_eV"], color="tab:blue")
    axes[0].set_ylabel("Energy RMSE (eV)"); axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(summary_df["model"], summary_df["force_rmse_eVA"], color="tab:orange")
    axes[1].set_ylabel("Force RMSE (eV/Å)"); axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / "cross_model_rmse.png", dpi=150)
    plt.close(fig)

    # ── Cache + rapports texte ────────────────────────────────────────────────
    cache: dict[str, dict] = {}
    for label, tables in run_data.items():
        edf = tables["energy"]; adf = tables["atom"]
        e_ref = edf["energy_DFT_eV"].to_numpy(float)
        e_dp  = edf["energy_DP_eV"].to_numpy(float)
        mae_e, rmse_e, r2_e = _stats(e_ref, e_dp)

        if n_atoms is None:
            cnt = adf.groupby("frame").size().reindex(edf["frame"]).to_numpy(float)
        else:
            cnt = np.full(len(edf), float(n_atoms))
        mask = np.isfinite(cnt) & (cnt > 0)
        er_pa = e_ref[mask] / cnt[mask]; ed_pa = e_dp[mask] / cnt[mask]
        err_pa = ed_pa - er_pa
        mae_pa, rmse_pa, r2_pa = _stats(er_pa, ed_pa)

        fmr = np.sqrt(adf["fx_DFT"]**2 + adf["fy_DFT"]**2 + adf["fz_DFT"]**2).to_numpy(float)
        fmd = np.sqrt(adf["fx_DP"]**2  + adf["fy_DP"]**2  + adf["fz_DP"]**2 ).to_numpy(float)
        mae_f, rmse_f, r2_f = _stats(fmr, fmd)
        ang = _angle_diff_deg(adf)

        cache[label] = dict(
            e_ref=e_ref, e_dp=e_dp, rmse_e=rmse_e, r2_e=r2_e,
            er_pa=er_pa, ed_pa=ed_pa, err_pa=err_pa, rmse_pa=rmse_pa, r2_pa=r2_pa,
            fmr=fmr, fmd=fmd, rmse_f=rmse_f, r2_f=r2_f, ang=ang, adf=adf,
        )

        rpt = reports_dir / f"{label}_stats.txt"
        with rpt.open("w", encoding="utf-8") as fh:
            fh.write(f"Model: {label}\n")
            fh.write(f"Energy MAE (eV): {mae_e:.6f}\n")
            fh.write(f"Energy RMSE (eV): {rmse_e:.6f}\n")
            fh.write(f"Energy R²: {r2_e:.6f}\n")
            fh.write(f"Energy/atom MAE (eV/atom): {mae_pa:.6f}\n")
            fh.write(f"Energy/atom RMSE (eV/atom): {rmse_pa:.6f}\n")
            fh.write(f"Energy/atom R²: {r2_pa:.6f}\n")
            for c in ["x", "y", "z"]:
                r = adf[f"f{c}_DFT"].to_numpy(float)
                p = adf[f"f{c}_DP"].to_numpy(float)
                m, rm, r2c = _stats(r, p)
                fh.write(f"Force {c.upper()} MAE (eV/Å): {m:.6f}\n")
                fh.write(f"Force {c.upper()} RMSE (eV/Å): {rm:.6f}\n")
                fh.write(f"Force {c.upper()} R²: {r2c:.6f}\n")
            fh.write(f"Force mag MAE (eV/Å): {mae_f:.6f}\n")
            fh.write(f"Force mag RMSE (eV/Å): {rmse_f:.6f}\n")
            fh.write(f"Force mag R²: {r2_f:.6f}\n")
            fh.write(f"Force angle mean (deg): {ang.mean():.6f}\n")
            fh.write(f"Force angle median (deg): {float(np.median(ang)):.6f}\n")
            fh.write(f"Force angle P90 (deg): {float(np.percentile(ang, 90)):.6f}\n")
        logger.info("  Rapport → %s", rpt)

    # ── Parité énergie ───────────────────────────────────────────────────────
    fig, axes = _make_panel_axes(n_models, 6, 6)
    for ax, label in zip(axes, labels):
        v = cache[label]
        lims = [min(v["e_ref"].min(), v["e_dp"].min()) - .1,
                max(v["e_ref"].max(), v["e_dp"].max()) + .1]
        ax.scatter(v["e_ref"], v["e_dp"],
                   c=np.abs(v["e_dp"] - v["e_ref"]), s=18,
                   cmap="viridis", alpha=0.9, edgecolors="none")
        ax.plot(lims, lims, "k-", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Energy DFT (eV)"); ax.set_ylabel("Energy DP (eV)"); ax.set_title(label)
        ax.text(.05, .95, f"RMSE = {v['rmse_e']:.4f} eV\nR² = {v['r2_e']:.4f}",
                transform=ax.transAxes, va="top", fontsize=11)
    _hide_extra(axes, n_models)
    fig.suptitle("Energy parity", y=1.02); fig.tight_layout()
    fig.savefig(plots_dir / "energy_parity.png", dpi=150); plt.close(fig)

    # ── Parité énergie/atome ──────────────────────────────────────────────────
    fig, axes = _make_panel_axes(n_models, 6, 5)
    for ax, label in zip(axes, labels):
        v = cache[label]
        lims = [min(v["er_pa"].min(), v["ed_pa"].min()) - .01,
                max(v["er_pa"].max(), v["ed_pa"].max()) + .01]
        ax.scatter(v["er_pa"], v["ed_pa"],
                   c=np.abs(v["ed_pa"] - v["er_pa"]) * 1000, s=18,
                   cmap="viridis", alpha=0.9, edgecolors="none")
        ax.plot(lims, lims, "k-", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("E/atom DFT (eV/atom)"); ax.set_ylabel("E/atom DP (eV/atom)"); ax.set_title(label)
        ax.text(.05, .95, f"RMSE = {v['rmse_pa']*1000:.2f} meV/atom\nR² = {v['r2_pa']:.4f}",
                transform=ax.transAxes, va="top", fontsize=11)
    _hide_extra(axes, n_models)
    fig.suptitle("Energy/atom parity", y=1.02); fig.tight_layout()
    fig.savefig(plots_dir / "energy_per_atom_parity.png", dpi=150); plt.close(fig)

    # ── Histogramme erreur énergie/atome ─────────────────────────────────────
    fig, axes = _make_panel_axes(n_models, 6, 4)
    for ax, label in zip(axes, labels):
        ax.hist(cache[label]["err_pa"], bins=80, color="steelblue", alpha=.8, edgecolor="black")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(label); ax.set_xlabel("E/atom error (DP−DFT) [eV/atom]"); ax.set_ylabel("Count")
    _hide_extra(axes, n_models)
    fig.suptitle("Energy/atom error distribution", y=1.02); fig.tight_layout()
    fig.savefig(plots_dir / "energy_per_atom_error_hist.png", dpi=150); plt.close(fig)

    # ── Parités Fx, Fy, Fz ───────────────────────────────────────────────────
    for comp in ["x", "y", "z"]:
        fig, axes = _make_panel_axes(n_models, 6, 6)
        for ax, label in zip(axes, labels):
            rc = cache[label]["adf"][f"f{comp}_DFT"].to_numpy(float)
            pc = cache[label]["adf"][f"f{comp}_DP"].to_numpy(float)
            mae_c, rmse_c, r2_c = _stats(rc, pc)
            lim = max(np.abs(rc).max(), np.abs(pc).max())
            ax.hexbin(rc, pc, gridsize=120, bins="log", cmap="viridis")
            ax.plot([-lim, lim], [-lim, lim], "k-", lw=1)
            ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"F{comp} DFT (eV/Å)"); ax.set_ylabel(f"F{comp} DP (eV/Å)"); ax.set_title(label)
            ax.text(.05, .95, f"MAE={mae_c:.4f}\nRMSE={rmse_c:.4f}\nR²={r2_c:.4f}",
                    transform=ax.transAxes, va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="white", alpha=.9, ec="0.8"))
        _hide_extra(axes, n_models)
        fig.suptitle(f"Force {comp.upper()} parity", y=1.02); fig.tight_layout()
        fig.savefig(plots_dir / f"force_{comp}_parity.png", dpi=150); plt.close(fig)

    # ── Parité |F| ────────────────────────────────────────────────────────────
    fig, axes = _make_panel_axes(n_models, 6, 6)
    for ax, label in zip(axes, labels):
        v = cache[label]
        lim = max(np.abs(v["fmr"]).max(), np.abs(v["fmd"]).max()) + .5
        ax.hexbin(v["fmr"], v["fmd"], gridsize=120, bins="log", cmap="viridis")
        ax.plot([0, lim], [0, lim], "k-", lw=1)
        ax.set_xlim([0, lim]); ax.set_ylim([0, lim]); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("|F| DFT (eV/Å)"); ax.set_ylabel("|F| DP (eV/Å)"); ax.set_title(label)
        ax.text(.05, .95, f"RMSE={v['rmse_f']:.4f}\nR²={v['r2_f']:.4f}",
                transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(boxstyle="round", fc="white", alpha=.9, ec="0.8"))
    _hide_extra(axes, n_models)
    fig.suptitle("Force magnitude parity", y=1.02); fig.tight_layout()
    fig.savefig(plots_dir / "force_magnitude_parity.png", dpi=150); plt.close(fig)

    # ── Histogramme angle forces ──────────────────────────────────────────────
    fig, axes = _make_panel_axes(n_models, 6, 4)
    for ax, label in zip(axes, labels):
        ang = cache[label]["ang"]
        ax.hist(ang, bins=90, color="steelblue", alpha=.85, edgecolor="black")
        ax.axvline(ang.mean(), color="darkred", linestyle="--", lw=1.5)
        ax.set_title(label); ax.set_xlabel("Angle (deg)"); ax.set_ylabel("Count")
        ax.text(.98, .95,
                f"Mean={ang.mean():.2f}°\nMedian={float(np.median(ang)):.2f}°\nP90={float(np.percentile(ang,90)):.2f}°",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=.9, ec="0.8"))
    _hide_extra(axes, n_models)
    fig.suptitle("Force-vector angle difference", y=1.02); fig.tight_layout()
    fig.savefig(plots_dir / "force_angle_hist.png", dpi=150); plt.close(fig)

    logger.info("  Plots → %s", plots_dir)
    logger.info("=== Step 3 terminé ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_STEPS = [1, 2, 3]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dp_model_analysis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name", required=True, metavar="NAME",
        help="Nom de base du modèle (ex: 'graph'). "
             "Le script cherche les dossiers NAME_1, NAME_2, NAME_3, NAME_4 … "
             "(uniquement un entier après l'underscore, ex: HW_AL_1 est ignoré).",
    )
    parser.add_argument(
        "--base-dir", default=".", metavar="PATH",
        help="Répertoire parent contenant les dossiers NAME_* (défaut: répertoire courant).",
    )
    parser.add_argument(
        "--trajectory", default=None, metavar="PATH",
        help="Chemin vers la trajectoire de référence (requis pour le step 2).",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Répertoire de sortie (défaut: <model_name>_analysis).",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=ALL_STEPS, choices=ALL_STEPS,
        metavar="N",
        help=f"Steps à exécuter (défaut: tous = {ALL_STEPS}).",
    )
    parser.add_argument(
        "--summary-window", type=int, default=1000, metavar="N",
        help="Fenêtre (derniers N pas) pour le résumé RMSE (défaut: 1000).",
    )
    parser.add_argument(
        "--n-atoms", type=int, default=None, metavar="N",
        help="Nombre d'atomes par frame (inféré automatiquement si omis).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args   = parser.parse_args(argv)
    logger = _setup_logging(getattr(logging, args.log_level))

    base_dir   = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir if args.output_dir else f"{args.model_name}_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(set(args.steps))

    logger.info("Modèle       : %s", args.model_name)
    logger.info("Base dir     : %s", base_dir)
    logger.info("Output dir   : %s", output_dir)
    logger.info("Steps        : %s", steps)

    if 1 in steps:
        step1_training_curves(base_dir, args.model_name, output_dir,
                               args.summary_window, logger)

    if 2 in steps:
        if not args.trajectory:
            parser.error("--trajectory est requis pour le step 2.")
        traj = Path(args.trajectory).expanduser().resolve()
        step2_inference(base_dir, args.model_name, traj, output_dir, logger)

    if 3 in steps:
        step3_correlation_plots(args.model_name, output_dir, logger,
                                 n_atoms=args.n_atoms)

    logger.info("Terminé. Résultats dans %s", output_dir)


if __name__ == "__main__":
    main()
