# Workflow_DeepMD

Construction d'un potentiel DeePMD (H, W, O — H2 incident sur surface de
tungstène, propre ou oxydée) par apprentissage actif : partir de zéro (AIMD),
entraîner un premier modèle, puis l'améliorer en boucle en identifiant où il
se trompe le plus (QBC) et en labellisant ces cas avec du vrai DFT.

*[English version below](#english)*

---

## Vue d'ensemble — deux boucles

```
                    ┌─────────────────────────────────────────────┐
                    │  BOOTSTRAP (une fois, avant d'avoir un modèle) │
                    └─────────────────────────────────────────────┘

  AIMD_dynamic/          data_sampling/              deepmd_toolkit/
  (POSCAR H2+surface  →  (nettoie, choisit un    →   (dpdata_convert.py
   thermalisée, AIMD)     sous-ensemble divers,        → dp train → graph.pb)
                          train/val/test)
                                                              │
                                                              ▼
                    ┌─────────────────────────────────────────────┐
                    │  BOUCLE D'APPRENTISSAGE ACTIF (répéter)       │
                    └─────────────────────────────────────────────┘

  Relaxation/            Lammps_dynamic/            QBC_AL_DeepMD/
  (une fois : construit  (fait tourner plein de  →  (comité de graph_N.pb,
   + relaxe la surface    trajectoires H2/surface     repère les frames où
   propre en DFT)         avec le(s) modèle(s)         ils sont le plus en
        │                 actuel(s))                    désaccord = les plus
        └────────────────────────────────────────────►  informatives)
                                                              │
                                                              ▼
                                                Single_Points_Calculations/
                                                (VASP single-point sur ces
                                                 frames → vrai E + F)
                                                              │
                                                              ▼
                                              data_sampling/ (re-échantillonne
                                              avec les nouvelles données)
                                                              │
                                                              ▼
                                              deepmd_toolkit/ (réentraîne
                                              graph_N+1.pb) → retour à
                                              Lammps_dynamic/ avec le modèle
                                              amélioré
```

**Par où commencer** : `Lammps_dynamic/` a besoin d'un modèle DeePMD
(`graph.pb`) pour tourner — donc si tu n'en as aucun, tu ne peux pas encore
entrer dans la boucle d'apprentissage actif. Pars de `AIMD_dynamic/` (aucun
modèle requis, juste du DFT) pour produire tes toutes premières données
d'entraînement, entraîne un premier modèle via `deepmd_toolkit/`, puis
seulement à partir de là la boucle du bas devient utilisable.

## Dossiers, dans l'ordre du flux de travail

### `Relaxation/` — construire et relaxer la surface propre (une fois)
Trois scripts à la main, dans l'ordre : `buildLAMMPSPOSCAR.py` (construit la
dalle BCC W, format LAMMPS) → `lammps2poscar.py` (→ POSCAR VASP) →
`run_relaxation.slurm` (relaxation VASP) → `contcar2LaampsDatafile.py`
(CONTCAR relaxé → data file LAMMPS). Le résultat sert de `LAMMPS_POSCAR` à
`Lammps_dynamic/`.
```bash
./buildLAMMPSPOSCAR.py -a 3.2398 --surface 110 --nx 2 --ny 2 --nlayers 5 --vacuum 15 -o LAMMPS_POSCAR
```

### `AIMD_dynamic/` — AIMD H2/surface depuis des snapshots thermalisés
Pioche une surface thermalisée au hasard (positions + vitesses réelles) dans
un pool de `POSCAR-*`, y ajoute H2 (position, orientation, vitesse
incidente — tout configurable), écrit un POSCAR par structure, prêt pour
`setup_aimd_calculations.py` (prépare les dossiers de calcul VASP + job
arrays SLURM, copié/adapté de `Single_Points_Calculations/`).
```bash
./GenerateAIMDPoscars_ThermalSurface.py --poscar-dir <pool>/POSCAR -n 200 \
    --height 3.0 --hh-distance 0.7414 -e 0.3 -T 45 -P random -o aimd_poscars/
./setup_aimd_calculations.py -i aimd_poscars/ -o AIMD_calculations \
    --incar-aimd ./INCAR_AIMD --kpoints ./KPOINTS --potcar ./POTCAR_HOW \
    --vdw-kernel ./vdw_kernel.bindat --cores 4 --mem 12G
```
Le pool `POSCAR-*` thermalisé vient d'une extraction externe à partir d'un
AIMD initial (positions/vitesses réelles de la trajectoire, un fichier par
pas de temps échantillonné).

### `Lammps_dynamic/` — trajectoires H2/surface en masse, avec un modèle DeePMD
`massif_lammps.sh` soumet en masse des trajectoires LAMMPS/DeepMD (mode ZPE
ou NoZPE pour H2, surface figée à 0K ou thermalisée). Nécessite un
`graph.pb` déjà entraîné.
```bash
./massif_lammps.sh -e 0.1 -e 0.2 -t 20000 -j 10 model_A
# ou avec un fichier de config (--config, voir config.example.conf)
```

### `QBC_AL_DeepMD/` — sélection par désaccord de comité (Query By Committee)
Prend un comité de modèles (`models/{name}_1/graph.pb`, `_2`, ...), scanne
les trajectoires LAMMPS (`--pool`), et repère les frames où les modèles du
comité sont le plus en désaccord — les plus informatives à labelliser en
DFT.
```bash
python run_qbc.py --models-dir ./models --model-name HW2O \
    --pool "./md_pools/**/*.traj" --thresh-low 0.05 --thresh-high 0.50
```

### `Single_Points_Calculations/` — labelliser en DFT les frames sélectionnées
Prend les POSCAR sélectionnés par QBC (ou par n'importe quelle autre source)
et prépare un calcul VASP single-point par structure (`setup_sp_calculations.py`
— même moteur que `AIMD_dynamic/setup_aimd_calculations.py`).
```bash
python setup_sp_calculations.py -i <qbc_output>/poscars -o SP_out \
    --incar-sp ./INCAR --kpoints ./KPOINTS --potcar ./POTCAR_HOW \
    --vdw-kernel ./vdw_kernel.bindat
```

### `data_sampling/` — nettoyer puis choisir un sous-ensemble diversifié
`AIMD_process.py` nettoie les trajectoires AIMD brutes (contrôle qualité,
détection dissociation/rebond/piégeage, cache `.traj`/`.npz`).
`Pipeline_smart_sampling/` choisit ensuite un sous-ensemble représentatif
(SOAP → réduction de dimension → échantillonnage par couverture → analyse →
split train/val → jeu de test).
```bash
python AIMD_process.py -i path/to/AIMD/ -o path/to/output --all-groups
cd Pipeline_smart_sampling && python run_pipeline.py   # édite config.yaml d'abord
```

### `deepmd_toolkit/` — convertir, entraîner, analyser
`dpdata_convert.py` (`.traj` → format `deepmd/npy`) → `dp train` via
`train_modelDPMD.slurm` (produit `graph.pb`) → `dp_model_analysis.py`
(courbes d'apprentissage, comparaison DFT vs. modèle, graphiques de parité).
```bash
python dpdata_convert.py --input-dir selected --base-dir graph_1 --type-map H W O
sbatch train_modelDPMD.slurm graph_1
python dp_model_analysis.py --model-name graph --trajectory selected/test.traj
```

## Environnements

Chaque dossier avec un `environment.txt` a son propre venv/conda, nommé
d'après le dossier (`deepmd_toolkit`, `data_sampling`) :
```bash
python -m venv <nom_du_dossier> && source <nom_du_dossier>/bin/activate
pip install -r environment.txt
```
`Lammps_dynamic/` et `AIMD_dynamic/` ne dépendent que de numpy (+ scipy pour
le mode ZPE de LAMMPS) — pas de venv dédié nécessaire, juste vérifier que
`numpy` est disponible dans le Python utilisé.

## Ordre de lecture conseillé

Chaque dossier a son propre README plus détaillé (options, formats de
sortie, cas particuliers) — celui-ci n'est qu'une carte générale :
`Relaxation/README.md`, `Single_Points_Calculations/README.md`,
`Lammps_dynamic/README.md`, `data_sampling/README.md`,
`data_sampling/Pipeline_smart_sampling/README.md`, `deepmd_toolkit/README.md`.
(`AIMD_dynamic/` et `QBC_AL_DeepMD/` : voir la docstring en tête de chaque
script pour l'instant, pas encore de README dédié.)

---

<a name="english"></a>
## English

Building a DeePMD potential (H, W, O — H2 incident on a tungsten surface,
clean or oxidized) via active learning: bootstrap from scratch (AIMD), train
a first model, then iteratively improve it by finding where it's most wrong
(QBC) and labeling those cases with real DFT.

See the French section above for the full diagram and per-folder
walkthrough — same content, and each subfolder's own README (linked above)
is already in English/bilingual. Two loops:

1. **Bootstrap** (once, before any model exists): `AIMD_dynamic/` (pure DFT,
   no model needed) → `data_sampling/` → `deepmd_toolkit/` → first `graph.pb`.
2. **Active learning loop** (repeat): `Relaxation/` (once) →
   `Lammps_dynamic/` (run MD with the current model) → `QBC_AL_DeepMD/`
   (find the most disagreed-upon frames) → `Single_Points_Calculations/`
   (DFT-label them) → `data_sampling/` (re-sample) → `deepmd_toolkit/`
   (retrain) → back to `Lammps_dynamic/` with the improved model.

Start with `AIMD_dynamic/` if you have no model yet — `Lammps_dynamic/`
requires an existing `graph.pb` to run at all.
