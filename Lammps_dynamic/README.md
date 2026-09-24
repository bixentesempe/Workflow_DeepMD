# LAMMPS massif H2/surface — massif_lammps.sh

*[English version below](#english) — [Version française plus bas](#francais)*

---

<a name="english"></a>
## English

### What this is

`massif_lammps.sh` mass-submits LAMMPS/DeepMD molecular-dynamics trajectories
of an H2 molecule impinging on a surface, for several (model, kinetic
energy) pairs, grouped into SLURM job arrays (1 CPU each, several
trajectories run sequentially per array task).

Two sampling modes for the initial state of H2:

- **ZPE mode** (default): H2 keeps its true internal vibration/rotation,
  sampled at the requested (v, J) quantum state in an analytic Morse
  potential fitted automatically from a reference curve. No external,
  pre-generated file is needed per energy/angle.
- **NoZPE mode** (`--no-zpe`): H2 is a rigid rotor (fixed H-H bond length),
  only the center-of-mass translational velocity is imposed from
  (E, theta, phi).

Independently, two sources for the surface, chosen **explicitly** via
`--surface-mode` (never guessed from which path option happens to be set —
if the wrong one is set for the active mode, a warning is printed and it is
ignored):

- **`--surface-mode cold`** (default): a single `LAMMPS_POSCAR`
  (`-o/--path2poscar`), frozen at zero velocity (0K) for every trajectory.
- **`--surface-mode thermal`** (+ `--thermal-surface-dir DIR`): a different
  snapshot from a pool of AIMD-derived `POSCAR-*` files at a given
  temperature/coverage (e.g. produced by a separate extraction script from
  `vasprun.xml` trajectories) is drawn at random for *each* trajectory —
  real thermal positions and velocities for the mobile surface atoms,
  instead of a frozen 0-velocity slab. See
  `GenerateLammpsDatafile_ThermalSurface.py`'s docstring for the expected
  `POSCAR-*` format.

### Directory layout

Everything lives in one self-contained, relocatable folder — clone or copy
it anywhere, nothing is hardcoded to a fixed path (e.g. `~/bin`).

```
<SCRIPTS_DIR>                       (this folder — auto-detected)
├── massif_lammps.sh                orchestrator (entry point)
├── GenerateZPELammpsDatafile_ZPE.py    ZPE-mode generator
├── GenerateLammpsDatafile_NoZPE.py     NoZPE-mode generator (also imported
│                                        by the ZPE and ThermalSurface
│                                        generators for shared code)
├── GenerateLammpsDatafile_ThermalSurface.py  thermalized-surface generator
│                                        (--thermal-surface-dir; wraps the
│                                        NoZPE/ZPE H2-placement logic above)
├── lammps_worker.slurm             SLURM worker (submitted by sbatch)
├── in.simulation.default           default LAMMPS input template
├── config.example.conf              example --config file (see "Config
│                                     file" below), copy & edit
├── lammps2poscar.py                 standalone utility, NOT part of the
│                                     pipeline (see "Other utilities" below)
└── init_cond/                       ZPE-mode sampler (Morse potential)
    ├── exe                          compiled Fortran sampler
    ├── init_cond.f                  its source (kept for reference/rebuild)
    ├── intrep.dat                   small parameter file required by exe
    └── morse_potasym.dat            universal H2 (r_HH) Morse-fitted curve

<CASE_DIR>                          (your simulation case — anywhere)
├── LAMMPS_POSCAR                   surface data file you provide
│                                    (IDs >= 3, H1/H2 use IDs 1 and 2)
│                                    -- not needed with --thermal-surface-dir
└── in.simulation                   optional: overrides in.simulation.default

<CASE_DIR>/models/  (or -p elsewhere)
└── <model_name>/graph.pb           one DeepMD model per name
```

`SCRIPTS_DIR` is auto-detected as the folder containing
`massif_lammps.sh` itself (override with `--scripts-dir`). `CASE_DIR`
defaults to the current directory (override with `-d/--case-dir`). Nothing
else needs to be configured to relocate the whole toolkit.

### Requirements

- Python 3 with `numpy` and `scipy` (used by both generators; `scipy` is
  needed only in ZPE mode, for the Morse fit).
- A LAMMPS binary built with the DeepMD-kit plugin (`pair_style deepmd`),
  reachable from `lammps_worker.slurm` (adjust the `module load` lines
  there to your cluster — currently `module load use.own` and
  `module load deepmd/3.0.3-cpu`).
- SLURM (`sbatch`) with array-job support.
- `init_cond/exe` is a precompiled Fortran binary (gfortran, dynamically
  linked against `libgfortran`); rebuild from `init_cond/init_cond.f` if
  your cluster's glibc/gfortran differ too much to run the shipped binary.

### Quick start

```bash
cd /path/to/my_case          # contains LAMMPS_POSCAR and models/model_A/graph.pb
/path/to/SCRIPTS_DIR/massif_lammps.sh -e 0.1 -e 0.2 -t 20000 -j 10 model_A
```

This submits, for `model_A` at 0.1 eV and 0.2 eV, job arrays of 2000 tasks
each (20000 trajectories / 10 per task), with H2 in its v=0/J=0
zero-point-energy state (ZPE mode default), normal incidence.

See `./massif_lammps.sh --help` for the full option list.

### Config file

As the option count grows, a long one-off command line gets unwieldy and
hard to reproduce later. `-c/--config FILE` loads a plain bash file
(`KEY=VALUE`, arrays as `KEY=(a b c)`) that is `source`d directly — no new
dependency (no YAML/JSON parser), and the keys are **exactly** the script's
internal variable names, so there is no translation layer to maintain.
Copy `config.example.conf`, edit it, then:

```bash
massif_lammps.sh --config my_run.conf
```

Resolution order: defaults → config file (overrides defaults) → CLI flags
(override the config file). Any flag given explicitly on the command line
wins over the file, **including** `-e/--energie` and the positional model
names — both *replace* the file's list rather than appending to it, exactly
like any scalar option:

```bash
# my_run.conf has energies=(0.1 0.2) -> this runs only 0.5, not all three
massif_lammps.sh --config my_run.conf -e 0.5
```

### Execution flow

1. `massif_lammps.sh` resolves `SCRIPTS_DIR`, `CASE_DIR`, `LAMMPS_POSCAR`,
   and the `in.simulation` template (from `CASE_DIR` if present, otherwise
   the shipped default), validates all prerequisites.
2. For each (model, energy) pair:
   a. Creates a scratch `CALCDIR`, copies `graph.pb`, writes `in.simulation`
      (with the requested timestep/nsteps substituted).
   b. Runs the Python generator **inside** `CALCDIR`, which writes
      `data_POSCAR_1.lammps` … `data_POSCAR_N.lammps` and `groups.lmp`
      (frozen/mobile atom groups, detected from `# fixed` comments in
      `LAMMPS_POSCAR` — or from `Selective dynamics` in the thermalized
      pool's `POSCAR-*` — and/or `--frozen-ids`).
      - With `--thermal-surface-dir`, a random `POSCAR-*` is picked
        **per trajectory** and converted on the fly (positions + real
        AIMD velocities, Å/fs → Å/ps) instead of reusing the same
        0K `LAMMPS_POSCAR` for all of them.
      - In ZPE mode, the generator first fits the Morse potential on
        `init_cond/morse_potasym.dat`, computes the exact vibrational
        energy of the requested level `v` (closed-form Morse formula),
        then calls `init_cond/exe` once (translation forced to zero) to
        sample N independent internal (r_HH, orientation, internal
        velocity) states — decoupled from energy/angle, since the
        molecule's internal potential doesn't depend on them. Each state
        is then combined with a translational center-of-mass velocity
        drawn from (E, theta, phi).
   c. Submits a SLURM job array via `sbatch`, exporting `CALCDIR`,
      `FINALDIR`, `MODEL_DIR`, `INPUT_SCRIPT`, `GROUPS_FILE`, `NTRAJ`,
      `TRAJ_PER_JOB` to `lammps_worker.slurm`.
3. Each array task (`lammps_worker.slurm`) loops over its assigned
   trajectory range, running `lmp -in in.simulation -var DATAFILE ...`
   for each one, then moves the results (`log_*`, `dump_*.lammpstrj`,
   `energies_*.lammpstrj`, `HHcom_*.dat`) to `FINALDIR` (shared storage).

### Output layout

```
/scratch/$USER/massif_<timestamp>/            work area (deleted by you)
└── CALCDIR_<E>[_T..P..]_<model>/              data files, logs (transient)

/scratch/$USER/Lammps/<results_name>/          final results (-n/--name)
└── CALCDIR_<E>[_T..P..]_<model>/
    ├── log_<i>.lammps
    ├── dump_<i>.lammpstrj
    ├── energies_<i>.lammpstrj
    ├── HHcom_<i>.dat
    └── slurm-<jobid>_<taskid>.out
```

### Physics notes (ZPE mode)

- `init_cond/morse_potasym.dat` gives the H2 internal potential V(r_HH),
  **with r in bohr** (atomic units) — this is the convention the Fortran
  sampler expects; don't feed it a table in Ångström. Its equilibrium
  distance (≈1.400 bohr ≈ 0.741 Å) matches the real H2 bond length, a
  useful sanity check if you ever refit it.
- The Morse parameters (D, a, re) are re-fitted from this file on every
  run (fast, no caching needed) rather than hardcoded, so replacing
  `morse_potasym.dat` (e.g. with a curve for a different isotope or a
  refined fit) is enough to change the physics — nothing else to edit.
- `-v/--vib-level` must stay below the level printed as `v_max` in the
  generator's output (the highest bound state of the fitted well);
  requesting more raises a clear error instead of a cryptic Fortran crash.
- `-j/--jrot` (rotational quantum number) is independent of `v` and has no
  upper bound enforced by the script.

### Other utilities (not part of the pipeline)

- `lammps2poscar.py`: standalone converter, LAMMPS data file → VASP POSCAR,
  useful to prepare a DFT relaxation from a LAMMPS structure. Not called by
  `massif_lammps.sh`.

---

<a name="francais"></a>
## Français

### Qu'est-ce que c'est

`massif_lammps.sh` soumet en masse des trajectoires de dynamique
moléculaire LAMMPS/DeepMD d'une molécule H2 incidente sur une surface,
pour plusieurs couples (modèle, énergie cinétique), regroupées en job
arrays SLURM (1 CPU chacun, plusieurs trajectoires enchaînées par tâche).

Deux modes d'échantillonnage de l'état initial de H2 :

- **Mode ZPE** (par défaut) : H2 garde sa vraie vibration/rotation
  interne, échantillonnée dans l'état quantique (v, J) demandé via un
  potentiel de Morse analytique ajusté automatiquement sur une courbe de
  référence. Aucun fichier externe pré-généré n'est nécessaire par
  énergie/angle.
- **Mode NoZPE** (`--no-zpe`) : H2 est traité comme un rotor rigide
  (distance H-H fixe), seule la vitesse de translation du centre de masse
  est imposée à partir de (E, theta, phi).

Indépendamment, deux sources pour la surface, choisies **explicitement**
via `--surface-mode` (jamais devinées d'après ce qui est rempli — si la
mauvaise option est définie pour le mode actif, un avertissement s'affiche
et elle est ignorée) :

- **`--surface-mode cold`** (par défaut) : un seul `LAMMPS_POSCAR`
  (`-o/--path2poscar`), figé à vitesse nulle (0K) pour toutes les
  trajectoires.
- **`--surface-mode thermal`** (+ `--thermal-surface-dir DIR`) : un
  snapshot différent est pioché au hasard, **pour chaque trajectoire**,
  dans un pool de `POSCAR-*` issus d'un AIMD à une température/coverage
  donnés (ex. produits par un script d'extraction séparé à partir de
  trajectoires `vasprun.xml`) — positions et vitesses
  thermiques réelles pour les atomes de surface mobiles, plutôt qu'une
  dalle figée à vitesse nulle. Voir la docstring de
  `GenerateLammpsDatafile_ThermalSurface.py` pour le format `POSCAR-*`
  attendu.

### Organisation des dossiers

Tout tient dans un seul dossier autonome et déplaçable — clone-le ou
copie-le n'importe où, rien n'est câblé en dur vers un chemin fixe
(comme `~/bin`).

```
<SCRIPTS_DIR>                       (ce dossier — auto-détecté)
├── massif_lammps.sh                orchestrateur (point d'entrée)
├── GenerateZPELammpsDatafile_ZPE.py    générateur mode ZPE
├── GenerateLammpsDatafile_NoZPE.py     générateur mode NoZPE (aussi importé
│                                        par les générateurs ZPE et
│                                        ThermalSurface, code partagé)
├── GenerateLammpsDatafile_ThermalSurface.py  générateur surface thermalisee
│                                        (--thermal-surface-dir ; reutilise
│                                        le placement H2 NoZPE/ZPE ci-dessus)
├── lammps_worker.slurm             worker SLURM (soumis par sbatch)
├── in.simulation.default           template LAMMPS par défaut
├── config.example.conf              exemple de fichier --config (voir
│                                     "Fichier de config" plus bas), a copier
├── lammps2poscar.py                 utilitaire autonome, PAS dans la
│                                     chaîne (voir "Autres utilitaires")
└── init_cond/                       sampler du mode ZPE (potentiel de Morse)
    ├── exe                          binaire Fortran compilé
    ├── init_cond.f                  son code source (gardé pour référence/recompilation)
    ├── intrep.dat                   petit fichier de paramètres requis par exe
    └── morse_potasym.dat            courbe de Morse universelle pour H2 (r_HH)

<CASE_DIR>                          (ton cas de simulation — n'importe où)
├── LAMMPS_POSCAR                   data file de la surface, fourni par toi
│                                    (IDs >= 3, H1/H2 utilisent les IDs 1 et 2)
│                                    -- pas necessaire avec --thermal-surface-dir
└── in.simulation                   optionnel : remplace in.simulation.default

<CASE_DIR>/models/  (ou -p ailleurs)
└── <nom_modele>/graph.pb           un modèle DeepMD par nom
```

`SCRIPTS_DIR` est auto-détecté comme le dossier contenant
`massif_lammps.sh` lui-même (surchargeable avec `--scripts-dir`).
`CASE_DIR` vaut par défaut le dossier courant (surchargeable avec
`-d/--case-dir`). Rien d'autre à configurer pour déplacer tout l'outillage.

### Prérequis

- Python 3 avec `numpy` et `scipy` (utilisés par les deux générateurs ;
  `scipy` n'est nécessaire qu'en mode ZPE, pour le fit Morse).
- Un binaire LAMMPS compilé avec le plugin DeepMD-kit (`pair_style deepmd`),
  accessible depuis `lammps_worker.slurm` (adapte les lignes `module load`
  à ton cluster — actuellement `module load use.own` et
  `module load deepmd/3.0.3-cpu`).
- SLURM (`sbatch`) avec support des job arrays.
- `init_cond/exe` est un binaire Fortran précompilé (gfortran, lié
  dynamiquement à `libgfortran`) ; recompile depuis
  `init_cond/init_cond.f` si le glibc/gfortran de ton cluster diffère trop
  pour exécuter le binaire fourni.

### Démarrage rapide

```bash
cd /chemin/vers/mon_cas      # contient LAMMPS_POSCAR et models/model_A/graph.pb
/chemin/vers/SCRIPTS_DIR/massif_lammps.sh -e 0.1 -e 0.2 -t 20000 -j 10 model_A
```

Ça soumet, pour `model_A` à 0.1 eV et 0.2 eV, des job arrays de 2000
tâches chacun (20000 trajectoires / 10 par tâche), avec H2 dans son état
fondamental v=0/J=0 (ZPE, mode par défaut), incidence normale.

Voir `./massif_lammps.sh --help` pour la liste complète des options.

### Fichier de config

Le nombre d'options augmentant, une longue commande tapée à la main devient
vite difficile à relire et à reproduire plus tard. `-c/--config FICHIER`
charge un simple fichier bash (`CLE=VALEUR`, tableaux en `CLE=(a b c)`)
directement `source`-é — aucune nouvelle dépendance (pas de parseur
YAML/JSON), et les clés sont **exactement** les noms de variables internes
du script, donc aucune couche de traduction à maintenir. Copie
`config.example.conf`, adapte-le, puis :

```bash
massif_lammps.sh --config mon_run.conf
```

Ordre de résolution : valeurs par défaut → fichier de config (écrase les
défauts) → flags CLI (écrasent le fichier). Un flag donné explicitement sur
la ligne de commande gagne toujours sur le fichier, **y compris**
`-e/--energie` et les modèles positionnels — les deux *remplacent* la liste
du fichier plutôt que de s'y ajouter, exactement comme n'importe quelle
option scalaire :

```bash
# mon_run.conf contient energies=(0.1 0.2) -> ne lance que 0.5, pas les trois
massif_lammps.sh --config mon_run.conf -e 0.5
```

### Déroulement de l'exécution

1. `massif_lammps.sh` résout `SCRIPTS_DIR`, `CASE_DIR`, `LAMMPS_POSCAR`,
   et le template `in.simulation` (celui de `CASE_DIR` s'il existe, sinon
   le défaut fourni), puis vérifie tous les prérequis.
2. Pour chaque couple (modèle, énergie) :
   a. Crée un `CALCDIR` sur le scratch, copie `graph.pb`, écrit
      `in.simulation` (avec le pas de temps/nombre de pas substitués).
   b. Lance le générateur Python **dans** `CALCDIR`, qui écrit
      `data_POSCAR_1.lammps` … `data_POSCAR_N.lammps` et `groups.lmp`
      (groupes d'atomes figés/mobiles, détectés via les commentaires
      `# fixed` du `LAMMPS_POSCAR` — ou via `Selective dynamics` dans le
      pool thermalisé de `POSCAR-*` — et/ou `--frozen-ids`).
      - Avec `--thermal-surface-dir`, un `POSCAR-*` est pioché au hasard
        **par trajectoire** et converti à la volée (positions + vraies
        vitesses AIMD, Å/fs → Å/ps) plutôt que de réutiliser le même
        `LAMMPS_POSCAR` figé à 0K pour toutes.
      - En mode ZPE, le générateur ajuste d'abord le potentiel de Morse
        sur `init_cond/morse_potasym.dat`, calcule l'énergie exacte du
        niveau vibrationnel `v` demandé (formule de Morse fermée), puis
        appelle une fois `init_cond/exe` (translation forcée à zéro) pour
        échantillonner N états internes indépendants (r_HH, orientation,
        vitesse interne) — découplés de l'énergie/angle, puisque le
        potentiel interne de la molécule n'en dépend pas. Chaque état est
        ensuite combiné avec une vitesse de translation du centre de
        masse tirée de (E, theta, phi).
   c. Soumet un job array SLURM via `sbatch`, en exportant `CALCDIR`,
      `FINALDIR`, `MODEL_DIR`, `INPUT_SCRIPT`, `GROUPS_FILE`, `NTRAJ`,
      `TRAJ_PER_JOB` vers `lammps_worker.slurm`.
3. Chaque tâche du job array (`lammps_worker.slurm`) boucle sur sa plage
   de trajectoires assignée, lance `lmp -in in.simulation -var DATAFILE ...`
   pour chacune, puis déplace les résultats (`log_*`, `dump_*.lammpstrj`,
   `energies_*.lammpstrj`, `HHcom_*.dat`) vers `FINALDIR` (stockage partagé).

### Organisation des résultats

```
/scratch/$USER/massif_<horodatage>/            zone de travail (a toi de nettoyer)
└── CALCDIR_<E>[_T..P..]_<modele>/              data files, logs (transitoire)

/scratch/$USER/Lammps/<results_name>/          resultats finaux (-n/--name)
└── CALCDIR_<E>[_T..P..]_<modele>/
    ├── log_<i>.lammps
    ├── dump_<i>.lammpstrj
    ├── energies_<i>.lammpstrj
    ├── HHcom_<i>.dat
    └── slurm-<jobid>_<taskid>.out
```

### Notes de physique (mode ZPE)

- `init_cond/morse_potasym.dat` donne le potentiel interne de H2 V(r_HH),
  **avec r en bohr** (unités atomiques) — c'est la convention attendue par
  le sampler Fortran ; ne lui donne pas une table en Ångström. Sa distance
  d'équilibre (≈1.400 bohr ≈ 0.741 Å) correspond à la vraie longueur de
  liaison H2, un bon test de cohérence si tu le refits un jour.
- Les paramètres de Morse (D, a, re) sont réajustés depuis ce fichier à
  chaque exécution (rapide, pas besoin de cache) plutôt que codés en dur :
  remplacer `morse_potasym.dat` (par une courbe pour un autre isotope ou
  un fit affiné) suffit à changer la physique, rien d'autre à modifier.
- `-v/--vib-level` doit rester sous le niveau affiché comme `v_max` dans
  la sortie du générateur (le plus haut état lié du puits ajusté) ;
  demander plus déclenche une erreur claire plutôt qu'un plantage Fortran
  cryptique.
- `-j/--jrot` (nombre quantique rotationnel) est indépendant de `v` et
  n'a pas de borne supérieure imposée par le script.

### Autres utilitaires (hors chaîne)

- `lammps2poscar.py` : convertisseur autonome, data file LAMMPS → POSCAR
  VASP, utile pour préparer une relaxation DFT à partir d'une structure
  LAMMPS. N'est pas appelé par `massif_lammps.sh`.
