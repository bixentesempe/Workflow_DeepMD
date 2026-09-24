# Single-point VASP calculations from QBC-selected structures

*[English version below](#english) — [Version française plus bas](#francais)*

---

<a name="english"></a>
## English

### What this is

Takes the structures picked by [`QBC_AL_DeepMD`](../QBC_AL_DeepMD/) (frames
from your LAMMPS MD trajectories that the DeePMD committee disagrees on
most) and turns them into VASP single-point (SP) calculations: one
POSCAR in, one energy+forces label out. `setup_sp_calculations.py` does
the file plumbing and SLURM script generation; you never hand-build a
calc directory.

```
QBC_AL_DeepMD/run_qbc.py            setup_sp_calculations.py              submit_all.sh            deepmd_toolkit/dpdata_convert.py
  (select disagreement    --->  (one VASP SP dir per POSCAR    --->   (sbatch job arrays   --->    (vasprun.xml -> labeled
   frames from MD pool)          + run_array.slurm + dirlist)          on the cluster)                DeePMD training set)
  <qbc_out>/poscars/*/POSCAR      SP_out/<name>/{POSCAR,INCAR,          SP_out/<name>/vasprun.xml
                                   KPOINTS,POTCAR,vdw_kernel.bindat}
```

Nothing here launches VASP by itself — you run each step by hand, in
order.

### Step 1 — select structures with QBC

Covered in [`../QBC_AL_DeepMD/`](../QBC_AL_DeepMD/). The output you need
is `<qbc_output_dir>/poscars/`, one subfolder per selected frame, each
containing a single `POSCAR`.

### Step 2 — VASP inputs: what's already here, what you choose

| File | Status | What it is |
|---|---|---|
| `INCAR` | ready to use | Single-point settings: `IBRION=-1`, `NSW=0` (no ionic step), `ISTART=0`/`ICHARG=2` (fresh SCF, no prior WAVECAR), rPBE-vdW (`GGA=RE`, `LUSE_VDW=.TRUE.`) to match the AIMD runs it's sampled from, `LWAVE=.FALSE.`/`LCHARG=.FALSE.` (only `vasprun.xml` is needed to recover E + F). Adjust `NPAR` to your cluster. |
| `KPOINTS` | example | Monkhorst-Pack 5×5×1 for a W(110) slab — **not a generic default**, re-derive it for your own cell size/symmetry. |
| `POTCAR_HW` | ready, pick one | Concatenated PAW-PBE potentials for **H, W** — pure metal + H projectile, no oxygen. |
| `POTCAR_HOW` | ready, pick one | Concatenated PAW-PBE potentials for **H, O, W** — system with adsorbed O "impurities". |
| `vdw_kernel.bindat` | ready to use | vdW-DF kernel table required because `INCAR` sets `LUSE_VDW=.TRUE.`. |

**The POTCAR species order must match the POSCAR species order** — VASP
does not check this for you, a mismatch silently assigns the wrong
pseudopotential to the wrong species. Copy whichever variant matches
your system's POSCAR:

```bash
cp POTCAR_HW  POTCAR    # H + W system
# or
cp POTCAR_HOW POTCAR    # H + O + W system
```

(You can also skip the copy and point `--potcar` directly at
`POTCAR_HW`/`POTCAR_HOW` in step 3.)

### Step 3 — generate the calculation directories

```bash
# Dry-run first: counts calcs, checks inputs exist, shows disk savings —
# creates nothing.
python setup_sp_calculations.py \
    -i ../QBC_AL_DeepMD/HW2O-QBC1/poscars \
    -o ./SP_HW2O_AL1 \
    --potcar ./POTCAR_HOW \
    --dry-run

# Then for real
python setup_sp_calculations.py \
    -i ../QBC_AL_DeepMD/HW2O-QBC1/poscars \
    -o ./SP_HW2O_AL1 \
    --potcar ./POTCAR_HOW
```

`--incar-sp`, `--kpoints` and `--vdw-kernel` default to the files in this
folder — pass them explicitly only if you're using different ones.

Useful flags:
- `--calcs-per-task N` — group N calculations per SLURM array task,
  executed in sequence. Needed once you have thousands of POSCARs: one
  task per calc would blow past the cluster's pending-job limit (see the
  table below).
- `--unique-names` — name calc dirs from the full relative POSCAR path
  instead of just the parent folder name. Use this if you're feeding in
  POSCARs from **more than one QBC run** — different runs can reuse the
  same `rank00000...` folder name, and without this flag the second run
  silently collides with (and the script refuses to overwrite) the
  first.
- `--copy-inputs` — copy `POTCAR`/`vdw_kernel.bindat` into every calc dir
  instead of symlinking them (default: symlink, ~9 MB saved per calc).
- `--exclude PATTERN` — skip paths matching a glob pattern (repeatable).
- `--time`, `--mem`, `--vasp-module` — SLURM resources / module for the
  generated `run_array.slurm`.

Re-running the same command is safe: already-prepared calc dirs
(recognized by having both `INCAR` and `POTCAR`) are skipped, so a
second POSCAR batch on the same `-o` only adds new calculations.

### Step 4 — launch

```bash
bash ./SP_HW2O_AL1/submit_all.sh
```

This generates and runs one or more `sbatch --array=...` calls, staying
under the cluster's `MaxArraySize` by splitting into several submissions
if needed (`OFFSET` tells each array which chunk range it owns).

Two SLURM limits to know about:

| Limit | What it caps | Check with | Worked around by |
|---|---|---|---|
| `MaxArraySize` | size of *one* `--array` | `scontrol show config \| grep MaxArraySize` | multiple `sbatch` calls with different `OFFSET` (handled automatically) |
| pending jobs / user | total queued jobs at once | `sacctmgr show assoc user=$USER format=user,maxsubmit` | `--calcs-per-task N` in step 3 (fewer, fatter tasks) |

The script warns at generation time if your calc count would exceed the
pending-job limit and suggests a `--calcs-per-task` value.

### What you get back

`SP_HW2O_AL1/<calc_name>/` contains `vasprun.xml` (energy + forces) and
`vasp.out`. Feed these into
[`../deepmd_toolkit/dpdata_convert.py`](../deepmd_toolkit/) to build the
next DeePMD training/validation set.

### Notes / pitfalls this README exists to spare you

- **POTCAR/POSCAR species order.** `POTCAR_HW` = H,W ; `POTCAR_HOW` =
  H,O,W. Pick the one matching your POSCAR's atom-type order — VASP
  won't error on a mismatch, it'll just mislabel every atom's
  pseudopotential.
- **Symlinked shared inputs.** By default `POTCAR` and
  `vdw_kernel.bindat` are relative symlinks into `<output_dir>/inputs/`
  (a frozen copy made once at generation time), not per-calc copies —
  ~9 MB saved per calculation. Editing the *source* file you passed to
  `--potcar` after generation does **not** retroactively change already
  generated calcs, since the frozen copy is what's linked.
- **Two QBC runs, same output dir.** If you're accumulating SP
  calculations from successive QBC/active-learning iterations into one
  `-o`, use `--unique-names` — different QBC runs can produce POSCARs
  with the same parent folder name (e.g. `rank00000_...`), which
  otherwise collide.
- **`#SBATCH -D <output_dir>`** in the generated `run_array.slurm` means
  logs always land in `<output_dir>/logs/`, regardless of where you run
  `submit_all.sh` from.

---

<a name="francais"></a>
## Français

### Qu'est-ce que c'est

Prend les structures sélectionnées par
[`QBC_AL_DeepMD`](../QBC_AL_DeepMD/) (les frames de tes trajectoires MD
LAMMPS sur lesquelles le comité DeePMD est le plus en désaccord) et les
transforme en calculs VASP single-point (SP) : un POSCAR en entrée, un
label énergie+forces en sortie. `setup_sp_calculations.py` s'occupe de
la plomberie fichiers et de la génération des scripts SLURM — tu ne
construis jamais un dossier de calcul à la main.

```
QBC_AL_DeepMD/run_qbc.py            setup_sp_calculations.py              submit_all.sh            deepmd_toolkit/dpdata_convert.py
 (sélectionne les frames   --->  (un dossier VASP SP par POSCAR   --->  (job arrays sbatch  --->    (vasprun.xml -> jeu
  en désaccord dans le MD)         + run_array.slurm + dirlist)          sur le cluster)               d'entraînement DeePMD)
  <qbc_out>/poscars/*/POSCAR       SP_out/<nom>/{POSCAR,INCAR,           SP_out/<nom>/vasprun.xml
                                    KPOINTS,POTCAR,vdw_kernel.bindat}
```

Rien ici ne lance VASP tout seul — tu exécutes chaque étape à la main,
dans l'ordre.

### Étape 1 — sélectionner les structures avec QBC

Couvert dans [`../QBC_AL_DeepMD/`](../QBC_AL_DeepMD/). Ce dont tu as
besoin en sortie : `<qbc_output_dir>/poscars/`, un sous-dossier par
frame sélectionnée, chacun contenant un `POSCAR`.

### Étape 2 — entrées VASP : ce qui est déjà là, ce que tu choisis

| Fichier | Statut | Ce que c'est |
|---|---|---|
| `INCAR` | prêt à l'emploi | Réglages single-point : `IBRION=-1`, `NSW=0` (pas d'étape ionique), `ISTART=0`/`ICHARG=2` (SCF neuf, pas de WAVECAR préalable), rPBE-vdW (`GGA=RE`, `LUSE_VDW=.TRUE.`) pour rester cohérent avec les AIMD dont ces structures sont issues, `LWAVE=.FALSE.`/`LCHARG=.FALSE.` (seul `vasprun.xml` est nécessaire pour récupérer E + F). Ajuste `NPAR` à ton cluster. |
| `KPOINTS` | exemple | Monkhorst-Pack 5×5×1 pour une dalle W(110) — **pas un défaut générique**, à redériver pour ta propre taille/symétrie de maille. |
| `POTCAR_HW` | prêt, à choisir | Potentiels PAW-PBE concaténés pour **H, W** — métal pur + H projectile, sans oxygène. |
| `POTCAR_HOW` | prêt, à choisir | Potentiels PAW-PBE concaténés pour **H, O, W** — système avec "impuretés" O adsorbées. |
| `vdw_kernel.bindat` | prêt à l'emploi | Table du noyau vdW-DF requise car `INCAR` active `LUSE_VDW=.TRUE.`. |

**L'ordre des espèces dans le POTCAR doit correspondre à celui du
POSCAR** — VASP ne vérifie pas ça pour toi, une incohérence assigne
silencieusement le mauvais pseudopotentiel à la mauvaise espèce. Copie
la variante qui correspond au POSCAR de ton système :

```bash
cp POTCAR_HW  POTCAR    # système H + W
# ou
cp POTCAR_HOW POTCAR    # système H + O + W
```

(Tu peux aussi sauter la copie et pointer `--potcar` directement vers
`POTCAR_HW`/`POTCAR_HOW` à l'étape 3.)

### Étape 3 — générer les dossiers de calcul

```bash
# Dry-run d'abord : compte les calculs, vérifie que les entrées existent,
# affiche l'espace disque économisé — ne crée rien.
python setup_sp_calculations.py \
    -i ../QBC_AL_DeepMD/HW2O-QBC1/poscars \
    -o ./SP_HW2O_AL1 \
    --potcar ./POTCAR_HOW \
    --dry-run

# Puis pour de vrai
python setup_sp_calculations.py \
    -i ../QBC_AL_DeepMD/HW2O-QBC1/poscars \
    -o ./SP_HW2O_AL1 \
    --potcar ./POTCAR_HOW
```

`--incar-sp`, `--kpoints` et `--vdw-kernel` pointent par défaut vers les
fichiers de ce dossier — ne les passe explicitement que si tu utilises
d'autres fichiers.

Options utiles :
- `--calcs-per-task N` — regroupe N calculs par tâche d'array SLURM,
  exécutés en séquence. Nécessaire dès que tu as des milliers de
  POSCARs : une tâche par calcul dépasserait vite la limite de jobs en
  attente du cluster (voir tableau ci-dessous).
- `--unique-names` — nomme les dossiers de calcul d'après le chemin
  relatif complet du POSCAR plutôt que le seul nom du dossier parent. À
  utiliser si tu combines des POSCARs de **plusieurs runs QBC** —
  différents runs peuvent réutiliser le même nom de dossier
  `rank00000...`, et sans cette option le second run entre en collision
  avec le premier (le script refuse alors d'écraser).
- `--copy-inputs` — copie `POTCAR`/`vdw_kernel.bindat` dans chaque
  dossier de calcul au lieu de les lier (défaut : lien symbolique, ~9 Mo
  économisés par calcul).
- `--exclude MOTIF` — ignore les chemins correspondant à un motif glob
  (répétable).
- `--time`, `--mem`, `--vasp-module` — ressources SLURM / module pour le
  `run_array.slurm` généré.

Relancer la même commande est sans risque : les dossiers de calcul déjà
préparés (reconnus par la présence d'`INCAR` **et** `POTCAR`) sont
ignorés, donc un second lot de POSCARs sur le même `-o` ne fait
qu'ajouter les nouveaux calculs.

### Étape 4 — lancer

```bash
bash ./SP_HW2O_AL1/submit_all.sh
```

Ceci génère et exécute un ou plusieurs appels `sbatch --array=...`, en
restant sous le `MaxArraySize` du cluster en découpant en plusieurs
soumissions si besoin (`OFFSET` indique à chaque array la plage de
chunks dont il est responsable).

Deux limites SLURM à connaître :

| Limite | Ce qu'elle plafonne | Vérifier avec | Contournée par |
|---|---|---|---|
| `MaxArraySize` | taille d'*un seul* `--array` | `scontrol show config \| grep MaxArraySize` | plusieurs appels `sbatch` avec un `OFFSET` différent (géré automatiquement) |
| jobs en attente / utilisateur | total de jobs en queue à la fois | `sacctmgr show assoc user=$USER format=user,maxsubmit` | `--calcs-per-task N` à l'étape 3 (moins de tâches, plus grosses) |

Le script avertit à la génération si ton nombre de calculs dépasserait
la limite de jobs en attente, et suggère une valeur de
`--calcs-per-task`.

### Ce que tu récupères

`SP_HW2O_AL1/<nom_calcul>/` contient `vasprun.xml` (énergie + forces) et
`vasp.out`. Utilise ces fichiers avec
[`../deepmd_toolkit/dpdata_convert.py`](../deepmd_toolkit/) pour
construire le prochain jeu d'entraînement/validation DeePMD.

### Notes / pièges que ce README t'évite

- **Ordre des espèces POTCAR/POSCAR.** `POTCAR_HW` = H,W ; `POTCAR_HOW` =
  H,O,W. Choisis celui qui correspond à l'ordre des types d'atomes de
  ton POSCAR — VASP ne signale aucune erreur en cas d'incohérence, il
  attribue juste le mauvais pseudopotentiel à chaque atome.
- **Entrées partagées liées par symlink.** Par défaut, `POTCAR` et
  `vdw_kernel.bindat` sont des liens symboliques relatifs vers
  `<output_dir>/inputs/` (une copie figée une seule fois à la
  génération), pas des copies par calcul — ~9 Mo économisés par calcul.
  Éditer le fichier *source* passé à `--potcar` après la génération ne
  change **pas** rétroactivement les calculs déjà générés, puisque c'est
  la copie figée qui est liée.
- **Deux runs QBC, même dossier de sortie.** Si tu accumules des calculs
  SP de plusieurs itérations QBC/active-learning successives dans un
  même `-o`, utilise `--unique-names` — différents runs QBC peuvent
  produire des POSCARs avec le même nom de dossier parent (ex.
  `rank00000_...`), ce qui entre sinon en collision.
- **`#SBATCH -D <output_dir>`** dans le `run_array.slurm` généré fait que
  les logs atterrissent toujours dans `<output_dir>/logs/`, peu importe
  d'où tu lances `submit_all.sh`.
