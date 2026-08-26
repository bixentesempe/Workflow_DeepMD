from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
#                         FRAMES
# =====================================================

# Dossier où chercher les .traj (défaut : dossier courant)
TRAJ_DIR = "."
TRAJ_FILES = sorted(Path(TRAJ_DIR).glob("*.traj"))

# =====================================================
#                  CHARGEMENT ET FUSION
# =====================================================

traj = []
if not TRAJ_FILES:
    raise FileNotFoundError(f"Aucun fichier *.traj trouvé dans '{TRAJ_DIR}'")
for f in TRAJ_FILES:
    frames = read(f, index=':')
    if isinstance(frames, list):
        traj.extend(frames)
    else:
        traj.append(frames)
    print(f"  {f} : {len(frames)} frames")

print(f"  Total fusionné : {len(traj)} frames")

# =====================================================
#                  INITIALISATION
# =====================================================

zsurf = 9.0892524

energies  = np.array([a.get_potential_energy() for a in traj])

# Distance H-H avec conditions périodiques (minimum image convention)
distances = np.array([a.get_distance(0, 1, mic=True) for a in traj])

# Positions z avec repliement dans la maille (wrap=True gère les PBC)
# Si un atome H est sorti de la maille par le haut/bas, wrap le ramène dedans.
def _z_above_surface(atoms, idx, zsurf):
    pos  = atoms.get_positions(wrap=True)
    z    = pos[idx, 2] - zsurf
    # Si z est négatif après wrap (atome sous la surface dans la maille),
    # on le ramène du bon côté en ajoutant la hauteur de cellule.
    if z < -zsurf:
        z += atoms.cell[2, 2]
    return z

z_H1   = np.array([_z_above_surface(a, 0, zsurf) for a in traj])
z_H2   = np.array([_z_above_surface(a, 1, zsurf) for a in traj])
z_mean = (z_H1 + z_H2) / 2

# =====================================================
# Histogrammes
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(energies, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Énergie (eV)')
axes[0].set_ylabel('Nombre de frames')
axes[0].set_title('Distribution des énergies')
axes[0].grid(True, alpha=0.3)

axes[1].hist(distances, bins=60, color='tomato', edgecolor='white', linewidth=0.4)
axes[1].set_xlabel('$d_{HH}$ (Å)')
axes[1].set_ylabel('Nombre de frames')
axes[1].set_title('Distribution des distances H-H')
axes[1].grid(True, alpha=0.3)

axes[2].hist(z_mean, bins=60, color='mediumseagreen', edgecolor='white', linewidth=0.4)
axes[2].set_xlabel('z moyen H$_2$ (Å)')
axes[2].set_ylabel('Nombre de frames')
axes[2].set_title('Distribution de la hauteur z')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogrammes.png', dpi=200, bbox_inches='tight')
plt.close()

# =====================================================
# Scatter : R_HH vs z, coloré par énergie
# =====================================================
fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(distances, z_mean, c=energies, cmap='plasma',
                s=4, alpha=0.7, rasterized=True)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Énergie (eV)', fontsize=12)

ax.set_xlabel('$d_{HH}$ (Å)', fontsize=13)
ax.set_ylabel('z moyen H$_2$ (Å)', fontsize=13)
ax.set_title('Espace de configuration H$_2$ / surface', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_rHH_z_energie.png', dpi=200, bbox_inches='tight')
plt.close()

# =====================================================
# Comparaison zones de configuration
# =====================================================

THRESHOLD = 3.0  # Angström

zone_proche  = np.sum(z_mean <= THRESHOLD)
zone_loin    = np.sum(z_mean >  THRESHOLD)
total_frames = len(traj)

print(f"\n{'='*45}")
print(f"  Répartition des frames par zone z :")
print(f"{'='*45}")
print(f"  z ≤ {THRESHOLD} Å (proche surface) : {zone_proche:6d} frames "
      f"({100*zone_proche/total_frames:.1f}%)")
print(f"  z >  {THRESHOLD} Å (loin surface)  : {zone_loin:6d} frames "
      f"({100*zone_loin/total_frames:.1f}%)")
print(f"  Total                       : {total_frames:6d} frames")
print(f"{'='*45}")

# Visualisation barplot
fig, ax = plt.subplots(figsize=(6, 5))
zones  = [f'z ≤ {THRESHOLD} Å\n(proche surface)',
          f'z > {THRESHOLD} Å\n(loin surface)']
counts = [zone_proche, zone_loin]
colors = ['tomato', 'steelblue']

bars = ax.bar(zones, counts, color=colors, edgecolor='white',
              linewidth=0.8, width=0.5)

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + total_frames*0.01,
            f'{count}\n({100*count/total_frames:.1f}%)',
            ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Nombre de frames', fontsize=12)
ax.set_title('Répartition des frames par zone de configuration', fontsize=12)
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('zones_configuration.png', dpi=200, bbox_inches='tight')
plt.close()

print("✅ Plots sauvegardés.")
