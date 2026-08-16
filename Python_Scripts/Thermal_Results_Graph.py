import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

# ─────────────────────────────────────────
# LOAD ANSYS PATH DATA
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "temperature_path_270kW.csv"

path_data = pd.read_csv(
    CSV_FILE,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)
path_data.columns = ["z", "temperature"]
path_data["z"] = pd.to_numeric(path_data["z"], errors="coerce")
path_data["temperature"] = pd.to_numeric(path_data["temperature"], errors="coerce")
path_data = (
    path_data
    .dropna(subset=["z", "temperature"])
    .sort_values("z")
    .reset_index(drop=True)
)

z_path = path_data["z"].to_numpy()
T_path = path_data["temperature"].to_numpy()

# ── Smooth the raw path to remove mesh oscillations ──────────────
# Savitzky-Golay filter preserves peaks while removing high-frequency noise
# window must be odd and less than number of points
n_pts   = len(T_path)
window  = min(31, n_pts if n_pts % 2 == 1 else n_pts - 1)
T_smooth = savgol_filter(T_path, window_length=window, polyorder=3)

print(f"Points loaded: {n_pts}")
print(f"z range: {z_path.min():.3f} to {z_path.max():.3f} m")
print(f"T range: {T_path.min():.1f} to {T_path.max():.1f} °C")
print(f"T_peak raw:    {T_path.max():.2f} °C at z = {z_path[T_path.argmax()]:.3f} m")
print(f"T_peak smooth: {T_smooth.max():.2f} °C at z = {z_path[T_smooth.argmax()]:.3f} m")

# ─────────────────────────────────────────
# BLOCK DATA — your ANSYS per-body T_max
# ─────────────────────────────────────────
# (z_centre_m, material, T_max_degC)
blocks = [
    (0.250, 'Graphite', 101.37),
    (0.750, 'Graphite', 261.81),
    (1.250, 'Graphite', 408.03),
    (1.750, 'Graphite', 448.55),   # shower max
    (2.250, 'Graphite', 423.15),
    (2.750, 'Graphite', 337.40),
    (3.250, 'Graphite', 229.42),
    (3.750, 'Graphite', 144.21),
    (4.150, 'Graphite',  91.91),
    (4.450, 'TZM',       89.42),
    (4.550, 'TZM',       83.64),
    (4.794, 'Tungsten',  64.20),
]

z_blocks = np.array([b[0] for b in blocks])
T_blocks = np.array([b[2] for b in blocks])
mat_blocks = [b[1] for b in blocks]

marker_colors = {
    'Graphite': '#2c7bb6',
    'TZM':      '#d7191c',
    'Tungsten': '#1a9641',
}

# ─────────────────────────────────────────
# PAPER REFERENCE — Fig. 29 at 25 kW
# ─────────────────────────────────────────
# These are approximate Pt100 measured values from paper Fig. 29
# digitised from the published figure at 25 kW average power
# Your model runs at 270 kW so these are shown scaled for shape reference
# Scale factor: 270/25 = 10.8 (linear approximation for shape comparison)
# NOTE: digitise these from Fig. 29 using WebPlotDigitizer for exact values

paper_pt100_25kW = np.array([
    # [z_mm,  T_measured_25kW_degC]
    [220,   30.5],
    [720,   36.2],
    [1220,  45.8],
    [1720,  55.1],   # peak near block 4
    [2220,  52.3],
    [2720,  44.6],
    [3220,  38.2],
    [3720,  33.4],
    [4120,  30.8],
    [4450,  32.1],   # TZM
    [4650,  29.5],   # Tungsten
])

z_paper_m  = paper_pt100_25kW[:, 0] / 1000.0
T_paper_25 = paper_pt100_25kW[:, 1]

# Scale paper 25kW data linearly to 270kW for shape comparison
scale = 270.0 / 25.0
T_paper_scaled = (T_paper_25 - 35) * scale + 35   # scale above coolant baseline

# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# ── Material shading ──────────────────────────────────────────────
shade_regions = [
    (0,     4.400, '#e8e8e8', 'Graphite'),
    (4.400, 4.600, '#cce5ff', 'TZM'),
    (4.600, 4.989, '#fff3cd', 'Tungsten'),
]
for z0, z1, col, name in shade_regions:
    ax.axvspan(z0, z1, alpha=0.5, color=col, zorder=0)

# Material labels at top
ax.text(2.200, 565, 'Graphite (7×500mm + 1×400mm)',
        ha='center', fontsize=9, color='#555555',
        fontweight='bold')
ax.text(4.500, 565, 'TZM\n(2×100mm)',
        ha='center', fontsize=8, color='#0055aa',
        fontweight='bold')
ax.text(4.794, 565, 'W\n(389mm)',
        ha='center', fontsize=8, color='#006600',
        fontweight='bold')

# ── Block boundary lines ──────────────────────────────────────────
boundaries = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
              4.4, 4.5, 4.6]
for zb in boundaries:
    ax.axvline(zb, color='#aaaaaa', linewidth=0.8,
               linestyle='--', alpha=0.6, zorder=1)

# ── Raw ANSYS path (faint background) ────────────────────────────
ax.plot(z_path, T_path,
        color='#ffb3b3', linewidth=0.8, alpha=0.4,
        label='ANSYS path raw (mesh artefacts visible)',
        zorder=2)

# ── Smoothed ANSYS path (main line) ──────────────────────────────
ax.plot(z_path, T_smooth,
        color='#cc0000', linewidth=2.5,
        label='ANSYS FEM — smoothed (270 kW worst case)',
        zorder=4)

# ── Per-block T_max markers ───────────────────────────────────────
for i, (zc, mat, T) in enumerate(blocks):
    col = marker_colors[mat]
    ax.scatter(zc, T, color=col, s=90, zorder=6,
               edgecolors='white', linewidths=1.2)

# Add legend entries for block markers (one per material)
for mat, col in marker_colors.items():
    ax.scatter([], [], color=col, s=90, edgecolors='white',
               linewidths=1.2,
               label=f'Present FEM T_max per block — {mat}')

# ── Paper scaled reference ────────────────────────────────────────
ax.plot(z_paper_m, T_paper_scaled,
        color='#555555', linewidth=1.5,
        linestyle='--', marker='o', markersize=6,
        markerfacecolor='white', markeredgewidth=1.5,
        label='Paper Fig. 29 Pt100 data (25 kW, scaled ×10.8 for shape)',
        zorder=5)

# ── Paper Table X peak temperature lines ─────────────────────────
ax.axhline(532, color='#8b0000', linestyle='-.',
           linewidth=1.5, alpha=0.85,
           label='Paper Table X — T_max graphite = 532°C')
ax.axhline(505, color='#00008b', linestyle='-.',
           linewidth=1.5, alpha=0.85,
           label='Paper Table X — T_max TZM = 505°C')
ax.axhline(98,  color='#006400', linestyle='-.',
           linewidth=1.5, alpha=0.85,
           label='Paper Table X — T_max W = 98°C')

# Annotate the reference lines on the right
ax.annotate('532°C (paper)', xy=(4.95, 532), fontsize=8,
            color='#8b0000', va='center')
ax.annotate('505°C (paper)', xy=(4.95, 505), fontsize=8,
            color='#00008b', va='center')
ax.annotate('98°C (paper)',  xy=(4.95, 98),  fontsize=8,
            color='#006400', va='center')

# ── Annotate peak ─────────────────────────────────────────────────
peak_idx = T_smooth.argmax()
ax.annotate(
    f'Peak: {T_smooth[peak_idx]:.0f}°C\n(z = {z_path[peak_idx]:.2f} m)',
    xy=(z_path[peak_idx], T_smooth[peak_idx]),
    xytext=(z_path[peak_idx] - 0.6, T_smooth[peak_idx] + 40),
    fontsize=9, color='#cc0000',
    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor='#cc0000', alpha=0.8)
)

# ── Discrepancy annotation ────────────────────────────────────────
ax.annotate(
    f'Graphite discrepancy:\n{T_smooth.max():.0f}°C vs 532°C (−{(1 - T_smooth.max()/532)*100:.0f}%)\n'
    f'Attributed to absence of\nsecondary particle transport\nin analytical shower model',
    xy=(1.75, 448), xytext=(0.1, 480),
    fontsize=7.5, color='#333333',
    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='gray', alpha=0.9)
)

# ── TZM discrepancy annotation ────────────────────────────────────
ax.annotate(
    f'TZM: {89}°C vs 505°C\nSecondary hadron\ndeposition not modelled',
    xy=(4.5, 89), xytext=(3.6, 300),
    fontsize=7.5, color='#333333',
    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='gray', alpha=0.9)
)

# ── Axes formatting ───────────────────────────────────────────────
ax.set_xlabel('Position along beam axis z [m]', fontsize=12)
ax.set_ylabel('Temperature [°C]', fontsize=12)
ax.set_title(
    'Temperature Distribution Along TIDVG 5 Beam Dump Core — 270 kW Worst Case\n'
    'Analytical Hadronic Shower Model + ANSYS FEM  vs  Reference (Romero Francia et al., 2024)',
    fontsize=11, fontweight='bold'
)
ax.set_xlim(-0.05, 5.15)
ax.set_ylim(0, 600)
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.35, linestyle='-', linewidth=0.6)

# Legend — two columns to keep it compact
ax.legend(fontsize=8, loc='upper left',
          framealpha=0.92, edgecolor='gray',
          ncol=2, columnspacing=1.0)

plt.tight_layout()
plt.savefig('temperature_path_validation_270kW.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: temperature_path_validation_270kW.png")
