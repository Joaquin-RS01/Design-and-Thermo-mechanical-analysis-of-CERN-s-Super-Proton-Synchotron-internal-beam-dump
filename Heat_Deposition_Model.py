import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma as gamma_dist

# ═══════════════════════════════════════════════════════════════════
# SPS BEAM DUMP — HEAT DEPOSITION MODEL
# Based on: Romero Francia et al., Phys. Rev. Accel. Beams 27, 043001 (2024)
# Physics: Hadronic shower — 450 GeV protons do NOT stop in target
# ═══════════════════════════════════════════════════════════════════
# TARGET DESIGN — exact TIDVG 5 geometry (paper Table I)
#
# Graphite : 8 × 500mm + 1 × 400mm = 4400mm  ✓
# TZM      : 2 × 100mm = 200mm               ✓
# Tungsten : 1 × 389mm                        ✓
# Total    : 4989mm ≈ 5.0m                    ✓
# Cross section: 200 × 96 mm                  ✓
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────
# 1. BEAM PARAMETERS — from paper Table VIII
# ─────────────────────────────────────────
P_beam_nominal = 164e3    # W — LHC filling supercycle
P_beam_worst   = 270e3    # W — FT production worst case
P_beam         = P_beam_worst   # ← change to P_beam_nominal for Case 1

# Beam spot — diluted by kicker magnets per paper Fig. 20
# Horizontal kicker (MKDH) sweeps beam wider in x
# Vertical kicker (MKDV) deflects beam downward in y
sigma_beam_x = 0.060    # m — horizontal sigma
sigma_beam_y = 0.025    # m — vertical sigma

# ─────────────────────────────────────────
# 2. TARGET GEOMETRY — paper Table I exact
# ─────────────────────────────────────────
block_W = 0.200    # m width  (x-direction)
block_H = 0.096    # m height (y-direction)

# Format: (name, length_m, density_kg/m³, lambda_int_m)
stack = [
    ('Graphite', 0.500, 1800,  0.381),   # block 1
    ('Graphite', 0.500, 1800,  0.381),   # block 2
    ('Graphite', 0.500, 1800,  0.381),   # block 3
    ('Graphite', 0.500, 1800,  0.381),   # block 4  ← shower maximum here
    ('Graphite', 0.500, 1800,  0.381),   # block 5
    ('Graphite', 0.500, 1800,  0.381),   # block 6
    ('Graphite', 0.500, 1800,  0.381),   # block 7
    ('Graphite', 0.500, 1800,  0.381),   # block 8
    ('Graphite', 0.400, 1800,  0.381),   # block 9 — shorter per Table I
    ('TZM',      0.100, 10100, 0.096),   # block 10
    ('TZM',      0.100, 10100, 0.096),   # block 11
    ('Tungsten', 0.389, 18800, 0.096),   # block 12
]

# Build z boundary positions
z_boundaries = [0.0]
for _, L, _, _ in stack:
    z_boundaries.append(z_boundaries[-1] + L)
L_total = z_boundaries[-1]

print(f"Total target length: {L_total:.3f} m")
print(f"Number of blocks: {len(stack)}")
print(f"Graphite length: {sum(L for n,L,_,_ in stack if n=='Graphite'):.3f} m")
print(f"TZM length:      {sum(L for n,L,_,_ in stack if n=='TZM'):.3f} m")
print(f"Tungsten length: {sum(L for n,L,_,_ in stack if n=='Tungsten'):.3f} m")

# ─────────────────────────────────────────
# 3. HADRONIC SHOWER AXIAL PROFILE f(z)
# ─────────────────────────────────────────
# At 450 GeV protons do NOT stop — hadronic cascade develops
# Shower profile follows Gamma distribution (Bock parameterisation)
# Peak confirmed at block 4 by FLUKA in paper Fig. 21

z_fine = np.linspace(0, L_total, 5000)

# Build piecewise interaction-length coordinate t(z)
# t = cumulative depth in units of lambda_int
t_fine = np.zeros_like(z_fine)
for i, z_val in enumerate(z_fine):
    cum_t = 0.0
    cum_z = 0.0
    for name, L, rho, lam in stack:
        if z_val <= cum_z + L:
            cum_t += (z_val - cum_z) / lam
            break
        else:
            cum_t += L / lam
            cum_z += L
    t_fine[i] = cum_t

print(f"\nTotal depth: {t_fine[-1]:.2f} interaction lengths")

# Shower maximum — centre of block 4 (z = 3.5 × 0.500 = 1.75m)
# Updated for 500mm blocks
t_shower_max = (3.5 * 0.500) / 0.381
z_shower_max = 3.5 * 0.500
print(f"Shower max at t = {t_shower_max:.2f} λ_int")
print(f"Shower max at z = {z_shower_max:.3f} m (centre of block 4)")

# Gamma distribution: f(t) = t^(a-1) * exp(-t) / Gamma(a)
a_shape = t_shower_max + 1.0
f_t = gamma_dist.pdf(t_fine, a=a_shape, scale=1.0)

# Convert from interaction-length space to physical space
# f(z) = f(t) / lambda_int(z)  [chain rule: dE/dz = dE/dt * dt/dz = f(t)/λ]
lambda_int_z = np.zeros_like(z_fine)
for i, z_val in enumerate(z_fine):
    cum = 0.0
    for name, L, rho, lam in stack:
        if z_val <= cum + L:
            lambda_int_z[i] = lam
            break
        cum += L

f_z = f_t / lambda_int_z

# Normalise so integral over full target = 1 [units: 1/m]
norm_z = np.trapezoid(f_z, z_fine)
f_z = f_z / norm_z
print(f"Axial normalisation: {np.trapezoid(f_z, z_fine):.6f} (should be 1.0)")

# ─────────────────────────────────────────
# 4. TRANSVERSE PROFILE g(x,y) — RECTANGULAR
# ─────────────────────────────────────────
# 2D Gaussian normalised over rectangular block face
# Integral over face = 1 [units: 1/m²]

Nx, Ny = 100, 50
x = np.linspace(-block_W/2, block_W/2, Nx)
y = np.linspace(-block_H/2, block_H/2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

G_2d = (np.exp(-X**2 / (2*sigma_beam_x**2)) *
        np.exp(-Y**2 / (2*sigma_beam_y**2)))

norm_rect = np.trapezoid(np.trapezoid(G_2d, y, axis=1), x)
G_2d_norm = G_2d / norm_rect   # [1/m²]

check = np.trapezoid(np.trapezoid(G_2d_norm, y, axis=1), x)
print(f"Transverse normalisation: {check:.6f} (should be 1.0)")

# ─────────────────────────────────────────
# 5. BLOCK-AVERAGED q''' FOR ANSYS
# ─────────────────────────────────────────
# q'''_avg = P_beam × f_avg_z × (1/A_block)
# f_avg_z  = average of f(z) over block length
# 1/A_block = spatial average of G_2d_norm over rectangular face

A_block = block_W * block_H   # m²
g_avg   = 1.0 / A_block        # [1/m²]

print(f"\nBlock face area: {A_block*1e6:.0f} mm²")
print(f"\n{'Block':>6} {'Material':>10} {'z_start':>10} {'z_end':>8} "
      f"{'q_avg [W/m³]':>15} {'q_avg [MW/m³]':>14}")

block_results = []
for b_idx, (name, L, rho, lam) in enumerate(stack):
    z_start = z_boundaries[b_idx]
    z_end   = z_boundaries[b_idx + 1]
    mask    = (z_fine >= z_start) & (z_fine < z_end)

    f_avg = np.trapezoid(f_z[mask], z_fine[mask]) / L
    q_avg = P_beam * f_avg * g_avg
    block_results.append(q_avg)

    print(f"{b_idx+1:>6} {name:>10} {z_start:>10.3f} {z_end:>8.3f} "
          f"{q_avg:>15.3e} {q_avg/1e6:>14.3f}")

# ─────────────────────────────────────────
# 6. ENERGY BALANCE VERIFICATION
# ─────────────────────────────────────────
Q_total = sum(block_results[i] * stack[i][1] * A_block
              for i in range(len(stack)))
print(f"\nEnergy balance: {Q_total/1e3:.3f} kW (should be {P_beam/1e3:.1f} kW)")

# ─────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'SPS Beam Dump TIDVG 5 — Heat Deposition Model (270 kW worst case)\n'
    'Ref: Romero Francia et al., Phys. Rev. Accel. Beams 27, 043001 (2024)',
    fontsize=10)

block_colors = {'Graphite': 'gray', 'TZM': 'royalblue', 'Tungsten': 'darkorange'}

# ── Plot 1: Axial shower profile ──────────────────────────────────
ax = axes[0]
ax.plot(z_fine * 100, f_z * P_beam / 1e3, 'b-', linewidth=2)
ax.set_xlabel('z [cm]')
ax.set_ylabel('Linear power density [kW/m]')
ax.set_title('Hadronic Shower Profile — Along Beam Axis')
ax.grid(True, alpha=0.4)

for b_idx, (name, L, rho, lam) in enumerate(stack):
    zb = z_boundaries[b_idx] * 100
    ax.axvline(zb, color=block_colors[name], linestyle='--',
               alpha=0.5, linewidth=0.8)
    ax.text(zb + 0.5, 0.90, f'B{b_idx+1}',
            transform=ax.get_xaxis_transform(), fontsize=6,
            color=block_colors[name])

# TZM and W start markers — updated indices for 12-block stack
ax.axvline(z_boundaries[9]  * 100, color='royalblue',
           linewidth=1.5, label='TZM start')
ax.axvline(z_boundaries[11] * 100, color='darkorange',
           linewidth=1.5, label='W start')
ax.axvline(z_shower_max * 100, color='red', linewidth=1.5,
           linestyle=':', label=f'Shower max z={z_shower_max*100:.0f}cm')
ax.legend(fontsize=8)

# ── Plot 2: Transverse beam profile ───────────────────────────────
ax2 = axes[1]
cf2 = ax2.contourf(X * 1000, Y * 1000, G_2d_norm, levels=50, cmap='hot')
plt.colorbar(cf2, ax=ax2, label='g(x,y) [1/m²]')
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('y [mm]')
ax2.set_title('Transverse Beam Profile\n(diluted, swept by kickers)')
ax2.set_aspect('equal')

# ── Plot 3: Block-averaged q''' bar chart ─────────────────────────
ax3 = axes[2]
block_labels = [f'B{i+1}\n{stack[i][0][:3]}' for i in range(len(stack))]
colors       = [block_colors[stack[i][0]] for i in range(len(stack))]
bars = ax3.bar(block_labels, [q/1e6 for q in block_results],
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Block')
ax3.set_ylabel("q‴ average [MW/m³]")
ax3.set_title("Block-Averaged Heat Generation\n(ANSYS Internal Heat Generation input)")
ax3.grid(True, alpha=0.4, axis='y')

for bar, q in zip(bars, block_results):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'{q/1e6:.1f}', ha='center', va='bottom', fontsize=6)

plt.tight_layout()
plt.savefig('SPS_TIDVG5_heat_deposition.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────
# 8. ANSYS INPUT SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*65)
print("ANSYS Internal Heat Generation — enter these values [W/m³]")
print("="*65)
print(f"{'Block':<8} {'Material':<12} {'q [W/m³]':<16} {'Note'}")
print("-"*65)
for i, (name, L, rho, lam) in enumerate(stack):
    if name == 'TZM':
        note = "← most thermomechanically loaded (paper Table XI)"
    elif i in [3, 4]:
        note = "← shower maximum region (paper Fig. 21)"
    elif name == 'Tungsten':
        note = "← validation: expect T~98°C (paper Table X)"
    else:
        note = ""
    print(f"Block {i+1:<3} {name:<12} {block_results[i]:<16.4e} {note}")
print("-"*65)
print(f"{'TOTAL POWER CHECK:':<30} {Q_total/1e3:.2f} kW  "
      f"(target: {P_beam/1e3:.1f} kW)")

# ─────────────────────────────────────────
# 9. CSV EXPORT FOR ANSYS EXTERNAL DATA
# ─────────────────────────────────────────
print("\nExporting CSV...")
rows = []
z_sample = z_fine[::5]
r_x      = x[::2]
r_y      = y[::2]

for j, z_val in enumerate(z_sample):
    iz = j * 5
    for kx, xv in enumerate(r_x):
        for ky, yv in enumerate(r_y):
            ix = kx * 2
            iy = ky * 2
            q_val = P_beam * f_z[iz] * G_2d_norm[ix, iy]
            rows.append([xv, yv, z_val, q_val])

df = pd.DataFrame(rows,
                  columns=['x [m]', 'y [m]', 'z [m]',
                           'Heat Generation [W/m3]'])
fname = f'SPS_TIDVG5_{int(P_beam/1e3)}kW.csv'
df.to_csv(fname, index=False)
print(f"Exported {len(df):,} points → {fname}")
print(f"Peak CSV value: {df['Heat Generation [W/m3]'].max():.3e} W/m³")
