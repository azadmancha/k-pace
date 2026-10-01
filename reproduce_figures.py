"""Generates Figure 1 and Figure 2 from the transition dataset."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_figure1():
    print("[1/2] Generating Figure 1: Biomechanical Reachable Set Comparison...")
    delta_t = 1.2
    t_react = 0.7
    a_max = 4.5
    v_max = 8.0
    p0 = np.array([50.0, 34.0])
    v0_fwd = np.array([5.0, 0.0])
    v0_side = np.array([0.0, 4.0])

    C_SPEAR = '#D95F02'
    C_KIN = '#1B9E77'
    C_DIST = '#E6AB02'
    C_HELPLESS = '#7570B3'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    fig.patch.set_facecolor('white')

    for ax, v0, title_suffix in [(ax1, v0_fwd, 'Running Forward'), (ax2, v0_side, 'Running Sideways')]:
        ax.set_facecolor('white')
        s0 = float(np.linalg.norm(v0))
        eff_dt = max(0, delta_t - t_react)
        center = p0 + v0 * min(delta_t, t_react)

        if eff_dt <= 0:
            sprint_r = 0
        elif s0 >= v_max:
            sprint_r = v_max * eff_dt
        else:
            t_acc = (v_max - s0) / a_max
            if eff_dt <= t_acc:
                sprint_r = s0 * eff_dt + 0.5 * a_max * eff_dt**2
            else:
                d_acc = s0 * t_acc + 0.5 * a_max * t_acc**2
                d_cruise = v_max * (eff_dt - t_acc)
                sprint_r = d_acc + d_cruise

        spear_circle = plt.Circle(center, v_max * delta_t, fill=False,
                                   color=C_SPEAR, ls='--', lw=2, alpha=0.8,
                                   label=f'Spearman ({v_max*delta_t:.1f}m radius)')
        ax.add_patch(spear_circle)

        v_angle = np.arctan2(v0[1], v0[0])
        boundary_xs, boundary_ys = [], []
        for theta in np.linspace(0, 2 * np.pi, 100):
            phi = abs(theta - v_angle)
            while phi > np.pi: phi -= 2 * np.pi
            while phi < -np.pi: phi += 2 * np.pi
            phi = abs(phi)
            k = 0.25
            r_reduction = k * (s0 / v_max) * (1.0 - np.cos(phi))
            r = sprint_r * (1.0 - r_reduction)
            boundary_xs.append(center[0] + r * np.cos(theta))
            boundary_ys.append(center[1] + r * np.sin(theta))

        ax.fill(boundary_xs, boundary_ys, alpha=0.18, color=C_KIN)
        ax.plot(boundary_xs, boundary_ys, color=C_KIN, lw=2.2,
                label=f'Kinematic (max {sprint_r:.1f}m)')

        ax.scatter(p0[0], p0[1], c='white', s=110, marker='X', edgecolors='black', linewidths=1.2, zorder=5, label='Actual position')
        ax.scatter(center[0], center[1], c=C_DIST, s=65, marker='o', edgecolors='black', linewidths=0.6, zorder=5,
                  label=f'Drift center ({t_react}s)')
        ax.plot([p0[0], center[0]], [p0[1], center[1]], ':', color='#666666', lw=1.5, alpha=0.8)

        ax.annotate('', xy=(p0[0] + v0[0] * 0.8, p0[1] + v0[1] * 0.8), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle='->', color=C_HELPLESS, lw=2.2))
        ax.text(p0[0] + v0[0] * 0.85, p0[1] + v0[1] * 0.85 + 0.8, f'v₀={s0:.0f}m/s',
                fontsize=9.5, color=C_HELPLESS, fontweight='bold')

        ax.set_xlim(35, 70)
        ax.set_ylim(20, 48)
        ax.set_aspect('equal')
        ax.plot([52.5, 52.5], [20, 48], color='#DDDDDD', ls='-', lw=1.2)
        ax.grid(True, linestyle=':', alpha=0.5, color='#CCCCCC')

        ax.set_title(f'Reachable Set: {title_suffix}', fontsize=11, fontweight='bold', pad=8, color='#0F2042')
        ax.set_xlabel('x (m)', fontsize=9.5)
        ax.set_ylabel('y (m)', fontsize=9.5)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.95, edgecolor='#CCCCCC')

    plt.suptitle(f'Kinematic vs Constant-Velocity Reachable Set (Δt={delta_t}s)',
                 fontsize=13, fontweight='bold', color='#0F2042', y=0.98)
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    out_path = 'figures/reachable_set_comparison.png'
    plt.savefig(out_path, dpi=300)
    print(f"  -> Saved: {out_path}")
    plt.close()

def generate_figure2():
    print("[2/2] Generating Figure 2: Empirical Threat Reduction vs Displacement...")
    data_path = 'data/batch_analysis_results.json' if os.path.exists('data/batch_analysis_results.json') else 'batch_analysis_results.json'
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at data/batch_analysis_results.json or {data_path}.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    goals = [d for d in data if d['actual_outcome'] == 'GOAL_CONCEDED']
    shots = [d for d in data if d['actual_outcome'] != 'GOAL_CONCEDED']

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    s_gd = [d['ghost_div'] for d in shots]
    s_kd = [d['kin_drop'] for d in shots]
    g_gd = [d['ghost_div'] for d in goals]
    g_kd = [d['kin_drop'] for d in goals]

    ax.scatter(s_gd, s_kd, color='#2CA25F', s=45, alpha=0.6, edgecolors='black', linewidth=0.5, label='Non-Scoring Transition Shots', zorder=3)
    ax.scatter(g_gd, g_kd, color='#A51E19', s=90, alpha=0.9, edgecolors='black', linewidth=1.0, label='Conceded Goals', zorder=4)

    ax.axhline(10.0, color='#D95F02', linestyle='--', linewidth=1.8, label='Culpability Threshold (10% Threat Drop)', zorder=2)
    ax.axhspan(0, 3.0, color='#E66101', alpha=0.08, zorder=1)
    ax.axhspan(3.0, 10.0, color='#5E3C99', alpha=0.06, zorder=1)

    ax.text(3.6, 1.2, 'Structural Overload Zone (84.1%)\nOptimal repositioning yields < 3% threat drop', fontsize=8.5, color='#A51E19', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D0D0D0', alpha=0.95))
    ax.text(0.7, 7.8, 'Tactical Dilemma Zone (15.9%)\n2v1 split-zone spatial conflict', fontsize=8.5, color='#5E3C99', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D0D0D0', alpha=0.95))

    ax.set_title('Counterfactual Threat Reduction vs. Optimal Repositioning Across Transition Events', fontsize=11, fontweight='bold', pad=10, color='#0F1937')
    ax.set_xlabel('Counterfactual Displacement to Threat-Minimizing Coordinate ||p* - p₀|| (m)', fontsize=9.5, labelpad=6)
    ax.set_ylabel('Counterfactual Threat Reduction ΔThreat (%)', fontsize=9.5, labelpad=6)

    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(-0.5, 26.0)
    ax.grid(True, linestyle=':', alpha=0.6, color='#CCCCCC')
    ax.legend(loc='upper right', fontsize=8.2, framealpha=0.95, edgecolor='#CCCCCC')

    plt.tight_layout()
    out_path = 'figures/all_transitions_threat_reduction.png'
    plt.savefig(out_path, dpi=300)
    print(f"  -> Saved: {out_path}")
    plt.close()

if __name__ == '__main__':
    generate_figure1()
    generate_figure2()
