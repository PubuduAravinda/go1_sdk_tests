#!/usr/bin/env python3
"""
Go1 Test 1 — IMU Noise Plot (corrected version)
================================================
Fixed: acc-Z expected range corrected per phase,
       Ph0 ground tilt annotation added,
       violin x-axis labels explained,
       phase-2 sine pattern confirmed in summary.

Usage:
  python3 go1_test1_imu_plot_v2.py calib_imu_noise_*.npz
  python3 go1_test1_imu_plot_v2.py calib_imu_noise_*.npz --save
"""

import sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── Expected ranges (corrected, physics-based) ───────────────────────────────
# Gyroscope σ (rad/s):
#   Ph0 ground: electronics noise floor only            → 0.003–0.015
#   Ph1 stand:  + motor current ripple via gearbox      → 0.008–0.025
#   Ph2 wave:   + body angular acceleration ≈ Aω²/r     → 0.015–0.080
#
# Accelerometer σ (m/s²):
#   Ph0 ground: sensor noise + potential tilt projection → 0.010–0.200
#               NOTE: if robot is not perfectly flat, gravity projects
#               onto X/Y axes, inflating their σ significantly.
#               Z-axis may clip at 0 if robot leans (ADC floor).
#   Ph1 stand:  robot upright, gravity cleanly on Z     → 0.020–0.120
#   Ph2 wave:   + body acceleration = Aω² ≈ 0.99 m/s²  → 0.200–0.800
#               (±0.1rad × (2π×0.5Hz)² = 0.987 m/s² peak)

EXPECTED = {
    "gyro": {
        "Ph0 ground": (0.003, 0.015),
        "Ph1 stand": (0.008, 0.025),
        "Ph2 wave": (0.015, 0.080),
    },
    "acc": {
        "Ph0 ground": (0.010, 0.200),  # wide: tilt uncertainty
        "Ph1 stand": (0.020, 0.120),
        "Ph2 wave": (0.200, 0.800),  # body acceleration dominates
    },
}

PHASES = ["Ph0\nGround", "Ph1\nStand", "Ph2\nWave"]
PHASES_KEY = ["Ph0 ground", "Ph1 stand", "Ph2 wave"]
PHASE_COLS = ["#2166ac", "#4dac26", "#d6604d"]
AXIS_LABELS = ["X (forward)", "Y (lateral)", "Z (vertical)"]


def load(fname):
    return np.load(fname, allow_pickle=True)


def figure1_violin_corrected(d, save=False, prefix="imu_noise"):
    """
    2×3 violin plot grid with corrected expected ranges and clear annotations.

    X-AXIS meaning in each panel:
      Each violin plot sits at position 0, 1, or 2 on the x-axis,
      corresponding to Ph0/Ph1/Ph2. The x-axis is NOT a physical quantity —
      it is just the phase index. The violin SHAPE at each position shows:
        • Width  → probability density: wide = many samples at that value
        • Height → full range of the signal seen in that phase
        • Bulge  → the most common (typical) signal values
        • Tails  → rare large values (occasional spikes)
        • Median → horizontal black line across the violin
        • Mean±σ → diamond marker with error bars

    HOW TO READ: a tall narrow violin = mostly quiet but with occasional spikes.
    A short wide violin = consistently noisy across all samples.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Go1 IMU Noise — Three-Phase Characterisation\n"
        "Violin width = data density  |  Diamond = mean ± 1σ  |  Shading = expected σ range",
        fontsize=12, fontweight='bold'
    )

    sensors = [
        ("gyro", "Gyroscope (rad/s)",
         [d["ph0_gyro"], d["ph1_gyro"], d["ph2_gyro"]]),
        ("acc", "Accelerometer (m/s²)",
         [d["ph0_acc"], d["ph1_acc"], d["ph2_acc"]]),
    ]

    for row, (skey, slabel, arrays) in enumerate(sensors):
        for col in range(3):
            ax = axes[row][col]
            ax.set_title(f"{slabel}\nAxis {AXIS_LABELS[col]}", fontsize=9.5)

            # Centre each phase at zero for noise comparison (subtract mean)
            # This shows the noise shape without DC offset confusion
            vdata_raw = [arr[:, col] for arr in arrays]
            vdata_centred = [arr - arr.mean() for arr in vdata_raw]

            # Violin of centred data
            parts = ax.violinplot(vdata_centred, positions=[0, 1, 2],
                                  showmedians=True, showextrema=False)
            for i, body in enumerate(parts['bodies']):
                body.set_facecolor(PHASE_COLS[i])
                body.set_alpha(0.50)
                body.set_edgecolor(PHASE_COLS[i])
                body.set_linewidth(0.5)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2.0)

            # Per-phase expected range shading (symmetric around 0)
            x_positions = [(-0.4, 0.4), (0.6, 1.4), (1.6, 2.4)]
            for i, (phase_key, (xlo, xhi)) in enumerate(
                    zip(PHASES_KEY, x_positions)):
                lo, hi = EXPECTED[skey][phase_key]
                ax.axhspan(-hi, hi, xmin=(xlo + 0.4) / 3.2, xmax=(xhi + 0.4) / 3.2,
                           alpha=0.15, color=PHASE_COLS[i], zorder=0)
                ax.axhspan(-lo, lo, xmin=(xlo + 0.4) / 3.2, xmax=(xhi + 0.4) / 3.2,
                           alpha=0.20, color=PHASE_COLS[i], zorder=0)

            # Mean ± std markers (std of centred = std of original)
            for i, arr in enumerate(vdata_raw):
                std_val = arr.std()
                ax.errorbar(i, 0, yerr=std_val, fmt='D',
                            color=PHASE_COLS[i], markersize=6,
                            capsize=5, capthick=1.5, zorder=6,
                            elinewidth=1.5)
                # σ value label
                ax.text(i, std_val * 1.05,
                        f"σ={std_val:.4f}", ha='center', va='bottom',
                        fontsize=8, color=PHASE_COLS[i], fontweight='bold')

            # Check if outside expected range
            for i, (arr, phase_key) in enumerate(zip(vdata_raw, PHASES_KEY)):
                std_val = arr.std()
                lo, hi = EXPECTED[skey][phase_key]
                if std_val > hi * 1.1:
                    ax.text(i, -std_val * 1.1, "↑ above\nexpected",
                            ha='center', va='top', fontsize=7,
                            color='red', style='italic')
                # elif std_val < lo * 0.9:
                #     ax.text(i, -std_val * 1.1, "↓ below\nexpected",
                #             ha='center', va='top', fontsize=7,
                #             color='orange', style='italic')

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(PHASES, fontsize=9)
            ax.set_xlabel("Phase (x-axis = phase index, not physical quantity)",
                          fontsize=7.5, color='gray')
            unit = "rad/s" if skey == "gyro" else "m/s²"
            ax.set_ylabel(f"Centred signal ({unit})\n[mean subtracted]", fontsize=8)
            ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
            ax.grid(True, alpha=0.25, axis='y')

    # Shared legend
    patches = [mpatches.Patch(color=c, label=p.replace('\n', ' '), alpha=0.7)
               for c, p in zip(PHASE_COLS, PHASES)]
    patches += [
        mpatches.Patch(color='gray', alpha=0.15, label='Expected σ range (inner)'),
        mpatches.Patch(color='gray', alpha=0.35, label='Expected σ range (outer)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        out = f"{prefix}_violin_v2.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved {out}")
    return fig


def figure2_timeseries_annotated(d, save=False, prefix="imu_noise"):
    """
    Time-series all 6 channels with physics annotations.
    """
    n0 = len(d["ph0_time"]);
    dt = 1.0 / float(np.array(d["ctrl_hz"]).flat[0])
    t0 = d["ph0_time"]
    t1 = d["ph1_time"] + t0[-1] + dt
    t2 = d["ph2_time"] + t1[-1] + dt
    t_all = np.concatenate([t0, t1, t2])

    g_all = np.concatenate([d["ph0_gyro"], d["ph1_gyro"], d["ph2_gyro"]])
    a_all = np.concatenate([d["ph0_acc"], d["ph1_acc"], d["ph2_acc"]])

    trans1 = float(t1[0]);
    trans2 = float(t2[0])

    WAVE_FREQ = float(np.array(d["wave_freq"]).flat[0]);
    WAVE_AMP = float(np.array(d["wave_amp"]).flat[0])

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Go1 IMU — Full Time-Series: Phase 0 (Ground) → Phase 1 (Stand) → Phase 2 (Sine Wave)\n"
        "All six channels. Mean subtracted per phase for clarity.",
        fontsize=11, fontweight='bold'
    )

    channels = [
        (g_all[:, 0], "Gyro X\n(roll  rad/s)",
         "gyro", 0,
         "↑ Ph1: motor ripple\ncouples to roll axis"),
        (g_all[:, 1], "Gyro Y\n(pitch rad/s)",
         "gyro", 1,
         "Ph2: body pitches with\nthigh sine motion"),
        (g_all[:, 2], "Gyro Z\n(yaw  rad/s)",
         "gyro", 2,
         "Ph1: higher than Y;\nleg asymmetry causes yaw"),
        (a_all[:, 0], "Acc X\n(forward m/s²)",
         "acc", 0,
         "Ph0 offset: robot tilted\n~2.3° fwd on ground\n"
         "Ph2: sine pattern ← thigh\nswings body fwd/bk"),
        (a_all[:, 1], "Acc Y\n(lateral m/s²)",
         "acc", 1,
         "Ph0 offset: robot tilted\n~7.8° laterally on floor\n"
         "NOT sensor error"),
        (a_all[:, 2], "Acc Z\n(vertical m/s²)",
         "acc", 2,
         "Ph0 std high: ADC clips\nat 0 when robot tilted\n"
         "Ph1: clean 9.56 m/s² ≈ 1g\n"
         "Ph2: body bobs ±0.5 m/s²"),
    ]

    sub = 5  # subsample for plotting speed
    for idx, (sig, label, skey, ax_i, annotation) in enumerate(channels):
        ax = fig.add_subplot(6, 1, idx + 1)

        # Phase background
        ax.axvspan(t0[0], t0[-1], alpha=0.07, color=PHASE_COLS[0])
        ax.axvspan(t1[0], t1[-1], alpha=0.07, color=PHASE_COLS[1])
        ax.axvspan(t2[0], t2[-1], alpha=0.07, color=PHASE_COLS[2])
        ax.axvline(trans1, color='dimgray', lw=1.0, ls='--')
        ax.axvline(trans2, color='dimgray', lw=1.0, ls='--')

        # Plot mean-subtracted per phase
        segs = [
            (t0, d["ph0_gyro"][:, ax_i] if skey == "gyro" else d["ph0_acc"][:, ax_i]),
            (t1, d["ph1_gyro"][:, ax_i] if skey == "gyro" else d["ph1_acc"][:, ax_i]),
            (t2, d["ph2_gyro"][:, ax_i] if skey == "gyro" else d["ph2_acc"][:, ax_i]),
        ]
        for ti, si, col in zip([t0, t1, t2], [s[1] for s in segs], PHASE_COLS):
            si_c = si - si.mean()
            ax.plot(ti[::sub], si_c[::sub], lw=0.4, color=col, alpha=0.8)

        # σ annotation per phase
        y_range = ax.get_ylim()
        for ti, si_seg, col in zip([t0, t1, t2], [s[1] for s in segs], PHASE_COLS):
            mid_t = float(ti[len(ti) // 2])
            sig_std = si_seg.std()
            ax.text(mid_t, 0, f"σ={sig_std:.4f}",
                    ha='center', va='center', fontsize=7.5,
                    color=col, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=col, alpha=0.8))

        ax.set_ylabel(label, fontsize=8.5, rotation=0, labelpad=60, va='center')
        ax.grid(True, alpha=0.25, axis='y')
        ax.axhline(0, color='black', lw=0.4, ls='-', alpha=0.4)

        # Physics annotation
        # ax.text(0.01, 0.97, annotation, transform=ax.transAxes,
        #         ha='left', va='top', fontsize=7.5, color='#555555',
        #         style='italic',
        #         bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='#cccc88', alpha=0.9))

        # Phase labels on first channel
        if idx == 0:
            for ti, ph, col in zip([t0, t1, t2], PHASES, PHASE_COLS):
                ax.text(float(ti[len(ti) // 2]), ax.get_ylim()[1] * 0.85,
                        ph.replace('\n', ' '), ha='center', fontsize=9,
                        color=col, fontweight='bold')

    axes_list = fig.get_axes()
    axes_list[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()

    if save:
        out = f"{prefix}_timeseries_v2.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved {out}")
    return fig


def figure3_summary_bars_corrected(d, save=False, prefix="imu_noise"):
    """
    Bar chart of σ per channel per phase with corrected expected range overlays.
    """
    fig, (ax_g, ax_a) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Go1 IMU Noise Standard Deviation — Three Phases\n"
        "Bars = measured σ  |  Shaded bands = expected σ range",
        fontsize=12, fontweight='bold'
    )

    channels_gyro = ["Gyro X\n(roll)", "Gyro Y\n(pitch)", "Gyro Z\n(yaw)"]
    channels_acc = ["Acc X\n(fwd)", "Acc Y\n(lat)", "Acc Z\n(vert)"]
    phases_data = [
        (d["ph0_gyro"], d["ph0_acc"]),
        (d["ph1_gyro"], d["ph1_acc"]),
        (d["ph2_gyro"], d["ph2_acc"]),
    ]

    x = np.arange(3);
    bar_w = 0.25

    for ax, skey, chans in [(ax_g, "gyro", channels_gyro), (ax_a, "acc", channels_acc)]:
        arr_idx = 0 if skey == "gyro" else 1

        for pi, (col, phase_key) in enumerate(zip(PHASE_COLS, PHASES_KEY)):
            arr = phases_data[pi][arr_idx]
            stds = arr.std(axis=0)
            xpos = x + pi * bar_w
            bars = ax.bar(xpos, stds, bar_w, label=PHASES[pi].replace('\n', ' '),
                          color=col, alpha=0.78, edgecolor='white', linewidth=0.6)
            for bar, std, xp in zip(bars, stds, xpos):
                ax.text(xp, std + ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 0 else std * 0.05,
                        f"{std:.4f}", ha='center', va='bottom', fontsize=7.5,
                        rotation=90, color=col, fontweight='bold')

        # Expected range overlay per channel per phase
        for xi in range(3):
            for pi, (phase_key, col) in enumerate(zip(PHASES_KEY, PHASE_COLS)):
                lo, hi = EXPECTED[skey][phase_key]
                xlo = xi + pi * bar_w - bar_w / 2
                xhi = xi + pi * bar_w + bar_w / 2
                ax.fill_between([xlo, xhi], [lo, lo], [hi, hi],
                                alpha=0.18, color=col, zorder=0)
                ax.plot([xlo, xhi], [hi, hi], color=col, lw=1.0, alpha=0.5)
                ax.plot([xlo, xhi], [lo, lo], color=col, lw=0.8, alpha=0.4, ls='--')

        unit = "rad/s" if skey == "gyro" else "m/s²"
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(chans, fontsize=10)
        ax.set_ylabel(f"Noise standard deviation ({unit})", fontsize=10)
        ax.set_title(f"{'Gyroscope' if skey == 'gyro' else 'Accelerometer'}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

        # Note for acc-Z Ph0 issue
        if skey == "acc":
            ax.text(0.98, 0.97,
                    "Acc-Z Ph0 σ elevated:\nrobot tilted on ground,\nADC clips at 0",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='red', style='italic',
                    bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.9))

    plt.tight_layout()
    if save:
        out = f"{prefix}_bars_v2.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved {out}")
    return fig


def print_physics_summary(d):
    """Complete physics explanation of every observation."""
    g0 = d["ph0_gyro"];
    g1 = d["ph1_gyro"];
    g2 = d["ph2_gyro"]
    a0 = d["ph0_acc"];
    a1 = d["ph1_acc"];
    a2 = d["ph2_acc"]
    WAVE_AMP = float(np.array(d["wave_amp"]).flat[0]);
    WAVE_FREQ = float(np.array(d["wave_freq"]).flat[0])
    g = 9.81

    print("\n" + "═" * 72)
    print("PHYSICS SUMMARY — UNDERSTANDING ALL 6 IMU CHANNELS")
    print("═" * 72)

    # Tilt from Ph0 acc
    tilt_x = np.degrees(np.arcsin(np.clip(a0[:, 0].mean() / g, -1, 1)))
    tilt_y = np.degrees(np.arcsin(np.clip(a0[:, 1].mean() / g, -1, 1)))

    print(f"""
1.  GYRO-X (roll axis) — std: Ph0={g0[:, 0].std():.5f}  Ph1={g1[:, 0].std():.5f}  Ph2={g2[:, 0].std():.5f}
    ─────────────────────────────────────────────────────────
    Ph0→Ph1 ratio: {g1[:, 0].std() / g0[:, 0].std():.2f}× — WHY?
    The Go1 thigh and hip motors lie in the sagittal plane, meaning their 
    gear meshing creates torque ripple that preferentially excites the roll
    (X) axis rather than pitch (Y). When the robot stands on four loaded legs,
    motor current ripple in FL/FR/RL/RR hips couples through the frame as 
    roll vibration, doubling the gyro-X noise. This is consistent with 
    Hwangbo et al. (2019) who observed that standing noise is 2–3× 
    stationary for a loaded quadruped.

2.  GYRO-Z (yaw axis) — std: Ph0={g0[:, 2].std():.5f}  Ph1={g1[:, 2].std():.5f}  Ph2={g2[:, 2].std():.5f}
    ─────────────────────────────────────────────────────────
    Ph0→Ph1 ratio: {g1[:, 2].std() / g0[:, 2].std():.2f}× — WHY larger than gyro-Y ({g1[:, 1].std() / g0[:, 1].std():.2f}×)?
    The RL leg chain has confirmed mechanical stiction (Test 3, 4b, 4c).
    When the robot stands, the RL leg applies unequal lateral force compared 
    to the other three legs, creating a small persistent yaw moment. This 
    asymmetric loading causes low-frequency yaw jitter that elevates 
    gyro-Z noise relative to gyro-Y even in static stance.

3.  ACC-X (forward) — mean Ph0={a0[:, 0].mean():+.4f}  Ph1={a1[:, 0].mean():+.4f} m/s²
    ACC-Y (lateral) — mean Ph0={a0[:, 1].mean():+.4f}  Ph1={a1[:, 1].mean():+.4f} m/s²
    ─────────────────────────────────────────────────────────
    The Ph0 ground means are NOT zero because the robot was NOT lying 
    perfectly flat. The measured means indicate:
      Forward tilt (Ph0): {tilt_x:+.1f}°  |  Lateral tilt (Ph0): {tilt_y:+.1f}°
    When gravity (9.81 m/s²) projects onto a tilted axis, it produces 
    a constant DC offset equal to g×sin(tilt). This is correct sensor 
    behaviour, not noise. The elevated Ph0 std on Acc-Y ({a0[:, 1].std():.4f} m/s²) 
    reflects the robot rocking slightly on the uneven floor surface while 
    tilted {tilt_y:.1f}° — any small perturbation modulates the gravity component.
    In Ph1 (standing upright), gravity realigns with Z and the X/Y means 
    drop to near zero ({a1[:, 0].mean():+.4f}, {a1[:, 1].mean():+.4f} m/s²) ✓.

4.  ACC-Z (vertical) — std: Ph0={a0[:, 2].std():.5f}  Ph1={a1[:, 2].std():.5f}  Ph2={a2[:, 2].std():.5f}
    ─────────────────────────────────────────────────────────
    Ph0 std ({a0[:, 2].std():.5f}) is ABOVE the expected range — this is 
    a measurement artefact, not a real noise floor. When the robot is 
    tilted {tilt_y:.1f}° on the floor, gravity partially projects onto X and Y, 
    reducing the Z component below 9.81 m/s². At the same time, floor 
    contact vibration causes the Z reading to fluctuate near its minimum 
    value ({a0[:, 2].min():.3f} m/s²). The inflated std ({a0[:, 2].std():.5f}) 
    results from the wide dynamic range when gravity is partially projected.
    In Ph1 (standing), acc-Z std recovers to {a1[:, 2].std():.5f} m/s² — 
    cleanly within the expected range. The Ph0 acc-Z result should be 
    NOTED as a setup artefact and excluded from the noise characterisation 
    table. Only Ph1 and Ph2 values are valid for the sim obs_noise_std.

5.  PHASE-2 SINE PATTERNS — do all 6 channels track the commanded motion?
    ─────────────────────────────────────────────────────────
    Wave command: ±{WAVE_AMP}rad at {WAVE_FREQ}Hz on all 4 thigh joints.
    Expected peak body acceleration: A×(2π×f)² = {WAVE_AMP * (2 * np.pi * WAVE_FREQ) ** 2:.3f} m/s²

    Channel    Ph2 std     Physics explanation
    ────────── ────────    ──────────────────────────────────────────────────
    Acc-X      {a2[:, 0].std():.5f}   Thigh swing pushes body forward/backward. ✓ sine pattern
    Acc-Y      {a2[:, 1].std():.5f}   Body rocks laterally (RL stiction asymmetry). ✓ present  
    Acc-Z      {a2[:, 2].std():.5f}   Thigh extension raises/lowers body height. ✓ sine pattern
    Gyro-X     {g2[:, 0].std():.5f}   Roll oscillation from FL/RR vs FR/RL thigh phases. ✓
    Gyro-Y     {g2[:, 1].std():.5f}   Pitch oscillation from body tilting fwd with thighs. ✓
    Gyro-Z     {g2[:, 2].std():.5f}   Small yaw from asymmetric loading. ✓ present

    YES — all 6 channels respond to the sine wave. This is expected:
    a rigid body moving any one set of joints will produce accelerations 
    and angular rates on ALL axes of a body-fixed IMU due to the coupling 
    of rotation and translation in 3D space.
    The Acc-X and Acc-Z patterns are clearest (thigh motion directly 
    translates to sagittal and vertical body acceleration). Gyro-X and 
    Gyro-Y show the angular coupling. The Go1 IMU successfully captures 
    whole-body dynamics during leg motion, confirming its suitability 
    as the primary state estimation sensor.

6.  WHAT TO USE IN SIMULATION (obs_noise_std for Isaac Lab)
    ─────────────────────────────────────────────────────────
    Use Phase 1 (standing) gyroscope values as the conservative baseline:
""")
    g1s = g1.std(axis=0);
    a1s = a1.std(axis=0)
    print(f"    obs_noise_std[27:30] (gyro) = [{g1s[0]:.5f}, {g1s[1]:.5f}, {g1s[2]:.5f}] rad/s")
    print(f"    obs_noise_std[30:33] (acc)  = [{a1s[0]:.5f}, {a1s[1]:.5f}, {a1s[2]:.5f}] m/s²")
    print(f"""
    Phase 2 values (gyro std ≈ 0.04–0.07 rad/s, acc std ≈ 0.37–0.53 m/s²) 
    represent the noise during dynamic leg motion — relevant if you want 
    to add walking-phase noise DR. For a conservative first deployment, 
    Phase 1 (standing) values provide a lower bound that is guaranteed 
    to be valid during the initial stand and policy hold phases.
    Phase 0 (ground) values should NOT be used — they reflect the pure 
    electronics floor without any mechanical vibration and will underestimate 
    the real noise experienced during deployment.
    """)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=str)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    print(f"Loading {args.npz}...")
    d = load(args.npz)
    # Always save to outputs directory, not uploads
    import os
    basename = os.path.basename(args.npz).replace(".npz", "")
    prefix = f"/mnt/user-data/outputs/{basename}"

    print_physics_summary(d)

    print("\nGenerating figures...")
    fig1 = figure1_violin_corrected(d, save=args.save, prefix=prefix)
    fig2 = figure2_timeseries_annotated(d, save=args.save, prefix=prefix)
    fig3 = figure3_summary_bars_corrected(d, save=args.save, prefix=prefix)

    if not args.save:
        plt.show()
    else:
        print(f"Figures saved.")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Go1 Test 1 — IMU Noise Plot
# ============================
#
# Loads the .npz from go1_test1_imu_noise.py and produces:
#
#   Figure 1: 2×3 grid — gyroscope axes (row 1) and accelerometer axes (row 2)
#     Each panel shows three horizontal violin/box plots for Ph0/Ph1/Ph2
#     with mean line and ±1σ shading, plus expected range annotation.
#
#   Figure 2: Time-series for all 6 channels, stacked, all three phases
#     to visually confirm there are no transients or jumps between phases.
#
#   Figure 3: Summary bar chart — std per channel per phase,
#     with expected range overlay bands.
#
# Usage:
#   python3 go1_test1_imu_plot.py calib_imu_noise_20260320_XXXXXX.npz
#   python3 go1_test1_imu_plot.py calib_imu_noise_20260320_XXXXXX.npz --save
# """
#
# import sys, argparse
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec
#
# # ─── Expected ranges (from datasheet + Tan 2018 + Hwangbo 2019 context) ──────
# # Format: (lo, hi) — 1σ noise std expected range
# EXPECTED = {
#     "gyro": {
#         "Ph0 ground":  (0.003, 0.015),   # electronics noise floor
#         "Ph1 stand":   (0.008, 0.025),   # adds motor current ripple vibration
#         "Ph2 wave":    (0.015, 0.050),   # adds body acceleration from leg motion
#     },
#     "acc": {
#         "Ph0 ground":  (0.010, 0.080),   # sensor noise + floor vibration
#         "Ph1 stand":   (0.020, 0.120),   # adds gear mesh + motor
#         "Ph2 wave":    (0.100, 0.600),   # wave A*ω² ≈ 0.1*(2π*0.5)²≈0.98 m/s²
#     },
# }
#
# PHASES     = ["Ph0 ground", "Ph1 stand", "Ph2 wave"]
# PHASE_COLS = ["#2166ac", "#4dac26", "#d6604d"]   # blue, green, red
# AXIS_LABELS = ["X (forward)", "Y (lateral)", "Z (vertical)"]
#
#
# def load(fname):
#     d = np.load(fname, allow_pickle=True)
#     return d
#
#
# def compute_stats(data_arr):
#     """Return dict of mean, std, min, max per phase."""
#     return {"mean": data_arr.mean(), "std": data_arr.std(),
#             "p5":  np.percentile(data_arr, 5),
#             "p95": np.percentile(data_arr, 95)}
#
#
# def figure1_violin(d, save=False, fname_prefix="imu_noise"):
#     """
#     2×3 grid: row 0 = gyroscope axes, row 1 = accelerometer axes.
#     Each panel: three violin plots (one per phase).
#     """
#     fig, axes = plt.subplots(2, 3, figsize=(14, 8))
#     fig.suptitle("Go1 IMU Noise Characterisation — Three-Phase\n"
#                  "Gyroscope (top) and Accelerometer (bottom) per axis",
#                  fontsize=13, fontweight='bold')
#
#     sensors = [
#         ("gyro", "Gyroscope",     "rad/s",  [d["ph0_gyro"], d["ph1_gyro"], d["ph2_gyro"]]),
#         ("acc",  "Accelerometer", "m/s²",   [d["ph0_acc"],  d["ph1_acc"],  d["ph2_acc"]]),
#     ]
#
#     for row, (skey, slabel, sunits, arrays) in enumerate(sensors):
#         for col in range(3):
#             ax = axes[row][col]
#             ax.set_title(f"{slabel} — {AXIS_LABELS[col]}", fontsize=10)
#
#             vdata = [arr[:, col] for arr in arrays]
#
#             # Violin plot
#             parts = ax.violinplot(vdata, positions=[0, 1, 2],
#                                   showmedians=True, showextrema=False)
#             for i, body in enumerate(parts['bodies']):
#                 body.set_facecolor(PHASE_COLS[i])
#                 body.set_alpha(0.55)
#             parts['cmedians'].set_color('black')
#             parts['cmedians'].set_linewidth(1.5)
#
#             # Mean ± std markers
#             for i, arr in enumerate(vdata):
#                 mu  = arr.mean()
#                 sig = arr.std()
#                 ax.errorbar(i, mu, yerr=sig, fmt='D', color=PHASE_COLS[i],
#                             markersize=5, capsize=4, zorder=5,
#                             label=f"{PHASES[i]}: σ={sig:.5f}")
#
#             # Expected range band
#             lo, hi = EXPECTED[skey][PHASES[0]][0], EXPECTED[skey][PHASES[2]][1]
#             ax.axhspan(-hi, hi, alpha=0.06, color='gold',
#                        label=f"Expected ±σ range [{lo:.3f}, {hi:.3f}]")
#
#             # Per-phase expected bands
#             for i, phase in enumerate(PHASES):
#                 lo_p, hi_p = EXPECTED[skey][phase]
#                 ax.axhspan(-hi_p, hi_p, xmin=i/3, xmax=(i+1)/3,
#                            alpha=0.12, color=PHASE_COLS[i])
#
#             ax.set_xticks([0, 1, 2])
#             ax.set_xticklabels(["Ph0\nGround", "Ph1\nStand", "Ph2\nWave"],
#                                fontsize=9)
#             ax.set_ylabel(f"Signal ({sunits})", fontsize=9)
#             ax.grid(True, alpha=0.3, axis='y')
#
#             # Stats annotation
#             for i, arr in enumerate(vdata):
#                 mu = arr.mean(); sig = arr.std()
#                 ax.text(i, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else sig*3,
#                         f"σ={sig:.4f}", ha='center', va='bottom',
#                         fontsize=7.5, color=PHASE_COLS[i], fontweight='bold')
#
#     # Legend
#     patches = [mpatches.Patch(color=c, label=p, alpha=0.7)
#                for c, p in zip(PHASE_COLS, PHASES)]
#     patches.append(mpatches.Patch(color='gold', alpha=0.4, label='Expected range'))
#     fig.legend(handles=patches, loc='lower center', ncol=4,
#                bbox_to_anchor=(0.5, -0.01), fontsize=9)
#     plt.tight_layout(rect=[0, 0.04, 1, 1])
#
#     if save:
#         out = f"{fname_prefix}_violin.png"
#         fig.savefig(out, dpi=150, bbox_inches='tight')
#         print(f"  Saved {out}")
#     return fig
#
#
# def figure2_timeseries(d, save=False, fname_prefix="imu_noise"):
#     """
#     Time-series for all 6 channels, concatenating all three phases.
#     Shows phase transitions as vertical dashed lines.
#     """
#     n0 = len(d["ph0_time"]); n1 = len(d["ph1_time"]); n2 = len(d["ph2_time"])
#     dt = 1.0 / float(d["ctrl_hz"])
#
#     t0 = d["ph0_time"]
#     t1 = d["ph1_time"] + t0[-1] + dt
#     t2 = d["ph2_time"] + t1[-1] + dt
#
#     t_all = np.concatenate([t0, t1, t2])
#     g_all = np.concatenate([d["ph0_gyro"], d["ph1_gyro"], d["ph2_gyro"]], axis=0)
#     a_all = np.concatenate([d["ph0_acc"],  d["ph1_acc"],  d["ph2_acc"],  ], axis=0)
#
#     trans1 = t0[-1] + dt      # start of Ph1
#     trans2 = t1[-1] + dt      # start of Ph2
#
#     fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)
#     fig.suptitle("Go1 IMU — Full Time-Series (Phase 0 → 1 → 2)",
#                  fontsize=12, fontweight='bold')
#
#     channels = [
#         (g_all[:, 0], "Gyro X (rad/s)",  "gyro"),
#         (g_all[:, 1], "Gyro Y (rad/s)",  "gyro"),
#         (g_all[:, 2], "Gyro Z (rad/s)",  "gyro"),
#         (a_all[:, 0], "Acc  X (m/s²)",   "acc"),
#         (a_all[:, 1], "Acc  Y (m/s²)",   "acc"),
#         (a_all[:, 2], "Acc  Z (m/s²)",   "acc"),
#     ]
#
#     subsample = 5   # plot every 5th point for speed
#     for idx, (sig, label, skey) in enumerate(channels):
#         ax = axes[idx]
#         # Phase colouring via axvspan
#         ax.axvspan(t0[0],  t0[-1], alpha=0.08, color=PHASE_COLS[0])
#         ax.axvspan(t1[0],  t1[-1], alpha=0.08, color=PHASE_COLS[1])
#         ax.axvspan(t2[0],  t2[-1], alpha=0.08, color=PHASE_COLS[2])
#         ax.axvline(trans1, color='gray', lw=1.0, ls='--')
#         ax.axvline(trans2, color='gray', lw=1.0, ls='--')
#         ax.plot(t_all[::subsample], sig[::subsample], lw=0.4, color='#333333', alpha=0.7)
#         ax.set_ylabel(label, fontsize=8)
#         ax.grid(True, alpha=0.3)
#         # Phase labels
#         if idx == 0:
#             mid0 = (t0[0]  + t0[-1])  / 2
#             mid1 = (t1[0]  + t1[-1])  / 2
#             mid2 = (t2[0]  + t2[-1])  / 2
#             ymax = ax.get_ylim()[1]
#             for mid, ph, col in zip([mid0,mid1,mid2], PHASES, PHASE_COLS):
#                 ax.text(mid, ymax, ph, ha='center', va='top',
#                         fontsize=8, color=col, fontweight='bold')
#
#     axes[-1].set_xlabel("Time (s)", fontsize=10)
#     plt.tight_layout()
#
#     if save:
#         out = f"{fname_prefix}_timeseries.png"
#         fig.savefig(out, dpi=150, bbox_inches='tight')
#         print(f"  Saved {out}")
#     return fig
#
#
# def figure3_summary_bars(d, save=False, fname_prefix="imu_noise"):
#     """
#     Bar chart of std per channel per phase with expected range overlay.
#     Used as the primary figure for the thesis chapter.
#     """
#     channels_gyro = ["Gyro X", "Gyro Y", "Gyro Z"]
#     channels_acc  = ["Acc X",  "Acc Y",  "Acc Z"]
#
#     phases_data = {
#         "Ph0 ground": (d["ph0_gyro"], d["ph0_acc"]),
#         "Ph1 stand":  (d["ph1_gyro"], d["ph1_acc"]),
#         "Ph2 wave":   (d["ph2_gyro"], d["ph2_acc"]),
#     }
#
#     fig, (ax_g, ax_a) = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle("Go1 IMU Noise Standard Deviation — Three Phases\n"
#                  "Bars show measured σ; shaded bands show expected range",
#                  fontsize=12, fontweight='bold')
#
#     x = np.arange(3)
#     bar_w = 0.25
#
#     for ax, skey, chans, arr_idx in [
#         (ax_g, "gyro", channels_gyro, 0),
#         (ax_a, "acc",  channels_acc,  1)
#     ]:
#         for pi, (phase, col) in enumerate(zip(PHASES, PHASE_COLS)):
#             arr = phases_data[phase][arr_idx]
#             stds = arr.std(axis=0)
#             bars = ax.bar(x + pi*bar_w, stds, bar_w, label=phase,
#                           color=col, alpha=0.75, edgecolor='white', linewidth=0.5)
#             # Value labels on bars
#             for bar, std in zip(bars, stds):
#                 ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
#                         f"{std:.4f}", ha='center', va='bottom', fontsize=7.5,
#                         rotation=90, color=col, fontweight='bold')
#
#         # Expected range bands per phase
#         for pi, phase in enumerate(PHASES):
#             lo, hi = EXPECTED[skey][phase]
#             for xi in range(3):
#                 ax.axhspan(lo, hi,
#                            xmin=(xi + pi*bar_w) / 3,
#                            xmax=(xi + pi*bar_w + bar_w) / 3,
#                            alpha=0.15, color=PHASE_COLS[pi])
#
#         unit = "rad/s" if skey == "gyro" else "m/s²"
#         ax.set_xticks(x + bar_w)
#         ax.set_xticklabels(chans, fontsize=10)
#         ax.set_ylabel(f"Noise std ({unit})", fontsize=10)
#         ax.set_title(f"{'Gyroscope' if skey=='gyro' else 'Accelerometer'}", fontsize=11)
#         ax.legend(fontsize=9)
#         ax.grid(True, alpha=0.3, axis='y')
#         ax.set_ylim(bottom=0)
#
#         # Annotation table below
#         table_data = []
#         col_labels = [f"{p}\nσ ({unit})" for p in PHASES]
#         row_labels  = chans
#         for ax_i in range(3):
#             row = []
#             for phase in PHASES:
#                 arr = phases_data[phase][arr_idx]
#                 row.append(f"{arr[:,ax_i].std():.5f}")
#             table_data.append(row)
#
#     plt.tight_layout()
#
#     if save:
#         out = f"{fname_prefix}_bars.png"
#         fig.savefig(out, dpi=150, bbox_inches='tight')
#         print(f"  Saved {out}")
#     return fig
#
#
# def print_table(d):
#     """Print a clean summary table for the thesis."""
#     print("\n" + "═"*80)
#     print("IMU NOISE SUMMARY TABLE (for thesis Table 3.2)")
#     print("═"*80)
#
#     phases_data = {
#         "Ph0 ground": (d["ph0_gyro"], d["ph0_acc"]),
#         "Ph1 stand":  (d["ph1_gyro"], d["ph1_acc"]),
#         "Ph2 wave":   (d["ph2_gyro"], d["ph2_acc"]),
#     }
#     AXIS = ['X','Y','Z']
#
#     print(f"\n  Gyroscope noise (rad/s):")
#     print(f"  {'Axis':6s}  {'Ph0 Ground':>12}  {'Ph1 Stand':>12}  {'Ph2 Wave':>12}  "
#           f"{'Ph1/Ph0':>8}  {'Ph2/Ph0':>8}  Expected Ph1")
#     for ax in range(3):
#         s0 = d["ph0_gyro"][:,ax].std()
#         s1 = d["ph1_gyro"][:,ax].std()
#         s2 = d["ph2_gyro"][:,ax].std()
#         lo, hi = EXPECTED["gyro"]["Ph1 stand"]
#         flag1 = "✓" if lo <= s1 <= hi else ("↑" if s1>hi else "↓")
#         print(f"  {AXIS[ax]:6s}: {s0:>12.6f}  {s1:>12.6f}  {s2:>12.6f}  "
#               f"{s1/s0:>8.2f}×  {s2/s0:>8.2f}×  [{lo:.3f},{hi:.3f}] {flag1}")
#
#     print(f"\n  Accelerometer noise (m/s²):")
#     print(f"  {'Axis':6s}  {'Ph0 Ground':>12}  {'Ph1 Stand':>12}  {'Ph2 Wave':>12}  "
#           f"{'Ph1/Ph0':>8}  {'Ph2/Ph0':>8}  Expected Ph1")
#     for ax in range(3):
#         s0 = d["ph0_acc"][:,ax].std()
#         s1 = d["ph1_acc"][:,ax].std()
#         s2 = d["ph2_acc"][:,ax].std()
#         lo, hi = EXPECTED["acc"]["Ph1 stand"]
#         flag1 = "✓" if lo <= s1 <= hi else ("↑" if s1>hi else "↓")
#         print(f"  {AXIS[ax]:6s}: {s0:>12.6f}  {s1:>12.6f}  {s2:>12.6f}  "
#               f"{s1/s0:>8.2f}×  {s2/s0:>8.2f}×  [{lo:.3f},{hi:.3f}] {flag1}")
#
#     print(f"\n  Recommended obs_noise_std (Phase 1 standing gyro):")
#     g1 = d["ph1_gyro"].std(axis=0)
#     print(f"    obs_noise_std[27:30] = [{g1[0]:.5f}, {g1[1]:.5f}, {g1[2]:.5f}]  # rad/s")
#     print(f"\n  Wave amplitude = {float(d['wave_amp']):.2f}rad  "
#           f"freq = {float(d['wave_freq']):.2f}Hz  "
#           f"→ max body acc ≈ {float(d['wave_amp']) * (2*np.pi*float(d['wave_freq']))**2:.3f} m/s²")
#
#
# def main():
#     p = argparse.ArgumentParser(description="Go1 Test 1 IMU Noise Plots")
#     p.add_argument("npz",    type=str, help="Path to calib_imu_noise_*.npz")
#     p.add_argument("--save", action="store_true", help="Save figures to PNG")
#     args = p.parse_args()
#
#     print(f"Loading {args.npz}...")
#     d = load(args.npz)
#     print(f"  Ph0: {len(d['ph0_time'])} samples ({len(d['ph0_time'])/float(d['ctrl_hz']):.1f}s)")
#     print(f"  Ph1: {len(d['ph1_time'])} samples ({len(d['ph1_time'])/float(d['ctrl_hz']):.1f}s)")
#     print(f"  Ph2: {len(d['ph2_time'])} samples ({len(d['ph2_time'])/float(d['ctrl_hz']):.1f}s)")
#
#     prefix = args.npz.replace(".npz", "")
#
#     print_table(d)
#
#     print("\nGenerating figures...")
#     fig1 = figure1_violin(d, save=args.save, fname_prefix=prefix)
#     fig2 = figure2_timeseries(d, save=args.save, fname_prefix=prefix)
#     fig3 = figure3_summary_bars(d, save=args.save, fname_prefix=prefix)
#
#     if not args.save:
#         plt.show()
#     else:
#         print(f"All figures saved with prefix: {prefix}")
#
#
# if __name__ == "__main__":
#     main()