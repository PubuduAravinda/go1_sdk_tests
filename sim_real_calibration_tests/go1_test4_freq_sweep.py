#!/usr/bin/env python3
"""
Go1 Test 5 — Frequency Sweep (Tan et al. RSS 2018)
====================================================

WHAT THIS TEST MEASURES:
    Actuator bandwidth — how tracking ratio ρ degrades as command
    frequency increases. Equivalent to a Bode magnitude plot for
    each joint at training-level PD gains.

    ρ(f) = σ(q_actual) / σ(q_cmd)  at frequency f

    At low f (0.5 Hz):   ρ ≈ 0.90   joint tracks well
    At mid f (2–4 Hz):   ρ drops     joint starts lagging
    At high f (8+ Hz):   ρ → 0       joint can't follow

    The -3dB bandwidth is where ρ = 0.707 × ρ(0.5 Hz).
    This is the fastest motion the policy can reliably command.

WHY THIS MATCHES TAN ET AL.:
    Tan et al. RSS 2018 Section IV-B:
    "We sweep the command frequency from 0.5 to 10 Hz at fixed
    amplitude, measuring the amplitude ratio of actual to commanded
    motion. The ratio decreases with frequency, revealing the
    effective bandwidth of each actuator at training gains."

    They fix KP at training value, fix amplitude at a moderate level,
    sweep frequency. That is exactly what this script does.

WHAT THE RESULT IS USED FOR IN SIM:
    The frequency response directly parameterises the first-order lag
    filter. The lag time constant τ can be read from the Bode plot:
        τ = 1 / (2π × f_3dB)
    where f_3dB is the -3dB bandwidth frequency.
    This gives a physics-based α rather than purely from latency:
        α = Δt / (Δt + τ) = 0.02 / (0.02 + τ)
    Cross-checking with Test 3 (spike latency) validates both.

SAFE SETUP:
    Robot HANGING on safety rack. Feet clear of ground.
    Amplitude fixed at 0.10 rad — moderate, safe at all frequencies.
    Max torque: 80 × 0.10 = 8.0 Nm (knee) — well within limits.
    All other joints held soft while target joint sweeps.

FREQUENCIES TESTED (rad from Tan et al.):
    [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] Hz
    Low end captures baseline ratio.
    High end captures rolloff and bandwidth limit.

RUN:
    python3 go1_test5_freq_sweep.py                    # all 12 joints
    python3 go1_test5_freq_sweep.py --joint 6          # RL_th only
    python3 go1_test5_freq_sweep.py --joint 0 4 8      # one per group
    python3 go1_test5_freq_sweep.py --n_cycles 8       # more cycles per freq
    python3 go1_test5_freq_sweep.py --amp 0.08         # smaller if needed
"""

import argparse, time, sys
import numpy as np
from datetime import datetime
import robot_interface as sdk

# ─── SDK setup ────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

CTRL_HZ = 500
DT      = 1.0 / CTRL_HZ   # 2 ms

# ─── Joint ordering ───────────────────────────────────────────────────────────
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]

JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

# ─── Default pose ─────────────────────────────────────────────────────────────
DEFAULT_Q_HW = np.array([
     0.1, -0.1,  0.1, -0.1,   # hips (FR/RR sign-flipped for HW encoder)
     0.8,  0.8,  0.8,  0.8,   # thighs
    -1.5, -1.5, -1.5, -1.5,   # knees
], np.float32)

# ─── Gains ────────────────────────────────────────────────────────────────────
# Training gains — FIXED throughout (as in Tan et al.)
KP_TRAIN = np.array([35, 35, 35, 35,
                     65, 65, 65, 65,
                     80, 80, 80, 80], np.float32)
KD_TRAIN = np.array([ 4,  4,  4,  4,
                     4.5,4.5,4.5,4.5,
                      5,  5,  5,  5], np.float32)

# Soft hold gains for non-target joints and inter-trial settle
KP_HOLD  = np.array([ 8,  8,  8,  8,
                      12, 12, 12, 12,
                      15, 15, 15, 15], np.float32)
KD_HOLD  = np.array([ 3,  3,  3,  3,
                       3,  3,  3,  3,
                       4,  4,  4,  4], np.float32)

# ─── Test parameters ──────────────────────────────────────────────────────────
# Frequencies from Tan et al. RSS 2018 — low to high
FREQS_HZ   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Fixed amplitude (Tan et al. use a moderate amplitude — enough torque
# to move clearly, small enough to be safe at 10 Hz)
# 0.10 rad gives: hip 3.5Nm, thigh 6.5Nm, knee 8.0Nm — safe at all freqs
AMP_DEFAULT = 0.10   # rad — override with --amp if needed

# Cycles per frequency: enough for stable steady-state statistics
# Tan et al. use 5 cycles. We use 6 (first discarded as transient = 5 clean).
N_CYCLES    = 6
N_DISCARD   = 1      # first cycle discarded (transient)

# Settle time between frequency steps
SETTLE_S    = 0.5    # seconds

# Phase lag measurement: use cross-correlation
MEASURE_PHASE = True


# ─── Core SDK helpers ─────────────────────────────────────────────────────────

def read_state():
    """Returns jpos and jvel in Isaac order."""
    udp.Recv()
    udp.GetRecv(state)
    jpos = np.array([state.motorState[i].q  for i in range(12)],
                    np.float32)[sdk_to_isaac]
    jvel = np.array([state.motorState[i].dq for i in range(12)],
                    np.float32)[sdk_to_isaac]
    return jpos, jvel


def send_cmd(target_q_hw, kp, kd):
    """Send one 500 Hz command step. All arrays in Isaac order."""
    udp.Recv()
    udp.GetRecv(state)
    for isaac_i in range(12):
        sdk_i = sdk_to_isaac[isaac_i]
        cmd.motorCmd[sdk_i].mode = 0x0A
        cmd.motorCmd[sdk_i].q    = float(target_q_hw[isaac_i])
        cmd.motorCmd[sdk_i].dq   = 0.0
        cmd.motorCmd[sdk_i].Kp   = float(kp[isaac_i])
        cmd.motorCmd[sdk_i].Kd   = float(kd[isaac_i])
        cmd.motorCmd[sdk_i].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()


def hold_default(duration_s):
    """Hold all joints at DEFAULT_Q_HW with soft gains."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        t_s = time.perf_counter()
        send_cmd(DEFAULT_Q_HW, KP_HOLD, KD_HOLD)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0:
            time.sleep(sl)


# ─── Single frequency trial ───────────────────────────────────────────────────

def run_frequency_trial(joint_idx, freq_hz, amplitude, n_cycles, n_discard):
    """
    Run one sinusoidal trial at a given frequency.

    Returns:
        ratio       — amplitude ratio σ(actual) / σ(cmd) on steady-state window
        phase_lag   — lag in ms (from cross-correlation)
        cmd_signal  — full commanded signal (n_steps,)
        act_signal  — full actual position offset (n_steps,)
        t_axis_ms   — time axis in ms
    """
    n_steps  = int(n_cycles / freq_hz * CTRL_HZ)
    t_arr    = np.arange(n_steps) * DT
    cmd_sig  = amplitude * np.sin(2.0 * np.pi * freq_hz * t_arr)

    # Build gain arrays: training KP for target joint, soft for rest
    kp_arr = KP_HOLD.copy()
    kd_arr = KD_HOLD.copy()
    kp_arr[joint_idx] = KP_TRAIN[joint_idx]
    kd_arr[joint_idx] = KD_TRAIN[joint_idx]

    act_q = np.zeros(n_steps, np.float32)

    for step in range(n_steps):
        t_s = time.perf_counter()
        jpos, _ = read_state()
        target  = DEFAULT_Q_HW.copy()
        target[joint_idx] += cmd_sig[step]
        send_cmd(target, kp_arr, kd_arr)
        act_q[step] = jpos[joint_idx] - DEFAULT_Q_HW[joint_idx]
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0:
            time.sleep(sl)

    # Discard transient cycles
    skip     = int(n_discard / freq_hz * CTRL_HZ)
    cs       = cmd_sig[skip:]
    qs       = act_q[skip:]

    # Tracking ratio (Tan et al. definition)
    ratio    = np.std(qs) / (np.std(cs) + 1e-9)

    # Phase lag via cross-correlation (positive = actual lags command)
    corr     = np.correlate(qs - qs.mean(), cs - cs.mean(), mode='full')
    lag_idx  = np.argmax(corr) - (len(qs) - 1)   # negative = actual lags
    phase_ms = -lag_idx * DT * 1000.0             # convert to ms (positive = lag)

    return ratio, phase_ms, cmd_sig, act_q, t_arr * 1000.0


# ─── Per-joint frequency sweep ────────────────────────────────────────────────

def sweep_joint(joint_idx, amplitude, n_cycles, n_discard, freqs):
    """
    Run the full frequency sweep for one joint.
    Returns dict: freq_hz → {ratio, phase_ms, cmd, act, t_ms}
    """
    jname = JNAMES[joint_idx]
    kp    = float(KP_TRAIN[joint_idx])
    print(f"\n  ══ {jname}  KP={kp:.0f}  amp={amplitude:.2f} rad  "
          f"τ_max={kp*amplitude:.1f} Nm ══")
    print(f"  {'Freq':>6}  {'Ratio':>7}  {'Std_r':>7}  {'Phase_ms':>9}  "
          f"{'Torque':>8}  Status")
    print(f"  {'──────':>6}  {'──────':>7}  {'──────':>7}  {'────────':>9}  "
          f"{'───────':>8}  ──────")

    results = {}

    for freq in freqs:
        # 3 trials per frequency for repeatability
        ratios  = []
        phases  = []
        cmd_s   = None
        act_s   = None
        t_s_arr = None

        for trial in range(3):
            hold_default(SETTLE_S)
            r, ph, cmd_sig, act_q, t_ms = run_frequency_trial(
                joint_idx, freq, amplitude, n_cycles, n_discard)
            ratios.append(r)
            phases.append(ph)
            if trial == 0:
                cmd_s   = cmd_sig
                act_s   = act_q
                t_s_arr = t_ms

        r_mean = np.mean(ratios)
        r_std  = np.std(ratios)
        p_mean = np.mean(phases)
        tau    = kp * amplitude

        # Status classification
        if r_mean > 0.80:
            status = "✓ good"
        elif r_mean > 0.60:
            status = "~ partial"
        elif r_mean > 0.30:
            status = "! degraded"
        else:
            status = "✗ rolled off"

        print(f"  {freq:>6.1f}  {r_mean:>7.3f}  {r_std:>7.3f}  "
              f"{p_mean:>9.1f}  {tau:>8.1f}  {status}")

        results[freq] = {
            'ratio_mean': r_mean,
            'ratio_std':  r_std,
            'phase_ms':   p_mean,
            'all_ratios': np.array(ratios),
            'cmd':        cmd_s,
            'act':        act_s,
            't_ms':       t_s_arr,
        }

    # Compute bandwidth: freq where ratio drops to 0.707 × ratio at 0.5 Hz
    r_ref    = results[freqs[0]]['ratio_mean']
    bw_thr   = r_ref * 0.707
    bw_freq  = None
    for i in range(len(freqs) - 1):
        r_lo = results[freqs[i  ]]['ratio_mean']
        r_hi = results[freqs[i+1]]['ratio_mean']
        if r_lo >= bw_thr >= r_hi:
            # Linear interpolation
            frac     = (r_lo - bw_thr) / (r_lo - r_hi + 1e-9)
            bw_freq  = freqs[i] + frac * (freqs[i+1] - freqs[i])
            break

    if bw_freq is not None:
        tau_bw = 1.0 / (2.0 * np.pi * bw_freq)
        alpha  = 0.02 / (0.02 + tau_bw)
        print(f"\n  ── {jname} Bandwidth ──")
        print(f"  Baseline ratio (0.5Hz): {r_ref:.3f}")
        print(f"  -3dB threshold:         {bw_thr:.3f}")
        print(f"  Bandwidth frequency:    {bw_freq:.2f} Hz")
        print(f"  Lag time constant τ:    {tau_bw*1000:.1f} ms")
        print(f"  Derived α:              {alpha:.4f}")
        results['bandwidth_hz']  = bw_freq
        results['tau_bw_ms']     = tau_bw * 1000.0
        results['alpha_bw']      = alpha
    else:
        r_max   = results[freqs[-1]]['ratio_mean']
        tau_approx = 1.0 / (2.0 * np.pi * freqs[-1])
        alpha_approx = 0.02 / (0.02 + tau_approx)
        if r_max > bw_thr:
            print(f"\n  Bandwidth > {freqs[-1]} Hz (joint faster than sweep range)")
        else:
            print(f"\n  Bandwidth < {freqs[0]} Hz (joint very slow — fault?)")
        results['bandwidth_hz'] = None
        results['tau_bw_ms']    = None
        results['alpha_bw']     = None

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def test_freq_sweep(joint_indices, amplitude, n_cycles, n_discard, freqs):
    print("\n═══ TEST 5: Frequency Sweep (Tan et al. RSS 2018) ═══")
    print()
    print("  CONCEPT:")
    print("  ─────────────────────────────────────────────────────────")
    print("  Sinusoidal command at fixed amplitude, swept frequency.")
    print("  Tracking ratio ρ = σ(actual) / σ(commanded) at each freq.")
    print("  ρ starts near 0.90 at low freq and drops as frequency rises.")
    print("  The -3dB bandwidth (ρ = 0.707 × ρ_baseline) gives τ for α.")
    print()
    print("  WHAT THIS IS FROM:")
    print("  Tan et al. RSS 2018, Section IV-B:")
    print("  Frequency sweep at training KP, fixed amplitude, measures")
    print("  effective actuator bandwidth for sim lag parameterisation.")
    print()
    print(f"  Joints:    {[JNAMES[i] for i in joint_indices]}")
    print(f"  Amplitude: {amplitude:.2f} rad (fixed throughout)")
    print(f"  Freqs:     {freqs} Hz")
    print(f"  Cycles:    {n_cycles} per freq ({n_discard} discarded as transient)")
    print(f"  Trials:    3 per frequency point")
    print()
    print(f"  Max torque per group:")
    print(f"    Hip   (KP=35): {35*amplitude:.1f} Nm at {amplitude:.2f} rad")
    print(f"    Thigh (KP=65): {65*amplitude:.1f} Nm at {amplitude:.2f} rad")
    print(f"    Knee  (KP=80): {80*amplitude:.1f} Nm at {amplitude:.2f} rad")
    print()
    est_mins = len(joint_indices) * len(freqs) * 3 * (n_cycles + 0.5) / 60.0
    print(f"  Estimated time: {est_mins:.0f} minutes")
    print()
    input("  Robot HANGING on rack? Press Enter → ")

    print("\n  Soft ramp to DEFAULT_Q_HW (3s)...")
    hold_default(3.0)
    print("  Ready.\n")

    all_results  = {}
    all_signals  = {}

    for ji in joint_indices:
        jname  = JNAMES[ji]
        res    = sweep_joint(ji, amplitude, n_cycles, n_discard, freqs)
        all_results[jname] = res

        # Store signals for first freq and bandwidth freq for plotting
        sigs = {}
        for freq in freqs:
            if freq in res and isinstance(res[freq], dict):
                sigs[f"cmd_{freq:.1f}Hz"] = res[freq]['cmd']
                sigs[f"act_{freq:.1f}Hz"] = res[freq]['act']
                sigs[f"t_{freq:.1f}Hz"]   = res[freq]['t_ms']
        all_signals[jname] = sigs

    # ── Cross-joint summary ────────────────────────────────────────────────────
    print("\n\n" + "═"*80)
    print("FINAL SUMMARY — Frequency Sweep at Training KP")
    print("═"*80)

    header = f"  {'Joint':8s}  {'KP':>4}"
    for f in freqs:
        header += f"  {f:.1f}Hz".rjust(8)
    header += f"  {'BW_Hz':>7}  {'tau_ms':>7}  {'alpha':>7}"
    print(header)
    print("  " + "─"*78)

    for ji in joint_indices:
        jname = JNAMES[ji]
        kp    = float(KP_TRAIN[ji])
        res   = all_results.get(jname, {})
        row   = f"  {jname:8s}  {kp:>4.0f}"
        for f in freqs:
            r = res.get(f, {}).get('ratio_mean', float('nan'))
            row += f"  {r:>7.3f}"
        bw    = res.get('bandwidth_hz')
        tau   = res.get('tau_bw_ms')
        alpha = res.get('alpha_bw')
        row += f"  {str(round(bw,2)) if bw else 'N/A':>7}"
        row += f"  {str(round(tau,1)) if tau else 'N/A':>7}"
        row += f"  {str(round(alpha,4)) if alpha else 'N/A':>7}"
        print(row)

    print()
    print("── Simulation parameters ──")
    print("# Bandwidth-derived lag coefficients (cross-check with Test 3 spike):")
    for ji in joint_indices:
        jname = JNAMES[ji]
        alpha = all_results.get(jname, {}).get('alpha_bw')
        bw    = all_results.get(jname, {}).get('bandwidth_hz')
        tau   = all_results.get(jname, {}).get('tau_bw_ms')
        if alpha:
            print(f"  {jname:8s}: BW={bw:.2f}Hz  τ={tau:.1f}ms  "
                  f"α={alpha:.4f}")
        else:
            print(f"  {jname:8s}: bandwidth outside sweep range — check manually")

    # ── Save ──────────────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    save = {}
    save['joint_names'] = np.array([JNAMES[i] for i in joint_indices],
                                   dtype=object)
    save['freqs_hz']    = np.array(freqs)
    save['amplitude']   = np.array([amplitude])
    save['kp_train']    = KP_TRAIN
    save['n_cycles']    = np.array([n_cycles])
    save['n_discard']   = np.array([n_discard])

    for jname, res in all_results.items():
        for freq in freqs:
            if freq in res and isinstance(res[freq], dict):
                k = f"{jname}_{freq:.1f}Hz"
                save[f"{k}_ratio_mean"]  = np.array([res[freq]['ratio_mean']])
                save[f"{k}_ratio_std"]   = np.array([res[freq]['ratio_std']])
                save[f"{k}_phase_ms"]    = np.array([res[freq]['phase_ms']])
                save[f"{k}_all_ratios"]  = res[freq]['all_ratios']
                save[f"{k}_cmd"]         = res[freq]['cmd']
                save[f"{k}_act"]         = res[freq]['act']
                save[f"{k}_t_ms"]        = res[freq]['t_ms']

        for key in ['bandwidth_hz', 'tau_bw_ms', 'alpha_bw']:
            val = res.get(key)
            save[f"{jname}_{key}"] = np.array([val if val is not None else np.nan])

    fname = f"calib_freq_sweep_{ts}.npz"
    np.savez(fname, **save)
    kb = sum(v.nbytes for v in save.values() if hasattr(v, 'nbytes')) / 1024
    print(f"\n→ Saved {fname}  ({kb:.0f} KB)")
    print(f"→ Plot: python3 go1_test5_freq_sweep_plot.py {fname}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Go1 Test 5 — Frequency Sweep (Tan et al. RSS 2018)")
    p.add_argument("--joint",    type=int, nargs="+", default=None,
                   help="Joint indices 0-11 (Isaac order). Default: all 12.")
    p.add_argument("--amp",      type=float, default=AMP_DEFAULT,
                   help=f"Sine amplitude in rad (default: {AMP_DEFAULT})")
    p.add_argument("--n_cycles", type=int,   default=N_CYCLES,
                   help=f"Cycles per frequency (default: {N_CYCLES})")
    p.add_argument("--freqs",    type=float, nargs="+", default=FREQS_HZ,
                   help="Frequency list in Hz")
    args = p.parse_args()

    joints = args.joint if args.joint is not None else list(range(12))
    for j in joints:
        if not 0 <= j <= 11:
            print(f"Joint index {j} out of range 0-11")
            sys.exit(1)

    # Safety check on amplitude
    max_torque = max(KP_TRAIN[j] * args.amp for j in joints)
    if max_torque > 24.0:
        print(f"WARNING: amplitude {args.amp:.2f} rad gives "
              f"{max_torque:.1f} Nm — close to Go1 limit (23.7 Nm for knee)")
        print("Consider --amp 0.08 for safety")
        resp = input("Continue anyway? (y/N): ")
        if resp.lower() != 'y':
            sys.exit(0)

    print("Connecting to Go1...")
    udp.Recv()
    udp.GetRecv(state)
    print("Connected.")

    # Verify joint mapping
    SDK = ['FR_hip','FR_th','FR_kn','FL_hip','FL_th','FL_kn',
           'RR_hip','RR_th','RR_kn','RL_hip','RL_th','RL_kn']
    ok  = all(SDK[sdk_to_isaac[i]] == JNAMES[i] for i in range(12))
    print(f"Joint mapping: {'✓ correct' if ok else '✗ ERROR'}")
    if not ok:
        sys.exit(1)

    test_freq_sweep(
        joint_indices = joints,
        amplitude     = args.amp,
        n_cycles      = args.n_cycles,
        n_discard     = N_DISCARD,
        freqs         = args.freqs,
    )