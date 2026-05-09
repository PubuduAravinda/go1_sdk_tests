#!/usr/bin/env python3
"""
Go1 Test 4c — Amplitude Ramp (Torque Threshold Sweep)
======================================================

MOTIVATION from RL_th result in Test 4b:
    At KP=10-60, amp=0.05 → τ_max=0.5-3.0 Nm  → ratio=0.001 (no movement)
    At KP=65,    amp=0.05 → τ_max=3.25 Nm      → ratio=0.548±0.399 (intermittent)

    The ±0.399 std is the key: motor SOMETIMES moves, SOMETIMES doesn't.
    This means there's a TORQUE THRESHOLD — a minimum force needed to break free.
    Below threshold: motor locked (stiction or gear jam)
    Above threshold: motor moves (intermittently because threshold is near τ_max)

THIS TEST:
    Holds KP fixed at training KP for each joint.
    Sweeps amplitude: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 rad
    Effective torque = KP × amplitude
    Finds the threshold: smallest amplitude where ratio > 0.10

    For RL_th (KP=65):
      amp=0.05 → τ=3.25 Nm → intermittent (ratio=0.548±0.399)
      amp=0.10 → τ=6.50 Nm → should be more consistent
      amp=0.15 → τ=9.75 Nm → should be reliable
      amp=0.20 → τ=13.0 Nm → should always work

    If ratio is STILL intermittent at amp=0.30 → τ=19.5 Nm:
      → Electrical fault, not mechanical stiction
      → 100% masking in sim

    If ratio converges to >0.80 at amp=0.15:
      → Gear stiction — motor is mechanically jammed
      → Sim fix: reduce initial joint DR, use larger action scale
      → Real fix: physically check/lubricate RL thigh joint

ALSO RUN FOR FR_hip:
    FR_hip showed normal latency (16ms) but high encoder noise.
    Amplitude ramp reveals whether it has torque-dependent noise:
      Linear noise (same at all amplitudes): electronic issue
      Amplitude-dependent noise: mechanical (gear backlash under load)

SAFE SETUP: robot hanging, amplitude goes up to 0.30 rad max.
    Max torque at KP=80 (knee), amp=0.30: 80×0.30 = 24 Nm
    This is AT the Go1 knee limit (23.7 Nm). Use amp_max=0.25 for knees.
    Hips/thighs: limit is higher, 0.30 is safe.

Run:
    python3 go1_test4c_amp_ramp.py --joint 6          # RL_th only (broken)
    python3 go1_test4c_amp_ramp.py --joint 1 6        # FR_hip + RL_th
    python3 go1_test4c_amp_ramp.py --joint 0 1 2 4 6  # full diagnostic
    python3 go1_test4c_amp_ramp.py                     # all 12 joints
"""

import argparse, time, sys
import numpy as np
from datetime import datetime
import robot_interface as sdk

# ─── SDK ──────────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

CTRL_HZ = 500
DT      = 1.0 / CTRL_HZ

sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]

JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

DEFAULT_Q_HW = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  0.8,  0.8,
    -1.5, -1.5, -1.5, -1.5,
], np.float32)

# Training KP/KD — FIXED for this test (amplitude is the variable)
KP_TRAIN = np.array([35,35,35,35,  65,65,65,65,  80,80,80,80], np.float32)
KD_TRAIN = np.array([ 4, 4, 4, 4, 4.5,4.5,4.5,4.5,  5, 5, 5, 5], np.float32)

KP_HOLD = np.array([ 8, 8, 8, 8,  12,12,12,12,  15,15,15,15], np.float32)
KD_HOLD = np.array([ 3, 3, 3, 3,   3, 3, 3, 3,   4, 4, 4, 4], np.float32)

# Amplitude ramp levels (rad) — torque = KP_TRAIN × amplitude
AMPS_HIP   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # max τ = 35×0.30 = 10.5 Nm
AMPS_THIGH = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # max τ = 65×0.30 = 19.5 Nm
AMPS_KNEE  = [0.05, 0.10, 0.15, 0.20, 0.25]         # max τ = 80×0.25 = 20.0 Nm (safe)

AMP_SETS = {
    **{i: AMPS_HIP   for i in range(0, 4)},
    **{i: AMPS_THIGH for i in range(4, 8)},
    **{i: AMPS_KNEE  for i in range(8, 12)},
}

SWEEP_FREQ   = 2.0   # Hz — walking frequency, most relevant
SWEEP_CYCLES = 5     # cycles per amplitude level
N_TRIALS     = 5     # trials per amplitude (to get reliable std)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_state():
    udp.Recv(); udp.GetRecv(state)
    jpos = np.array([state.motorState[i].q      for i in range(12)], np.float32)[sdk_to_isaac]
    jtau = np.array([state.motorState[i].tauEst for i in range(12)], np.float32)[sdk_to_isaac]
    return jpos, jtau


def send_cmd(target_q, kp, kd):
    udp.Recv(); udp.GetRecv(state)
    for i in range(12):
        s = sdk_to_isaac[i]
        cmd.motorCmd[s].mode = 0x0A
        cmd.motorCmd[s].q    = float(target_q[i])
        cmd.motorCmd[s].dq   = 0.0
        cmd.motorCmd[s].Kp   = float(kp[i])
        cmd.motorCmd[s].Kd   = float(kd[i])
        cmd.motorCmd[s].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def hold(duration_s):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        t_s = time.perf_counter()
        send_cmd(DEFAULT_Q_HW, KP_HOLD, KD_HOLD)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


def run_amplitude_trial(joint_idx, amplitude, freq=SWEEP_FREQ):
    """
    One amplitude trial at training KP.
    Returns amplitude_ratio, noise_std, peak_tau, actual signal, cmd signal.
    """
    n_steps = int(SWEEP_CYCLES / freq * CTRL_HZ)
    t_arr   = np.arange(n_steps) * DT
    cmd_sig = amplitude * np.sin(2 * np.pi * freq * t_arr)

    # Use training KP for target joint, soft hold for all others
    kp_arr = KP_HOLD.copy()
    kd_arr = KD_HOLD.copy()
    kp_arr[joint_idx] = KP_TRAIN[joint_idx]
    kd_arr[joint_idx] = KD_TRAIN[joint_idx]

    act_q   = np.zeros(n_steps, np.float32)
    act_tau = np.zeros(n_steps, np.float32)

    for step in range(n_steps):
        t_s = time.perf_counter()
        jpos, jtau = read_state()
        target = DEFAULT_Q_HW.copy()
        target[joint_idx] += cmd_sig[step]
        send_cmd(target, kp_arr, kd_arr)
        act_q[step]   = jpos[joint_idx] - DEFAULT_Q_HW[joint_idx]
        act_tau[step] = jtau[joint_idx]
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    # Skip first cycle (transient)
    skip = int(1.0 / freq * CTRL_HZ)
    cs   = cmd_sig[skip:]
    qs   = act_q[skip:]

    amp_ratio  = np.std(qs) / (np.std(cs) + 1e-9)
    noise_std  = np.std(qs - cs * amp_ratio)  # residual noise after tracking
    peak_tau   = float(np.abs(act_tau).max())

    return amp_ratio, noise_std, peak_tau, act_q, cmd_sig


# ─── Main test ────────────────────────────────────────────────────────────────

def test_amp_ramp(joint_indices, n_trials=N_TRIALS):
    print("\n═══ TEST 4c: Amplitude Ramp (Torque Threshold Sweep) ═══")
    print()
    print("  KP fixed at TRAINING KP. Amplitude swept: 0.05 → 0.30 rad")
    print("  Effective torque = KP_train × amplitude")
    print()
    print("  For RL_th (KP=65):")
    for amp in AMPS_THIGH:
        tau = 65 * amp
        print(f"    amp={amp:.2f} → τ = 65 × {amp:.2f} = {tau:.1f} Nm")
    print()
    print("  Expected behaviours:")
    print("    Healthy:     ratio rises smoothly to ~0.90 from amp=0.05")
    print("    Stiction:    ratio=0.001 at low amp, jumps to >0.80 above threshold")
    print("    Electrical:  ratio stays ~0.001 at ALL amplitudes")
    print("    Noisy(FR):   ratio OK but noise_std grows with amplitude")
    print()
    input("  Robot HANGING. Press Enter to start → ")

    print("\n  Soft ramp to DEFAULT_Q_HW (3s)...")
    hold(3.0)
    print("  Ready.\n")

    all_results = {}
    all_signals = {}

    for ji in joint_indices:
        jname   = JNAMES[ji]
        amps    = AMP_SETS[ji]
        kp      = float(KP_TRAIN[ji])
        print(f"\n  ══ {jname}  KP={kp:.0f} ══")
        print(f"  Max torque: {kp * max(amps):.1f} Nm  at amp={max(amps):.2f} rad")

        jresults = {}
        jsignals = {}

        for amp in amps:
            tau_max = kp * amp
            ratios = []; noise_stds = []; taus = []
            signals_act = []

            for trial in range(n_trials):
                hold(0.3)
                ratio, noise_std, peak_tau, act_q, cmd_sig = \
                    run_amplitude_trial(ji, amp)
                ratios.append(ratio)
                noise_stds.append(noise_std)
                taus.append(peak_tau)
                signals_act.append(act_q)

            r_mean  = np.mean(ratios)
            r_std   = np.std(ratios)
            n_mean  = np.mean(noise_stds)
            t_mean  = np.mean(taus)

            # Classify this amplitude level
            if r_mean > 0.80 and r_std < 0.05:
                char = "✓ tracking well"
            elif r_mean > 0.80 and r_std >= 0.05:
                char = "~ tracking but noisy"
            elif r_mean > 0.40:
                char = "~ partial tracking"
            elif r_mean > 0.10:
                char = "! weak response"
            else:
                char = "✗ no response"

            print(f"    amp={amp:.2f} τ={tau_max:5.1f}Nm: "
                  f"ratio={r_mean:.3f}±{r_std:.3f}  "
                  f"noise={n_mean:.4f}  "
                  f"τ_peak={t_mean:.2f}Nm  {char}")

            jresults[amp] = {
                "ratio_mean": r_mean, "ratio_std": r_std,
                "noise_std": n_mean,  "tau_mean": t_mean,
                "tau_max_cmd": tau_max,
                "all_ratios": np.array(ratios),
            }
            jsignals[f"amp{amp:.2f}_cmd"] = cmd_sig
            jsignals[f"amp{amp:.2f}_act"] = signals_act[0]

        all_results[jname] = jresults
        all_signals[jname] = jsignals

        # ── Diagnosis ────────────────────────────────────────────────────────
        print(f"\n  ── {jname} Diagnosis ──")

        # Find torque threshold
        threshold_amp = None
        threshold_tau = None
        for amp in amps:
            r = jresults[amp]
            if r["ratio_mean"] > 0.40:
                threshold_amp = amp
                threshold_tau = r["tau_max_cmd"]
                break

        # Check if noise grows with amplitude (mechanical vs electronic)
        low_noise  = np.mean([jresults[a]["noise_std"] for a in amps[:2]])
        high_noise = np.mean([jresults[a]["noise_std"] for a in amps[-2:]])
        noise_grows = high_noise > low_noise * 2.0

        # Ratio at max amplitude
        ratio_max_amp = jresults[max(amps)]["ratio_mean"]
        std_max_amp   = jresults[max(amps)]["ratio_std"]

        if ratio_max_amp < 0.05:
            # Never responded at any amplitude
            print(f"  ELECTRICAL FAULT: ratio<0.05 at ALL amplitudes (max τ={kp*max(amps):.0f}Nm)")
            print(f"  → Motor not responding to any torque command")
            print(f"  → 100% masking in sim")
            print(f"  → Check motor driver/cable before replacement")
            sim_action = "100% MASK"
            sim_kp_lo  = 0.0

        elif threshold_amp is not None and threshold_tau is not None and ratio_max_amp > 0.70:
            # Stiction — needs threshold torque then works fine
            print(f"  STICTION/GEAR JAM: needs τ > {threshold_tau:.1f} Nm to move")
            print(f"  → Below {threshold_amp:.2f} rad at KP={kp:.0f}: motor locked")
            print(f"  → Above {threshold_amp:.2f} rad: motor tracks {'well' if std_max_amp<0.1 else 'but inconsistently'}")
            print(f"  → Sim fix: action scale large enough that commands exceed threshold")
            print(f"  → Real fix: physically check/clean RL thigh joint housing")
            sim_action = f"mask at small actions (<{threshold_amp:.2f}rad)"
            sim_kp_lo  = 0.40

        elif ratio_max_amp > 0.40 and std_max_amp > 0.20:
            # Intermittent — sometimes works
            dropout_rate = np.mean([jresults[a]["ratio_mean"] < 0.10 for a in amps])
            print(f"  INTERMITTENT DROPOUT: ratio inconsistent (std={std_max_amp:.3f})")
            print(f"  → Motor sometimes responds, sometimes doesn't")
            print(f"  → Kim masking: randomise between 0 and normal torque per episode")
            print(f"  → Dropout rate ≈ {dropout_rate*100:.0f}% — use this as masking probability")
            sim_action = f"Kim mask, p={dropout_rate:.2f}"
            sim_kp_lo  = 0.40

        else:
            # Noise character
            if noise_grows:
                print(f"  MECHANICAL NOISE: noise grows with amplitude ({low_noise:.4f}→{high_noise:.4f})")
                print(f"  → Gear backlash under load — normal for worn gears")
                print(f"  → Sim fix: KP DR [0.70, 1.20] + obs_noise_std ×1.5")
                sim_action = "KP DR + noise"
                sim_kp_lo  = 0.70
            else:
                print(f"  HEALTHY: ratio={ratio_max_amp:.3f} at max amplitude")
                print(f"  → Electronic noise only (not amplitude-dependent)")
                lo = max(0.75, ratio_max_amp - 0.15)
                hi = min(1.25, ratio_max_amp + 0.15)
                print(f"  → KP DR range: [{lo:.2f}, {hi:.2f}]")
                sim_action = f"KP DR [{max(0.75,ratio_max_amp-0.15):.2f},{min(1.25,ratio_max_amp+0.15):.2f}]"
                sim_kp_lo  = max(0.75, ratio_max_amp - 0.15)

        all_results[jname]["diagnosis"] = sim_action
        all_results[jname]["kp_lo_rec"] = sim_kp_lo

    # ══ Cross-joint summary ════════════════════════════════════════════════════
    print("\n\n" + "═"*70)
    print("FINAL SUMMARY — Amplitude Ramp at Training KP")
    print("═"*70)
    print(f"  {'joint':8s}  {'KP':>5}  {'@0.05rad':>9}  {'@max_amp':>9}  {'std@max':>8}  diagnosis")
    print(f"  {'─────':8s}  {'──':>5}  {'────────':>9}  {'────────':>9}  {'───────':>8}  ─────────")

    for ji in joint_indices:
        jname = JNAMES[ji]
        amps  = AMP_SETS[ji]
        jres  = all_results.get(jname, {})
        kp    = float(KP_TRAIN[ji])

        r_low = jres.get(0.05, {}).get("ratio_mean", float("nan"))
        r_hi  = jres.get(max(amps), {}).get("ratio_mean", float("nan"))
        s_hi  = jres.get(max(amps), {}).get("ratio_std", float("nan"))
        diag  = jres.get("diagnosis", "?")

        print(f"  {jname:8s}: {kp:>5.0f}  {r_low:>9.3f}  {r_hi:>9.3f}  {s_hi:>8.3f}  {diag}")

    # ── go1_env.py update block ────────────────────────────────────────────────
    print("\n── go1_env.py update ──\n")
    print("# Per-joint sim parameters from Test 4c amplitude ramp")
    print("# Paste into Go1Env.__init__() and _pre_physics_step()\n")

    # Masking list
    mask_always = []
    mask_prob   = {}
    kp_lo_dict  = {}

    for ji in joint_indices:
        jname = JNAMES[ji]
        r     = all_results.get(jname, {})
        diag  = r.get("diagnosis", "")
        kp_lo = r.get("kp_lo_rec", 0.80)
        kp_lo_dict[ji] = kp_lo

        if "100% MASK" in diag:
            mask_always.append(ji)
        elif "Kim mask" in diag:
            try:
                p = float(diag.split("p=")[1])
            except Exception:
                p = 0.20
            mask_prob[ji] = p

    # KP_rand_lo per group (use minimum across healthy joints in each group)
    def group_kp_lo(indices):
        vals = [kp_lo_dict.get(i, 0.80) for i in indices
                if i in joint_indices and kp_lo_dict.get(i, 0) > 0.05]
        return min(vals) if vals else 0.80

    hip_lo   = group_kp_lo(range(0,4))
    thigh_lo = group_kp_lo(range(4,8))
    knee_lo  = group_kp_lo(range(8,12))

    print(f"# KP domain randomisation (from measured amplitude ratios):")
    print(f"self._kp_hip_lo   = {hip_lo:.2f}")
    print(f"self._kp_thigh_lo = {thigh_lo:.2f}")
    print(f"self._kp_knee_lo  = {knee_lo:.2f}")
    print(f"self._kp_hi       = 1.20")
    print()

    if mask_always:
        jnames_m = [JNAMES[i] for i in mask_always]
        print(f"# Broken motors — always masked (electrical fault):")
        print(f"self._always_masked = {mask_always}  # {jnames_m}")
        print()

    if mask_prob:
        print(f"# Intermittent motors — Kim masking per episode:")
        for ji, p in mask_prob.items():
            print(f"self._mask_prob[{ji}] = {p:.2f}  # {JNAMES[ji]}")
        print()

    print("# In _pre_physics_step(), add BEFORE lag filter:")
    print("# --- Kim masking ---")
    if mask_always:
        for ji in mask_always:
            print(f"actions[:, {ji}] = 0.0   # {JNAMES[ji]}: electrical fault")
    if mask_prob:
        for ji, p in mask_prob.items():
            print(f"# {JNAMES[ji]}: mask in {p*100:.0f}% of episodes (set in _reset_idx)")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    save = {}

    for jname, jres in all_results.items():
        for amp, adata in jres.items():
            if not isinstance(amp, float): continue
            k = f"{jname}_amp{amp:.2f}"
            save[f"{k}_ratio_mean"]  = np.array([adata["ratio_mean"]])
            save[f"{k}_ratio_std"]   = np.array([adata["ratio_std"]])
            save[f"{k}_noise_std"]   = np.array([adata["noise_std"]])
            save[f"{k}_tau_mean"]    = np.array([adata["tau_mean"]])
            save[f"{k}_tau_max_cmd"] = np.array([adata["tau_max_cmd"]])
            save[f"{k}_all_ratios"]  = adata["all_ratios"]

    for jname, jsig in all_signals.items():
        for sig_k, sig_v in jsig.items():
            save[f"{jname}_{sig_k}"] = sig_v

    save["joint_names"] = np.array([JNAMES[i] for i in joint_indices], dtype=object)
    save["kp_train"]    = KP_TRAIN
    save["sweep_freq"]  = np.array([SWEEP_FREQ])

    fname = f"calib_amp_ramp_{ts}.npz"
    np.savez(fname, **save)
    kb = sum(v.nbytes for v in save.values() if hasattr(v,'nbytes')) / 1024
    print(f"\n→ Saved {fname}  ({kb:.0f} KB)")

    print("\n── Plot ratio vs torque ──")
    print(f"python3 - << 'EOF'")
    print(f"import numpy as np, matplotlib.pyplot as plt")
    print(f"d = np.load('{fname}', allow_pickle=True)")
    print(f"joints = [j.decode() for j in d['joint_names']]")
    print(f"kp = d['kp_train']")
    print(f"fig, axes = plt.subplots(2, (len(joints)+1)//2, figsize=(16,8))")
    print(f"axes = axes.flatten()")
    print(f"for idx, jn in enumerate(joints):")
    print(f"    ax = axes[idx]")
    print(f"    amps=[]; ratios=[]; stds=[]")
    print(f"    for key in sorted(d.files):")
    print(f"        if key.startswith(jn+'_amp') and key.endswith('_ratio_mean'):")
    print(f"            amp=float(key.split('_amp')[1].split('_')[0])")
    print(f"            ji=[j.decode() for j in d['joint_names']].index(jn)")
    print(f"            amps.append(amp*float(d['kp_train'][ji]))")
    print(f"            ratios.append(float(d[key]))")
    print(f"            stds.append(float(d[key.replace('mean','std')]))")
    print(f"    if amps:")
    print(f"        ax.errorbar(amps, ratios, yerr=stds, fmt='o-', capsize=4)")
    print(f"        ax.axhline(0.80,color='g',ls='--',alpha=0.5,label='healthy')")
    print(f"        ax.axhline(0.10,color='r',ls='--',alpha=0.5,label='broken')")
    print(f"        ax.set_title(jn); ax.set_ylim(-0.05,1.2)")
    print(f"        ax.set_xlabel('Torque (Nm)'); ax.set_ylabel('Amplitude ratio')")
    print(f"plt.suptitle('Amplitude Ramp — ratio vs torque at 2Hz')")
    print(f"plt.tight_layout(); plt.show()")
    print(f"EOF")


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test 4c — Amplitude ramp at training KP")
    p.add_argument("--joint",    type=int, nargs="+", default=None,
                   help="Joint indices 0-11. Default: all.")
    p.add_argument("--n_trials", type=int, default=5,
                   help="Trials per amplitude level. Default 5.")
    p.add_argument("--freq",     type=float, default=2.0,
                   help="Sweep frequency Hz. Default 2.0.")
    args = p.parse_args()

    SWEEP_FREQ = args.freq
    joints     = args.joint if args.joint is not None else list(range(12))

    for j in joints:
        if not 0 <= j <= 11:
            print(f"Joint {j} out of range"); sys.exit(1)

    print("Connecting to Go1...")
    udp.Recv(); udp.GetRecv(state)
    print("Connected.")

    SDK = ['FR_hip','FR_th','FR_kn','FL_hip','FL_th','FL_kn',
           'RR_hip','RR_th','RR_kn','RL_hip','RL_th','RL_kn']
    ok = all(SDK[sdk_to_isaac[i]] == JNAMES[i] for i in range(12))
    print(f"Mapping: {'✓' if ok else '✗ ERROR'}");
    if not ok: sys.exit(1)

    test_amp_ramp(joints, n_trials=args.n_trials)