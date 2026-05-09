#!/usr/bin/env python3
"""
Go1 Test 3 — Latency Spike Test (Tan et al. RSS 2018)
======================================================

CONCEPT:
    The Go1 sends commands at 500Hz (one packet every 2ms).
    When you send "move to position X", the motor doesn't get there instantly.
    There is a delay: CAN bus (1ms) + motor drive response (2-4ms) = 3-8ms total.
    This delay is what we measure.

HOW THE TEST WORKS (one joint, one trial):
    Step 1: Hold joint at DEFAULT position for 0.5s → stable baseline
    Step 2: Measure baseline encoder value (average of 50 readings)
    Step 3: Record timestamp T_cmd
    Step 4: Send SPIKE command — target = default + SPIKE_AMP, KP = training KP
            This creates: τ = KP × error = 35 × 0.3 = 10.5 Nm (hip)
            The joint WILL move — this is intentional
    Step 5: Hold spike command for 20 steps (40ms) then return to default
    Step 6: While sending spike, log encoder at each 500Hz step
    Step 7: Find the FIRST step where encoder moved > threshold from baseline
    Step 8: latency_ms = step_index × 2ms

WHY PREVIOUS VERSION FAILED:
    Previous: KP_HANG=8 × 0.08 rad = 0.64 Nm × 2ms = 0.000051 rad movement
    Threshold: 0.002 rad
    0.000051 << 0.002 → never detected, measured gravity drift instead

    Fixed:    KP_SPIKE=35 × 0.30 rad = 10.5 Nm × 40ms → clearly visible
    Expected: ~0.05-0.15 rad movement → far above threshold

WHAT THE RESULT MEANS:
    latency_ms = 5ms  →  α = 0.02 / (0.02 + 0.005) = 0.80
    latency_ms = 10ms →  α = 0.02 / (0.02 + 0.010) = 0.667
    latency_ms = 15ms →  α = 0.02 / (0.02 + 0.015) = 0.571
    latency_ms = 20ms →  α = 0.02 / (0.02 + 0.020) = 0.500

    α goes into go1_env.py as self._lag_alpha
    Higher α = less lag in sim. Lower α = more lag in sim.
    Real Go1 CAN hardware: ~3-8ms expected → α should be 0.71-0.87

SETUP:
    Robot hanging on safety rack, feet off ground.
    Each joint will move ±0.3 rad briefly during its spike.
    Safe: returns to default after each trial.

Run:
    python3 go1_test3_latency.py                    # all 12 joints
    python3 go1_test3_latency.py --joint 1          # FR_hip only
    python3 go1_test3_latency.py --joint 1 5 9      # FR hip/thigh/knee
    python3 go1_test3_latency.py --n_trials 30      # more averages
"""

import argparse
import time
import sys
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
DT      = 1.0 / CTRL_HZ   # 2ms per step

# ─── Joint mapping ─────────────────────────────────────────────────────────────
# sdk_to_isaac[sdk_idx] = isaac_idx
# Also used as sdk_of_isaac (symmetric mapping)
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]

JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

# ─── Default joint positions (Isaac order, hardware signs) ─────────────────────
# FR_hip and RR_hip are sign-flipped vs Isaac Lab convention
DEFAULT_Q_HW = np.array([
     0.1, -0.1,  0.1, -0.1,   # FL_hip FR_hip RL_hip RR_hip (FR/RR flipped)
     0.8,  0.8,  0.8,  0.8,   # FL_th  FR_th  RL_th  RR_th
    -1.5, -1.5, -1.5, -1.5,   # FL_kn  FR_kn  RL_kn  RR_kn
], np.float32)

# ─── Gains ────────────────────────────────────────────────────────────────────
# SOFT hold between spikes (safe for hanging)
KP_HOLD = np.array([ 8, 8, 8, 8,  12,12,12,12,  15,15,15,15], np.float32)
KD_HOLD = np.array([ 3, 3, 3, 3,   3, 3, 3, 3,   4, 4, 4, 4], np.float32)

# SPIKE gains = same as training gains (strong enough to actually move the joint)
# These must be high enough that τ = KP × error >> encoder noise
# Hip: 35 × 0.30 = 10.5 Nm  → visible in ~5ms
# Thigh: 65 × 0.25 = 16.3 Nm → visible in ~3ms
# Knee: 80 × 0.20 = 16.0 Nm → visible in ~3ms
KP_SPIKE = np.array([35,35,35,35, 65,65,65,65, 80,80,80,80], np.float32)
KD_SPIKE = np.array([ 4, 4, 4, 4,  4.5,4.5,4.5,4.5, 5,5,5,5], np.float32)

# Spike amplitude per joint group (large enough to move, small enough to be safe)
# Direction: positive for all (actual motion direction handles itself via sign)
SPIKE_AMP = np.array([
    0.30, 0.30, 0.30, 0.30,   # hips   — 0.30 rad = 17°, safe range
    0.25, 0.25, 0.25, 0.25,   # thighs — 0.25 rad = 14°
    0.20, 0.20, 0.20, 0.20,   # knees  — 0.20 rad = 11°
], np.float32)

SPIKE_STEPS = 20    # 40ms spike duration (enough to see response clearly)
SETTLE_STEPS = 300  # 600ms hold between trials (let joint return to default)
PRE_STEPS    = 50   # 100ms baseline measurement before spike
POST_STEPS   = 200  # 400ms log window after spike (for response detection)
THRESHOLD_SIGMA = 4.0   # std multiplier for response detection


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_state():
    udp.Recv(); udp.GetRecv(state)
    jpos_sdk = np.array([state.motorState[i].q  for i in range(12)], np.float32)
    jvel_sdk = np.array([state.motorState[i].dq for i in range(12)], np.float32)
    return jpos_sdk[sdk_to_isaac], jvel_sdk[sdk_to_isaac]


def send_cmd(target_q_hw, kp, kd):
    """Send one 500Hz command step. target_q_hw is in Isaac order, HW signs."""
    udp.Recv(); udp.GetRecv(state)
    for isaac_i in range(12):
        sdk_i = sdk_to_isaac[isaac_i]   # symmetric mapping
        cmd.motorCmd[sdk_i].mode = 0x0A
        cmd.motorCmd[sdk_i].q    = float(target_q_hw[isaac_i])
        cmd.motorCmd[sdk_i].dq   = 0.0
        cmd.motorCmd[sdk_i].Kp   = float(kp[isaac_i])
        cmd.motorCmd[sdk_i].Kd   = float(kd[isaac_i])
        cmd.motorCmd[sdk_i].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def hold(duration_s):
    """Hold at DEFAULT_Q_HW with soft gains."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        t_s = time.perf_counter()
        send_cmd(DEFAULT_Q_HW, KP_HOLD, KD_HOLD)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


def one_step(target_q_hw, kp, kd):
    """Send exactly one command step and sleep one DT."""
    t_s = time.perf_counter()
    send_cmd(target_q_hw, kp, kd)
    sl = DT - (time.perf_counter() - t_s)
    if sl > 0: time.sleep(sl)


# ─── Single trial ─────────────────────────────────────────────────────────────

def run_trial(joint_idx):
    """
    Run one latency spike trial on a single joint.
    Returns latency_ms, or None if response not detected.

    Timeline of one trial:
    ┌─────────────────────────────────────────────────────────────────┐
    │ SETTLE(600ms) → PRE(100ms) → SPIKE(40ms) → HOLD(400ms)         │
    │                 measure      detect        confirm              │
    │                 baseline     first move                         │
    └─────────────────────────────────────────────────────────────────┘
    """

    # 1. Settle at default
    for _ in range(SETTLE_STEPS):
        one_step(DEFAULT_Q_HW, KP_HOLD, KD_HOLD)

    # 2. Measure baseline (mean and std of current joint position)
    pre_q = []
    for _ in range(PRE_STEPS):
        jpos, _ = read_state()
        pre_q.append(jpos[joint_idx])
        one_step(DEFAULT_Q_HW, KP_HOLD, KD_HOLD)
    baseline  = np.mean(pre_q)
    noise_std = np.std(pre_q)
    threshold = max(noise_std * THRESHOLD_SIGMA, 0.001)  # at least 1mm

    # 3. Build spike target: only move the target joint
    spike_target = DEFAULT_Q_HW.copy()
    spike_target[joint_idx] += SPIKE_AMP[joint_idx]

    # 4. Send spike and log encoder at each step
    # Record the timestamp BEFORE first spike command
    q_log = []     # encoder position at each step
    t_log = []     # timestamp in ms from spike start

    t_spike = time.perf_counter()

    # SPIKE phase — strong KP drives joint to spike_target
    for step in range(SPIKE_STEPS + POST_STEPS):
        t_s = time.perf_counter()

        jpos, _ = read_state()
        q_log.append(jpos[joint_idx])
        t_log.append((time.perf_counter() - t_spike) * 1000.0)

        if step < SPIKE_STEPS:
            send_cmd(spike_target, KP_SPIKE, KD_SPIKE)  # spike
        else:
            send_cmd(DEFAULT_Q_HW, KP_HOLD,  KD_HOLD)   # return

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    # 5. Find first response: first step where |q - baseline| > threshold
    q_arr  = np.array(q_log)
    t_arr  = np.array(t_log)
    dev    = np.abs(q_arr - baseline)
    moving = np.where(dev > threshold)[0]

    if len(moving) > 0:
        latency_ms = t_arr[moving[0]]
        peak_dev   = dev.max()
        return latency_ms, peak_dev, q_arr, t_arr, baseline, threshold
    else:
        return None, dev.max(), q_arr, t_arr, baseline, threshold


# ─── Main test ────────────────────────────────────────────────────────────────

def test_latency(joint_indices, n_trials=20):
    print("\n═══ TEST 3: Latency Spike ═══")
    print()
    print("  CONCEPT:")
    print("  ─────────────────────────────────────────────────────────")
    print("  1. Hold joint at default position (still, 600ms)")
    print("  2. Record baseline encoder value")
    print("  3. Send STRONG spike command: target = default + 0.20-0.30 rad")
    print("     KP = training gains (35/65/80) → τ = 10-16 Nm")
    print("     This WILL move the joint — intentional")
    print("  4. At each 2ms step: log encoder position")
    print("  5. Find first step where encoder moved > 4σ above baseline")
    print("  6. latency_ms = that step's timestamp")
    print("  7. α = 0.02 / (0.02 + latency_ms/1000)")
    print()
    print("  WHAT PREVIOUS VERSION DID WRONG:")
    print("  KP_HANG=8 × 0.08 rad = 0.64 Nm → moved 0.00005 rad → invisible")
    print("  Measured gravity drift (759ms, 91ms) not actual latency")
    print()
    print("  Setup: robot HANGING on rack, legs free to swing")
    print(f"  Joints to test: {[JNAMES[i] for i in joint_indices]}")
    print(f"  Trials per joint: {n_trials}")
    print(f"  Spike amplitude: hips={SPIKE_AMP[0]:.2f}rad  thighs={SPIKE_AMP[4]:.2f}rad  knees={SPIKE_AMP[8]:.2f}rad")
    print(f"  Spike KP: hips={KP_SPIKE[0]:.0f}  thighs={KP_SPIKE[4]:.0f}  knees={KP_SPIKE[8]:.0f}")
    print()
    print("  NOTE: Each trial lasts ~1.4s. Joint moves briefly then returns.")
    print(f"  Total time: {len(joint_indices) * n_trials * 1.4 / 60:.1f} minutes")
    print()
    input("  Robot hanging? Press Enter to start → ")

    print("\n  Soft ramping to DEFAULT_Q_HW (3s)...")
    hold(3.0)
    print("  Ready.\n")

    all_results = {}
    all_q_logs  = {}

    for ji in joint_indices:
        jname = JNAMES[ji]
        print(f"  ─── {jname} (Isaac[{ji}]) ───")
        print(f"  Spike: +{SPIKE_AMP[ji]:.2f} rad  KP={KP_SPIKE[ji]:.0f}  "
              f"τ≈{KP_SPIKE[ji]*SPIKE_AMP[ji]:.1f} Nm for {SPIKE_STEPS*2:.0f}ms")

        trial_lats  = []
        trial_q_all = []

        for trial in range(n_trials):
            result = run_trial(ji)
            lat_ms, peak_dev, q_arr, t_arr, baseline, threshold = result

            trial_q_all.append(q_arr)

            if lat_ms is not None:
                trial_lats.append(lat_ms)
                alpha = 0.02 / (0.02 + lat_ms / 1000.0)
                print(f"    trial {trial+1:2d}: latency={lat_ms:.1f}ms  "
                      f"peak_dev={peak_dev:.4f}rad  α={alpha:.3f}")
            else:
                print(f"    trial {trial+1:2d}: NO RESPONSE  "
                      f"peak_dev={peak_dev:.4f}rad  threshold={threshold:.4f}rad")

        # Summary for this joint
        print()
        if len(trial_lats) >= 3:
            lat_arr   = np.array(trial_lats)
            lat_mean  = lat_arr.mean()
            lat_std   = lat_arr.std()
            lat_med   = np.median(lat_arr)
            alpha_rec = 0.02 / (0.02 + lat_mean / 1000.0)
            hit_rate  = len(trial_lats) / n_trials * 100
            print(f"  {jname} RESULT:")
            print(f"    Mean latency:   {lat_mean:.1f} ± {lat_std:.1f} ms")
            print(f"    Median latency: {lat_med:.1f} ms")
            print(f"    Detection rate: {hit_rate:.0f}%  ({len(trial_lats)}/{n_trials})")
            print(f"    → α = 0.02 / (0.02 + {lat_mean/1000:.4f}) = {alpha_rec:.3f}")
            all_results[jname] = {"mean_ms": lat_mean, "std_ms": lat_std,
                                   "median_ms": lat_med, "alpha": alpha_rec,
                                   "hit_rate": hit_rate, "all_ms": lat_arr}
        elif len(trial_lats) > 0:
            lat_mean  = np.mean(trial_lats)
            alpha_rec = 0.02 / (0.02 + lat_mean / 1000.0)
            print(f"  {jname}: {len(trial_lats)} detections (low — increase SPIKE_AMP or n_trials)")
            print(f"    Mean: {lat_mean:.1f}ms  α={alpha_rec:.3f}")
            all_results[jname] = {"mean_ms": lat_mean, "std_ms": np.std(trial_lats),
                                   "median_ms": lat_mean, "alpha": alpha_rec,
                                   "hit_rate": len(trial_lats)/n_trials*100,
                                   "all_ms": np.array(trial_lats)}
        else:
            print(f"  {jname}: NO detections.")
            print(f"    Possible causes:")
            print(f"      - Motor off or unplugged")
            print(f"      - Spike amplitude too small (increase SPIKE_AMP[{ji}])")
            print(f"      - Threshold too large (baseline encoder very noisy)")
            all_results[jname] = {"mean_ms": np.nan, "std_ms": np.nan,
                                   "median_ms": np.nan, "alpha": 0.80,
                                   "hit_rate": 0.0, "all_ms": np.array([])}

        all_q_logs[jname] = np.array(trial_q_all) if trial_q_all else np.zeros((1, SPIKE_STEPS+POST_STEPS))
        print()

    # ─── Final summary ────────────────────────────────────────────────────────
    print("═" * 65)
    print("LATENCY SUMMARY")
    print("═" * 65)
    print(f"  {'joint':8s}  {'lat_ms':>8}  {'lat_std':>8}  {'alpha':>7}  {'hit%':>6}")
    for jn, r in all_results.items():
        na = "nan" if np.isnan(r['mean_ms']) else f"{r['mean_ms']:.1f}"
        ns = "nan" if np.isnan(r['std_ms'])  else f"{r['std_ms']:.1f}"
        print(f"  {jn:8s}: {na:>8}  {ns:>8}  {r['alpha']:>7.3f}  {r['hit_rate']:>5.0f}%")

    print()
    print("Expected for healthy Go1 joints: 3-8ms  α=0.71-0.87")
    print("Expected for noisy FR motor:     higher latency or low hit rate")
    print()

    # Per-joint alpha vector
    full_alphas = []
    for jn in JNAMES:
        r = all_results.get(jn, {})
        full_alphas.append(round(r.get("alpha", 0.80), 3))
    print("── go1_env.py update ──")
    print(f"# Replace single self._lag_alpha = 0.6 with per-joint vector:")
    print(f"self._lag_alpha_per_joint = {full_alphas}")
    print()
    print("In _pre_physics_step(), change:")
    print("  self._lag_pos = (α * target + (1-α) * self._lag_pos)")
    print("to:")
    print("  α = torch.tensor(self._lag_alpha_per_joint, device=self.device)")
    print("  self._lag_pos = (α * target + (1-α) * self._lag_pos)")

    # Save
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    save = {}
    for jn, r in all_results.items():
        save[f"{jn}_lat_all_ms"]   = r["all_ms"]
        save[f"{jn}_lat_mean_ms"]  = np.array([r["mean_ms"]])
        save[f"{jn}_lat_std_ms"]   = np.array([r["std_ms"]])
        save[f"{jn}_alpha"]        = np.array([r["alpha"]])
        save[f"{jn}_hit_rate"]     = np.array([r["hit_rate"]])
    for jn, qlog in all_q_logs.items():
        save[f"{jn}_q_trials"] = qlog
    save["spike_amp"]   = SPIKE_AMP
    save["spike_steps"] = np.array([SPIKE_STEPS])
    save["dt_ms"]       = np.array([DT * 1000])
    save["joint_names"] = np.array(JNAMES, dtype=object)

    fname = f"calib_latency_{ts}.npz"
    np.savez(fname, **save)
    kb = sum(v.nbytes for v in save.values() if hasattr(v,'nbytes')) / 1024
    print(f"\n→ Saved {fname}  ({kb:.0f} KB)")
    print()
    print("To plot one joint's trials:")
    print(f"  python3 -c \"")
    print(f"  import numpy as np, matplotlib.pyplot as plt")
    print(f"  d=np.load('{fname}',allow_pickle=True)")
    print(f"  q=d['FL_hip_q_trials']")
    print(f"  [plt.plot(q[i],'gray',alpha=0.3) for i in range(len(q))]")
    print(f"  plt.axhline(q[0].mean(),color='blue',label='baseline')")
    print(f"  plt.xlabel('step (2ms each)'); plt.ylabel('joint pos rad')")
    print(f"  plt.title('FL_hip spike trials'); plt.legend(); plt.show()\"")


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Go1 Test 3 — Latency Spike")
    p.add_argument("--joint",    type=int,   nargs="+", default=None,
                   help="Joint indices (Isaac order 0-11). Default: all 12.")
    p.add_argument("--n_trials", type=int,   default=20,
                   help="Number of spike trials per joint (default 20)")
    args = p.parse_args()

    joints = args.joint if args.joint is not None else list(range(12))

    # Validate
    for j in joints:
        if not 0 <= j <= 11:
            print(f"Joint index {j} out of range 0-11"); sys.exit(1)

    print("Connecting to Go1...")
    udp.Recv(); udp.GetRecv(state)
    print("Connected.")

    # Verify mapping
    SDK = ['FR_hip','FR_th','FR_kn','FL_hip','FL_th','FL_kn',
           'RR_hip','RR_th','RR_kn','RL_hip','RL_th','RL_kn']
    ok = all(SDK[sdk_to_isaac[i]] == JNAMES[i] for i in range(12))
    print(f"Joint mapping: {'✓' if ok else '✗ ERROR'}")
    if not ok: sys.exit(1)

    test_latency(joints, n_trials=args.n_trials)