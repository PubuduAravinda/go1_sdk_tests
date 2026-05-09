#!/usr/bin/env python3
"""
Go1 Hardware Calibration Suite v2  — Bugs fixed 2026-03-19
===========================================================
Fixed:
  1. isaac_to_sdk was wrong (inverse permutation bug).
     sdk_to_isaac and sdk_of_isaac are THE SAME array — symmetric mapping.
  2. FR_hip and RR_hip sign flips missing in DEFAULT_Q_HW.
     Real hardware: FR/RR hips have reversed encoder vs Isaac.
  3. IMU zeros: Go1 needs a heartbeat packet to keep broadcasting state.
     Fix: zero-torque damping command (KP=0, KD=1) every step.

Tests:
  1  IMU noise      robot still on table, NO holding force, 5 min
  2  Encoder noise  robot hanging, soft hold, 2 min
  3  Latency spike  robot hanging, one-step impulse per joint
  4  Freq sweep     robot hanging, sinusoidal cmd per joint
  5  Friction curve one joint free to rotate, velocity sweeps
  6  FR profile     robot hanging, FR leg only

Run:  python3 go1_calibration.py --test 1
      python3 go1_calibration.py --test all   (tests 1,2,3,4)
      python3 go1_calibration.py --test 6
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

# ─── Joint ordering ──────────────────────────────────────────────────────────
#
# Unitree SDK motor order (index → joint name):
#   0=FR_hip  1=FR_th  2=FR_kn
#   3=FL_hip  4=FL_th  5=FL_kn
#   6=RR_hip  7=RR_th  8=RR_kn
#   9=RL_hip  10=RL_th 11=RL_kn
#
# Isaac Lab order:
#   0=FL_hip 1=FR_hip 2=RL_hip 3=RR_hip
#   4=FL_th  5=FR_th  6=RL_th  7=RR_th
#   8=FL_kn  9=FR_kn  10=RL_kn 11=RR_kn
#
# sdk_to_isaac[i] = "to get Isaac joint i, read SDK motor sdk_to_isaac[i]"
# ALSO used for writing: "to command Isaac joint i, write to SDK motor sdk_to_isaac[i]"
# This mapping is SELF-INVERSE (symmetric), so sdk_to_isaac == sdk_of_isaac.
#
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
sdk_of_isaac = sdk_to_isaac   # SAME array — mapping is symmetric

JNAMES_ISAAC = ['FL_hip','FR_hip','RL_hip','RR_hip',
                'FL_th', 'FR_th', 'RL_th', 'RR_th',
                'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']
JNAMES_SDK   = ['FR_hip','FR_th', 'FR_kn',
                'FL_hip','FL_th', 'FL_kn',
                'RR_hip','RR_th', 'RR_kn',
                'RL_hip','RL_th', 'RL_kn']

# ─── Default joint positions ──────────────────────────────────────────────────
#
# DEFAULT_Q_ISAAC: values in Isaac Lab convention (all hips +0.1 = legs outward)
# DEFAULT_Q_HW:    same physical pose in real-hardware convention
#                  FR_hip and RR_hip have reversed encoder → sign flip needed
#
DEFAULT_Q_ISAAC = np.array([
     0.1,  0.1,  0.1,  0.1,   # FL_hip FR_hip RL_hip RR_hip
     0.8,  0.8,  0.8,  0.8,   # FL_th  FR_th  RL_th  RR_th
    -1.5, -1.5, -1.5, -1.5,   # FL_kn  FR_kn  RL_kn  RR_kn
], np.float32)

DEFAULT_Q_HW = DEFAULT_Q_ISAAC.copy()
DEFAULT_Q_HW[1] = -DEFAULT_Q_ISAAC[1]   # FR_hip: reversed encoder on real Go1
DEFAULT_Q_HW[3] = -DEFAULT_Q_ISAAC[3]   # RR_hip: reversed encoder on real Go1

# Soft gains for hanging — low KP so max torque ≈ KP×0.3rad ≈ 2Nm (well under PowerProtect)
KP_HANG = np.array([ 8, 8, 8, 8,  12,12,12,12,  15,15,15,15], np.float32)
KD_HANG = np.array([ 3, 3, 3, 3,   3, 3, 3, 3,   4, 4, 4, 4], np.float32)
KD_HEARTBEAT = 1.0   # Nm·s/rad — just enough to keep motors from free-swinging


# ─── Core SDK helpers ─────────────────────────────────────────────────────────

def read_state():
    """Read joint/IMU state. Does NOT send any command."""
    udp.Recv()
    udp.GetRecv(state)
    jpos_sdk = np.array([state.motorState[i].q      for i in range(12)], np.float32)
    jvel_sdk = np.array([state.motorState[i].dq     for i in range(12)], np.float32)
    jtau_sdk = np.array([state.motorState[i].tauEst for i in range(12)], np.float32)
    # Reorder SDK→Isaac
    jpos = jpos_sdk[sdk_to_isaac]
    jvel = jvel_sdk[sdk_to_isaac]
    jtau = jtau_sdk[sdk_to_isaac]
    gyro = np.array(state.imu.gyroscope,     np.float32)
    acc  = np.array(state.imu.accelerometer, np.float32)
    return jpos, jvel, jtau, gyro, acc


def send_heartbeat():
    """
    Zero-torque heartbeat — keeps Go1 broadcasting state without any holding force.
    KP=0: no position stiffness, no torque from position error.
    KD=small: gentle damping only, prevents free-swing oscillation.

    REQUIRED for Test 1 (IMU). Without this, robot stops broadcasting
    and all reads return stale zeros.
    """
    udp.Recv()
    udp.GetRecv(state)
    for sdk_i in range(12):
        cmd.motorCmd[sdk_i].mode = 0x0A
        cmd.motorCmd[sdk_i].q    = float(state.motorState[sdk_i].q)  # target = current pos
        cmd.motorCmd[sdk_i].dq   = 0.0
        cmd.motorCmd[sdk_i].Kp   = 0.0                # NO stiffness
        cmd.motorCmd[sdk_i].Kd   = KD_HEARTBEAT       # tiny damping only
        cmd.motorCmd[sdk_i].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()


def send_soft_hold_step(target_q_hw, kp=None, kd=None):
    """
    Send ONE step of soft position hold.
    target_q_hw: desired joint positions in Isaac order WITH hardware sign convention.
    Uses sdk_of_isaac (= sdk_to_isaac) to map Isaac→SDK motor index correctly.
    """
    if kp is None: kp = KP_HANG
    if kd is None: kd = KD_HANG
    udp.Recv()
    udp.GetRecv(state)
    for isaac_i in range(12):
        sdk_i = sdk_of_isaac[isaac_i]          # ← FIXED: use sdk_of_isaac, not inverse
        cmd.motorCmd[sdk_i].mode = 0x0A
        cmd.motorCmd[sdk_i].q    = float(target_q_hw[isaac_i])
        cmd.motorCmd[sdk_i].dq   = 0.0
        cmd.motorCmd[sdk_i].Kp   = float(kp[isaac_i])
        cmd.motorCmd[sdk_i].Kd   = float(kd[isaac_i])
        cmd.motorCmd[sdk_i].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()


def soft_hold(target_q_hw, duration_s, kp=None, kd=None):
    """Hold target position for duration_s seconds."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        t_s = time.perf_counter()
        send_soft_hold_step(target_q_hw, kp, kd)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


def save_npz(fname, **arrays):
    np.savez(fname, **arrays)
    kb = sum(v.nbytes for v in arrays.values() if hasattr(v, 'nbytes')) / 1024
    print(f"  → Saved {fname}  ({kb:.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — IMU Noise (passive read + heartbeat)
# ══════════════════════════════════════════════════════════════════════════════
def test_imu_noise(duration_s=300):
    print("\n═══ TEST 1: IMU Noise ═══")
    print("  Robot must be COMPLETELY STILL on flat surface or hanging without swing.")
    print("  A zero-torque heartbeat is sent to keep robot broadcasting state.")
    print("  KP=0 on heartbeat → NO holding force applied.")
    print(f"  Duration: {duration_s:.0f}s = {duration_s/60:.1f} min")
    input("\n  Press Enter when robot is still → ")

    N        = int(duration_s * CTRL_HZ)
    gyro_log = np.zeros((N, 3), np.float32)
    acc_log  = np.zeros((N, 3), np.float32)
    t_log    = np.zeros(N, np.float32)

    t0 = time.perf_counter()
    print("  Collecting... (heartbeat sent, no holding force)")
    for i in range(N):
        t_s = time.perf_counter()

        send_heartbeat()           # keeps robot broadcasting — KP=0
        _, _, _, gyro, acc = read_state()
        gyro_log[i] = gyro
        acc_log[i]  = acc
        t_log[i]    = time.perf_counter() - t0

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

        if i > 0 and i % (CTRL_HZ * 30) == 0:
            g = gyro_log[:i].std(axis=0)
            a = acc_log[:i].std(axis=0)
            print(f"  t={t_log[i]:.0f}s  gyro_std=[{g[0]:.5f},{g[1]:.5f},{g[2]:.5f}]  "
                  f"acc_std=[{a[0]:.5f},{a[1]:.5f},{a[2]:.5f}]")

    gyro_bias = gyro_log.mean(axis=0)
    gyro_std  = gyro_log.std(axis=0)
    acc_mean  = acc_log.mean(axis=0)
    acc_std   = acc_log.std(axis=0)
    acc_norm  = max(float(np.linalg.norm(acc_mean)), 1e-3)
    grav_std  = acc_std / acc_norm

    print(f"\n  ── Results ──")
    print(f"  Gyro bias  (rad/s): [{gyro_bias[0]:+.5f}, {gyro_bias[1]:+.5f}, {gyro_bias[2]:+.5f}]")
    print(f"  Gyro std   (rad/s): [{gyro_std[0]:.5f}, {gyro_std[1]:.5f}, {gyro_std[2]:.5f}]  ← obs[27:30]")
    print(f"  Acc mean   (m/s²):  [{acc_mean[0]:+.4f}, {acc_mean[1]:+.4f}, {acc_mean[2]:+.4f}]")
    print(f"  Acc std    (m/s²):  [{acc_std[0]:.5f}, {acc_std[1]:.5f}, {acc_std[2]:.5f}]")
    print(f"  ProjGrav std:       [{grav_std[0]:.5f}, {grav_std[1]:.5f}, {grav_std[2]:.5f}]  ← obs[30:33]")
    gs = gyro_std.round(5).tolist(); pg = grav_std.round(5).tolist()
    print(f"\n  ── go1_env.py copy-paste ──")
    print(f"  # gyro noise (measured {datetime.now().strftime('%Y-%m-%d')})")
    print(f"  # obs[27:30]: {gs}")
    print(f"  # obs[30:33]: {pg}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_npz(f"calib_imu_{ts}.npz",
             gyro=gyro_log, acc=acc_log, time=t_log,
             gyro_bias=gyro_bias, gyro_std=gyro_std,
             acc_mean=acc_mean, acc_std=acc_std, grav_std=grav_std)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Encoder Noise
# ══════════════════════════════════════════════════════════════════════════════
def test_encoder_noise(duration_s=120):
    print("\n═══ TEST 2: Encoder Noise ═══")
    print("  Robot HANGING on rack. Soft hold at DEFAULT_Q_HW (hardware signs applied).")
    print(f"  DEFAULT_Q_HW (Isaac order): {DEFAULT_Q_HW.tolist()}")
    print("  Note FR_hip=-0.1 and RR_hip=-0.1 (sign flips from Isaac convention)")
    print(f"  KP_HANG = {KP_HANG.tolist()} — very soft, safe for hanging")
    input("\n  Press Enter to start soft hold → ")

    print("  Ramping to DEFAULT_Q_HW (5s)...")
    soft_hold(DEFAULT_Q_HW, duration_s=5.0)

    N        = int(duration_s * CTRL_HZ)
    jpos_log = np.zeros((N, 12), np.float32)
    jvel_log = np.zeros((N, 12), np.float32)
    t_log    = np.zeros(N, np.float32)

    t0 = time.perf_counter()
    print(f"  Collecting {duration_s:.0f}s of encoder data...")
    for i in range(N):
        t_s  = time.perf_counter()
        jpos, jvel, _, _, _ = read_state()
        send_soft_hold_step(DEFAULT_Q_HW)
        jpos_log[i] = jpos
        jvel_log[i] = jvel
        t_log[i]    = time.perf_counter() - t0
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)
        if i > 0 and i % (CTRL_HZ * 30) == 0:
            print(f"  t={t_log[i]:.0f}s  max_jpos_std={jpos_log[:i].std(axis=0).max():.5f}")

    jpos_std = jpos_log.std(axis=0)
    jvel_std = jvel_log.std(axis=0)
    jpos_off = (jpos_log - DEFAULT_Q_HW).mean(axis=0)

    print(f"\n  ── Results ──")
    print(f"  {'joint':8s}  {'jpos_std':>10}  {'jvel_std':>10}  {'offset':>10}")
    for i, n in enumerate(JNAMES_ISAAC):
        flag = "  ← HIGH" if jpos_std[i] > 0.015 else ""
        print(f"  {n:8s}: {jpos_std[i]:>10.5f}  {jvel_std[i]:>10.5f}  {jpos_off[i]:>+10.5f}{flag}")

    print(f"\n  ── go1_env.py copy-paste ──")
    print(f"  jpos_noise = {jpos_std.round(5).tolist()}")
    print(f"  jvel_noise = {jvel_std.round(5).tolist()}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_npz(f"calib_encoder_{ts}.npz",
             jpos=jpos_log, jvel=jvel_log, time=t_log,
             jpos_std=jpos_std, jvel_std=jvel_std,
             jpos_offset=jpos_off, default_q_hw=DEFAULT_Q_HW,
             joint_names=np.array(JNAMES_ISAAC, dtype=object))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Latency Spike (Tan et al. 2018)
# ══════════════════════════════════════════════════════════════════════════════
def test_latency_spike(joint_idx=None, n_spikes=20):
    print("\n═══ TEST 3: Latency Spike (Tan 2018) ═══")
    print("  One-step impulse per joint → measures CAN + motor response delay.")
    print("  Robot HANGING. Spike = +0.08 rad for ONE control step (2ms).")
    input("  Press Enter when ready → ")

    joints    = list(range(12)) if joint_idx is None else [joint_idx]
    SPIKE_AMP = 0.08  # rad
    SETTLE    = 250   # steps
    POST      = 200   # steps to log after spike

    results = {}
    for ji in joints:
        jname = JNAMES_ISAAC[ji]
        print(f"\n  → {jname}")
        lats = []

        for trial in range(n_spikes):
            # Settle at default
            soft_hold(DEFAULT_Q_HW, duration_s=SETTLE * DT)

            # Baseline
            pre = []
            for _ in range(20):
                jpos, _, _, _, _ = read_state()
                pre.append(jpos[ji])
                send_heartbeat()
                time.sleep(DT)
            baseline  = np.mean(pre)
            threshold = max(np.std(pre) * 5.0, 0.002)

            # ONE spike step
            spike_q = DEFAULT_Q_HW.copy()
            # For FR_hip and RR_hip, the sign is already handled in DEFAULT_Q_HW
            spike_q[ji] += SPIKE_AMP
            t_spike = time.perf_counter()
            send_soft_hold_step(spike_q)
            time.sleep(DT)

            # Immediately return
            for _ in range(3):
                send_soft_hold_step(DEFAULT_Q_HW)
                time.sleep(DT)

            # Log response
            resp_q = []; resp_t = []
            for step in range(POST):
                jpos, _, _, _, _ = read_state()
                send_soft_hold_step(DEFAULT_Q_HW)
                resp_q.append(jpos[ji])
                resp_t.append((time.perf_counter() - t_spike) * 1000)
                time.sleep(DT)

            # First response
            dev = np.abs(np.array(resp_q) - baseline)
            idx = np.where(dev > threshold)[0]
            if len(idx) > 0:
                lats.append(resp_t[idx[0]])

        if lats:
            mu  = np.mean(lats); sd = np.std(lats)
            alpha = 0.02 / (0.02 + mu / 1000.0)
            results[jname] = {"mean": mu, "std": sd, "alpha": alpha,
                               "raw": np.array(lats)}
            print(f"    {mu:.1f} ± {sd:.1f} ms   α = {alpha:.3f}")
        else:
            print(f"    NO RESPONSE — motor may not respond to single spike")
            results[jname] = {"mean": np.nan, "std": np.nan, "alpha": 0.6,
                               "raw": np.array([])}

    print(f"\n  ── Summary ──")
    alphas = [results.get(n, {}).get("alpha", 0.6) for n in JNAMES_ISAAC]
    print(f"  self._lag_alpha_per_joint = {[round(a, 3) for a in alphas]}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d  = {}
    for jn, r in results.items():
        d[f"{jn}_latency_ms"] = r["raw"]
        d[f"{jn}_alpha"]      = np.array([r["alpha"]])
    save_npz(f"calib_latency_{ts}.npz", **d)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Frequency Sweep (Hwangbo actuator characterisation)
# ══════════════════════════════════════════════════════════════════════════════
def test_frequency_sweep(joint_idx=None, amplitude=0.08, cycles=5):
    print("\n═══ TEST 4: Frequency Sweep ═══")
    print(f"  Sin cmd at [0.5,1.0,2.0,4.0,6.0] Hz, ±{amplitude} rad per joint.")
    print("  Robot HANGING. Measures effective KP fraction per joint per frequency.")
    freqs  = [0.5, 1.0, 2.0, 4.0, 6.0]
    joints = list(range(12)) if joint_idx is None else [joint_idx]
    input("  Press Enter when ready → ")

    all_results = {}
    for ji in joints:
        jname = JNAMES_ISAAC[ji]
        print(f"\n  → {jname}")
        jres = {}

        for freq in freqs:
            n_steps = int(cycles / freq * CTRL_HZ)
            t_arr   = np.arange(n_steps) * DT
            cmd_sig = amplitude * np.sin(2 * np.pi * freq * t_arr)
            act_q   = np.zeros(n_steps, np.float32)

            t0 = time.perf_counter()
            for step in range(n_steps):
                t_s   = time.perf_counter()
                jpos, _, _, _, _ = read_state()
                tq = DEFAULT_Q_HW.copy()
                tq[ji] += cmd_sig[step]
                send_soft_hold_step(tq)
                act_q[step] = jpos[ji] - DEFAULT_Q_HW[ji]
                sl = DT - (time.perf_counter() - t_s)
                if sl > 0: time.sleep(sl)

            skip      = int(1.0 / freq * CTRL_HZ)
            cs        = cmd_sig[skip:]; qs = act_q[skip:]
            amp_ratio = np.std(qs) / (np.std(cs) + 1e-6)
            corr      = np.correlate(qs - qs.mean(), cs - cs.mean(), mode='full')
            lag_ms    = (np.argmax(corr) - (len(qs) - 1)) * DT * 1000

            print(f"    {freq:.1f}Hz: ratio={amp_ratio:.3f}  lag={lag_ms:.1f}ms")
            jres[freq] = {"amp_ratio": amp_ratio, "lag_ms": lag_ms,
                          "cmd": cmd_sig, "actual": act_q}
            soft_hold(DEFAULT_Q_HW, duration_s=1.0)

        if 2.0 in jres:
            kp = jres[2.0]["amp_ratio"]
            lo, hi = max(0.1, kp - 0.15), min(1.5, kp + 0.15)
            print(f"  → KP@2Hz = {kp:.3f}×nominal   DR range [{lo:.2f},{hi:.2f}]")
        all_results[jname] = jres

    print(f"\n  ── Summary ──")
    print(f"  {'joint':8s}  {'@0.5Hz':>8}  {'@2Hz':>8}  {'@4Hz':>8}  {'lag@2Hz':>10}")
    for jn, jres in all_results.items():
        r5 = jres.get(0.5, {}).get("amp_ratio", float("nan"))
        r2 = jres.get(2.0, {}).get("amp_ratio", float("nan"))
        r4 = jres.get(4.0, {}).get("amp_ratio", float("nan"))
        lg = jres.get(2.0, {}).get("lag_ms",    float("nan"))
        print(f"  {jn:8s}: {r5:>8.3f}  {r2:>8.3f}  {r4:>8.3f}  {lg:>10.1f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d  = {}
    for jn, jres in all_results.items():
        for freq, data in jres.items():
            k = f"{jn}_{freq:.1f}Hz"
            d[f"{k}_cmd"]       = data["cmd"]
            d[f"{k}_actual"]    = data["actual"]
            d[f"{k}_amp_ratio"] = np.array([data["amp_ratio"]])
            d[f"{k}_lag_ms"]    = np.array([data["lag_ms"]])
    save_npz(f"calib_freqsweep_{ts}.npz", **d)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Friction Curve
# ══════════════════════════════════════════════════════════════════════════════
def test_friction_curve(joint_idx=5):
    jname = JNAMES_ISAAC[joint_idx]
    print(f"\n═══ TEST 5: Friction Curve — {jname} ═══")
    print("  Constant velocity sweeps → Coulomb + viscous friction.")
    input("  Press Enter when ready → ")

    vels    = [-3,-2,-1.5,-1,-0.5,-0.3, 0.3, 0.5, 1, 1.5, 2, 3]
    tau_out = []; vel_out = []

    for v in vels:
        n = int(2.0 * CTRL_HZ)
        tau_buf, vel_buf = [], []
        for step in range(n):
            t_s  = time.perf_counter()
            jpos, jvel, jtau, _, _ = read_state()
            tq = DEFAULT_Q_HW.copy()
            tq[joint_idx] = jpos[joint_idx] + v * DT
            kp = KP_HANG.copy(); kp[joint_idx] = 3.0
            kd = KD_HANG.copy(); kd[joint_idx] = 6.0
            send_soft_hold_step(tq, kp, kd)
            if step > n // 2:
                tau_buf.append(jtau[joint_idx])
                vel_buf.append(jvel[joint_idx])
            sl = DT - (time.perf_counter() - t_s)
            if sl > 0: time.sleep(sl)
        tau_out.append(np.mean(tau_buf))
        vel_out.append(np.mean(vel_buf))
        print(f"  v_cmd={v:+.1f}  actual={vel_out[-1]:+.3f}  tau={tau_out[-1]:+.4f} Nm")
        soft_hold(DEFAULT_Q_HW, duration_s=0.5)

    A = np.column_stack([np.sign(vel_out), vel_out])
    b_c, b_v = np.linalg.lstsq(A, tau_out, rcond=None)[0]
    print(f"\n  Coulomb b_c = {b_c:.4f} Nm")
    print(f"  Viscous b_v = {b_v:.4f} Nm·s/rad")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_npz(f"calib_friction_{jname}_{ts}.npz",
             tau=np.array(tau_out), vel=np.array(vel_out),
             v_cmd=np.array(vels),
             b_coulomb=np.array([b_c]), b_viscous=np.array([b_v]))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — FR Motor Impairment Profile (Kim et al. 2024)
# ══════════════════════════════════════════════════════════════════════════════
def test_fr_motor_profile(n_trials=30):
    print("\n═══ TEST 6: FR Motor Impairment Profile ═══")
    print("  Step responses on FR hip, thigh, knee to detect dropout vs compliance.")
    print("  Robot HANGING, FR leg free.")
    input("  Press Enter when ready → ")

    for ji in [1, 5, 9]:  # FR_hip, FR_th, FR_kn in Isaac order
        jname = JNAMES_ISAAC[ji]
        print(f"\n  → {jname}")
        amp_results = {}

        for amp in [0.05, 0.10, 0.15, 0.20]:
            achieved = []; dropouts = 0

            for trial in range(n_trials):
                soft_hold(DEFAULT_Q_HW, duration_s=200 * DT)

                tq = DEFAULT_Q_HW.copy()
                tq[ji] += amp
                taus = []; qvals = []

                for step in range(150):
                    jpos, _, jtau, _, _ = read_state()
                    send_soft_hold_step(tq)
                    taus.append(jtau[ji])
                    qvals.append(jpos[ji])
                    time.sleep(DT)

                frac = (qvals[-1] - DEFAULT_Q_HW[ji]) / (amp + 1e-6)
                achieved.append(frac)
                tau_a = np.array(taus)
                if tau_a[:50].mean() > 0.5 and tau_a[50:100].mean() < 0.2:
                    dropouts += 1

            do = dropouts / n_trials * 100
            ac = np.mean(achieved)
            print(f"    amp={amp:.2f}: achieved={ac:.3f}±{np.std(achieved):.3f}  dropout={do:.0f}%")
            amp_results[amp] = {"achieved_mean": ac, "dropout_rate": do,
                                 "achieved_all": np.array(achieved)}

        avg_ac   = np.mean([v["achieved_mean"] for v in amp_results.values()])
        max_drop = max(v["dropout_rate"] for v in amp_results.values())
        print(f"\n  Diagnosis for {jname}:")
        if max_drop > 10:
            print(f"  *** INTERMITTENT DROPOUT ({max_drop:.0f}%) → use Kim joint masking in sim")
        elif avg_ac < 0.70:
            print(f"  *** LOW COMPLIANCE (avg {avg_ac:.3f}) → KP_rand_lo = {avg_ac:.2f}")
        else:
            print(f"  Normal — KP_rand_lo = 0.80 appropriate")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        d  = {}
        for a, r in amp_results.items():
            d[f"amp_{a:.2f}_achieved"]  = r["achieved_all"]
            d[f"amp_{a:.2f}_drop_rate"] = np.array([r["dropout_rate"]])
        save_npz(f"calib_fr_{jname}_{ts}.npz", **d)


# ─── Entry point ──────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║          Go1 Calibration Suite v2  (bugs fixed 2026-03-19)             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  1  IMU noise      still on table, heartbeat only, 5 min               ║
║  2  Encoder noise  hanging, soft hold DEFAULT_Q_HW, 2 min              ║
║  3  Latency spike  hanging, 0.08 rad impulse per joint                 ║
║  4  Freq sweep     hanging, sin cmd per joint                          ║
║  5  Friction       one joint free, velocity sweeps                     ║
║  6  FR profile     hanging, FR leg only                                ║
║  all               1,2,3,4 in sequence                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test",  type=str, default="1")
    p.add_argument("--joint", type=int, default=None)
    p.add_argument("--dur",   type=float, default=None)
    args = p.parse_args()

    print_banner()

    # Verify mapping correctness at startup
    print("Verifying joint mapping...")
    SDK = ['FR_hip','FR_th','FR_kn','FL_hip','FL_th','FL_kn',
           'RR_hip','RR_th','RR_kn','RL_hip','RL_th','RL_kn']
    ok = all(SDK[sdk_to_isaac[i]] == JNAMES_ISAAC[i] for i in range(12))
    if ok:
        print("  sdk_to_isaac mapping: ✓ correct\n")
    else:
        print("  sdk_to_isaac mapping: ✗ ERROR — check mapping!")
        for i in range(12):
            s = sdk_to_isaac[i]
            match = "✓" if SDK[s] == JNAMES_ISAAC[i] else "✗"
            print(f"    Isaac[{i}]={JNAMES_ISAAC[i]:8s}  SDK[{s}]={SDK[s]:8s} {match}")
        sys.exit(1)

    print("Connecting to Go1...")
    udp.Recv(); udp.GetRecv(state)
    print("Connected.\n")

    run = args.test.split(",") if "," in args.test else [args.test]
    if "all" in run: run = ["1","2","3","4"]

    for t in run:
        if   t=="1": test_imu_noise(args.dur or 300)
        elif t=="2": test_encoder_noise(args.dur or 120)
        elif t=="3": test_latency_spike(args.joint)
        elif t=="4": test_frequency_sweep(args.joint)
        elif t=="5": test_friction_curve(args.joint or 5)
        elif t=="6": test_fr_motor_profile()
        else:        print(f"Unknown test: {t}")

    print("\nDone.")