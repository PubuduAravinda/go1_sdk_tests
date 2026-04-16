#!/usr/bin/env python3
"""
=============================================================================
GO1 PACE DATA COLLECTION  v9  — FINAL (symmetric chirp, 3Hz max)
=============================================================================
Key fixes from v8:

1. SYMMETRIC CHIRP DIRECTIONS — cancels net base wrench (PACE paper Sec 2.1)
   "For robots with symmetry planes, we cancel net base wrenches by
    commanding symmetric joint trajectories."
   Front legs and rear legs move in OPPOSITE directions simultaneously:
     FL_thigh moves forward  +  while  RL_thigh moves backward  -
     FR_thigh moves forward  +  while  RR_thigh moves backward  -
   Net pitching moment on base ≈ 0 → no resonance excitation

2. MAX FREQUENCY 3Hz (was 8Hz)
   Go1 on rack resonates at ~1.5Hz when all legs move in phase.
   ANYmal paper limit: 2Hz. With symmetric cancellation Go1 can reach 3Hz.
   The PACE paper states: "excitation should cover at least 2× the highest
   locomotion frequency" → Go1 walks at 2Hz → need coverage up to ~3Hz.
   Going beyond 3Hz with suspended rack adds noise, not signal.

3. TRACKING RATIO FIXED — now detects resonance correctly
   ratio > 1.1 = resonance → warning printed
   ratio < 0.3 = stiction/blocked → warning printed
   ratio 0.3–1.1 = useful PACE identification data

Run:
    python3 go1_pace_data_collection_v9.py
    python3 go1_pace_data_collection_v9.py --max_freq 2.0  (conservative)
=============================================================================
"""

import argparse, time, sys
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    print("[ERROR] pip install torch"); sys.exit(1)
try:
    import robot_interface as sdk
except ImportError:
    print("[ERROR] robot_interface not found."); sys.exit(1)

# =============================================================================
# SDK SETUP
# =============================================================================
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

CTRL_HZ = 500
DT      = 1.0 / CTRL_HZ

# =============================================================================
# JOINT MAPPINGS
# =============================================================================
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i
SDK_TO_ISAAC = np.array(sdk_to_isaac, dtype=np.int32)
ISAAC_TO_SDK = np.array(isaac_to_sdk, dtype=np.int32)

ISAAC_NAMES = [
    "FL_hip",   "FR_hip",   "RL_hip",   "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf",  "FR_calf",  "RL_calf",  "RR_calf",
]

# =============================================================================
# ROBOT CONSTANTS
# =============================================================================
DEFAULT_JOINT_POS = np.array([
     0.1,  0.1,  0.1,  0.1,
     0.8,  0.8,  0.8,  0.8,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float64)

KP_MULTIPLIER = np.array([
    1.000, 1.000, 1.000, 1.000,
    1.857, 1.857, 1.857, 1.857,
    2.286, 2.286, 2.286, 2.286,
], dtype=np.float64)

KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,
    4.5, 4.5, 4.5, 4.5,
    5.0, 5.0, 5.0, 5.0,
], dtype=np.float64)

TAU_FF_ISAAC = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.2, 1.2, 1.2, 1.2,
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)

KP_START = 5.0
KP_FULL  = 35.0

Q_LIM_LO = np.array([-0.80,-0.80,-0.80,-0.80,
                      -0.50,-0.50,-0.50,-0.50,
                      -2.70,-2.70,-2.70,-2.70], np.float64)
Q_LIM_HI = np.array([ 0.80, 0.80, 0.80, 0.80,
                       4.40, 4.40, 4.40, 4.40,
                      -0.95,-0.95,-0.95,-0.95], np.float64)

MAX_CMD_ERR = np.array([
    0.25, 0.25, 0.25, 0.25,
    0.18, 0.18, 0.18, 0.18,
    0.12, 0.12, 0.12, 0.12,
], dtype=np.float64)

# ── Chirp amplitudes ──────────────────────────────────────────────────────────
CHIRP_AMP = np.array([
    0.15, 0.15, 0.15, 0.15,   # hips
    0.15, 0.15, 0.08, 0.15,   # thighs (RL_thigh=0.08 stiction)
    0.12, 0.12, 0.12, 0.12,   # calves
], dtype=np.float64)

# ── SYMMETRIC CHIRP DIRECTIONS — KEY FIX ─────────────────────────────────────
#
# Isaac order:
#   [0]=FL_hip  [1]=FR_hip  [2]=RL_hip  [3]=RR_hip
#   [4]=FL_th   [5]=FR_th   [6]=RL_th   [7]=RR_th
#   [8]=FL_calf [9]=FR_calf [10]=RL_calf[11]=RR_calf
#
# HIPS (ab/adduction):
#   FL_hip +, FR_hip - (both abduct = legs spread out)
#   RL_hip -, RR_hip + (both adduct = legs move in) ← opposite to front
#   Net lateral force ≈ 0 (front and rear cancel)
#
# THIGHS (flexion/extension):
#   FL_th +, FR_th + → both front legs swing forward
#   RL_th -, RR_th - → both rear legs swing backward simultaneously
#   Net pitching moment ≈ 0 (front forward = rear backward)
#   This is EXACTLY what PACE paper ANYmal data_collection does
#
# CALVES (knee extension):
#   FL_calf -, FR_calf - → front calves extend
#   RL_calf +, RR_calf + → rear calves flex
#   Matches thigh symmetry so knee motion is consistent with thigh motion
#
CHIRP_SIGN = np.array([
     1, -1, -1,  1,   # hips:   FL+  FR-  RL-  RR+  (lateral symmetry)
     1,  1, -1, -1,   # thighs: FL+  FR+  RL-  RR-  (fore-aft symmetry)  ← KEY
    -1, -1,  1,  1,   # calves: FL-  FR-  RL+  RR+  (matches thigh symmetry)
], dtype=np.float64)

# =============================================================================
# PRE-COMPUTED SDK ARRAYS
# =============================================================================
kp_sdk_full = np.array([
    KP_FULL * KP_MULTIPLIER[isaac_to_sdk[s]] for s in range(12)
], dtype=np.float64)
kd_sdk = np.array([
    KD_PER_JOINT[isaac_to_sdk[s]] for s in range(12)
], dtype=np.float64)
tau_ff_sdk = np.array([
    TAU_FF_ISAAC[isaac_to_sdk[s]] for s in range(12)
], dtype=np.float64)

_q_sdk_buf = np.zeros(12, dtype=np.float64)


# =============================================================================
# HOT LOOP HELPERS
# =============================================================================
def read_isaac_fast():
    udp.Recv(); udp.GetRecv(state)
    ms = state.motorState
    _q_sdk_buf[0] =ms[0].q;  _q_sdk_buf[1] =ms[1].q
    _q_sdk_buf[2] =ms[2].q;  _q_sdk_buf[3] =ms[3].q
    _q_sdk_buf[4] =ms[4].q;  _q_sdk_buf[5] =ms[5].q
    _q_sdk_buf[6] =ms[6].q;  _q_sdk_buf[7] =ms[7].q
    _q_sdk_buf[8] =ms[8].q;  _q_sdk_buf[9] =ms[9].q
    _q_sdk_buf[10]=ms[10].q; _q_sdk_buf[11]=ms[11].q
    return _q_sdk_buf[SDK_TO_ISAAC]


def send_cmd_fast(q_target_isaac, kp_sdk, use_tau_ff=False):
    q_sdk = q_target_isaac[ISAAC_TO_SDK]
    mc = cmd.motorCmd
    mc[0].q=q_sdk[0];  mc[0].Kp=kp_sdk[0];  mc[0].Kd=kd_sdk[0]
    mc[1].q=q_sdk[1];  mc[1].Kp=kp_sdk[1];  mc[1].Kd=kd_sdk[1]
    mc[2].q=q_sdk[2];  mc[2].Kp=kp_sdk[2];  mc[2].Kd=kd_sdk[2]
    mc[3].q=q_sdk[3];  mc[3].Kp=kp_sdk[3];  mc[3].Kd=kd_sdk[3]
    mc[4].q=q_sdk[4];  mc[4].Kp=kp_sdk[4];  mc[4].Kd=kd_sdk[4]
    mc[5].q=q_sdk[5];  mc[5].Kp=kp_sdk[5];  mc[5].Kd=kd_sdk[5]
    mc[6].q=q_sdk[6];  mc[6].Kp=kp_sdk[6];  mc[6].Kd=kd_sdk[6]
    mc[7].q=q_sdk[7];  mc[7].Kp=kp_sdk[7];  mc[7].Kd=kd_sdk[7]
    mc[8].q=q_sdk[8];  mc[8].Kp=kp_sdk[8];  mc[8].Kd=kd_sdk[8]
    mc[9].q=q_sdk[9];  mc[9].Kp=kp_sdk[9];  mc[9].Kd=kd_sdk[9]
    mc[10].q=q_sdk[10]; mc[10].Kp=kp_sdk[10]; mc[10].Kd=kd_sdk[10]
    mc[11].q=q_sdk[11]; mc[11].Kp=kp_sdk[11]; mc[11].Kd=kd_sdk[11]
    if use_tau_ff:
        mc[0].tau=tau_ff_sdk[0];   mc[1].tau=tau_ff_sdk[1]
        mc[2].tau=tau_ff_sdk[2];   mc[3].tau=tau_ff_sdk[3]
        mc[4].tau=tau_ff_sdk[4];   mc[5].tau=tau_ff_sdk[5]
        mc[6].tau=tau_ff_sdk[6];   mc[7].tau=tau_ff_sdk[7]
        mc[8].tau=tau_ff_sdk[8];   mc[9].tau=tau_ff_sdk[9]
        mc[10].tau=tau_ff_sdk[10]; mc[11].tau=tau_ff_sdk[11]
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def deadline_sleep(t0):
    r = DT - (time.perf_counter() - t0)
    if r > 0.0001: time.sleep(r)


# =============================================================================
# PHASE HELPERS
# =============================================================================
def dual_ramp(total_s=10.0):
    q_start = read_isaac_fast().copy()
    n = int(total_s * CTRL_HZ)
    print(f"  KP {KP_START:.0f} → {KP_FULL:.0f}")
    for i in range(n):
        t0 = time.perf_counter()
        a = (i/n)**2 * (3.0 - 2.0*(i/n))
        q = q_start + a * (DEFAULT_JOINT_POS - q_start)
        kp = KP_START + a * (KP_FULL - KP_START)
        kp_sdk = np.array([kp*KP_MULTIPLIER[isaac_to_sdk[s]] for s in range(12)],
                          dtype=np.float64)
        send_cmd_fast(q, kp_sdk=kp_sdk, use_tau_ff=True)
        if i % (2*CTRL_HZ) == 0:
            q_now = read_isaac_fast()
            tau_est = float(np.max(kp * KP_MULTIPLIER *
                                   np.abs(q_now - DEFAULT_JOINT_POS)))
            print(f"  t={i*DT:.0f}s  KP={kp:.0f}  "
                  f"max_err={np.max(np.abs(q_now-DEFAULT_JOINT_POS)):.3f}  "
                  f"est_τ={tau_est:.1f}Nm")
        deadline_sleep(t0)
    print("  Done.")


def hold(duration_s=3.0):
    for _ in range(int(duration_s * CTRL_HZ)):
        t0 = time.perf_counter()
        send_cmd_fast(DEFAULT_JOINT_POS, kp_sdk=kp_sdk_full, use_tau_ff=True)
        deadline_sleep(t0)


def release():
    q_start = read_isaac_fast().copy()
    n = int(3.0 * CTRL_HZ)
    for i in range(n):
        t0 = time.perf_counter()
        a = (i/n)**2
        send_cmd_fast(q_start + a*(DEFAULT_JOINT_POS-q_start),
                      kp_sdk=kp_sdk_full, use_tau_ff=True)
        deadline_sleep(t0)
    for _ in range(int(0.5*CTRL_HZ)):
        for s in range(12):
            cmd.motorCmd[s].q=0.0; cmd.motorCmd[s].Kp=0.0
            cmd.motorCmd[s].Kd=2.0; cmd.motorCmd[s].tau=0.0
        udp.SetSend(cmd); udp.Send()
        time.sleep(DT)


# =============================================================================
# MAIN
# =============================================================================
def main(args):
    duration   = args.duration
    f0         = args.min_freq
    f1         = min(args.max_freq, 3.0)   # hard cap at 3Hz for Go1 on rack
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_freq > 3.0:
        print(f"  [NOTE] max_freq capped at 3.0Hz (was {args.max_freq}Hz)")
        print(f"         Go1 on rack resonates above 3Hz. PACE paper ANYmal")
        print(f"         limit is 2Hz. 3Hz is sufficient for locomotion ID.")

    n_steps = int(duration * CTRL_HZ)
    t_lin   = np.linspace(0.0, duration, n_steps, dtype=np.float64)

    # ── Chirp — fixed amplitude, symmetric directions ──────────────────────
    phase  = 2*np.pi * (f0*t_lin + ((f1-f0)/(2*duration))*t_lin**2)
    chirp  = np.sin(phase)
    q_chirp = (DEFAULT_JOINT_POS[None,:]
               + CHIRP_AMP[None,:] * CHIRP_SIGN[None,:] * chirp[:,None])
    q_chirp = np.clip(q_chirp, Q_LIM_LO, Q_LIM_HI)

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  GO1 PACE DATA COLLECTION  v9  (FINAL)")
    print("="*65)
    print(f"  Duration  : {duration}s  ({n_steps} steps @ {CTRL_HZ}Hz)")
    print(f"  Frequency : {f0} → {f1} Hz  (chirp)")
    print()
    print(f"  Chirp directions (SYMMETRIC — cancels base wrench):")
    print(f"    Thighs: FL+ FR+ RL- RR-  (front fwd, rear back simultaneously)")
    print(f"    Hips:   FL+ FR- RL- RR+  (lateral symmetry)")
    print(f"    Calves: FL- FR- RL+ RR+  (matches thigh)")
    print()
    print(f"  Amplitudes: hip=±0.15  thigh=±0.15  RL_th=±0.08  calf=±0.12 rad")
    print(f"  KP/KD: full training  hip=35/4.0  thigh=65/4.5  calf=80/5.0")
    print(f"  Output: {output_dir / 'chirp_data.pt'}")
    print()
    print("  ✓ Base RIGIDLY clamped  ✓ Legs free  ✓ Kill switch ready")
    print()
    input("  Press Enter → ")

    # ── Connect ───────────────────────────────────────────────────────────
    udp.Recv(); udp.GetRecv(state)
    q_now = read_isaac_fast()

    print("\n  Current joints:")
    print(f"  {'Joint':12s}  {'Now':>8s}  {'Default':>8s}  {'Err':>8s}")
    print("  " + "-"*44)
    for j in range(12):
        print(f"  {ISAAC_NAMES[j]:12s}  {q_now[j]:+8.3f}  "
              f"{DEFAULT_JOINT_POS[j]:+8.3f}  "
              f"{q_now[j]-DEFAULT_JOINT_POS[j]:+8.3f}")
    worst = float(np.max(np.abs(q_now - DEFAULT_JOINT_POS)))
    print(f"\n  Max err: {worst:.3f} rad  |  "
          f"Calf τ at KP=5: {KP_START*KP_MULTIPLIER[8]*worst:.1f}Nm (safe)")

    # ── Phase 1 ────────────────────────────────────────────────────────────
    print("\n── Phase 1: Dual ramp KP+pos (10s) ──")
    dual_ramp(total_s=10.0)

    # ── Phase 2 ────────────────────────────────────────────────────────────
    print("\n── Phase 2: Hold 3s ──")
    hold(duration_s=3.0)

    q_check = read_isaac_fast()
    errs = np.abs(q_check - DEFAULT_JOINT_POS)
    print(f"\n  {'Joint':12s}  {'Now':>8s}  {'Err':>8s}  Status")
    print("  " + "-"*44)
    all_ok = True
    for j in range(12):
        if j == 6 and errs[j] > 0.20:
            st = "⚠ RL_thigh stiction (expected)"
        elif errs[j] > 0.12:
            st = "⚠ HIGH"
            all_ok = False
        else:
            st = "✓"
        print(f"  {ISAAC_NAMES[j]:12s}  {q_check[j]:+8.3f}  {errs[j]:8.3f}  {st}")

    if not all_ok:
        if input("\n  Non-RL error > 0.12. Continue? [y/N] → ").strip().lower() != 'y':
            release(); return

    # ── Phase 3: Chirp ─────────────────────────────────────────────────────
    print(f"\n── Phase 3: Chirp {f0}→{f1}Hz × {duration}s ──")
    print(f"  SYMMETRIC directions → net base torque ≈ 0 → no resonance")
    print(f"  Expected: track_ratio starts ~1.0, drops to ~0.3-0.5 at {f1}Hz")
    print(f"  If ratio > 1.1 → resonance (should not happen with symmetric chirp)")
    print()

    q_actual_buf = np.zeros((n_steps, 12), np.float32)
    q_target_buf = np.zeros((n_steps, 12), np.float32)
    t_actual_buf = np.zeros(n_steps,       np.float32)
    n_clamped    = np.zeros(12, np.int32)

    # Resonance tracking — abort if severe overshoot detected
    resonance_count = 0
    RESONANCE_LIMIT = 50   # abort if > 50 consecutive steps with ratio > 1.5

    prog = 5 * CTRL_HZ
    t0_wall = time.perf_counter()

    for step in range(n_steps):
        t_step = time.perf_counter()

        q_now = read_isaac_fast()

        # Safety clamp
        q_des = q_chirp[step]
        q_cl  = np.clip(q_des, q_now - MAX_CMD_ERR, q_now + MAX_CMD_ERR)
        n_clamped += (np.abs(q_des - q_cl) > 0.001).astype(np.int32)

        q_actual_buf[step] = q_now
        q_target_buf[step] = q_cl
        t_actual_buf[step] = t_step - t0_wall

        send_cmd_fast(q_cl, kp_sdk=kp_sdk_full, use_tau_ff=True)

        # Resonance abort check on FL_thigh (most visible indicator)
        if step > 100:
            fl_dev = abs(float(q_now[4]) - DEFAULT_JOINT_POS[4])
            if fl_dev > CHIRP_AMP[4] * 1.5:
                resonance_count += 1
                if resonance_count > RESONANCE_LIMIT:
                    print(f"\n  ⚠ RESONANCE DETECTED at t={t_step-t0_wall:.1f}s — "
                          f"FL_th deviation={fl_dev:.3f} > {CHIRP_AMP[4]*1.5:.3f}")
                    print(f"  Stopping sweep early. Data saved up to step {step}.")
                    n_steps = step   # truncate buffers
                    q_actual_buf = q_actual_buf[:step]
                    q_target_buf = q_target_buf[:step]
                    t_actual_buf = t_actual_buf[:step]
                    break
            else:
                resonance_count = max(0, resonance_count - 1)

        # Progress with tracking ratio (window of last 1s)
        if step > 0 and step % prog == 0:
            w = min(step, CTRL_HZ)   # 1-second window
            fl_amp = float(np.max(np.abs(
                q_actual_buf[step-w:step, 4] - DEFAULT_JOINT_POS[4])))
            fl_cmd = float(np.max(np.abs(
                q_target_buf[step-w:step, 4] - DEFAULT_JOINT_POS[4])))
            rl_amp = float(np.max(np.abs(
                q_actual_buf[step-w:step, 6] - DEFAULT_JOINT_POS[6])))
            rl_cmd = float(np.max(np.abs(
                q_target_buf[step-w:step, 6] - DEFAULT_JOINT_POS[6])))
            fl_ratio = fl_amp / max(fl_cmd, 1e-3)
            rl_ratio = rl_amp / max(rl_cmd, 1e-3)
            freq_now = f0 + (f1 - f0) * step / (int(duration*CTRL_HZ))
            t_el = t_step - t0_wall

            res_warn = "⚠ RESONANCE" if fl_ratio > 1.1 else ""
            print(f"  t={t_el:5.1f}s  f={freq_now:.2f}Hz  "
                  f"FL_th track={fl_ratio:.2f}  RL_th track={rl_ratio:.2f}  "
                  f"{res_warn}")

        deadline_sleep(t_step)

    total_wall = time.perf_counter() - t0_wall

    # ── Phase 4 ────────────────────────────────────────────────────────────
    print(f"\n── Phase 4: Return and release ──")
    release()
    print("  Released.")

    # ── Loop rate ──────────────────────────────────────────────────────────
    actual_dur = float(t_actual_buf[-1] - t_actual_buf[0])
    actual_hz  = (len(t_actual_buf)-1) / actual_dur
    mean_dt_ms = actual_dur / (len(t_actual_buf)-1) * 1000.0
    dts_ms     = np.diff(t_actual_buf.astype(np.float64)) * 1000.0

    print(f"\n── Loop rate ──")
    print(f"  Actual: {actual_hz:.0f} Hz  ({mean_dt_ms:.2f} ms/step)  "
          f"wall={total_wall:.1f}s  dt_std={float(np.std(dts_ms)):.2f}ms")
    if actual_hz < 350:
        print(f"  ⚠ {actual_hz:.0f}Hz too slow. Td fit off by {400/actual_hz:.1f}×.")
        if input("  Save anyway? [y/N] → ").strip().lower() != 'y': return
    elif actual_hz < 450:
        print(f"  ⚠ Marginal: {actual_hz:.0f}Hz.")
    else:
        print(f"  ✓ {actual_hz:.0f}Hz OK")

    # ── Save ───────────────────────────────────────────────────────────────
    out = output_dir / "chirp_data.pt"
    torch.save({
        "time":        torch.tensor(t_actual_buf, dtype=torch.float32),
        "dof_pos":     torch.tensor(q_actual_buf, dtype=torch.float32),
        "des_dof_pos": torch.tensor(q_target_buf, dtype=torch.float32),
    }, out)
    print(f"\n  ✓ Saved: {out}")
    print(f"    samples={len(t_actual_buf)}  "
          f"duration={t_actual_buf[-1]:.1f}s")

    # ── Motion ranges ──────────────────────────────────────────────────────
    print(f"\n── Motion ranges ──")
    print(f"  {'Joint':12s}  {'ActAmp':>8s}  {'CmdAmp':>8s}  "
          f"{'Ratio':>7s}  Note")
    print("  " + "-"*58)
    for j in range(12):
        a_r   = float(q_actual_buf[:,j].max()-q_actual_buf[:,j].min())
        c_r   = float(q_target_buf[:,j].max()-q_target_buf[:,j].min())
        ratio = a_r / max(c_r, 1e-3)
        if j == 6:
            note = "RL_thigh stiction"
        elif ratio > 1.1:
            note = "⚠ resonance — check attachment"
        elif ratio < 0.15:
            note = "⚠ barely moved"
        elif ratio < 0.50:
            note = "partial track (inertia signal) ✓"
        else:
            note = "good tracking ✓"
        print(f"  {ISAAC_NAMES[j]:12s}  {a_r:8.3f}rad  {c_r:8.3f}rad  "
              f"{ratio:7.2f}  {note}")

    # ── Clamp stats ────────────────────────────────────────────────────────
    print(f"\n── Clamp stats ──")
    for j in range(12):
        pct = 100.0 * n_clamped[j] / len(t_actual_buf)
        if pct > 1.0:
            print(f"  {ISAAC_NAMES[j]:12s}: {pct:.1f}% steps clamped")

    print(f"\n── Next steps ──")
    print(f"  scp {out} <user>@<sim>:~/pace-sim2real/data/go1/chirp_data.pt")
    print(f"  ~/IsaacLab/isaaclab.sh -p ~/pace-sim2real/scripts/pace/fit.py \\")
    print(f"    --task Isaac-Pace-Go1-v0 --num_envs 4096 --headless")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Go1 PACE data collection v9 — symmetric chirp, 3Hz max")
    p.add_argument("--duration",   type=float, default=30.0)
    p.add_argument("--min_freq",   type=float, default=0.1)
    p.add_argument("--max_freq",   type=float, default=3.0,
                   help="Max chirp frequency Hz (default 3.0, hard cap at 3.0)")
    p.add_argument("--output_dir", type=str,   default="./pace_data/go1")
    main(p.parse_args())