#!/usr/bin/env python3
"""
=============================================================================
GO1 PACE DATA COLLECTION  v10  — PER-LEG CHIRP  (delay-identifiable)
=============================================================================
Key change from v9:

PROBLEM WITH v9:
  4-leg simultaneous chirp → resonance above 3Hz → max 3Hz → delay not visible
  16ms delay creates only 17° phase shift at 3Hz → CMA-ES can't separate
  delay from inertia → Td collapses to ~0.9 steps (1.8ms) instead of 8 steps

FIX — PER-LEG CHIRP:
  Chirp ONE leg at a time, other 3 legs held at default (rigid)
  Single leg momentum = 1/4 of 4-leg → base stays stable up to 6Hz
  At 6Hz: 16ms delay = 34° phase shift → clearly visible to CMA-ES
  Repeat for all 4 legs, concatenate into one .pt file

DATA STRUCTURE:
  FL leg active (30s):  FL_hip/thigh/calf chirp  |  FR/RL/RR flat at default
  FR leg active (30s):  FR_hip/thigh/calf chirp  |  FL/RL/RR flat at default
  RL leg active (30s):  RL_hip/thigh/calf chirp  |  FL/FR/RR flat at default
  RR leg active (30s):  RR_hip/thigh/calf chirp  |  FL/FR/RL flat at default
  ─────────────────────────────────────────────────────────────────────────
  Total: 120s, 60000 steps, all 12 joint columns, time continuous

  PACE fit.py loads this as one long trajectory — no modification needed.
  Flat columns (non-active legs) contribute ~0 loss and don't hurt the fit.
  Each leg segment contains delay-identifiable data at 6Hz.

WHY 6Hz:
  delay = 16ms = 8 steps at 500Hz
  At 6Hz: phase shift = 360° × 0.016 × 6 = 34.6°  ← clearly visible
  At 3Hz: phase shift = 17.3°  ← lost in noise (what caused Td→0.9 in run 1)
  At 6Hz single leg: torque = Ia × amp × (2π×6)² ≈ 0.015 × 0.08 × 1421 = 1.7Nm ✓

Run:
    python3 go1_pace_data_collection_v10.py
    python3 go1_pace_data_collection_v10.py --leg_duration 40 --max_freq 6.0
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

# Per-leg joint indices in Isaac order
# FL: hip=0, thigh=4, calf=8
# FR: hip=1, thigh=5, calf=9
# RL: hip=2, thigh=6, calf=10
# RR: hip=3, thigh=7, calf=11
LEG_JOINTS = {
    "FL": [0, 4, 8],
    "FR": [1, 5, 9],
    "RL": [2, 6, 10],
    "RR": [3, 7, 11],
}

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

# Safety clamp — tighter at full KP
# τ_max = KP_eff × MAX_CMD_ERR
# hip:   35 × 0.20 = 7.0 Nm  ✓
# thigh: 65 × 0.15 = 9.8 Nm  ✓
# calf:  80 × 0.10 = 8.0 Nm  ✓
MAX_CMD_ERR = np.array([
    0.20, 0.20, 0.20, 0.20,
    0.15, 0.15, 0.15, 0.15,
    0.10, 0.10, 0.10, 0.10,
], dtype=np.float64)

# Chirp amplitudes — SMALLER than v9 to handle 6Hz safely
# At 6Hz: acc = amp × (2π×6)² = amp × 1421
# Torque = Ia × acc: for Ia=0.015, amp=0.08: τ = 0.015 × 0.08 × 1421 = 1.7Nm ✓
CHIRP_AMP = np.array([
    0.12, 0.12, 0.12, 0.12,   # hips   ±0.12 rad
    0.10, 0.10, 0.05, 0.10,   # thighs ±0.10 (RL_thigh=0.05 — stiction)
    0.08, 0.08, 0.08, 0.08,   # calves ±0.08 rad
], dtype=np.float64)

# Pre-computed SDK arrays
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


def dual_ramp(total_s=10.0):
    """Ramp KP and position to default simultaneously."""
    q_start = read_isaac_fast().copy()
    n = int(total_s * CTRL_HZ)
    worst = float(np.max(np.abs(q_start - DEFAULT_JOINT_POS)))
    print(f"  Ramp {total_s:.0f}s: KP {KP_START:.0f}→{KP_FULL:.0f}  "
          f"max_err={worst:.3f}rad")
    for i in range(n):
        t0 = time.perf_counter()
        a = (i/n)**2 * (3.0 - 2.0*(i/n))
        q = q_start + a * (DEFAULT_JOINT_POS - q_start)
        kp = KP_START + a * (KP_FULL - KP_START)
        kp_sdk = np.array([kp*KP_MULTIPLIER[isaac_to_sdk[s]]
                           for s in range(12)], dtype=np.float64)
        send_cmd_fast(q, kp_sdk=kp_sdk, use_tau_ff=True)
        if i % (2*CTRL_HZ) == 0:
            q_now = read_isaac_fast()
            print(f"  t={i*DT:.0f}s  KP={kp:.0f}  "
                  f"max_err={np.max(np.abs(q_now-DEFAULT_JOINT_POS)):.3f}")
        deadline_sleep(t0)
    print("  Done.")


def hold_default(duration_s=2.0):
    for _ in range(int(duration_s * CTRL_HZ)):
        t0 = time.perf_counter()
        send_cmd_fast(DEFAULT_JOINT_POS, kp_sdk=kp_sdk_full, use_tau_ff=True)
        deadline_sleep(t0)


def release():
    q_s = read_isaac_fast().copy()
    n = int(2.0 * CTRL_HZ)
    for i in range(n):
        t0 = time.perf_counter()
        send_cmd_fast(q_s + (i/n)**2*(DEFAULT_JOINT_POS-q_s),
                      kp_sdk=kp_sdk_full, use_tau_ff=True)
        deadline_sleep(t0)
    for _ in range(int(0.5*CTRL_HZ)):
        for s in range(12):
            cmd.motorCmd[s].q=0.0; cmd.motorCmd[s].Kp=0.0
            cmd.motorCmd[s].Kd=2.0; cmd.motorCmd[s].tau=0.0
        udp.SetSend(cmd); udp.Send()
        time.sleep(DT)


# =============================================================================
# SINGLE LEG CHIRP
# =============================================================================
def chirp_one_leg(leg_name, f0, f1, duration, t_offset):
    """
    Chirp one leg's 3 joints, hold other 9 joints at default.

    leg_name: "FL", "FR", "RL", or "RR"
    t_offset: cumulative time offset for continuous timestamp
    Returns (q_actual, q_target, t_actual) arrays of shape [N, 12] / [N]
    """
    active_joints = LEG_JOINTS[leg_name]   # e.g. [0, 4, 8] for FL
    n_steps = int(duration * CTRL_HZ)
    t_lin   = np.linspace(0.0, duration, n_steps, dtype=np.float64)

    # Chirp signal
    phase = 2*np.pi * (f0*t_lin + ((f1-f0)/(2*duration))*t_lin**2)
    chirp = np.sin(phase)

    # Build trajectory: only active joints move, others hold default
    q_traj = np.tile(DEFAULT_JOINT_POS, (n_steps, 1))   # [N, 12] all at default
    for j in active_joints:
        # Hip sign: FR(1) and RR(3) use negative sign
        sign = -1.0 if j in [1, 3] else 1.0
        # Calf uses negative direction
        if j in [8, 9, 10, 11]:
            sign = -1.0
        q_traj[:, j] = DEFAULT_JOINT_POS[j] + CHIRP_AMP[j] * sign * chirp
    q_traj = np.clip(q_traj, Q_LIM_LO, Q_LIM_HI)

    # Buffers
    q_actual_buf = np.zeros((n_steps, 12), np.float32)
    q_target_buf = np.zeros((n_steps, 12), np.float32)
    t_actual_buf = np.zeros(n_steps,       np.float32)
    n_clamped    = np.zeros(12, np.int32)

    print(f"\n  Chirping {leg_name} leg "
          f"(joints: {[ISAAC_NAMES[j] for j in active_joints]})")
    print(f"  Freq: {f0}→{f1}Hz  Duration: {duration}s  "
          f"Amp: hip=±{CHIRP_AMP[active_joints[0]]:.2f}  "
          f"thigh=±{CHIRP_AMP[active_joints[1]]:.2f}  "
          f"calf=±{CHIRP_AMP[active_joints[2]]:.2f} rad")
    if leg_name == "RL":
        print(f"  RL_thigh: ±{CHIRP_AMP[6]:.2f} rad (stiction fault — reduced)")

    t_wall_start = time.perf_counter()
    prog = 5 * CTRL_HZ

    for step in range(n_steps):
        t_step = time.perf_counter()

        q_now = read_isaac_fast()

        # Safety clamp on active joints only
        q_des = q_traj[step].copy()
        q_cl  = q_des.copy()
        for j in active_joints:
            lo = q_now[j] - MAX_CMD_ERR[j]
            hi = q_now[j] + MAX_CMD_ERR[j]
            q_cl[j] = np.clip(q_des[j], lo, hi)
            if abs(q_des[j] - q_cl[j]) > 0.001:
                n_clamped[j] += 1

        q_actual_buf[step] = q_now
        q_target_buf[step] = q_cl
        t_actual_buf[step] = t_offset + (t_step - t_wall_start)

        send_cmd_fast(q_cl, kp_sdk=kp_sdk_full, use_tau_ff=True)

        if step > 0 and step % prog == 0:
            freq_now = f0 + (f1-f0)*step/n_steps
            t_el = t_step - t_wall_start
            # Show active joint tracking
            j_th = active_joints[1]   # thigh of this leg
            a_th = float(q_now[j_th])
            c_th = float(q_cl[j_th])
            # 1-second tracking ratio
            w = min(step, CTRL_HZ)
            amp_act = float(np.max(np.abs(
                q_actual_buf[step-w:step, j_th] - DEFAULT_JOINT_POS[j_th])))
            amp_cmd = float(np.max(np.abs(
                q_target_buf[step-w:step, j_th] - DEFAULT_JOINT_POS[j_th])))
            ratio = amp_act / max(amp_cmd, 1e-3)
            res = "⚠ RES" if ratio > 1.1 else ""
            print(f"  t={t_el:5.1f}s  f={freq_now:.2f}Hz  "
                  f"{leg_name}_th={a_th:+.3f}(cmd={c_th:+.3f})  "
                  f"track={ratio:.2f}  {res}")

        deadline_sleep(t_step)

    total = time.perf_counter() - t_wall_start
    actual_hz = (n_steps-1) / float(t_actual_buf[-1] - t_actual_buf[0]
                                    + 1e-9)

    # Motion range check
    for j in active_joints:
        a_r = float(q_actual_buf[:,j].max()-q_actual_buf[:,j].min())
        c_r = float(q_target_buf[:,j].max()-q_target_buf[:,j].min())
        ratio = a_r / max(c_r, 1e-3)
        note = "⚠ STICTION" if ratio < 0.15 else ("⚠ LOW" if ratio < 0.40 else "✓")
        print(f"  {ISAAC_NAMES[j]:12s}: actual={a_r:.3f}  cmd={c_r:.3f}  "
              f"ratio={ratio:.2f}  {note}  clamped={n_clamped[j]}")

    print(f"  Loop rate: {actual_hz:.0f}Hz  wall={total:.1f}s")

    return q_actual_buf, q_target_buf, t_actual_buf


# =============================================================================
# MAIN
# =============================================================================
def main(args):
    leg_duration = args.leg_duration
    f0           = args.min_freq
    f1           = min(args.max_freq, 6.0)   # hard cap at 6Hz for single-leg
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_freq > 6.0:
        print(f"  [NOTE] max_freq capped at 6.0Hz (was {args.max_freq}Hz)")

    total_steps = int(leg_duration * CTRL_HZ) * 4

    print("\n" + "="*65)
    print("  GO1 PACE DATA COLLECTION  v10  — PER-LEG CHIRP")
    print("="*65)
    print(f"  Strategy    : ONE leg at a time → stable base → 6Hz possible")
    print(f"  Frequency   : {f0} → {f1} Hz per leg")
    print(f"  Per-leg dur : {leg_duration}s  ({int(leg_duration*CTRL_HZ)} steps)")
    print(f"  Total dur   : {leg_duration*4:.0f}s  ({total_steps} steps)")
    print()
    print(f"  WHY THIS WORKS FOR DELAY IDENTIFICATION:")
    print(f"    At 6Hz: 16ms delay = 34° phase shift → CMA-ES can see it")
    print(f"    At 3Hz: 16ms delay = 17° phase shift → lost in noise")
    print(f"    Single leg → 4× less momentum → base stays stable at 6Hz")
    print()
    print(f"  Amplitudes: hip=±0.12  thigh=±0.10  RL_th=±0.05  calf=±0.08 rad")
    print(f"  Output    : {output_dir / 'chirp_data.pt'}")
    print()
    print("  ✓ Base RIGIDLY clamped  ✓ Legs FREE  ✓ Kill switch ready")
    print()
    input("  Press Enter when ready → ")

    # Connect
    udp.Recv(); udp.GetRecv(state)

    print("\n── Startup ramp (10s) ──")
    dual_ramp(total_s=10.0)
    hold_default(duration_s=2.0)

    # Check Phase 2 stability
    q_check = read_isaac_fast()
    errs = np.abs(q_check - DEFAULT_JOINT_POS)
    print(f"\n  Hold check — max err: {np.max(errs):.3f} rad  "
          f"(RL_th: {errs[6]:.3f})")

    # ── Collect per-leg data ────────────────────────────────────────────────
    all_actual  = []
    all_target  = []
    all_time    = []
    t_offset    = 0.0

    leg_order = ["FL", "FR", "RR", "RL"]   # RL last — stiction won't affect others

    for leg in leg_order:
        print(f"\n{'='*50}")
        print(f"  LEG {leg}  ({leg_order.index(leg)+1}/4)")
        print(f"{'='*50}")

        # Short pause between legs
        if len(all_actual) > 0:
            print("  Pause 3s before next leg...")
            hold_default(duration_s=3.0)

        qa, qt, ta = chirp_one_leg(leg, f0, f1, leg_duration, t_offset)

        all_actual.append(qa)
        all_target.append(qt)
        all_time.append(ta)

        # Update time offset for continuous timestamps
        t_offset = float(ta[-1]) + DT   # +DT so no gap

    # ── Release ────────────────────────────────────────────────────────────
    print(f"\n── Release ──")
    release()
    print("  Released.")

    # ── Concatenate all legs ──────────────────────────────────────────────
    q_actual_all = np.concatenate(all_actual, axis=0)   # [4N, 12]
    q_target_all = np.concatenate(all_target, axis=0)   # [4N, 12]
    t_actual_all = np.concatenate(all_time,   axis=0)   # [4N]

    total_actual_s = float(t_actual_all[-1])
    actual_hz = len(t_actual_all) / total_actual_s

    print(f"\n── Combined dataset ──")
    print(f"  Total samples : {len(t_actual_all)}")
    print(f"  Total duration: {total_actual_s:.1f}s")
    print(f"  Mean rate     : {actual_hz:.0f} Hz")

    if actual_hz < 350:
        print(f"  ⚠ {actual_hz:.0f}Hz too slow. Td fit off by {400/actual_hz:.1f}×.")
        if input("  Save anyway? [y/N] → ").strip().lower() != 'y':
            return

    # ── Save ───────────────────────────────────────────────────────────────
    out = output_dir / "chirp_data.pt"
    torch.save({
        "time":        torch.tensor(t_actual_all, dtype=torch.float32),
        "dof_pos":     torch.tensor(q_actual_all, dtype=torch.float32),
        "des_dof_pos": torch.tensor(q_target_all, dtype=torch.float32),
    }, out)
    print(f"\n  ✓ Saved: {out}")
    print(f"    samples={len(t_actual_all)}  duration={total_actual_s:.1f}s")

    # ── Motion summary ─────────────────────────────────────────────────────
    print(f"\n── Motion ranges per joint ──")
    print(f"  {'Joint':12s}  {'Active_leg':>10s}  {'ActRange':>9s}  "
          f"{'CmdRange':>9s}  {'Ratio':>6s}  Note")
    print("  " + "-"*65)
    leg_of = {0:"FL",1:"FR",2:"RL",3:"RR",
              4:"FL",5:"FR",6:"RL",7:"RR",
              8:"FL",9:"FR",10:"RL",11:"RR"}
    for j in range(12):
        a_r   = float(q_actual_all[:,j].max()-q_actual_all[:,j].min())
        c_r   = float(q_target_all[:,j].max()-q_target_all[:,j].min())
        ratio = a_r / max(c_r, 1e-3)
        if j == 6:
            note = "RL_thigh stiction"
        elif ratio < 0.15:
            note = "⚠ barely moved"
        elif ratio < 0.40:
            note = "partial (inertia signal) ✓"
        else:
            note = "✓"
        print(f"  {ISAAC_NAMES[j]:12s}  {leg_of[j]:>10s}  "
              f"{a_r:9.3f}rad  {c_r:9.3f}rad  {ratio:6.2f}  {note}")

    print(f"\n── Next steps ──")
    print(f"  scp {out} <user>@<sim>:~/pace-sim2real/data/go1/chirp_data.pt")
    print()
    print(f"  ~/IsaacLab/isaaclab.sh -p ~/pace-sim2real/scripts/pace/fit.py \\")
    print(f"    --task Isaac-Pace-Go1-v0 --num_envs 4096 --headless")
    print()
    print(f"  Expected improvement: Td should converge to 7-10 steps (14-20ms)")
    print(f"  Score target: < 0.030 (vs 0.046 from 3Hz data)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Go1 PACE v10 — per-leg chirp for delay identification")
    p.add_argument("--leg_duration", type=float, default=30.0,
                   help="Duration per leg in seconds (default 30, total=120s)")
    p.add_argument("--min_freq",     type=float, default=0.1)
    p.add_argument("--max_freq",     type=float, default=6.0,
                   help="Max frequency Hz (default 6.0, hard cap at 6.0)")
    p.add_argument("--output_dir",   type=str,   default="./pace_data/go1")
    main(p.parse_args())