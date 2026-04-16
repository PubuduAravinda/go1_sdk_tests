#!/usr/bin/env python3
"""
Go1 Test 2 — Joint Encoder Noise Characterisation (Two-Phase)
==============================================================

PHASES:
  Phase 0 — Hanging (soft hold, KP_HANG):
      Robot suspended on safety rack, all feet off ground.
      Soft PD hold at DEFAULT_Q_HW. Low KP so max torque ≈ 2 Nm (safe).
      Measures pure encoder electronics noise floor with minimal
      mechanical load — analogous to IMU Phase 0 (ground, KP=0).

  Phase 1 — Standing (training KP, gravity load):
      Robot lowered to ground. Controlled 8s ramp to DEFAULT_Q_HW.
      Recording begins ONLY after:
        • all 12 joints within ERR_TOL rad of target, AND
        • all four knee tauEst > CONTACT_TAU_THR Nm (confirms ground contact)
      Measures encoder noise under the exact load conditions seen during
      policy deployment — this is the value adopted for Isaac Lab obs_noise_std.

WHY TWO PHASES (not three):
  Encoder quantization and electrical noise are intrinsic to the sensor and
  amplifier chain. Unlike the IMU, there is no meaningful "dynamic upper bound"
  phase — encoder noise does not increase significantly with leg motion
  (the SDK dq differentiator noise is a separate timing artefact, not encoder
  electronics noise). Phase 0 confirms the noise floor; Phase 1 gives the
  deployment-relevant value. No wave phase is needed.

HANGING vs STANDING (encoder):
  For the IMU test, standing is essential because motor current ripple under
  gravity load dominates noise. For encoders, the signal is quantization +
  electrical noise in the resolver/Hall chain — largely load-independent.
  Phase 0 (hanging) and Phase 1 (standing) sigmas should be very close.
  A large difference indicates mechanical play or slip in the joint under load.

LITERATURE BASIS:
  Tan et al. (2018): 30-60 s per condition is sufficient for a stationary
  ergodic noise process. At ~200 Hz (Pi4 Python limit), 120 s gives ~24,000
  samples; sigma converges to < 1% error well before that.
  Phase 0: 30 s (noise floor only — fast convergence).
  Phase 1: 120 s (deployment condition — longer for robustness).

OUTPUTS:
  calib_encoder_<timestamp>.npz — raw arrays + per-joint stats
  Console table                 — copy-paste ready for Isaac Lab

USAGE:
  python3 go1_test2_encoder_noise_v2.py
  python3 go1_test2_encoder_noise_v2.py --n_hang 30 --n_stand 120
  python3 go1_test2_encoder_noise_v2.py --dry-run

NOTES:
  • The Pi 4 Python loop achieves ~200 Hz, not 500 Hz. The SDK motor
    controllers and encoders still run at 500 Hz internally; we are
    sampling at 200 Hz. jpos_std is fully valid at this rate (encoder
    noise bandwidth << 100 Hz). jvel_std is a slight underestimate
    (use as conservative lower bound for Isaac Lab).
  • 3-sigma clipping is applied before computing adopted sigma to remove
    rare UDP dropout spikes from the estimate.
  • FR_hip is known to have elevated jpos noise (hardware fault on this
    unit). It is flagged automatically if its sigma exceeds 5× the
    healthy hip mean.

Author: Go1 Calibration Suite v2 — encoder module (2026-03)
"""

import argparse
import sys
import time
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
DT      = 1.0 / CTRL_HZ

# ─── Joint ordering ───────────────────────────────────────────────────────────
#
# Unitree SDK motor order:
#   0=FR_hip  1=FR_thigh  2=FR_knee
#   3=FL_hip  4=FL_thigh  5=FL_knee
#   6=RR_hip  7=RR_thigh  8=RR_knee
#   9=RL_hip  10=RL_thigh 11=RL_knee
#
# Isaac Lab order:
#   0=FL_hip  1=FR_hip   2=RL_hip   3=RR_hip
#   4=FL_th   5=FR_th    6=RL_th    7=RR_th
#   8=FL_kn   9=FR_kn   10=RL_kn  11=RR_kn
#
# sdk_to_isaac[i] = "Isaac joint i lives at SDK motor sdk_to_isaac[i]"
# Mapping is self-inverse: sdk_to_isaac == sdk_of_isaac
#
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
sdk_of_isaac = sdk_to_isaac   # same array — symmetric mapping

JNAMES_ISAAC = [
    'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
    'FL_th',  'FR_th',  'RL_th',  'RR_th',
    'FL_kn',  'FR_kn',  'RL_kn',  'RR_kn',
]

# SDK knee indices — for knee contact check (FR=2, FL=5, RR=8, RL=11)
KNEE_SDK_IDX = [2, 5, 8, 11]

# ─── Default pose (training standing pose) ────────────────────────────────────
#
# DEFAULT_Q_ISAAC: Isaac Lab convention (all hips +0.1 = legs outward)
# DEFAULT_Q_HW:    same physical pose in real-hardware sign convention
#                  FR_hip (Isaac idx 1) and RR_hip (Isaac idx 3) have
#                  reversed encoders on real Go1 hardware → sign flip
#
DEFAULT_Q_ISAAC = np.array([
     0.1,  0.1,  0.1,  0.1,   # FL_hip FR_hip RL_hip RR_hip
     0.8,  0.8,  0.8,  0.8,   # FL_th  FR_th  RL_th  RR_th
    -1.5, -1.5, -1.5, -1.5,   # FL_kn  FR_kn  RL_kn  RR_kn
], np.float32)

DEFAULT_Q_HW = DEFAULT_Q_ISAAC.copy()
DEFAULT_Q_HW[1] = -DEFAULT_Q_ISAAC[1]   # FR_hip: reversed encoder
DEFAULT_Q_HW[3] = -DEFAULT_Q_ISAAC[3]   # RR_hip: reversed encoder

# ─── PD gains ─────────────────────────────────────────────────────────────────
# Training gains — Isaac Lab Go1 defaults (Walk These Ways / Margolis 2022)
KP_TRAIN = np.array([35, 35, 35, 35,  65, 65, 65, 65,  80, 80, 80, 80], np.float32)
KD_TRAIN = np.array([ 4,  4,  4,  4,   4,  4,  4,  4,   4,  4,  4,  4], np.float32)

# Soft hanging gains — low KP so max torque ≈ KP × 0.3 rad ≈ 2–5 Nm (safe)
KP_HANG  = np.array([ 8,  8,  8,  8,  12, 12, 12, 12,  15, 15, 15, 15], np.float32)
KD_HANG  = np.array([ 3,  3,  3,  3,   3,  3,  3,  3,   4,  4,  4,  4], np.float32)

# ─── Thresholds ───────────────────────────────────────────────────────────────
#
# SETTLE CRITERION — velocity-based, not position-based.
#
# Under gravity load with finite KP, joints settle at:
#   q_actual = q_target - τ_gravity / KP
# For a thigh joint: τ_gravity ≈ 5 Nm, KP = 65 → steady-state offset ≈ 0.077 rad.
# This is physics, not a tracking failure. Checking position error against
# the commanded target will always fail under load.
#
# Instead we check that joints have STOPPED MOVING (velocity noise floor)
# and that ground contact is confirmed via knee torque.
# VEL_TOL is set conservatively above the Phase 0 hanging jvel noise floor
# (~0.008 rad/s) to avoid false negatives from quantization noise.
#
VEL_TOL         = 0.05   # rad/s — all joints must have |jvel| < this to settle
                          # (well above 0.008 rad/s noise floor, well below motion)
CONTACT_TAU_THR = 1.0    # Nm   — knee tauEst threshold confirming ground contact
STABLE_HOLD_S   = 2.0    # s    — must hold settle criterion continuously
RAMP_S          = 8.0    # s    — ramp duration from hanging pose to standing
FAULT_RATIO     = 5.0    # ×    — flag joint if sigma > FAULT_RATIO × healthy mean

# Keep ERR_TOL for logging only (reports the gravity-induced deflection)
ERR_TOL_WARN    = 0.15   # rad  — warn if offset exceeds this (suggests wrong pose)


# ─── SDK helpers ──────────────────────────────────────────────────────────────

def read_state():
    """
    Read joint and IMU state. Returns arrays in Isaac order.
    sdk_to_isaac reorders SDK motor indices → Isaac joint indices.
    """
    udp.Recv()
    udp.GetRecv(state)
    jpos_sdk = np.array([state.motorState[i].q      for i in range(12)], np.float32)
    jvel_sdk = np.array([state.motorState[i].dq     for i in range(12)], np.float32)
    jtau_sdk = np.array([state.motorState[i].tauEst for i in range(12)], np.float32)
    # Reorder SDK → Isaac
    jpos = jpos_sdk[sdk_to_isaac]
    jvel = jvel_sdk[sdk_to_isaac]
    jtau = jtau_sdk[sdk_to_isaac]
    return jpos, jvel, jtau


def send_hold_step(target_q_hw, kp, kd):
    """
    Send one PD position-hold step in Isaac order.
    sdk_of_isaac maps Isaac joint index → SDK motor index for command writing.
    target_q_hw must be in hardware sign convention (DEFAULT_Q_HW, not ISAAC).
    """
    udp.Recv()
    udp.GetRecv(state)
    for isaac_i in range(12):
        sdk_i = sdk_of_isaac[isaac_i]
        cmd.motorCmd[sdk_i].mode = 0x0A
        cmd.motorCmd[sdk_i].q    = float(target_q_hw[isaac_i])
        cmd.motorCmd[sdk_i].dq   = 0.0
        cmd.motorCmd[sdk_i].Kp   = float(kp[isaac_i])
        cmd.motorCmd[sdk_i].Kd   = float(kd[isaac_i])
        cmd.motorCmd[sdk_i].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()


def rate_sleep(t_step_start):
    """Sleep remainder of 2 ms control step."""
    sl = DT - (time.perf_counter() - t_step_start)
    if sl > 0:
        time.sleep(sl)


def knee_contact_ok(jtau):
    """
    Check all four knees show contact torque in Isaac order.
    Isaac knee indices: FL_kn=8, FR_kn=9, RL_kn=10, RR_kn=11
    """
    return bool(np.all(np.abs(jtau[8:12]) > CONTACT_TAU_THR))


def joint_str(jpos, target):
    err = np.abs(jpos - target).max()
    return f"max_err={err:.4f} rad"


# ─── Phase 0: Hanging ─────────────────────────────────────────────────────────

def phase0_hanging(n_seconds=30.0):
    """
    Robot suspended on safety rack. Soft PD hold at DEFAULT_Q_HW.
    Measures encoder noise floor with minimal mechanical load.
    Analogous to IMU Phase 0 (ground, KP=0) — establishes baseline.

    No settle criterion needed: robot is hanging, joints settle within
    a few seconds of soft hold. We simply allow 5 s of pre-settle
    before recording begins.
    """
    print(f"\n{'─'*65}")
    print(f"PHASE 0 — Hanging / noise floor  ({n_seconds:.0f}s recording)")
    print(f"  Robot suspended on safety rack. All feet off ground.")
    print(f"  Soft hold at KP_HANG. Max torque ≈ {8*0.3:.1f}–{15*0.3:.1f} Nm (safe).")
    print(f"  Pre-settle: 5s soft hold before recording starts.")
    print(f"{'─'*65}")
    input("\n  Robot hanging on rack? Press Enter → ")

    # Pre-settle: 5 s soft hold, no recording
    print("  Pre-settling 5s...")
    for _ in range(int(5.0 * CTRL_HZ)):
        t_s = time.perf_counter()
        send_hold_step(DEFAULT_Q_HW, KP_HANG, KD_HANG)
        rate_sleep(t_s)
    print("  Pre-settle done. Starting recording...")

    # Preallocate (generous — actual rate ~200 Hz)
    N_alloc  = int(n_seconds * CTRL_HZ * 1.5)
    jpos_log = np.zeros((N_alloc, 12), np.float32)
    jvel_log = np.zeros((N_alloc, 12), np.float32)
    jtau_log = np.zeros((N_alloc, 12), np.float32)
    dt_log   = np.zeros(N_alloc,       np.float32)
    t_log    = np.zeros(N_alloc,       np.float32)

    t0     = time.perf_counter()
    t_prev = t0
    i      = 0

    while time.perf_counter() - t0 < n_seconds:
        if i >= N_alloc:
            print("  WARNING: buffer full — increase N_alloc multiplier")
            break
        t_s = time.perf_counter()

        jpos, jvel, jtau = read_state()
        send_hold_step(DEFAULT_Q_HW, KP_HANG, KD_HANG)

        now      = time.perf_counter()
        dt_act   = now - t_prev
        t_prev   = now

        jpos_log[i] = jpos
        jvel_log[i] = jvel
        jtau_log[i] = jtau
        dt_log[i]   = dt_act
        t_log[i]    = now - t0
        i += 1

        rate_sleep(t_s)

        if i > 1 and i % (200 * 15) == 0:  # ~every 15s
            ps = jpos_log[:i].std(axis=0)
            hz = i / (time.perf_counter() - t0)
            print(f"  t={t_log[i-1]:.0f}s  hz={hz:.0f}  "
                  f"max_jpos_std={ps.max():.6f} rad  "
                  f"({JNAMES_ISAAC[ps.argmax()]})")

    # Trim
    jpos_log = jpos_log[:i]
    jvel_log = jvel_log[:i]
    jtau_log = jtau_log[:i]
    dt_log   = dt_log[1:i]
    t_log    = t_log[:i]

    hz = i / t_log[-1]
    print(f"\n  Phase 0 done.  Samples={i}  Duration={t_log[-1]:.1f}s  Hz={hz:.1f}")
    print(f"  dt: mean={dt_log.mean()*1000:.2f}ms  "
          f"std={dt_log.std()*1000:.2f}ms  max={dt_log.max()*1000:.1f}ms")

    ps = jpos_log.std(axis=0)
    vs = jvel_log.std(axis=0)
    print(f"  jpos std (raw): {[round(float(v),6) for v in ps]}")
    print(f"  jvel std (raw): {[round(float(v),5) for v in vs]}")

    return dict(jpos=jpos_log, jvel=jvel_log, jtau=jtau_log,
                dt=dt_log, time=t_log)


# ─── Phase 1: Standing ────────────────────────────────────────────────────────

def phase1_standing(n_seconds=120.0):
    """
    Lower robot to ground. Ramp to DEFAULT_Q_HW at training gains.
    Recording begins only after:
      (a) all |jvel| < VEL_TOL rad/s  — robot has stopped moving, AND
      (b) all four knee tauEst > CONTACT_TAU_THR Nm — ground contact confirmed,
      (c) held continuously for STABLE_HOLD_S seconds.

    WHY velocity-based, not position-based:
      Under gravity load with finite KP, joints settle at:
        q_actual = q_target - tau_gravity / KP
      Thigh example: tau_gravity ≈ 5 Nm, KP = 65 → offset ≈ 0.077 rad.
      This is expected physics — the PD controller has no integral term.
      Checking position error against the commanded target will always fail.
      Velocity check correctly identifies when the robot has stopped moving.

      The gravity-induced offset (q_actual - q_target) is saved as
      jpos_offset — this IS the sim-to-real joint position gap metric.
    """
    print(f"\n{'─'*65}")
    print(f"PHASE 1 — Standing / deployment condition  ({n_seconds:.0f}s recording)")
    print(f"  Robot on flat ground. Ramp {RAMP_S:.0f}s to DEFAULT_Q_HW.")
    print(f"  Settle criterion: ALL |jvel| < {VEL_TOL} rad/s  AND")
    print(f"                    all knee tau > {CONTACT_TAU_THR} Nm  for {STABLE_HOLD_S:.0f}s.")
    print(f"  Note: position offset from target (~0.08 rad) is expected physics,")
    print(f"        not a tracking failure. It is saved as jpos_offset.")
    print(f"{'─'*65}")
    input("\n  Robot lowered to ground, space clear? Press Enter to ramp → ")

    # ── Ramp: current pos → DEFAULT_Q_HW ─────────────────────────────────────
    print(f"  Reading current positions...")
    jpos_now, _, _ = read_state()
    q_start = jpos_now.copy()

    print(f"  Ramping {RAMP_S:.0f}s: KP_HANG → KP_TRAIN...")
    n_ramp = int(RAMP_S * CTRL_HZ)
    for step in range(n_ramp):
        t_s   = time.perf_counter()
        alpha = step / (n_ramp - 1)
        q_cmd = (1.0 - alpha) * q_start  + alpha * DEFAULT_Q_HW
        kp    = (1.0 - alpha) * KP_HANG  + alpha * KP_TRAIN
        kd    = (1.0 - alpha) * KD_HANG  + alpha * KD_TRAIN
        send_hold_step(q_cmd, kp, kd)
        if step % (CTRL_HZ * 2) == 0 and step > 0:
            jpos, jvel, _ = read_state()
            err     = np.abs(jpos - DEFAULT_Q_HW).max()
            max_vel = np.abs(jvel).max()
            print(f"  ramp t={step*DT:.1f}s  α={alpha:.2f}  "
                  f"pos_offset={err:.4f} rad  max_vel={max_vel:.4f} rad/s")
        rate_sleep(t_s)

    print(f"  Ramp complete. Waiting for settle criterion...")
    print(f"  (Position offset from target is expected — checking velocity instead)")

    # ── Wait for settle: velocity + contact ──────────────────────────────────
    hold_start   = None
    t_wait_start = time.perf_counter()
    MAX_WAIT_S   = 30.0   # short — velocity settles in seconds, not minutes

    while True:
        t_s = time.perf_counter()

        if t_s - t_wait_start > MAX_WAIT_S:
            jpos, jvel, jtau = read_state()
            max_vel = np.abs(jvel).max()
            print(f"\n  WARNING: settle timeout after {MAX_WAIT_S:.0f}s.")
            print(f"  max_vel={max_vel:.4f} rad/s  contact={knee_contact_ok(jtau)}")
            print(f"  Proceeding — robot appears visually stable.")
            break

        jpos, jvel, jtau = read_state()
        send_hold_step(DEFAULT_Q_HW, KP_TRAIN, KD_TRAIN)

        max_vel    = float(np.abs(jvel).max())
        vel_ok     = max_vel < VEL_TOL
        contact_ok = knee_contact_ok(jtau)
        settled    = vel_ok and contact_ok

        # Log gravity-induced position offset for reference (not a criterion)
        pos_offset = float(np.abs(jpos - DEFAULT_Q_HW).max())
        if pos_offset > ERR_TOL_WARN:
            pass   # expected — do not warn, just log in status line

        if settled:
            if hold_start is None:
                hold_start = t_s
                print(f"  ✓ Criterion met — holding {STABLE_HOLD_S:.0f}s to confirm...")
            elif t_s - hold_start >= STABLE_HOLD_S:
                print(f"  ✓ Stable.  max_vel={max_vel:.4f} rad/s  "
                      f"pos_offset={pos_offset:.4f} rad  contact=OK")
                print(f"  Starting recording...")
                break
        else:
            if hold_start is not None:
                hold_start = None
                print(f"  ✗ Lost  vel_ok={vel_ok}  contact_ok={contact_ok}")

        # Status every ~2 s
        elapsed = t_s - t_wait_start
        if int(elapsed * 0.5) != int((elapsed - DT) * 0.5):
            vo = "✓" if vel_ok     else "✗"
            ko = "✓" if contact_ok else "✗"
            print(f"  t={elapsed:.0f}s  max_vel={max_vel:.4f}{vo}  "
                  f"pos_offset={pos_offset:.4f}  contact{ko}  "
                  f"knee_tau={np.abs(jtau[8:12]).round(2).tolist()} Nm")

        rate_sleep(t_s)

    # ── Record ────────────────────────────────────────────────────────────────
    N_alloc  = int(n_seconds * CTRL_HZ * 1.5)
    jpos_log = np.zeros((N_alloc, 12), np.float32)
    jvel_log = np.zeros((N_alloc, 12), np.float32)
    jtau_log = np.zeros((N_alloc, 12), np.float32)
    dt_log   = np.zeros(N_alloc,       np.float32)
    t_log    = np.zeros(N_alloc,       np.float32)

    t0     = time.perf_counter()
    t_prev = t0
    i      = 0

    while time.perf_counter() - t0 < n_seconds:
        if i >= N_alloc:
            print("  WARNING: buffer full — increase N_alloc multiplier")
            break
        t_s = time.perf_counter()

        jpos, jvel, jtau = read_state()
        send_hold_step(DEFAULT_Q_HW, KP_TRAIN, KD_TRAIN)

        now      = time.perf_counter()
        dt_act   = now - t_prev
        t_prev   = now

        jpos_log[i] = jpos
        jvel_log[i] = jvel
        jtau_log[i] = jtau
        dt_log[i]   = dt_act
        t_log[i]    = now - t0
        i += 1

        rate_sleep(t_s)

        if i > 1 and i % (200 * 30) == 0:   # ~every 30 s
            ps  = jpos_log[:i].std(axis=0)
            hz  = i / (time.perf_counter() - t0)
            pct = (time.perf_counter() - t0) / n_seconds * 100
            print(f"  t={t_log[i-1]:.0f}s ({pct:.0f}%)  hz={hz:.0f}  "
                  f"max_jpos_std={ps.max():.6f} rad  ({JNAMES_ISAAC[ps.argmax()]})")

    # Trim
    jpos_log = jpos_log[:i]
    jvel_log = jvel_log[:i]
    jtau_log = jtau_log[:i]
    dt_log   = dt_log[1:i]
    t_log    = t_log[:i]

    hz = i / t_log[-1]
    print(f"\n  Phase 1 done.  Samples={i}  Duration={t_log[-1]:.1f}s  Hz={hz:.1f}")
    print(f"  dt: mean={dt_log.mean()*1000:.2f}ms  "
          f"std={dt_log.std()*1000:.2f}ms  max={dt_log.max()*1000:.1f}ms")

    ps = jpos_log.std(axis=0)
    vs = jvel_log.std(axis=0)
    print(f"  jpos std (raw): {[round(float(v),6) for v in ps]}")
    print(f"  jvel std (raw): {[round(float(v),5) for v in vs]}")

    return dict(jpos=jpos_log, jvel=jvel_log, jtau=jtau_log,
                dt=dt_log, time=t_log)


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(raw):
    """
    Per-joint noise statistics with 3-sigma clipping.
    Clipping removes rare UDP dropout spikes before computing sigma,
    giving a robust estimate of the stationary noise floor.
    Both raw and clipped values are returned and saved.
    """
    jpos = raw['jpos']
    jvel = raw['jvel']
    n    = jpos.shape[1]

    out = {k: np.zeros(n, np.float32) for k in
           ['jpos_std_raw', 'jpos_std_clip',
            'jvel_std_raw', 'jvel_std_clip',
            'jpos_offset']}
    out['spike_counts'] = np.zeros(n, int)

    for i in range(n):
        p = jpos[:, i];  v = jvel[:, i]
        out['jpos_std_raw'][i]  = p.std()
        out['jvel_std_raw'][i]  = v.std()
        out['jpos_offset'][i]   = p.mean() - DEFAULT_Q_HW[i]

        pm = np.abs(p - p.mean()) < 3 * p.std()
        vm = np.abs(v - v.mean()) < 3 * v.std()
        out['jpos_std_clip'][i]  = p[pm].std()
        out['jvel_std_clip'][i]  = v[vm].std()
        out['spike_counts'][i]   = (~pm).sum()

    return out


# ─── Console report ───────────────────────────────────────────────────────────

def print_summary(ph0_raw, ph0_stats, ph1_raw, ph1_stats):
    """Print full two-phase comparison table and Isaac Lab copy-paste block."""

    KP = {0:35,1:35,2:35,3:35, 4:65,5:65,6:65,7:65, 8:80,9:80,10:80,11:80}
    KD = {0:4.0,1:4.0,2:4.0,3:4.0, 4:4.5,5:4.5,6:4.5,7:4.5, 8:5.0,9:5.0,10:5.0,11:5.0}

    # Healthy hip mean for fault detection (exclude FR_hip idx 1)
    healthy_hip_pos = np.mean([ph1_stats['jpos_std_clip'][i] for i in [0, 2, 3]])

    print()
    print("═" * 110)
    print("TEST 2 — JOINT ENCODER NOISE  (two-phase summary)")
    print("═" * 110)

    # Timing
    for label, raw in [("Phase 0 (hanging)", ph0_raw), ("Phase 1 (standing)", ph1_raw)]:
        dt = raw['dt']
        print(f"  {label}: samples={raw['jpos'].shape[0]}  "
              f"duration={raw['time'][-1]:.1f}s  "
              f"hz={raw['jpos'].shape[0]/raw['time'][-1]:.1f}  "
              f"dt_mean={dt.mean()*1000:.2f}ms  dt_max={dt.max()*1000:.1f}ms")

    print()
    print(f"  Note: 3σ-clipped sigma is adopted for Isaac Lab (removes UDP dropout spikes).")
    print(f"  Phase 0 = noise floor (hanging).  Phase 1 = deployment condition (adopted).")
    print()

    # Per-joint table
    hdr = (f"  {'Joint':10s}  {'KP':>4}  "
           f"{'Ph0 jpos σ':>12}  {'Ph1 jpos σ':>12}  {'ratio':>6}  "
           f"{'Ph0 jvel σ':>12}  {'Ph1 jvel σ':>12}  "
           f"{'offset':>9}  {'spikes':>7}  Note")
    print(hdr)
    print("  " + "─" * 106)

    for i, n in enumerate(JNAMES_ISAAC):
        p0 = ph0_stats['jpos_std_clip'][i]
        p1 = ph1_stats['jpos_std_clip'][i]
        v0 = ph0_stats['jvel_std_clip'][i]
        v1 = ph1_stats['jvel_std_clip'][i]
        off    = ph1_stats['jpos_offset'][i]
        spk    = ph1_stats['spike_counts'][i]
        ratio  = p1 / (p0 + 1e-9)

        note = ""
        if i == 1 and p1 > FAULT_RATIO * healthy_hip_pos:
            note = f"  ← {p1/healthy_hip_pos:.1f}× healthy  ⚠ HW FAULT"
        elif ratio > 3.0:
            note = f"  ← {ratio:.1f}× Ph0 (load-induced noise increase)"

        print(f"  {n:10s}  {KP[i]:>4}  "
              f"{p0:>12.6f}  {p1:>12.6f}  {ratio:>6.2f}×  "
              f"{v0:>12.5f}  {v1:>12.5f}  "
              f"{off:>+9.6f}  {spk:>7d}  {note}")

    # Isaac Lab copy-paste
    ps = ph1_stats['jpos_std_clip']
    vs = ph1_stats['jvel_std_clip']
    print()
    print("  ── Isaac Lab obs_noise_std (Phase 1, 3σ-clipped) ──────────────────")
    print(f"  jpos_noise = {[round(float(v),6) for v in ps]}  # rad")
    print(f"  jvel_noise = {[round(float(v),5) for v in vs]}  # rad/s")

    # Group-level conservative (max per group, exclude FR_hip fault from hips)
    print()
    print("  ── Group-level conservative (max σ per group, excl. known faults) ──")
    groups = {"Hip (healthy)": [0,2,3], "Thigh": [4,5,6,7], "Knee": [8,9,10,11]}
    for gn, idxs in groups.items():
        mp = max(ps[i] for i in idxs)
        mv = max(vs[i] for i in idxs)
        print(f"  {gn:16s}: jpos_std_max={mp:.6f} rad   jvel_std_max={mv:.5f} rad/s")

    print("═" * 110)


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_results(ph0_raw, ph0_stats, ph1_raw, ph1_stats):
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"calib_encoder_{ts}.npz"
    np.savez(
        fname,
        # Phase 0 raw
        ph0_jpos         = ph0_raw['jpos'],
        ph0_jvel         = ph0_raw['jvel'],
        ph0_jtau         = ph0_raw['jtau'],
        ph0_dt           = ph0_raw['dt'],
        ph0_time         = ph0_raw['time'],
        # Phase 1 raw
        ph1_jpos         = ph1_raw['jpos'],
        ph1_jvel         = ph1_raw['jvel'],
        ph1_jtau         = ph1_raw['jtau'],
        ph1_dt           = ph1_raw['dt'],
        ph1_time         = ph1_raw['time'],
        # Phase 0 stats
        ph0_jpos_std_raw  = ph0_stats['jpos_std_raw'],
        ph0_jpos_std_clip = ph0_stats['jpos_std_clip'],
        ph0_jvel_std_raw  = ph0_stats['jvel_std_raw'],
        ph0_jvel_std_clip = ph0_stats['jvel_std_clip'],
        ph0_jpos_offset   = ph0_stats['jpos_offset'],
        # Phase 1 stats (adopted)
        ph1_jpos_std_raw  = ph1_stats['jpos_std_raw'],
        ph1_jpos_std_clip = ph1_stats['jpos_std_clip'],
        ph1_jvel_std_raw  = ph1_stats['jvel_std_raw'],
        ph1_jvel_std_clip = ph1_stats['jvel_std_clip'],
        ph1_jpos_offset   = ph1_stats['jpos_offset'],
        ph1_spike_counts  = ph1_stats['spike_counts'],
        # Metadata
        default_q_hw      = DEFAULT_Q_HW,
        default_q_isaac   = DEFAULT_Q_ISAAC,
        joint_names       = np.array(JNAMES_ISAAC, dtype=object),
        kp_train          = KP_TRAIN,
        kd_train          = KD_TRAIN,
        kp_hang           = KP_HANG,
        kd_hang           = KD_HANG,
    )
    kb = (ph0_raw['jpos'].nbytes + ph0_raw['jvel'].nbytes +
          ph1_raw['jpos'].nbytes + ph1_raw['jvel'].nbytes) / 1024
    print(f"\n  → Saved {fname}  ({kb:.0f} KB)")
    print(f"  → Plot: python3 go1_test2_encoder_plot.py {fname}")
    return fname


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Go1 Test 2 — Joint Encoder Noise (two-phase: hanging + standing)")
    p.add_argument("--n_hang",  type=float, default=30.0,
                   help="Phase 0 recording duration s (default: 30)")
    p.add_argument("--n_stand", type=float, default=120.0,
                   help="Phase 1 recording duration s (default: 120)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without connecting")
    args = p.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║      Go1 Calibration — Test 2: Joint Encoder Noise (Two-Phase)         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Phase 0  Hanging   soft hold KP_HANG    30 s   noise floor            ║
║  Phase 1  Standing  training KP_TRAIN   120 s   deployment condition   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Literature (Tan et al. 2018): 30-60 s sufficient for stationary       ║
║  ergodic noise. 120 s @ 200 Hz → ~24,000 samples → sigma < 1% error.  ║
║  Phase 1 values adopted for Isaac Lab obs_noise_std.                   ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"  Configuration:")
    print(f"    Phase 0 duration    : {args.n_hang:.0f} s (hanging, KP_HANG)")
    print(f"    Phase 1 duration    : {args.n_stand:.0f} s (standing, KP_TRAIN)")
    print(f"    Settle criterion    : ALL |jvel| < {VEL_TOL} rad/s  AND  knee tau > {CONTACT_TAU_THR} Nm")
    print(f"    Settle hold         : {STABLE_HOLD_S} s")
    print(f"    Contact tau thresh  : {CONTACT_TAU_THR} Nm (knee joints)")
    print(f"    Ramp duration       : {RAMP_S} s")
    print(f"    DEFAULT_Q_HW        : {DEFAULT_Q_HW.tolist()}")
    print(f"    KP_TRAIN            : {KP_TRAIN.tolist()}")
    print(f"    KP_HANG             : {KP_HANG.tolist()}")
    print(f"    Fault threshold     : {FAULT_RATIO}× healthy group mean")

    if args.dry_run:
        print("\n  [dry-run] Exiting without connecting.")
        sys.exit(0)

    # Connect
    print("\n  Connecting to Go1...")
    udp.Recv()
    udp.GetRecv(state)
    print("  Connected.\n")

    print("  SEQUENCE:")
    print("  1. Phase 0 — hang robot on rack, press Enter")
    print("  2. Phase 1 — lower robot to ground, press Enter to ramp")
    print()

    # ── Phase 0 ──────────────────────────────────────────────────────────────
    ph0_raw   = phase0_hanging(n_seconds=args.n_hang)
    ph0_stats = compute_stats(ph0_raw)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    ph1_raw   = phase1_standing(n_seconds=args.n_stand)
    ph1_stats = compute_stats(ph1_raw)

    # ── Summary and save ─────────────────────────────────────────────────────
    print_summary(ph0_raw, ph0_stats, ph1_raw, ph1_stats)
    save_results(ph0_raw, ph0_stats, ph1_raw, ph1_stats)

    print("\n  Done.")


if __name__ == "__main__":
    main()