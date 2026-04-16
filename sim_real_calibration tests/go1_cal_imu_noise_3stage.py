#!/usr/bin/env python3
"""
Go1 Test 1 — IMU Noise Characterisation (Three-Phase)
=======================================================

PHASES:
  Phase 0 — Ground (stationary, KP=0 KD=2):
      Robot lying on ground, motors off.
      Measures pure sensor noise floor with zero mechanical vibration.
      Records until N_STATIC samples collected.

  Phase 1 — Standing (KP ramped to training values):
      Controlled 8s ramp from lying to standing pose.
      Data recording begins ONLY after:
        • all 12 joints within ERR_TOL rad of target, AND
        • all knee tauEst > CONTACT_TAU_THR Nm (all feet loaded)
      This ensures IMU noise is measured at a stable, loaded stance.

  Phase 2 — Standing + sine wave (thigh/knee oscillation):
      Same KP as Phase 1.
      Sinusoidal offset applied to thigh joints (SDK indices 1,4,7,10).
      Data recording begins only after sine has run for at least one
      full cycle (ensuring steady-state oscillation, not transient).
      Records for N_WAVE seconds of clean oscillation.

WHY THREE PHASES:
  Stationary (Ph0): establishes the sensor electronics noise floor.
  Standing (Ph1):   adds vibration from motor current ripple and gearbox
                    under static load — the relevant baseline for deployment.
  Walking-proxy (Ph2): adds dynamic body acceleration from leg motion —
                    closest proxy to the IMU noise during actual locomotion
                    without requiring the robot to walk unsupported.

IMU CHANNELS (6 total, all logged):
  accelerometer[0,1,2]  m/s²   body frame X(forward) Y(left) Z(up)
  gyroscope[0,1,2]      rad/s  body frame X(roll) Y(pitch) Z(yaw)

EXPECTED RANGES (from datasheet and prior literature):
  Gyroscope stationary:  ~0.006-0.015 rad/s std
  Gyroscope standing:    ~0.010-0.020 rad/s std (2-3× stationary)
  Gyroscope wave:        ~0.015-0.040 rad/s std (body acceleration couples in)
  Accelerometer static:  ~0.02-0.10 m/s² std
  Accelerometer wave:    ~0.10-0.50 m/s² std (sine acceleration ≈ Aω² ≈ 0.4 m/s²)

Run:
  python3 go1_test1_imu_noise.py
  python3 go1_test1_imu_noise.py --n_static 30 --n_stand 30 --n_wave 30
"""

import argparse, math, time, sys
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
isaac_to_sdk = [0] * 12
for i, s in enumerate(sdk_to_isaac):
    isaac_to_sdk[s] = i

JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

# ─── Poses ───────────────────────────────────────────────────────────────────
# SDK-order default (lying) and standing poses
DEFAULT_Q_SDK = {
    'FR_hip':  0.0, 'FR_thigh': 0.8, 'FR_knee': -1.5,
    'FL_hip':  0.0, 'FL_thigh': 0.8, 'FL_knee': -1.5,
    'RR_hip':  0.0, 'RR_thigh': 0.8, 'RR_knee': -1.5,
    'RL_hip':  0.0, 'RL_thigh': 0.8, 'RL_knee': -1.5,
}
STAND_Q_SDK = {
    'FR_hip': -0.05, 'FR_thigh': 0.70, 'FR_knee': -1.40,
    'FL_hip':  0.05, 'FL_thigh': 0.70, 'FL_knee': -1.40,
    'RR_hip': -0.05, 'RR_thigh': 0.70, 'RR_knee': -1.40,
    'RL_hip':  0.05, 'RL_thigh': 0.70, 'RL_knee': -1.40,
}
SDK_JOINT_NAMES = ['FR_hip','FR_thigh','FR_knee',
                   'FL_hip','FL_thigh','FL_knee',
                   'RR_hip','RR_thigh','RR_knee',
                   'RL_hip','RL_thigh','RL_knee']
DEFAULT_Q_ARR = np.array([DEFAULT_Q_SDK[n] for n in SDK_JOINT_NAMES], np.float32)
STAND_Q_ARR   = np.array([STAND_Q_SDK[n]   for n in SDK_JOINT_NAMES], np.float32)

# SDK thigh indices (for sine wave): FR=1, FL=4, RR=7, RL=10
THIGH_SDK_IDX = [1, 4, 7, 10]

# ─── Gains ───────────────────────────────────────────────────────────────────
# Per Isaac-joint: [hips(0-3), thighs(4-7), knees(8-11)]
KP_TRAIN_ISAAC = np.array([35,35,35,35,  65,65,65,65,  80,80,80,80], np.float32)
KD_TRAIN_ISAAC = np.array([ 4, 4, 4, 4, 4.5,4.5,4.5,4.5,  5, 5, 5, 5], np.float32)
KP_HANG_ISAAC  = np.array([ 8, 8, 8, 8,  12,12,12,12,  15,15,15,15], np.float32)
KD_HANG_ISAAC  = np.array([ 3, 3, 3, 3,   3, 3, 3, 3,   4, 4, 4, 4], np.float32)

# Convert to SDK order for sending
KP_TRAIN_SDK = KP_TRAIN_ISAAC[sdk_to_isaac]
KD_TRAIN_SDK = KD_TRAIN_ISAAC[sdk_to_isaac]
KP_HANG_SDK  = KP_HANG_ISAAC[sdk_to_isaac]
KD_HANG_SDK  = KD_HANG_ISAAC[sdk_to_isaac]

# ─── Thresholds ──────────────────────────────────────────────────────────────
ERR_TOL          = 0.06    # rad — all joints must be within this to start recording
CONTACT_TAU_THR  = 1.0     # Nm — knee tauEst threshold for contact confirmation
STABLE_HOLD_S    = 2.0     # seconds joints must be stable before phase recording starts
WAVE_AMP         = 0.10    # rad — sine amplitude on thighs
WAVE_FREQ        = 0.5     # Hz — sine frequency (slow enough for quasi-static check)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_state():
    udp.Recv(); udp.GetRecv(state)
    acc  = np.array(state.imu.accelerometer, np.float32)
    gyro = np.array(state.imu.gyroscope,     np.float32)
    jpos_sdk = np.array([state.motorState[i].q      for i in range(12)], np.float32)
    jtau_sdk = np.array([state.motorState[i].tauEst for i in range(12)], np.float32)
    return acc, gyro, jpos_sdk, jtau_sdk


def send_cmd_sdk(target_sdk, kp_sdk, kd_sdk, tau_ff_sdk=None):
    """Send command in SDK joint order directly."""
    udp.Recv(); udp.GetRecv(state)
    ff = tau_ff_sdk if tau_ff_sdk is not None else np.zeros(12)
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(target_sdk[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(kp_sdk[i])
        cmd.motorCmd[i].Kd   = float(kd_sdk[i])
        cmd.motorCmd[i].tau  = float(ff[i])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def knee_contact(jtau_sdk):
    """Check contact on all four knees via tauEst threshold (SDK indices 2,5,8,11)."""
    return all(abs(jtau_sdk[i]) > CONTACT_TAU_THR for i in [2, 5, 8, 11])


def joint_max_err(jpos_sdk, target_sdk):
    return float(np.abs(jpos_sdk - target_sdk).max())


def imu_str(acc, gyro):
    return (f"acc=[{acc[0]:+.3f} {acc[1]:+.3f} {acc[2]:+.3f}]m/s²  "
            f"gyro=[{gyro[0]:+.4f} {gyro[1]:+.4f} {gyro[2]:+.4f}]rad/s")


# ─── Phase 0: Ground (stationary, KP=0) ──────────────────────────────────────

def phase0_ground(n_seconds=30.0):
    """
    Robot lying on ground, motors off (KP=0, KD=2).
    Pure sensor noise floor with zero mechanical vibration.
    Records for n_seconds seconds.
    """
    print(f"\n{'─'*60}")
    print(f"PHASE 0 — Ground / stationary ({n_seconds:.0f}s)")
    print(f"  Robot should be lying flat on the floor.")
    print(f"  KP=0, KD=2 — motors offer only damping, no holding force.")
    print(f"{'─'*60}")
    input("  Robot on floor, power on? Press Enter → ")

    N    = int(n_seconds * CTRL_HZ)
    acc_log  = np.zeros((N, 3), np.float32)
    gyro_log = np.zeros((N, 3), np.float32)
    t_log    = np.zeros(N, np.float32)

    print(f"  Recording {n_seconds:.0f}s...")
    t0 = time.perf_counter()

    for i in range(N):
        t_s = time.perf_counter()
        udp.Recv(); udp.GetRecv(state)
        # Motors off — only KD damping
        for j in range(12):
            cmd.motorCmd[j].mode = 0x0A; cmd.motorCmd[j].q   = 0.0
            cmd.motorCmd[j].dq   = 0.0;  cmd.motorCmd[j].Kp  = 0.0
            cmd.motorCmd[j].Kd   = 2.0;  cmd.motorCmd[j].tau = 0.0
        safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()

        acc_log[i]  = state.imu.accelerometer
        gyro_log[i] = state.imu.gyroscope
        t_log[i]    = time.perf_counter() - t0

        if i % (CTRL_HZ * 10) == 0 and i > 0:
            t_el = t_log[i]
            print(f"  t={t_el:.0f}s  {imu_str(acc_log[i], gyro_log[i])}")

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    acc_std  = acc_log.std(axis=0)
    gyro_std = gyro_log.std(axis=0)
    print(f"\n  Phase 0 results:")
    print(f"    acc  std: [{acc_std[0]:.5f}  {acc_std[1]:.5f}  {acc_std[2]:.5f}] m/s²")
    print(f"    gyro std: [{gyro_std[0]:.5f} {gyro_std[1]:.5f} {gyro_std[2]:.5f}] rad/s")
    return acc_log, gyro_log, t_log


# ─── Phase 1: Stand up and hold ───────────────────────────────────────────────

def phase1_stand(n_seconds=30.0, ramp_s=8.0):
    """
    Stand up from ground via controlled ramp, then record IMU once stable.
    Recording starts only after all joints within ERR_TOL AND all knees loaded.
    """
    print(f"\n{'─'*60}")
    print(f"PHASE 1 — Standing ({n_seconds:.0f}s recording after stabilisation)")
    print(f"  Ramp: {ramp_s:.0f}s.  Accept: max_err<{ERR_TOL}rad, all knee_tau>{CONTACT_TAU_THR}Nm")
    print(f"{'─'*60}")
    input("  Lift robot to 5-10cm above ground? Press Enter to ramp → ")

    # ── Ramp up ──────────────────────────────────────────────────────────────
    print(f"  Ramping {ramp_s:.0f}s...")
    t0 = time.perf_counter(); step = 0
    while True:
        t_s   = time.perf_counter()
        alpha = min(1.0, (t_s - t0) / ramp_s)
        q_now = DEFAULT_Q_ARR + alpha * (STAND_Q_ARR - DEFAULT_Q_ARR)
        kp_now = KP_HANG_SDK + alpha * (KP_TRAIN_SDK - KP_HANG_SDK)
        kd_now = KD_HANG_SDK + alpha * (KD_TRAIN_SDK - KD_HANG_SDK)
        acc, gyro, jpos_sdk, jtau_sdk = read_state()
        send_cmd_sdk(q_now, kp_now, kd_now)
        if step % (CTRL_HZ * 2) == 0:
            err = joint_max_err(jpos_sdk, STAND_Q_ARR)
            norm = max(float(np.linalg.norm(acc)), 0.1)
            tilt = float(np.degrees(np.arccos(min(1.0, abs(acc[2])/norm))))
            print(f"  t={(t_s-t0):.1f}s α={alpha:.2f} max_err={err:.3f}rad tilt={tilt:.1f}°")
        if alpha >= 1.0: break
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)
        step += 1

    # ── Wait for stable pose ──────────────────────────────────────────────────
    print(f"  Waiting for max_err<{ERR_TOL} AND all knee contact...")
    hold_start = None
    t_wait0 = time.perf_counter()
    while True:
        t_s = time.perf_counter()
        acc, gyro, jpos_sdk, jtau_sdk = read_state()
        send_cmd_sdk(STAND_Q_ARR, KP_TRAIN_SDK, KD_TRAIN_SDK)

        err       = joint_max_err(jpos_sdk, STAND_Q_ARR)
        contact   = knee_contact(jtau_sdk)
        primary_ok = (err < ERR_TOL) and contact

        if primary_ok:
            if hold_start is None:
                hold_start = t_s
                print(f"  ✓ Conditions met — holding {STABLE_HOLD_S:.0f}s for stability...")
            elif (t_s - hold_start) >= STABLE_HOLD_S:
                print(f"  ✓ Stable. Starting recording.")
                break
        else:
            if hold_start is not None:
                hold_start = None  # dropped out

        if (t_s - t_wait0) > 60.0:
            print(f"  Timeout waiting for stability. max_err={err:.4f} contact={contact}")
            print(f"  Proceeding with current pose.")
            break

        if int((t_s - t_wait0) * 0.5) != int(((t_s-t_wait0) - DT) * 0.5):
            norm = max(float(np.linalg.norm(acc)), 0.1)
            tilt = float(np.degrees(np.arccos(min(1.0, abs(acc[2])/norm))))
            ko   = "✓" if contact else "✗"
            eo   = "✓" if err<ERR_TOL else "✗"
            print(f"  err={err:.4f}{eo} contact{ko} tilt={tilt:.1f}°")

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    # ── Record ────────────────────────────────────────────────────────────────
    N        = int(n_seconds * CTRL_HZ)
    acc_log  = np.zeros((N, 3), np.float32)
    gyro_log = np.zeros((N, 3), np.float32)
    err_log  = np.zeros(N, np.float32)
    t_log    = np.zeros(N, np.float32)

    t0 = time.perf_counter()
    for i in range(N):
        t_s = time.perf_counter()
        acc, gyro, jpos_sdk, jtau_sdk = read_state()
        send_cmd_sdk(STAND_Q_ARR, KP_TRAIN_SDK, KD_TRAIN_SDK)
        acc_log[i]  = acc
        gyro_log[i] = gyro
        err_log[i]  = joint_max_err(jpos_sdk, STAND_Q_ARR)
        t_log[i]    = time.perf_counter() - t0
        if i % (CTRL_HZ * 10) == 0 and i > 0:
            print(f"  t={t_log[i]:.0f}s  {imu_str(acc_log[i], gyro_log[i])}  max_err={err_log[i]:.4f}")
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    acc_std  = acc_log.std(axis=0)
    gyro_std = gyro_log.std(axis=0)
    print(f"\n  Phase 1 results:")
    print(f"    acc  std: [{acc_std[0]:.5f}  {acc_std[1]:.5f}  {acc_std[2]:.5f}] m/s²")
    print(f"    gyro std: [{gyro_std[0]:.5f} {gyro_std[1]:.5f} {gyro_std[2]:.5f}] rad/s")
    return acc_log, gyro_log, err_log, t_log


# ─── Phase 2: Standing + sine wave ────────────────────────────────────────────

def phase2_wave(n_seconds=30.0):
    """
    Apply sine wave to thigh joints while standing.
    Records IMU once at least one full wave cycle has completed (steady-state).
    Wave: target_thigh = stand_thigh + WAVE_AMP * sin(2π * WAVE_FREQ * t)
    Applied to SDK thigh indices [1, 4, 7, 10].
    """
    print(f"\n{'─'*60}")
    print(f"PHASE 2 — Standing + sine wave ({n_seconds:.0f}s recording after 1 cycle)")
    print(f"  Wave: ±{WAVE_AMP}rad at {WAVE_FREQ}Hz on all thigh joints")
    print(f"  Recording starts after 1 full cycle ({1/WAVE_FREQ:.1f}s)")
    print(f"{'─'*60}")
    print(f"  Starting sine wave (robot should still be standing)...")

    ONE_CYCLE_S = 1.0 / WAVE_FREQ
    N           = int(n_seconds * CTRL_HZ)
    acc_log     = np.zeros((N, 3), np.float32)
    gyro_log    = np.zeros((N, 3), np.float32)
    err_log     = np.zeros(N, np.float32)
    t_log       = np.zeros(N, np.float32)
    wave_log    = np.zeros(N, np.float32)   # commanded wave offset for reference

    # Run one full cycle first (warm-up, no recording)
    print(f"  Warm-up cycle ({ONE_CYCLE_S:.1f}s)...")
    n_warmup = int(ONE_CYCLE_S * CTRL_HZ)
    t_wave_start = time.perf_counter()
    for i in range(n_warmup):
        t_s  = time.perf_counter()
        wave_t  = t_s - t_wave_start
        wave_offset = WAVE_AMP * math.sin(2 * math.pi * WAVE_FREQ * wave_t)
        target  = STAND_Q_ARR.copy()
        for si in THIGH_SDK_IDX:
            target[si] += wave_offset
        acc, gyro, jpos_sdk, jtau_sdk = read_state()
        send_cmd_sdk(target, KP_TRAIN_SDK, KD_TRAIN_SDK)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    # Confirm still standing before recording
    acc, gyro, jpos_sdk, jtau_sdk = read_state()
    err = joint_max_err(jpos_sdk, STAND_Q_ARR)
    contact = knee_contact(jtau_sdk)
    print(f"  Post-warmup: max_err={err:.4f} contact={knee_contact(jtau_sdk)}")
    if not contact:
        print("  WARNING: foot contact lost during warmup — check robot is still standing")

    # Record n_seconds of steady-state oscillation
    print(f"  Recording {n_seconds:.0f}s...")
    t0 = time.perf_counter()
    for i in range(N):
        t_s  = time.perf_counter()
        wave_t  = t_s - t_wave_start   # continuous from wave start
        wave_offset = WAVE_AMP * math.sin(2 * math.pi * WAVE_FREQ * wave_t)
        target  = STAND_Q_ARR.copy()
        for si in THIGH_SDK_IDX:
            target[si] += wave_offset

        acc, gyro, jpos_sdk, jtau_sdk = read_state()
        send_cmd_sdk(target, KP_TRAIN_SDK, KD_TRAIN_SDK)

        acc_log[i]  = acc
        gyro_log[i] = gyro
        err_log[i]  = joint_max_err(jpos_sdk, STAND_Q_ARR)
        wave_log[i] = wave_offset
        t_log[i]    = time.perf_counter() - t0

        if i % (CTRL_HZ * 10) == 0 and i > 0:
            print(f"  t={t_log[i]:.0f}s  {imu_str(acc_log[i], gyro_log[i])}  wave={wave_offset:+.3f}rad")

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    # Return to standing (remove wave)
    print(f"  Returning to static stand...")
    for _ in range(int(2.0 * CTRL_HZ)):
        t_s = time.perf_counter()
        send_cmd_sdk(STAND_Q_ARR, KP_TRAIN_SDK, KD_TRAIN_SDK)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    acc_std  = acc_log.std(axis=0)
    gyro_std = gyro_log.std(axis=0)
    print(f"\n  Phase 2 results:")
    print(f"    acc  std: [{acc_std[0]:.5f}  {acc_std[1]:.5f}  {acc_std[2]:.5f}] m/s²")
    print(f"    gyro std: [{gyro_std[0]:.5f} {gyro_std[1]:.5f} {gyro_std[2]:.5f}] rad/s")
    return acc_log, gyro_log, err_log, wave_log, t_log


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Go1 Test 1 — IMU Noise (3 phases)")
    p.add_argument("--n_static", type=float, default=30.0, help="Phase 0 duration s")
    p.add_argument("--n_stand",  type=float, default=30.0, help="Phase 1 duration s")
    p.add_argument("--n_wave",   type=float, default=30.0, help="Phase 2 duration s")
    p.add_argument("--ramp_s",   type=float, default=8.0,  help="Stand-up ramp s")
    args = p.parse_args()

    print("Connecting to Go1...")
    udp.Recv(); udp.GetRecv(state)
    print("Connected.")
    print()
    print("IMU NOISE TEST — 3 PHASES")
    print("  Phase 0: Robot on ground, motors damping only")
    print("  Phase 1: Robot standing, training KP")
    print("  Phase 2: Robot standing + ±0.1rad thigh sine wave")
    print()
    print(f"  Duration: {args.n_static:.0f}s + {args.n_stand:.0f}s + {args.n_wave:.0f}s")
    print(f"  Acceptance (Phases 1/2): max_err<{ERR_TOL}rad all knee_tau>{CONTACT_TAU_THR}Nm stable {STABLE_HOLD_S:.0f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Phase 0 ──────────────────────────────────────────────────────────────
    acc0, gyro0, t0_log = phase0_ground(n_seconds=args.n_static)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    acc1, gyro1, err1, t1_log = phase1_stand(n_seconds=args.n_stand, ramp_s=args.ramp_s)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    acc2, gyro2, err2, wave2, t2_log = phase2_wave(n_seconds=args.n_wave)

    # ── Consolidated summary ──────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("FINAL SUMMARY — IMU NOISE THREE-PHASE")
    print("═"*70)
    AXIS = ['X','Y','Z']
    for ax in range(3):
        print(f"\n  Accelerometer axis {AXIS[ax]} (m/s²):")
        print(f"    Ph0 ground:  mean={acc0[:,ax].mean():+.4f}  std={acc0[:,ax].std():.5f}")
        print(f"    Ph1 stand:   mean={acc1[:,ax].mean():+.4f}  std={acc1[:,ax].std():.5f}  "
              f"ratio={acc1[:,ax].std()/max(acc0[:,ax].std(),1e-9):.2f}×")
        print(f"    Ph2 wave:    mean={acc2[:,ax].mean():+.4f}  std={acc2[:,ax].std():.5f}  "
              f"ratio={acc2[:,ax].std()/max(acc0[:,ax].std(),1e-9):.2f}×")
    for ax in range(3):
        print(f"\n  Gyroscope axis {AXIS[ax]} (rad/s):")
        print(f"    Ph0 ground:  mean={gyro0[:,ax].mean():+.6f}  std={gyro0[:,ax].std():.6f}")
        print(f"    Ph1 stand:   mean={gyro1[:,ax].mean():+.6f}  std={gyro1[:,ax].std():.6f}  "
              f"ratio={gyro1[:,ax].std()/max(gyro0[:,ax].std(),1e-9):.2f}×")
        print(f"    Ph2 wave:    mean={gyro2[:,ax].mean():+.6f}  std={gyro2[:,ax].std():.6f}  "
              f"ratio={gyro2[:,ax].std()/max(gyro0[:,ax].std(),1e-9):.2f}×")

    print()
    print("── go1_env.py obs_noise_std (use Phase 1 standing values) ──")
    print("# obs[27:30] = gyro standing std")
    g1 = gyro1.std(axis=0)
    print(f"obs_noise_std[27:30] = [{g1[0]:.5f}, {g1[1]:.5f}, {g1[2]:.5f}]  # rad/s")

    # ── Save ──────────────────────────────────────────────────────────────────
    fname = f"calib_imu_noise_{ts}.npz"
    np.savez(fname,
        # Phase 0
        ph0_acc=acc0, ph0_gyro=gyro0, ph0_time=t0_log,
        # Phase 1
        ph1_acc=acc1, ph1_gyro=gyro1, ph1_err=err1, ph1_time=t1_log,
        # Phase 2
        ph2_acc=acc2, ph2_gyro=gyro2, ph2_err=err2,
        ph2_wave=wave2, ph2_time=t2_log,
        # Config
        wave_amp=np.array([WAVE_AMP]), wave_freq=np.array([WAVE_FREQ]),
        err_tol=np.array([ERR_TOL]),
        stand_q_sdk=STAND_Q_ARR,
        ctrl_hz=np.array([CTRL_HZ]),
    )
    kb = sum(v.nbytes for v in [acc0,gyro0,acc1,gyro1,acc2,gyro2]) / 1024
    print(f"\n→ Saved {fname}  ({kb:.0f} KB)")
    print(f"  Plot: python3 go1_test1_imu_plot.py {fname}")


if __name__ == "__main__":
    main()