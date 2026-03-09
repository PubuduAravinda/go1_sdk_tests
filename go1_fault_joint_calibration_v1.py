#!/usr/bin/env python3
"""
Per-Joint KP Calibration for Go1 Policy Deployment
=====================================================
Problem: Mechanically worn Go1 has joints that can't reach default standing
position at uniform KP. Weak/stiff joints lag behind, causing:
  - RR foot hanging in air (can't reach target)
  - Policy observes wrong delta → feeds bad obs → bad gait

Solution: Per-joint KP boosting — measure tracking error per joint,
increase KP independently until each joint reaches target position.

Output: PER_JOINT_KP_MULT array + measured DEFAULT_JOINT_POS for deploy.py

Phases:
  0 (0-10s):   Passive drop — measure natural rest position
  1 (10-50s):  Uniform KP ramp 5→35 — measure which joints still lag
  2 (50-120s): Per-joint KP boost — push weak joints to target
  3 (120-135s): Final measure — average actual positions at calibrated KP
"""

import time
import numpy as np
import robot_interface as sdk

# ─── TARGET DEFAULT (what policy was trained with) ────────────────────────────
TARGET_DEFAULT = np.array([
    0.1, 0.1, 0.1, 0.1,        # hips   FL FR RL RR (Isaac order)
    0.8, 0.8, 0.8, 0.8,        # thighs
   -1.5,-1.5,-1.5,-1.5,        # knees
], dtype=np.float32)

# ─── JOINT NAMES (Isaac order, inferred from sdk_to_isaac mapping) ───────────
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0]*12
for i in range(12): isaac_to_sdk[sdk_to_isaac[i]] = i

SDK_NAMES   = ["FR_hip","FR_thigh","FR_knee","FL_hip","FL_thigh","FL_knee",
               "RR_hip","RR_thigh","RR_knee","RL_hip","RL_thigh","RL_knee"]
ISAAC_NAMES = [None]*12
for sdk_i, isaac_i in enumerate(sdk_to_isaac):
    ISAAC_NAMES[isaac_i] = SDK_NAMES[sdk_i]

# ─── CALIBRATION CONFIG ───────────────────────────────────────────────────────
KP_BASE_START   = 5.0
KP_BASE_MAX     = 35.0
KP_RAMP_TIME    = 40.0   # seconds to ramp base KP 5→35

# Per-joint KP boost limits (safety ceiling)
# Go1 motor max torque: hip=23.7Nm, thigh=23.7Nm, knee=35.6Nm
# Don't exceed KP * max_error > max_torque
KP_MAX_PER_JOINT = np.array([
    40., 40., 40., 40.,    # hips   (23.7Nm / 0.5rad ≈ 47, keep 40 safe)
    70., 70., 70., 70.,    # thighs (23.7Nm / 0.3rad ≈ 79, keep 70)
    80., 80., 80., 80.,    # knees  (35.6Nm / 0.4rad ≈ 89, keep 80)
], dtype=np.float32)

KD_PER_JOINT = np.array([
    4.0,4.0,4.0,4.0,       # hips
    4.5,4.5,4.5,4.5,       # thighs
    4.5,4.5,4.5,4.5,       # knees
], dtype=np.float32)

THIGH_TAU_FF = 1.2  # Nm gravity feedforward for thighs

# Boost step: if joint error > threshold, increase its KP by this amount
BOOST_STEP      = 3.0    # KP increase per boost iteration
BOOST_INTERVAL  = 3.0    # seconds between boost steps
ERROR_THRESHOLD = 0.05   # rad — joint is "converged" when |error| < this

# Tilt safety — pause boost if robot tilting too much
TILT_LIMIT_DEG  = 35.0
MEASURE_TIME    = 15.0   # final averaging window

# ─── INIT ─────────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# Per-joint KP multiplier — starts at 1.0, gets boosted per joint
kp_mult    = np.ones(12, dtype=np.float32)
# Built-in thigh boost from training (policy assumes KP*1.4 for thighs)
kp_mult[4:8] = 1.4

TARGET_SDK  = TARGET_DEFAULT[isaac_to_sdk]
KD_SDK      = KD_PER_JOINT[isaac_to_sdk]

print("\n" + "="*70)
print("PER-JOINT KP CALIBRATION  —  Go1 deployment tuning")
print("="*70)
print("Place robot on ground. Script will:")
print("  Phase 0 (0-10s):    Passive — measure natural rest")
print("  Phase 1 (10-50s):   Uniform KP ramp 5→35")
print("  Phase 2 (50-120s):  Per-joint boost — push weak joints to target")
print("  Phase 3 (120-135s): Final measure — average calibrated position")
print("="*70 + "\n")

# State
phase          = 0
current_kp     = KP_BASE_START
last_boost_t   = 0.0
final_samples  = []
passive_samples= []
ramp_start_t   = None

t0 = time.time()

try:
    while True:
        time.sleep(0.002)  # 500Hz loop

        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception:
            continue

        t = time.time() - t0

        # ── Read state ────────────────────────────────────────────────────────
        q_sdk   = np.array([state.motorState[i].q for i in range(12)])
        q_isaac = q_sdk[sdk_to_isaac]
        error   = TARGET_DEFAULT - q_isaac   # positive = joint needs to move toward target

        gx = state.imu.accelerometer[0]
        gy = state.imu.accelerometer[1]
        gz = state.imu.accelerometer[2]
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
        tilt_deg = np.degrees(np.arccos(min(1.0, abs(gz) / g_mag)))

        # ── Phase transitions ─────────────────────────────────────────────────
        if t < 10.0:
            phase = 0   # passive
        elif t < 10.0 + KP_RAMP_TIME:
            if phase < 1:
                phase = 1
                ramp_start_t = time.time()
                print(f"\n[t={t:.1f}s] PHASE 1: KP ramp start")
        elif t < 10.0 + KP_RAMP_TIME + 70.0:
            if phase < 2:
                phase = 2
                print(f"\n[t={t:.1f}s] PHASE 2: Per-joint boost start")
                print(f"  Initial KP mult: {np.round(kp_mult,2)}")
        elif t < 10.0 + KP_RAMP_TIME + 70.0 + MEASURE_TIME:
            if phase < 3:
                phase = 3
                print(f"\n[t={t:.1f}s] PHASE 3: Final measurement")
        else:
            break

        # ── Motor commands ─────────────────────────────────────────────────────
        if phase == 0:
            # Passive — light damping only
            for i in range(12):
                cmd.motorCmd[i].mode = 0x0A
                cmd.motorCmd[i].q    = 0.0
                cmd.motorCmd[i].dq   = 0.0
                cmd.motorCmd[i].Kp   = 0.0
                cmd.motorCmd[i].Kd   = 2.0
                cmd.motorCmd[i].tau  = 0.0
            passive_samples.append(q_isaac.copy())

        else:
            # KP ramp for phases 1-3
            if phase == 1:
                frac = min((time.time() - ramp_start_t) / KP_RAMP_TIME, 1.0)
                current_kp = KP_BASE_START + frac * (KP_BASE_MAX - KP_BASE_START)
            else:
                current_kp = KP_BASE_MAX

            # Compute per-joint KP (base * per-joint multiplier, capped)
            kp_isaac = np.clip(current_kp * kp_mult, 0, KP_MAX_PER_JOINT)
            kp_sdk   = kp_isaac[isaac_to_sdk]

            # Thigh gravity feedforward
            tau_isaac = np.zeros(12)
            tau_isaac[4:8] = THIGH_TAU_FF
            tau_sdk = tau_isaac[isaac_to_sdk]

            for i in range(12):
                cmd.motorCmd[i].mode = 0x0A
                cmd.motorCmd[i].q    = float(TARGET_SDK[i])
                cmd.motorCmd[i].dq   = 0.0
                cmd.motorCmd[i].Kp   = float(kp_sdk[i])
                cmd.motorCmd[i].Kd   = float(KD_SDK[i])
                cmd.motorCmd[i].tau  = float(tau_sdk[i])

            # ── Phase 2: per-joint boost ──────────────────────────────────────
            if phase == 2 and t - last_boost_t > BOOST_INTERVAL:
                if tilt_deg < TILT_LIMIT_DEG:
                    boosted = []
                    for i in range(12):
                        if abs(error[i]) > ERROR_THRESHOLD:
                            new_mult = kp_mult[i] + BOOST_STEP / current_kp
                            max_mult = KP_MAX_PER_JOINT[i] / current_kp
                            if kp_mult[i] < max_mult:
                                kp_mult[i] = min(new_mult, max_mult)
                                boosted.append(f"{ISAAC_NAMES[i]}(err={error[i]:+.2f}→KP={kp_mult[i]*current_kp:.0f})")
                    if boosted:
                        print(f"  [t={t:.0f}s] Boost: {', '.join(boosted)}")
                    last_boost_t = t
                else:
                    print(f"  [t={t:.0f}s] Tilt {tilt_deg:.1f}° > limit — pausing boost")
                    last_boost_t = t

            # ── Phase 3: collect samples ──────────────────────────────────────
            if phase == 3:
                final_samples.append(q_isaac.copy())

        try:
            safe.PowerProtect(cmd, state, 9)
            udp.SetSend(cmd)
            udp.Send()
        except Exception:
            pass

        # ── Status every 2s ───────────────────────────────────────────────────
        if int(t * 0.5) != int((t - 0.002) * 0.5):
            max_err_i = np.argmax(np.abs(error))
            worst = f"{ISAAC_NAMES[max_err_i]}={error[max_err_i]:+.3f}rad"
            converged = np.sum(np.abs(error) < ERROR_THRESHOLD)
            print(f"t={t:6.1f}s | ph={phase} | KP={current_kp:.0f} | "
                  f"tilt={tilt_deg:.1f}° | converged={converged}/12 | worst={worst}")

except KeyboardInterrupt:
    print("\nAborted.")

# ─── RESULTS ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CALIBRATION RESULTS")
print("="*70)

passive_mean = np.mean(passive_samples, axis=0) if passive_samples else np.zeros(12)
final_mean   = np.mean(final_samples,   axis=0) if final_samples else np.zeros(12)
final_std    = np.std(final_samples,    axis=0) if final_samples else np.zeros(12)

print(f"\n{'#':>2}  {'Joint':>14}  {'Target':>7}  {'Passive':>8}  {'Final':>8}  {'Error':>7}  {'KP_mult':>8}  {'EffKP':>6}")
for i in range(12):
    err = TARGET_DEFAULT[i] - final_mean[i]
    converged = "✓" if abs(err) < ERROR_THRESHOLD else "⚠"
    print(f"{i:>2}  {ISAAC_NAMES[i]:>14}  {TARGET_DEFAULT[i]:>7.3f}  "
          f"{passive_mean[i]:>8.3f}  {final_mean[i]:>8.3f}  "
          f"{err:>+7.3f}  {kp_mult[i]:>8.2f}  "
          f"{kp_mult[i]*KP_BASE_MAX:>6.1f}  {converged}")

print(f"\n{'='*70}")
print("Paste these into go1_deploy.py:")
print(f"{'='*70}")

# New defaults = where joints actually settled (if converged) else target
new_defaults = np.where(
    np.abs(TARGET_DEFAULT - final_mean) < 0.15,
    TARGET_DEFAULT,           # use training default if close enough
    final_mean                # use measured if still offset
)

print(f"\n# Per-joint KP multiplier (calibrated for this robot)")
print(f"KP_MULTIPLIER = np.array([")
print(f"    {kp_mult[0]:.2f}, {kp_mult[1]:.2f}, {kp_mult[2]:.2f}, {kp_mult[3]:.2f},   # hips")
print(f"    {kp_mult[4]:.2f}, {kp_mult[5]:.2f}, {kp_mult[6]:.2f}, {kp_mult[7]:.2f},  # thighs")
print(f"    {kp_mult[8]:.2f}, {kp_mult[9]:.2f}, {kp_mult[10]:.2f}, {kp_mult[11]:.2f},  # knees")
print(f"], dtype=np.float32)")

print(f"\n# Measured standing position (where robot actually rests at calibrated KP)")
print(f"DEFAULT_JOINT_POS = np.array([")
h = final_mean[0:4]
th = final_mean[4:8]
kn = final_mean[8:12]
print(f"    {h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}, {h[3]:.3f},   # hips")
print(f"    {th[0]:.3f}, {th[1]:.3f}, {th[2]:.3f}, {th[3]:.3f},  # thighs")
print(f"    {kn[0]:.3f}, {kn[1]:.3f}, {kn[2]:.3f}, {kn[3]:.3f},  # knees")
print(f"], dtype=np.float32)")
print(f"{'='*70}")