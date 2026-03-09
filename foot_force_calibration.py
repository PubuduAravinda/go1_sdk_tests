#!/usr/bin/env python3
"""
Torque-Based Contact Calibration  (with per-joint KP/KD standing)
===================================================================
Phase 1 (0-20s):  HANG  — passive, KP=0
Phase 2 (20-50s): STAND — per-joint KP/KD matching deploy script, full ramp

Per-joint gains match go1_deploy.py exactly:
  Hips:   KP=base,      KD=4.0
  Thighs: KP=base*1.4,  KD=4.5   (higher stiffness, gravity feedforward)
  Knees:  KP=base,      KD=4.5

RL leg shown to need more force — per-joint tau feedforward helps it hold.
"""

import time
import numpy as np
import robot_interface as sdk

# ─── CONFIG — mirrors go1_deploy.py exactly ──────────────────────────────────
KP_START      = 5.0
KP_STEP       = 3.0
KP_MAX        = 35.0
RAMP_INTERVAL = 4.0   # faster ramp for calibration

KP_MULTIPLIER = np.array([
    1.0, 1.0, 1.0, 1.0,   # hips
    1.4, 1.4, 1.4, 1.4,   # thighs — 40% boost
    1.0, 1.0, 1.0, 1.0,   # knees
], dtype=np.float32)

KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,   # hips
    4.5, 4.5, 4.5, 4.5,   # thighs
    4.5, 4.5, 4.5, 4.5,   # knees
], dtype=np.float32)

# Gravity feedforward on thighs (same as deploy)
TAU_ISAAC = np.array([
    0.0, 0.0, 0.0, 0.0,   # hips
    1.2, 1.2, 1.2, 1.2,   # thighs
    0.0, 0.0, 0.0, 0.0,   # knees
], dtype=np.float32)

sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0]*12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

DEFAULT_ISAAC = np.array([0.1,0.1,0.1,0.1, 0.8,0.8,0.8,0.8, -1.5,-1.5,-1.5,-1.5])
DEFAULT_SDK   = DEFAULT_ISAAC[isaac_to_sdk]
KD_SDK        = KD_PER_JOINT[isaac_to_sdk]
KP_MULT_SDK   = KP_MULTIPLIER[isaac_to_sdk]
TAU_SDK       = TAU_ISAAC[isaac_to_sdk]

# Contact detection — SDK knee indices
KNEE_IDX_SDK = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}
foot_names   = ["FR", "FL", "RR", "RL"]

HANG_DURATION  = 20.0
STAND_DURATION = 30.0   # longer so we get full KP ramp data

# ─── INIT ────────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*70)
print("CONTACT CALIBRATION  —  per-joint KP/KD matching deploy script")
print("="*70)
print(f"Phase 1 (0-{HANG_DURATION:.0f}s):    HANG on rack, KP=0 (passive)")
print(f"Phase 2 ({HANG_DURATION:.0f}-{HANG_DURATION+STAND_DURATION:.0f}s): STAND on ground, KP ramps {KP_START:.0f}→{KP_MAX:.0f}")
print(f"  Thigh KP = base × 1.4  |  Thigh KD = 4.5  |  Thigh tau = 1.2 Nm")
print("="*70 + "\n")

hang_tau  = {k: [] for k in foot_names}
stand_tau = {k: [] for k in foot_names}
t0         = time.time()
phase      = "HANG"
ramp_start = None
current_kp = 0.0

try:
    while True:
        time.sleep(0.01)
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"UDP error: {e}")
            break

        t = time.time() - t0

        # ── Phase 1: hang ────────────────────────────────────────────────────
        if t < HANG_DURATION:
            phase = "HANG"
            for i in range(12):
                cmd.motorCmd[i].mode = 0x0A
                cmd.motorCmd[i].q    = 0.0
                cmd.motorCmd[i].dq   = 0.0
                cmd.motorCmd[i].Kp   = 0.0
                cmd.motorCmd[i].Kd   = 2.0
                cmd.motorCmd[i].tau  = 0.0

        # ── Phase 2: stand ───────────────────────────────────────────────────
        elif t < HANG_DURATION + STAND_DURATION:
            if phase != "STAND":
                phase      = "STAND"
                ramp_start = time.time()
                print("\n>>> PHASE 2: Place robot on ground now! KP ramping up... <<<\n")

            t_stand    = time.time() - ramp_start
            ramp_lvl   = min(int(t_stand // RAMP_INTERVAL),
                             int((KP_MAX - KP_START) / KP_STEP))
            current_kp = KP_START + ramp_lvl * KP_STEP

            for i in range(12):
                cmd.motorCmd[i].mode = 0x0A
                cmd.motorCmd[i].q    = float(DEFAULT_SDK[i])
                cmd.motorCmd[i].dq   = 0.0
                cmd.motorCmd[i].Kp   = float(current_kp * KP_MULT_SDK[i])
                cmd.motorCmd[i].Kd   = float(KD_SDK[i])
                cmd.motorCmd[i].tau  = float(TAU_SDK[i])
        else:
            break

        try:
            safe.PowerProtect(cmd, state, 9)
            udp.SetSend(cmd)
            udp.Send()
        except Exception as e:
            print(f"UDP send error: {e}")
            break

        # ── Sample knee tauEst ───────────────────────────────────────────────
        knee_taus = {k: abs(state.motorState[idx].tauEst)
                     for k, idx in KNEE_IDX_SDK.items()}

        if phase == "HANG":
            for k in foot_names:
                hang_tau[k].append(knee_taus[k])
            remaining = HANG_DURATION - t
        else:
            # Only collect stand samples at KP >= 20 (stable standing)
            if current_kp >= 20.0:
                for k in foot_names:
                    stand_tau[k].append(knee_taus[k])
            remaining = HANG_DURATION + STAND_DURATION - t

        # ── Live display every 0.5s ──────────────────────────────────────────
        if int(t * 2) != int((t - 0.01) * 2):
            tau_str = "  ".join(f"{k}:{knee_taus[k]:5.2f}Nm" for k in foot_names)
            thigh_kp = f"thighKP={current_kp*1.4:.0f}" if phase == "STAND" else ""
            kp_str   = f"baseKP={current_kp:.0f} {thigh_kp}" if phase == "STAND" else "KP=0 (passive)"
            print(f"[{phase}] t={t:5.1f}s  {tau_str}  {kp_str}  (rem:{remaining:.0f}s)")

except KeyboardInterrupt:
    print("\nAborted.")

# ─── ANALYSIS ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RESULTS  (stand samples collected only at baseKP >= 20)")
print("="*70)

if not any(hang_tau[k] for k in foot_names):
    print("No hang data.")
elif not any(stand_tau[k] for k in foot_names):
    print("No stable stand data — robot may not have reached KP=20.")
    print("Hang max values:")
    for k in foot_names:
        if hang_tau[k]:
            h = np.array(hang_tau[k])
            print(f"  {k}: max={np.max(h):.3f} mean={np.mean(h):.3f} Nm")
    print("\nSafe threshold estimate: 1.0 Nm (based on hang noise floor ~0.37 Nm)")
else:
    print(f"\n{'Foot':>4}  {'HANG_max':>9} {'HANG_mean':>10}  {'STAND_min':>10} {'STAND_mean':>11}  {'Thresh':>8}  {'Sep?':>12}")
    all_hang_max  = []
    all_stand_min = []

    for k in foot_names:
        h = np.array(hang_tau[k])
        s = np.array(stand_tau[k])
        hmax   = np.max(h)
        smin   = np.min(s)
        thresh = (hmax + smin) / 2.0
        all_hang_max.append(hmax)
        all_stand_min.append(smin)
        sep = "CLEAR" if smin > hmax * 2 else "marginal"
        print(f"{k:>4}  {hmax:>9.3f} {np.mean(h):>10.3f}  {smin:>10.3f} {np.mean(s):>11.3f}  {thresh:>8.2f}  {sep}")

    global_thresh = (max(all_hang_max) + min(all_stand_min)) / 2.0
    print(f"\nRecommended: KNEE_TAU_THRESHOLD = {global_thresh:.1f} Nm")
    print(f"\n>>> Update go1_deploy.py: <<<")
    print(f"  KNEE_TAU_THRESHOLD = {global_thresh:.1f}")