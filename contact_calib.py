#!/usr/bin/env python3
"""
Go1 Contact Threshold Calibration
Run this BEFORE deploying policy to measure real knee torque baselines.

Protocol:
  Phase 1 (HANG):  Robot hanging in air, default pose, full KP
                   → measures tauEst at each knee with no ground contact
                   → this is self-weight baseline (typically ~30-90 Nm depending on joint)

  Phase 2 (FLOOR): User places robot on ground (or lowers slowly)
                   → measures tauEst jump when feet contact ground
                   → threshold = hang_mean + 0.5*(floor_mean - hang_mean)

  Output: CONTACT_THRESHOLD per leg, saved to contact_threshold.npz
          Use directly in go1_deploy.py as KNEE_TAU_THRESHOLD

  Why this matters:
    Sim contact sensor: net_forces_w (N, physics ground truth, binary 0/nonzero)
    Real knee tauEst:   always non-zero from self-weight (10-100+ Nm in hang)
    Without calibration: KNEE_TAU_THRESHOLD=1.0 means ALL feet always "in contact"
    With calibration:   threshold sits between hang and floor → true binary contact
"""

import time
import numpy as np
import robot_interface as sdk

# ── Match your deploy script exactly ─────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,
    0.8,  0.8,  0.8,  0.8,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i

KP_MULTIPLIER = np.array([
    1.000, 1.000, 1.000, 1.000,
    1.857, 1.857, 1.857, 1.857,
    2.286, 2.286, 2.286, 2.286,
], dtype=np.float32)
KD_PER_JOINT = np.array([4., 4., 4., 4., 4.5, 4.5, 4.5, 4.5, 5., 5., 5., 5.], np.float32)
KP_BASE = 35.0

# Knee SDK indices for Go1 (actual hardware order: FR FL RR RL)
KNEE_SDK_IDX  = {"FL": 5, "FR": 2, "RL": 11, "RR": 8}   # SDK indices of knee joints
KNEE_LEGS     = ["FL", "FR", "RL", "RR"]
CTRL_HZ       = 500

# ── SDK setup ─────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*65)
print("Go1 Contact Threshold Calibration")
print("="*65)
print()
print("Step 1: Robot should be HANGING IN AIR in default pose.")
print("        (Supported from harness, legs dangling freely)")
print("        Press ENTER when robot is hanging and stable...")
input()

# ── Hold at default pose with full KP ────────────────────────────────────────
def send_default(kp_base=KP_BASE):
    udp.Recv(); udp.GetRecv(state)
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(kp_base * KP_MULTIPLIER[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd   = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()

# Ramp KP first
print("\nRamping KP to full training values over 4s...")
t0 = time.perf_counter()
while time.perf_counter() - t0 < 4.0:
    alpha = min(1.0, (time.perf_counter()-t0) / 4.0)
    kp = 5.0 + alpha * (KP_BASE - 5.0)
    send_default(kp)
    time.sleep(1.0/CTRL_HZ)
print(f"  KP at full: hip={KP_BASE*KP_MULTIPLIER[0]:.0f}  "
      f"thigh={KP_BASE*KP_MULTIPLIER[4]:.0f}  knee={KP_BASE*KP_MULTIPLIER[8]:.0f}")

# ── Phase 1: Measure HANG baseline ───────────────────────────────────────────
print("\nPhase 1: Collecting HANG baseline (5s)...")
print("  (Robot should be freely hanging — no ground contact)")
hang_samples = []
t0 = time.perf_counter()
n_samples = 0
while time.perf_counter() - t0 < 5.0:
    send_default()
    udp.Recv(); udp.GetRecv(state)
    tau = {leg: abs(state.motorState[idx].tauEst)
           for leg, idx in KNEE_SDK_IDX.items()}
    hang_samples.append([tau[l] for l in KNEE_LEGS])
    if n_samples % (CTRL_HZ // 2) == 0:
        print(f"  t={time.perf_counter()-t0:.1f}s  "
              f"tauEst: FL={tau['FL']:.1f}  FR={tau['FR']:.1f}  "
              f"RL={tau['RL']:.1f}  RR={tau['RR']:.1f}  Nm")
    n_samples += 1
    time.sleep(1.0/CTRL_HZ)

hang_arr  = np.array(hang_samples)   # [N, 4]  FL FR RL RR
hang_mean = hang_arr.mean(axis=0)
hang_std  = hang_arr.std(axis=0)
print(f"\nHANG baseline:")
for i, leg in enumerate(KNEE_LEGS):
    print(f"  {leg}: mean={hang_mean[i]:.2f} Nm  std={hang_std[i]:.2f}  "
          f"range=[{hang_arr[:,i].min():.2f},{hang_arr[:,i].max():.2f}]")

# ── Phase 2: Measure FLOOR ────────────────────────────────────────────────────
print("\nPhase 2: Place robot on ground NOW.")
print("  Lower slowly so all 4 feet contact simultaneously.")
print("  Press ENTER immediately when all feet are on the ground...")
input()

print("Collecting FLOOR readings (5s)...")
floor_samples = []
t0 = time.perf_counter()
n_samples = 0
while time.perf_counter() - t0 < 5.0:
    send_default()
    udp.Recv(); udp.GetRecv(state)
    tau = {leg: abs(state.motorState[idx].tauEst)
           for leg, idx in KNEE_SDK_IDX.items()}
    floor_samples.append([tau[l] for l in KNEE_LEGS])
    if n_samples % (CTRL_HZ // 2) == 0:
        print(f"  t={time.perf_counter()-t0:.1f}s  "
              f"tauEst: FL={tau['FL']:.1f}  FR={tau['FR']:.1f}  "
              f"RL={tau['RL']:.1f}  RR={tau['RR']:.1f}  Nm")
    n_samples += 1
    time.sleep(1.0/CTRL_HZ)

floor_arr  = np.array(floor_samples)
floor_mean = floor_arr.mean(axis=0)
floor_std  = floor_arr.std(axis=0)
print(f"\nFLOOR readings:")
for i, leg in enumerate(KNEE_LEGS):
    print(f"  {leg}: mean={floor_mean[i]:.2f} Nm  std={floor_std[i]:.2f}  "
          f"range=[{floor_arr[:,i].min():.2f},{floor_arr[:,i].max():.2f}]")

# ── Compute thresholds ────────────────────────────────────────────────────────
# Threshold = hang_mean + margin_above_noise + 30% of (floor-hang) gap
# At least 2 sigma above hang noise, at most halfway to floor
margin    = np.maximum(hang_mean + 2*hang_std,
                       hang_mean + 0.3 * (floor_mean - hang_mean))
threshold = np.round(margin, 1)
gap       = floor_mean - hang_mean

print("\n" + "="*65)
print("CONTACT THRESHOLD CALIBRATION RESULTS")
print("="*65)
print(f"\n{'Leg':4s}  {'Hang':>8s}  {'Floor':>8s}  {'Gap':>8s}  {'Threshold':>10s}  Status")
print("-"*60)
for i, leg in enumerate(KNEE_LEGS):
    ok = gap[i] > 5.0
    print(f"{leg:4s}  {hang_mean[i]:8.1f}  {floor_mean[i]:8.1f}  "
          f"{gap[i]:8.1f}  {threshold[i]:10.1f}  "
          f"{'✓ clear' if ok else '⚠ small gap — check joint'}")

if gap.min() < 5.0:
    print("\n⚠ WARNING: Some legs have <5Nm contact gap.")
    print("  Possible causes: KP too low, joint fault, or partial contact.")
    print("  Try with robot weight fully on ground before using these thresholds.")

# Single global threshold (most conservative = max of per-leg thresholds)
global_thresh = float(threshold.max())
print(f"\nGlobal threshold (conservative, use if per-leg not implemented):")
print(f"  KNEE_TAU_THRESHOLD = {global_thresh:.1f}")
print(f"\nPer-leg thresholds (recommended, use in go1_deploy.py):")
print(f"  KNEE_TAU_THRESHOLD_FL = {threshold[0]:.1f}")
print(f"  KNEE_TAU_THRESHOLD_FR = {threshold[1]:.1f}")
print(f"  KNEE_TAU_THRESHOLD_RL = {threshold[2]:.1f}")
print(f"  KNEE_TAU_THRESHOLD_RR = {threshold[3]:.1f}")

# ── Test binary detection ─────────────────────────────────────────────────────
print("\nPhase 3: Binary contact test (10s).")
print("  Lift individual feet while watching detection.")
print("  Expected: ● = contact (above threshold)  ○ = air (below threshold)")
print()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 10.0:
    send_default()
    udp.Recv(); udp.GetRecv(state)
    tau_vals = {leg: abs(state.motorState[idx].tauEst)
                for leg, idx in KNEE_SDK_IDX.items()}
    contact_str = "".join(
        "●" if tau_vals[leg] > threshold[i] else "○"
        for i, leg in enumerate(KNEE_LEGS))
    tau_str = "  ".join(f"{leg}={tau_vals[leg]:.0f}" for leg in KNEE_LEGS)
    print(f"\r  [{contact_str}] FL FR RL RR   {tau_str} Nm        ", end="", flush=True)
    time.sleep(0.05)  # 20Hz display

print()

# ── Save results ─────────────────────────────────────────────────────────────
np.savez("contact_threshold.npz",
    hang_mean=hang_mean, hang_std=hang_std,
    floor_mean=floor_mean, floor_std=floor_std,
    threshold=threshold,
    global_threshold=np.array([global_thresh]),
    legs=np.array(KNEE_LEGS))

print(f"\nSaved: contact_threshold.npz")
print(f"\nAdd to go1_deploy.py:")
print(f"  KNEE_TAU_THRESHOLD = {global_thresh:.1f}  # calibrated {time.strftime('%Y-%m-%d')}")
print(f"  # Per-leg: FL={threshold[0]:.1f}  FR={threshold[1]:.1f}  "
      f"RL={threshold[2]:.1f}  RR={threshold[3]:.1f}")
print()
print("Done. Landing robot...")
for i in range(12):
    cmd.motorCmd[i].Kp = 20.0
udp.SetSend(cmd); udp.Send()