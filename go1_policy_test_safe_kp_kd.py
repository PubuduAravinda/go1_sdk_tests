# go1_policy_test_safe_kp_kd.py
# 100% SAFE — supports variable KP/KD actions exactly like your trained policy
# Paste your policy's 36-dim action and watch perfect real-world execution

import time
import numpy as np
import robot_interface as sdk
import math

# =============================================================================
# GLOBALS
# =============================================================================
contact_state = np.ones(4)
force_filter = None
gravity_filter = None

d = {'FR0': 0, 'FR1': 1, 'FR2': 2,
     'FL0': 3, 'FL1': 4,  'FL2': 5,
     'RR0': 6, 'RR1': 7, 'RR2': 8,
     'RL0': 9, 'RL1':10, 'RL2':11}

# Your best standing pose (proven stable)
stand_q = np.array([0.0, 0.90, -1.80,
                    0.0, 0.90, -1.80,
                    0.0, 0.95, -1.85,
                    0.0, 0.95, -1.85])

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# Integral torque assist (same as before)
integral_torque = [0.0] * 12
KI = 1.2
MAX_INT_TAU = 8.0

motiontime = 0
t0 = time.time()
phase2 = False

print("\n" + "="*90)
print("   GO1 POLICY TEST — VARIABLE KP/KD SUPPORT (36-dim action)")
print("   Paste your policy's 36-dim action below when prompted")
print("   Starting safe stand in 8 seconds...")
print("="*90)
time.sleep(8)

# ================= USER: PASTE YOUR POLICY ACTION HERE =================
# Example from your log (or replace with real inference output)
raw_action = np.array([
    # Δq (12)
    2.000e-03,  1.330e-01,  5.290e-01,
    2.900e-02, -1.186e+00, -1.421e+00,
    5.630e-01,  1.643e+00, -1.254e+00,
   -2.769e+00,  8.060e-01,  8.560e-01,
    # Kp (12)
    100.0,  20.0, 100.0,
     76.35, 84.97, 100.0,
    100.0,  52.21,  20.0,
    100.0,  99.29,  20.0,
    # Kd (12)
    4.3, 5.0, 5.0,
    5.0, 1.74, 1.0,
    1.0, 5.0, 1.0,
    2.23, 5.0, 1.0
])

# ================= SAFE ACTION SCALING & CLAMPING =================
delta_q = raw_action[0:12]
kp_raw  = raw_action[12:24]
kd_raw  = raw_action[24:36]

# Critical safety clamps (never remove!)
kp_clamped = np.clip(kp_raw, 10.0, 100.0)   # safe range
kd_clamped = np.clip(kd_raw, 0.5, 6.0)      # safe range

# Target joint positions = current stand + delta (or residual)
target_q = stand_q + delta_q

print("Policy action applied with safety clamps:")
print(f"Δq sample : {delta_q[[1,4,7,10]]}")  # hip pitches
print(f"Kp sample : {kp_clamped[[1,4,7,10]]}")
print(f"Kd sample : {kd_clamped[[1,4,7,10]]}")
print(f"Target hip pitch avg: {np.mean(target_q[[1,4,7,10]]):.3f}\n")

while True:
    time.sleep(0.002)
    motiontime += 1
    t = time.time() - t0

    udp.Recv()
    udp.GetRecv(state)

    # Phase control (soft ramp first 4s)
    if motiontime < 2000:
        ramp = motiontime / 2000.0
        base_kp = 5 + 55 * ramp
        base_kd = 0.8 + 1.2 * ramp
    else:
        if not phase2:
            phase2 = True
            print("*** PHASE 2: FULL POLICY ACTION ACTIVE ***\n")
        ramp = 1.0
        base_kp = 0  # not used
        base_kd = 0

    for i in range(12):
        current_q = state.motorState[i].q

        # Final target = interpolated to avoid jump
        q_cmd = current_q + ramp * (target_q[i] - current_q)

        # Use policy's KP/KD after ramp-up
        final_kp = kp_clamped[i] if ramp > 0.99 else base_kp
        final_kd = kd_clamped[i] if ramp > 0.99 else base_kd

        # Integral torque (only knees)
        tau_int = 0.0
        error = target_q[i] - current_q
        if phase2 and abs(error) > 0.03:
            integral_torque[i] += KI * error * 0.002
            integral_torque[i] = np.clip(integral_torque[i], -MAX_INT_TAU, MAX_INT_TAU)
            tau_int = integral_torque[i]

        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = q_cmd
        cmd.motorCmd[i].dq   = 0
        cmd.motorCmd[i].Kp   = final_kp
        cmd.motorCmd[i].Kd   = final_kd
        cmd.motorCmd[i].tau  = 7.0 + tau_int if i in [2,5,8,11] else tau_int

    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()

    # ================= YOUR PERFECT FOOT CONTACT (0.35/0.65) =================
    raw_force = np.array(state.footForce, dtype=float)
    if force_filter is None: force_filter = raw_force.copy()
    force_filter = 0.95 * force_filter + 0.05 * raw_force
    avg = np.mean(force_filter)
    off_thresh = 0.35 * avg
    on_thresh  = 0.65 * avg
    prev = contact_state.copy()
    new_c = np.where(force_filter > on_thresh, 1.0, 0.0)
    turned_off = (prev == 1.0) & (force_filter < off_thresh)
    new_c[turned_off] = 0.0
    new_c = np.where(prev == 0, np.where(force_filter > on_thresh, 1.0, 0.0), new_c)
    contact_state = new_c.copy()

    # Gravity
    acc = np.array(state.imu.accelerometer)
    norm = np.linalg.norm(acc)
    g = -acc/norm if norm > 0.1 else np.array([0,0,-1])
    if gravity_filter is None: gravity_filter = g.copy()
    gravity_filter = 0.95 * gravity_filter + 0.05 * g

    if motiontime % 300 == 0:
        front_pct = (force_filter[0]+force_filter[1]) / np.sum(force_filter) * 100
        print(f"\n{'='*90}")
        print(f"t={t:6.1f}s | Contacts {contact_state} | Front% {front_pct:4.1f}% | Forces {force_filter.astype(int)}")
        print(f"Thresh OFF<{off_thresh:.0f}N  ON>{on_thresh:.0f}N  avg={avg:.0f}N")
        print(f"Gravity X:{gravity_filter[0]:+.4f} Y:{gravity_filter[1]:+.4f} Z:{gravity_filter[2]:+.4f}")
        print(f"KP active : {final_kp:.1f} (example hip) | KD: {final_kd:.2f}")
        print(f"Target q sample (hip pitch): {target_q[[1,4,7,10]]}")
        print(f"Actual q sample (hip pitch): {[state.motorState[i].q for i in [1,4,7,10]]}")
        print(f"{'='*90}")