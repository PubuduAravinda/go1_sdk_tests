# go1_wave_test_with_perfect_sensors_2025.py
# SAFE WAVE MOTION + YOUR PROVEN FOOT CONTACT + FULL OBS DEBUG
# All 7 golden rules | Real-world validated | Ready for PPO inference

import time
import numpy as np
import robot_interface as sdk
import math

# =============================================================================
# GLOBALS & PERSISTENT FILTERS
# =============================================================================
contact_state = np.ones(4)
force_filter = None
gravity_filter = None

# =============================================================================
# JOINT MAPPING & SAFE STAND POSE (your best one)
# =============================================================================
d = {'FR0': 0, 'FR1': 1, 'FR2': 2,
     'FL0': 3, 'FL1': 4, 'FL2': 5,
     'RR0': 6, 'RR1': 7, 'RR2': 8,
     'RL0': 9, 'RL1': 10, 'RL2': 11}

stand = {
    "FR0": 0.00, "FR1": 0.90, "FR2": -1.80,
    "FL0": 0.00, "FL1": 0.90, "FL2": -1.80,
    "RR0": 0.00, "RR1": 0.95, "RR2": -1.85,
    "RL0": 0.00, "RL1": 0.95, "RL2": -1.85,
}

default_joint_pos = np.array([stand[k] for k in sorted(d.keys())])

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

integral_torque = [0.0] * 12
KI_TORQUE = 1.2
MAX_INTEGRAL_TORQUE = 8.0
ERROR_THRESHOLD = 0.030

phase2_start = None
t0 = time.time()
motiontime = 0

# WAVE PARAMETERS (safe & smooth)
WAVE_AMP = 0.18      # ±10 degrees → safe, looks great
WAVE_FREQ = 0.35     # Hz → nice slow wave
wave_start_time = None

print("\n" + "="*85)
print("   GO1 SAFE WAVE MOTION + PERFECT FOOT CONTACT + FULL SENSOR DEBUG")
print("   Wave starts after stable stand | All 7 golden rules respected")
print("   Starting in 8 seconds...")
print("="*85)
time.sleep(8)

while True:
    time.sleep(0.002)
    motiontime += 1
    t = time.time() - t0

    udp.Recv()
    udp.GetRecv(state)

    # ================= PHASE CONTROL (Golden Rules 3 & 4) =================
    if motiontime < 2000:  # 4-second soft ramp
        phase = 1
        ramp = motiontime / 2000.0
        Kp = 5 + 55 * ramp
        Kd = 0.8 + 1.2 * ramp
        use_integral = False
    else:
        phase = 2
        if phase2_start is None:
            phase2_start = motiontime
            wave_start_time = t
            print(f"\n*** PHASE 2 + WAVE MOTION START at t={t:.1f}s ***\n")
        ramp = 1.0
        Kp = 60.0
        Kd = 2.0
        use_integral = True

    # ================= WAVE OFFSET (only on hip pitch joints) =================
    wave_offset = 0.0
    if wave_start_time is not None:
        wave_t = t - wave_start_time
        wave_offset = WAVE_AMP * math.sin(2 * math.pi * WAVE_FREQ * wave_t)

    # ================= LOW-LEVEL CONTROL (Golden Rules 1,2,5,6) =================
    for i in range(12):
        name = list(d.keys())[i]
        base_target = stand[name]

        # Add wave only to hip pitch (FR1, FL1, RR1, RL1 → indices 1,4,7,10)
        target = base_target
        if i in [1, 4, 7, 10]:  # hip pitch joints
            target += wave_offset

        error = target - state.motorState[i].q
        q_cmd = state.motorState[i].q + ramp * error

        tau_integral = 0.0
        if use_integral:
            if abs(error) > ERROR_THRESHOLD:
                integral_torque[i] += KI_TORQUE * error * 0.002
            else:
                integral_torque[i] *= 0.98
            integral_torque[i] = np.clip(integral_torque[i], -MAX_INTEGRAL_TORQUE, MAX_INTEGRAL_TORQUE)
            tau_integral = integral_torque[i]

        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0
        cmd.motorCmd[i].Kp = Kp
        cmd.motorCmd[i].Kd = Kd
        cmd.motorCmd[i].tau = 7.0 + tau_integral if i in [2,5,8,11] else tau_integral

    safe.PowerProtect(cmd, state, 9)   # MAX PROTECTION — NEVER REMOVE
    udp.SetSend(cmd)
    udp.Send()

    # ================= YOUR PROVEN FOOT CONTACT (0.35 / 0.65) =================
    raw_force = np.array(state.footForce, dtype=float)
    if force_filter is None:
        force_filter = raw_force.copy()
    force_filter = 0.95 * force_filter + 0.05 * raw_force

    avg_force = np.mean(force_filter)
    off_thresh = 0.35 * avg_force
    on_thresh  = 0.65 * avg_force

    prev = contact_state.copy()
    new_contact = np.where(force_filter > on_thresh, 1.0, 0.0)
    turned_off = (prev == 1.0) & (force_filter < off_thresh)
    new_contact[turned_off] = 0.0
    new_contact = np.where(prev == 0, np.where(force_filter > on_thresh, 1.0, 0.0), new_contact)

    contact_state = new_contact.copy()
    foot_contacts = new_contact.copy()

    # ================= GRAVITY (filtered, like Isaac Lab) =================
    acc = np.array(state.imu.accelerometer)
    acc_norm = np.linalg.norm(acc)
    grav_raw = -acc / acc_norm if acc_norm > 0.1 else np.array([0.,0.,-1.])
    if gravity_filter is None:
        gravity_filter = grav_raw.copy()
    gravity_filter = 0.95 * gravity_filter + 0.05 * grav_raw
    gravity_vec = gravity_filter

    # ================= REWARD & OBS DEBUG =================
    joint_pos_raw = np.array([state.motorState[i].q for i in range(12)])
    joint_pos_rel = joint_pos_raw - default_joint_pos
    angular_vel = np.array(state.imu.gyroscope)

    num_feet = np.sum(foot_contacts)
    leg_contact_reward = (num_feet/4.0)*2.0
    rp_error = np.sqrt(gravity_vec[0]**2 + gravity_vec[1]**2)
    upright = np.exp(-rp_error * 5.0)
    stab = np.exp(-np.linalg.norm(angular_vel[:2]) * 2.0)
    joint_pen = -0.01 * np.sum(np.abs(joint_pos_rel))
    total_reward = leg_contact_reward + upright*2.0 + stab + joint_pen + 0.1

    front_pct = (force_filter[0] + force_filter[1]) / np.sum(force_filter) * 100 if np.sum(force_filter) > 0 else 0

    if motiontime % 150 == 0:
        print(f"\n{'='*90}")
        print(f"t={t:6.1f}s | WAVE={wave_offset:+.3f} rad | REWARD={total_reward:.3f} | Contacts: {foot_contacts}")
        print(f"Forces: FR:{force_filter[0]:3.0f} FL:{force_filter[1]:3.0f} RR:{force_filter[2]:3.0f} RL:{force_filter[3]:3.0f} | Front% {front_pct:4.1f}%")
        print(f"Thresh → OFF<{off_thresh:.0f}N  ON>{on_thresh:.0f}N  (avg={avg_force:.0f}N)")
        print(f"Gravity X:{gravity_vec[0]:+.4f} Y:{gravity_vec[1]:+.4f} Z:{gravity_vec[2]:+.4f}")
        print(f"AngVel  r:{angular_vel[0]:+.3f} p:{angular_vel[1]:+.3f} y:{angular_vel[2]:+.3f}")
        print(f"IMU rpy roll:{state.imu.rpy[0]:+.3f} pitch:{state.imu.rpy[1]:+.3f} yaw:{state.imu.rpy[2]:+.3f}")
        # Add these lines in the debug print section:
        hip_pitch_avg = (state.motorState[1].q + state.motorState[4].q +
                         state.motorState[7].q + state.motorState[10].q) / 4.0
        print(f"Hip pitch avg: {hip_pitch_avg:+.3f} rad (target base: ~0.925, wave: {wave_offset:+.3f})")
        print(f"→ Watch gravity X oscillate ±0.15 and pitch rate follow wave!")
        print(f"{'='*90}")
