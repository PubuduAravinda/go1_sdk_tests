import time
import numpy as np
import robot_interface as sdk
import sys

# ─── CONFIG ─────────────────────────────────────────────────────────────────
KP_START = 5.0
KD_FIXED = 1.5
KP_STEP = 5.0
RAMP_INTERVAL_SECONDS = 3.0
RAMP_MAX_LEVEL = 6

# Real stand pose (default_q)
stand_q = np.array([0.0, 0.77, -1.5, 0.0, 0.77, -1.5, 0.0, 0.77, -1.5, 0.0, 0.77, -1.5])

# Short joint names for table
joint_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_th", "FR_th", "RL_th", "RR_th", "FL_cal", "FR_cal", "RL_cal", "RR_cal"]

# Tuned trot sequence — reduced hip abduction to prevent crossing
trot_sequence = np.array([
    [-0.20,  0.20, -0.20,  0.20,  0.65, -0.65,  0.65, -0.65,  0.85, -0.85,  0.85, -0.85],
    [-0.10,  0.10, -0.10,  0.10,  0.45, -0.45,  0.45, -0.45,  0.65, -0.65,  0.65, -0.65],
    [ 0.20, -0.20,  0.20, -0.20, -0.65,  0.65, -0.65,  0.65, -0.85,  0.85, -0.85,  0.85],
    [ 0.10, -0.10,  0.10, -0.10, -0.45,  0.45, -0.45,  0.45, -0.65,  0.65, -0.65,  0.65],
    [ 0.00,  0.00,  0.00,  0.00,  0.15, -0.15,  0.15, -0.15,  0.35, -0.35,  0.35, -0.35],
    [ 0.00,  0.00,  0.00,  0.00,  0.08, -0.08,  0.08, -0.08,  0.15, -0.15,  0.15, -0.15],
    [-0.03,  0.03, -0.03,  0.03,  0.00,  0.00,  0.00,  0.00,  0.08, -0.08,  0.08, -0.08],
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
])

# ─── INITIALIZATION ─────────────────────────────────────────────────────────
udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd = sdk.LowCmd()
state = sdk.LowState()

udp.InitCmdData(cmd)

print("\n" + "="*90)
print("MANUAL TROT — PER-STEP TABLE — NO PD ERROR")
print("Starting in 10 seconds...")
print("="*90 + "\n")

time.sleep(10)

step = 0
t0 = time.time()
sequence_idx = 0

while True:
    time.sleep(0.05)  # your slow setting — good for careful observation
    step += 1
    t = time.time() - t0

    try:
        udp.Recv()
        udp.GetRecv(state)
    except Exception as e:
        print(f"UDP RECV ERROR: {e}", flush=True)
        break

    real_joint_pos = np.array([state.motorState[i].q for i in range(12)])
    real_joint_vel = np.array([state.motorState[i].dq for i in range(12)])
    real_joint_tauEst = np.array([state.motorState[i].tauEst for i in range(12)])

    # KP ramp
    current_level = int(t // RAMP_INTERVAL_SECONDS)
    ramp_level = min(RAMP_MAX_LEVEL, current_level)
    current_kp = KP_START + ramp_level * KP_STEP
    current_kd = KD_FIXED

    # ─── MANUAL TROT SEQUENCE ───────────────────────────────────────────────
    delta_q = trot_sequence[sequence_idx % len(trot_sequence)]
    sequence_idx += 1

    # Reorder to real order
    real_delta_q = np.zeros(12)
    real_delta_q[0] = delta_q[1]
    real_delta_q[1] = delta_q[5]
    real_delta_q[2] = delta_q[9]
    real_delta_q[3] = delta_q[0]
    real_delta_q[4] = delta_q[4]
    real_delta_q[5] = delta_q[8]
    real_delta_q[6] = delta_q[3]
    real_delta_q[7] = delta_q[7]
    real_delta_q[8] = delta_q[11]
    real_delta_q[9] = delta_q[2]
    real_delta_q[10] = delta_q[6]
    real_delta_q[11] = delta_q[10]

    target_q = stand_q + real_delta_q

    # ─── Send pure position command ─────────────────────────────────────────
    for i in range(12):
        q_cmd = target_q[i]
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0.0
        cmd.motorCmd[i].Kp = current_kp
        cmd.motorCmd[i].Kd = current_kd
        cmd.motorCmd[i].tau = 0.0

    # ─── PER-STEP TABLE (every step) ────────────────────────────────────────
    current_trot_step = (sequence_idx - 1) % len(trot_sequence) + 1
    print(f"t={t:5.1f}s | kp={current_kp:.1f} | Trot step {current_trot_step}/{len(trot_sequence)}")
    print(" idx | joint     | real_curr_q | delta_q | default_q | target_q | error   | real_dq ")
    print("-----|-----------|-------------|---------|-----------|----------|---------|---------")
    for i in range(12):
        name = joint_names[i]
        real_curr = real_joint_pos[i]
        delt     = real_delta_q[i]
        default  = stand_q[i]
        targ     = target_q[i]
        err      = targ - real_curr
        dq       = real_joint_vel[i]
        print(f" {i:3d} | {name:9} | {real_curr:+11.3f} | {delt:+7.3f} | {default:+9.3f} | {targ:+8.3f} | {err:+7.3f} | {dq:+7.3f}")

    # Quick summary
    gravity = -np.array(state.imu.accelerometer) / max(np.linalg.norm(state.imu.accelerometer), 0.1)
    tilt = np.degrees(np.sqrt(gravity[0]**2 + gravity[1]**2))
    forces = state.footForce
    print(f"gravity: x={gravity[0]:+5.3f} y={gravity[1]:+5.3f} z={gravity[2]:+5.3f} | tilt={tilt:5.1f}°")
    print(f"forces: FL={forces[0]:3.0f} FR={forces[1]:3.0f} RL={forces[2]:3.0f} RR={forces[3]:3.0f}")
    print("-" * 80)

    try:
        udp.SetSend(cmd)
        udp.Send()
    except Exception as e:
        print(f"UDP SEND ERROR: {e}", flush=True)
        break