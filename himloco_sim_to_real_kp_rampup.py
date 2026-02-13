# himloco_sim_to_real.py — MATCH SIM POSE + OBS NORMALIZATION + RAMP

import time
import torch
import numpy as np
import robot_interface as sdk
import sys

policy_path = "/home/pi/unitree_legged_sdk/example_py/himloco_policy_full.pt"
encoder_path = "/home/pi/unitree_legged_sdk/example_py/himloco_encoder.pt"

policy = torch.jit.load(policy_path)
policy.eval()
policy.to("cpu")

encoder = torch.jit.load(encoder_path)
encoder.eval()
encoder.to("cpu")

# Match sim default_joint_pos
stand_q = np.array([0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 0.1, 0.8, -1.5])

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*90)
print("HIMLOCO DEPLOYMENT — MATCH SIM POSE + OBS NORMALIZATION + RAMP")
print("Manual mode switch first: L2+A → L2+B → L1+L2+Start")
print("Starting in 10 seconds...")
print("="*90 + "\n")
time.sleep(10)

prev_action = torch.zeros(12, dtype=torch.float32)
obs_history = torch.zeros(5, 45, dtype=torch.float32)
step = 0
t0 = time.time()
ramp_level = 0
KP_START = 25.0
KD_START = 2.0
KP_STEP = 5.0
KD_STEP = 0.3
RAMP_INTERVAL_SECONDS = 3.0
RAMP_MAX_LEVEL = 10  # cap Kp=75
TILT_THRESHOLD = 30.0  # degrees
scale = 1.0  # disabled (1.0 = no scale)

while True:
    time.sleep(0.002)
    step += 1
    t = time.time() - t0

    try:
        udp.Recv()
        udp.GetRecv(state)
    except Exception as e:
        print(f"UDP RECV ERROR: {e}", flush=True)
        break

    acc = np.array(state.imu.accelerometer)
    norm = max(np.linalg.norm(acc), 0.1)
    gravity = -acc / norm
    gravity_z = gravity[2]

    tilt_angle = np.degrees(np.sqrt(gravity[0]**2 + gravity[1]**2))

    joint_pos = np.array([state.motorState[i].q for i in range(12)])
    joint_vel = np.array([state.motorState[i].dq for i in range(12)])

    current_level = int(t // RAMP_INTERVAL_SECONDS)
    ramp_level = min(RAMP_MAX_LEVEL, current_level)

    current_kp = KP_START + ramp_level * KP_STEP
    current_kd = KD_START + ramp_level * KD_STEP

    # Tilt safety — reduce scale if tilt high
    if tilt_angle > TILT_THRESHOLD:
        scale = 0.5  # reduce to half
    else:
        scale = 1.0

    if step % 50 == 1:
        contact_str = "".join("●" if state.footForce[i] > 20 else "○" for i in range(4))
        print(f"\n{'='*90}", flush=True)
        print(f"t={t:5.1f}s | POLICY | "
              f"g=({gravity[0]:+.3f},{gravity[1]:+.3f},{gravity[2]:+.3f}) | tilt {tilt_angle:.1f}° | feet {contact_str}", flush=True)
        print(f"hip_q : {joint_pos[1]:6.3f} | knee_q : {joint_pos[2]:6.3f}", flush=True)
        print(f"elapsed time {t:.1f}s | ramp_level {ramp_level} | kp {current_kp:.1f} | kd {current_kd:.1f} | scale {scale:.1f}", flush=True)
        print(f"{'='*90}", flush=True)

    commands = torch.tensor([1.2, 0.0, 0.0], dtype=torch.float32)

    joint_pos_t = torch.tensor(joint_pos) - torch.tensor(stand_q)  # NORMALIZE to deltas

    joint_vel_t = torch.tensor(joint_vel, dtype=torch.float32)
    ang_vel_t = torch.tensor(state.imu.gyroscope, dtype=torch.float32)
    gravity_t = torch.tensor(gravity, dtype=torch.float32)

    current_base = torch.cat([commands, joint_pos_t, joint_vel_t, ang_vel_t, gravity_t], dim=0)

    obs_history = torch.roll(obs_history, shifts=-1, dims=0)
    obs_history[-1] = torch.cat([current_base, prev_action])

    history_flat = obs_history.flatten().unsqueeze(0)

    with torch.no_grad():
        embedding = encoder(history_flat)

    full_obs = torch.cat([current_base, prev_action, embedding.squeeze(0)], dim=0).unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        action = policy(full_obs)

    raw_delta_q = action.squeeze(0).numpy()

    delta_q = raw_delta_q * scale  # disabled (scale = 1.0 normal, 0.5 if tilt high)

    prev_action = torch.tensor(delta_q, dtype=torch.float32)

    target_q = stand_q + delta_q

    for i in range(12):
        err = target_q[i] - joint_pos[i]
        q_cmd = target_q[i] + 0.15 * err

        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0.0
        cmd.motorCmd[i].Kp = current_kp
        cmd.motorCmd[i].Kd = current_kd
        cmd.motorCmd[i].tau = 0.0

    if step % 50 == 1:
        print(f"→ raw_delta_q: {[f'{x:.3f}' for x in raw_delta_q]}", flush=True)
        print(f"→ scaled delta_q: {[f'{x:.3f}' for x in delta_q]}", flush=True)
        print(f"→ target_q hip {target_q[1]:+6.3f} knee {target_q[2]:+6.3f}", flush=True)
        print(f"→ current_q hip {joint_pos[1]:+6.3f} knee {joint_pos[2]:+6.3f}", flush=True)
        print(f"→ ERR hip {target_q[1] - joint_pos[1]:+6.3f} knee {target_q[2] - joint_pos[2]:+6.3f}", flush=True)
        print(f"→ FOOT FORCES: {state.footForce}", flush=True)

    try:
        safe.PowerProtect(cmd, state, 9)
        udp.SetSend(cmd)
        udp.Send()
    except Exception as e:
        print(f"UDP SEND ERROR: {e}", flush=True)
        break