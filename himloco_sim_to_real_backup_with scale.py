# himloco_sim_to_real.py — VERY GENTLE RAMP + TILT SAFETY

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

d = {'FR0':0, 'FR1':1, 'FR2':2, 'FL0':3, 'FL1':4, 'FL2':5, 'RR0':6, 'RR1':7, 'RR2':8, 'RL0':9, 'RL1':10,'RL2':11}

stand = {
    "FR0": 0.00, "FR1": 0.85, "FR2": -1.75,
    "FL0": 0.00, "FL1": 0.85, "FL2": -1.75,
    "RR0": 0.00, "RR1": 0.95, "RR2": -1.85,
    "RL0": 0.00, "RL1": 0.95, "RL2": -1.85,
}

stand_q = [stand[list(d.keys())[i]] for i in range(12)]

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

ALPHA = 0.05
MAX_DELTA_Q = 0.60
KP_SAFE = 70.0
KD_SAFE = 3.0

prev_action = torch.zeros(12, dtype=torch.float32)
obs_history = torch.zeros(5, 45, dtype=torch.float32)
step = 0
t0 = time.time()
policy_active = False
ramp_step = 0

integral = [0.0] * 12
KI = 0.00008
MAX_INTEGRAL = 0.25

print("\n" + "="*90)
print("HIMLOCO DEPLOYMENT — VERY GENTLE RAMP + TILT SAFETY")
print("Starting in 5 seconds...")
print("="*90 + "\n")
time.sleep(5)

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

    joint_pos = [state.motorState[i].q for i in range(12)]
    joint_vel = [state.motorState[i].dq for i in range(12)]

    if step % 50 == 1:
        contact_str = "".join("●" if state.footForce[i] > 20 else "○" for i in range(4))
        print(f"\n{'='*90}", flush=True)
        print(f"t={t:5.1f}s | {'POLICY' if policy_active else 'SAFE STAND'} | "
              f"g=({gravity[0]:+.3f},{gravity[1]:+.3f},{gravity[2]:+.3f}) | feet {contact_str}", flush=True)
        print(f"hip_q : {joint_pos[1]:6.3f} | knee_q : {joint_pos[2]:6.3f}", flush=True)
        print(f"{'='*90}", flush=True)

    # === MANUAL STAND PHASE ===
    if not policy_active:
        ramp = min(1.0, step / 2000.0)
        Kp = 5 + 55 * ramp
        Kd = 0.8 + 1.2 * ramp

        for i in range(12):
            name = list(d.keys())[i]
            target = stand[name]
            error = target - joint_pos[i]
            integral[i] += KI * error
            integral[i] = max(min(integral[i], MAX_INTEGRAL), -MAX_INTEGRAL)
            final_target = target + integral[i]
            q_cmd = joint_pos[i] + ramp * (final_target - joint_pos[i])

            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q = q_cmd
            cmd.motorCmd[i].dq = 0.0
            cmd.motorCmd[i].Kp = Kp
            cmd.motorCmd[i].Kd = Kd
            cmd.motorCmd[i].tau = 7.0 if i in [2,5,8,11] else 0.0

        if step % 50 == 1:
            print("→ In SAFE STAND phase — gentle ramp", flush=True)

        if step > 3000 and gravity_z < -0.82:
            print("\n" + "="*90, flush=True)
            print(f"ROBOT UPRIGHT & STABLE — ACTIVATING FULL HIMLOCO POLICY at t={t:.1f}s", flush=True)
            print("→ Setting policy_active = True", flush=True)
            print("="*90 + "\n", flush=True)
            policy_active = True
            integral = [0.0] * 12
            obs_history.fill_(0.0)
            ramp_step = 0

    # === POLICY PHASE ===
    elif policy_active:
        print("DEBUG: Entering POLICY block", flush=True)
        ramp_step += 1
        policy_weight = min(1.0, ramp_step / 2500.0)  # ramp over ~5 s

        scale = 0.5 + 2.0 * policy_weight
        Kp = 30 + 40 * policy_weight
        Kd = 1.5 + 1.5 * policy_weight
        err_factor = 0.4 + 0.6 * policy_weight

        commands = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        joint_pos_t = torch.tensor(joint_pos, dtype=torch.float32)
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

        raw_delta_q = action.squeeze(0).numpy() * scale

        delta_q = []
        for i in range(12):
            prev = prev_action[i].item()
            curr = raw_delta_q[i]
            filtered = prev * (1 - ALPHA) + curr * ALPHA
            clamped = max(-MAX_DELTA_Q, min(MAX_DELTA_Q, filtered))
            delta_q.append(clamped)

        delta_q = torch.tensor(delta_q)
        prev_action = delta_q.clone()

        policy_target_q = torch.tensor(stand_q, dtype=torch.float32) + delta_q

        target_q = policy_weight * policy_target_q + (1 - policy_weight) * torch.tensor(stand_q, dtype=torch.float32)

        kp_list = [Kp] * 12
        kd_list = [Kd] * 12

        for i in range(12):
            err = target_q[i].item() - joint_pos[i]
            q_cmd = joint_pos[i] + err_factor * err

            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q = q_cmd
            cmd.motorCmd[i].dq = 0.0
            cmd.motorCmd[i].Kp = kp_list[i]
            cmd.motorCmd[i].Kd = kd_list[i]
            cmd.motorCmd[i].tau = 7.0 if i in [2, 5, 8, 11] else 0.0

        if step % 50 == 1:
            print(
                f"→ ramp_step {ramp_step} | policy_weight {policy_weight:.2f} | scale {scale:.2f} | Kp {Kp:.1f} | Kd {Kd:.1f} | err_factor {err_factor:.2f}",
                flush=True)
            print(f"→ commands: {commands}", flush=True)
            print(f"→ joint_pos_t: {joint_pos_t}", flush=True)
            print(f"→ joint_vel_t: {joint_vel_t}", flush=True)
            print(f"→ ang_vel_t: {ang_vel_t}", flush=True)
            print(f"→ gravity_t: {gravity_t}", flush=True)
            print(f"→ embedding: {embedding}", flush=True)
            print(f"→ full_obs shape: {full_obs.shape}", flush=True)
            print(f"→ action: {action}", flush=True)
            print(f"→ SCALED RAW DELTA_Q: {[f'{x:+.3f}' for x in raw_delta_q]}", flush=True)
            print(f"→ FINAL DELTA_Q: {[f'{x:+.3f}' for x in delta_q]}", flush=True)
            print(f"→ TARGET_Q hip {target_q[1]:+6.3f} knee {target_q[2]:+6.3f}", flush=True)
            print(f"→ CURRENT_Q hip {joint_pos[1]:+6.3f} knee {joint_pos[2]:+6.3f}", flush=True)
            print(f"→ ERR hip {target_q[1] - joint_pos[1]:+6.3f} knee {target_q[2] - joint_pos[2]:+6.3f}", flush=True)
            print(f"→ FOOT FORCES: {state.footForce}", flush=True)

    try:
        safe.PowerProtect(cmd, state, 9)
        udp.SetSend(cmd)
        udp.Send()
    except Exception as e:
        print(f"UDP SEND ERROR: {e}", flush=True)
        break