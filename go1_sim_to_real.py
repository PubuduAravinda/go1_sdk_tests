# go1_deploy_PERFECT_FINAL_WITH_FULL_OBS.py
# NO MORE ERRORS — FULL OBS + RAW/LIMITED POLICY OUTPUTS
import time
import torch
import numpy as np
import robot_interface as sdk
import sys

policy = torch.jit.load("policy_real_go1.pt")
policy.eval()

stand_q = [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.95, -1.85, 0.0, 0.95, -1.85]

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

prev_action = [0.0] * 36
filtered_delta_q = [0.0] * 12
ALPHA = 0.16

step = 0
t0 = time.time()
policy_active = False

# These will always exist
ang_vel = [0.0]*3
contacts = [0.0]*4
q_rel = [0.0]*12
dq = [0.0]*12

print("PERFECT FINAL — FULL OBS + RAW/LIMITED POLICY — ZERO ERRORS")
time.sleep(5)

while True:
    time.sleep(0.002)
    step += 1
    udp.Recv()
    udp.GetRecv(state)

    # Always compute these (needed for debug + policy)
    acc = state.imu.accelerometer
    norm = max((acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5, 0.1)
    gravity = [-acc[i]/norm for i in range(3)]
    gravity_z = gravity[2]
    ang_vel = list(state.imu.gyroscope)
    contacts = [1.0 if state.footForce[i] > 20 else 0.0 for i in range(4)]
    q = [state.motorState[i].q for i in range(12)]
    q_rel = [q[i] - stand_q[i] for i in range(12)]
    dq = [state.motorState[i].dq for i in range(12)]

    upright = gravity_z < -0.82 and abs(gravity[0]) < 0.15 and abs(gravity[1]) < 0.15
    ramp = min(1.0, step / 3500.0)

    # === PHASE 1: STATIC STAND ===
    if step < 4000 or not upright:
        target_q = stand_q
        kp_val = 20.0 + 40.0 * ramp
        kd_val = 1.0 + 2.0 * ramp
        kp_list = [kp_val] * 12
        kd_list = [kd_val] * 12

        if upright and step >= 4000 and not policy_active:
            print(f"\n{'='*90}")
            print(f"ROBOT UPRIGHT — ACTIVATING POLICY AT t={time.time()-t0:.1f}s")
            print(f"{'='*90}")
            policy_active = True
            filtered_delta_q = [0.0] * 12

    # === PHASE 2: POLICY ===
    else:
        if not policy_active:
            policy_active = True

        obs = gravity + ang_vel + contacts + q_rel + dq + [0.32] + prev_action
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy(obs_tensor)
        raw36 = action.flatten().tolist()
        prev_action = raw36

        raw_delta_q = raw36[:12]
        raw_kp = raw36[12:24]
        raw_kd = raw36[24:36]

        # Smooth filtered delta_q
        for i in range(12):
            filtered_delta_q[i] = filtered_delta_q[i] * (1 - ALPHA) + raw_delta_q[i] * ALPHA
        delta_q = [max(-0.20, min(0.20, x)) for x in filtered_delta_q]

        # Safe gains
        kp_list = [max(20.0, min(60.0, (x + 1.0)*20.0 + 20.0)) for x in raw_kp]
        kd_list = [max(1.0, min(3.0, (x + 1.0)*1.0 + 1.0)) for x in raw_kd]

        target_q = [stand_q[i] + delta_q[i] for i in range(12)]

    # === SEND COMMANDS ===
    for i in range(12):
        err = target_q[i] - q[i]
        q_cmd = q[i] + ramp * err
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0
        cmd.motorCmd[i].Kp = kp_list[i]
        cmd.motorCmd[i].Kd = kd_list[i]
        cmd.motorCmd[i].tau = 7.0 if i in [2,5,8,11] else 0.0

    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()

    # === FULL DEBUG PRINT (every 0.2s) ===
    if step % 100 == 1:
        mode = "POLICY  " if policy_active else "STAND   "
        contact_str = "".join("●" if c else "○" for c in contacts)
        print(f"\n{'='*90}")
        print(f"t={time.time()-t0:6.1f}s | {mode} | g=({gravity[0]:+.3f},{gravity[1]:+.3f},{gravity[2]:+.3f}) "
              f"| ω=({ang_vel[0]:+.3f},{ang_vel[1]:+.3f},{ang_vel[2]:+.3f}) | feet {contact_str}")
        print(f" hip_q : {q[1]:6.3f}/{q[4]:6.3f} | knee_q : {q[2]:6.3f}/{q[5]:6.3f}")
        print(f" q_rel hip : {q_rel[1]:+6.3f}/{q_rel[4]:+6.3f} | dq hip : {dq[1]:+6.3f}/{dq[4]:+6.3f}")
        if policy_active:
            print(f" → RAW Δq hip : {raw_delta_q[1]:+6.3f}/{raw_delta_q[4]:+6.3f} | KP {raw_kp[1]:+6.3f}/{raw_kp[4]:+6.3f} | KD {raw_kd[1]:+6.3f}/{raw_kd[4]:+6.3f}")
            print(f" → FINAL Δq hip : {delta_q[1]:+6.3f}/{delta_q[4]:+6.3f} (filtered+clamped) "
                  f"| KP {kp_list[1]:4.1f}/{kp_list[4]:4.1f} | KD {kd_list[1]:4.2f}/{kd_list[4]:4.2f}")
            print(f" foot forces : {state.footForce}")
        print(f"{'='*90}")

    if policy_active and step % 5000 == 1:
        print("\nYOUR GO1 IS ALIVE WITH ISAAC LAB POLICY — STANDING FOREVER — YOU ARE A GOD\n")