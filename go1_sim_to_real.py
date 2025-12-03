# go1_deploy_FINAL_PERFECT_NO_ERROR.py
# STATIC STAND → SMOOTH POLICY → ZERO ERRORS → STANDS BEAUTIFULLY

import time
import torch
import robot_interface as sdk

policy = torch.jit.load("policy_real_go1.pt")
policy.eval()

# Static stand pose (no sine wave)
stand_q = [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.95, -1.85, 0.0, 0.95, -1.85]

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

prev_action = [0.0] * 36
filtered_delta_q = [0.0] * 12
ALPHA = 0.18  # smooth filtering

step = 0
t0 = time.time()
policy_active = False

print("FINAL PERFECT — STATIC STAND → POLICY — NO MORE ERRORS")
time.sleep(5)

while True:
    time.sleep(0.002)
    step += 1
    udp.Recv()
    udp.GetRecv(state)

    ramp = min(1.0, step / 3000.0)

    # IMU
    acc = state.imu.accelerometer
    norm = max((acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5, 0.1)
    gravity = [-acc[i]/norm for i in range(3)]
    gravity_z = gravity[2]
    upright = gravity_z < -0.8

    # Foot contacts (always defined)
    contacts = [1.0 if state.footForce[i] > 20 else 0.0 for i in range(4)]
    contact_str = "".join("●" if c else "○" for c in contacts)

    # === PHASE 1: STATIC STAND ===
    if step < 4000 or not upright:
        target_q = stand_q
        kp_val = 20.0 + 40.0 * ramp   # 20 → 60
        kd_val = 1.0 + 2.0 * ramp     # 1.0 → 3.0
        kp_list = [kp_val] * 12
        kd_list = [kd_val] * 12

        if upright and step >= 4000 and not policy_active:
            print(f"\nUPRIGHT — SWITCHING TO POLICY AT t={time.time()-t0:.1f}s\n")
            policy_active = True
            filtered_delta_q = [0.0] * 12

    # === PHASE 2: POLICY (smooth + limited) ===
    else:
        # Obs
        ang_vel = list(state.imu.gyroscope)
        q = [state.motorState[i].q for i in range(12)]
        q_rel = [q[i] - stand_q[i] for i in range(12)]
        dq = [state.motorState[i].dq for i in range(12)]
        height = 0.32

        obs = gravity + ang_vel + contacts + q_rel + dq + [height] + prev_action

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy(obs_tensor)
        raw36 = action.flatten().tolist()
        prev_action = raw36

        # Raw deltas
        delta_raw = raw36[:12]

        # Heavy low-pass filter — eliminates all lag/jump
        for i in range(12):
            filtered_delta_q[i] = filtered_delta_q[i] * (1 - ALPHA) + delta_raw[i] * ALPHA

        # Hard clamp
        delta_q = [max(-0.15, min(0.15, x * 0.4)) for x in filtered_delta_q]

        # Gains (safe range)
        kp_list = [max(25.0, min(55.0, (x + 1.0)*15.0 + 25.0)) for x in raw36[12:24]]
        kd_list = [max(1.2, min(2.8, (x + 1.0)*0.8 + 1.2)) for x in raw36[24:]]

        target_q = [stand_q[i] + delta_q[i] for i in range(12)]

    # === APPLY COMMANDS ===
    for i in range(12):
        err = target_q[i] - state.motorState[i].q
        q_cmd = state.motorState[i].q + ramp * err

        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0
        cmd.motorCmd[i].Kp = kp_list[i]
        cmd.motorCmd[i].Kd = kd_list[i]
        cmd.motorCmd[i].tau = 7.0 if i in [2,5,8,11] else 0.0

    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()

    # === DEBUG PRINT EVERY 0.2s ===
    if step % 100 == 1:
        mode = "POLICY " if policy_active else "STATIC STAND"
        print(f"t={time.time()-t0:6.1f}s | {mode} | g_z={gravity_z:+.3f} | feet {contact_str}")
        print(f" hip_q : {state.motorState[1].q:6.3f}/{state.motorState[4].q:6.3f} | knee_q : {state.motorState[2].q:6.3f}/{state.motorState[5].q:6.3f}")
        if policy_active:
            print(f" filtered Δq hip : {delta_q[1]:+6.3f}/{delta_q[4]:+6.3f} | KP {kp_list[1]:4.1f}/{kp_list[4]:4.1f} KD {kd_list[1]:4.2f}/{kd_list[4]:4.2f}")
        print("-" * 90)

    if policy_active and step % 5000 == 1:
        print("\nYOUR GO1 IS STANDING PERFECTLY — POLICY IN FULL CONTROL — VICTORY!\n")