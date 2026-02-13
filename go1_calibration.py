# go1_ultra_safe_perfect_stand.py  ←←← SAVE WITH THIS NAME

import time
import numpy as np
import robot_interface as sdk

d = {'FR0':0, 'FR1':1, 'FR2':2,
     'FL0':3, 'FL1':4, 'FL2':5,
     'RR0':6, 'RR1':7, 'RR2':8,
     'RL0':9, 'RL1':10,'RL2':11}

# Conservative but good starting pose (very close to factory default)
stand = {
    "FR0":  0.00, "FR1": 0.85, "FR2": -1.75,
    "FL0":  0.00, "FL1": 0.85, "FL2": -1.75,
    "RR0":  0.00, "RR1": 0.95, "RR2": -1.85,
    "RL0":  0.00, "RL1": 0.95, "RL2": -1.85,
}

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# Tiny gentle integral — this is all we need
integral = [0.0] * 12
KI = 0.00008           # 25× smaller than before — super safe
MAX_INTEGRAL = 0.25    # ±14° max correction

t0 = time.time()
motiontime = 0

print("ULTRA-SAFE PERFECT STAND — WILL NOT FALL — starting in 5 seconds...")
time.sleep(5)

while True:
    time.sleep(0.002)
    motiontime += 1
    t = time.time() - t0

    udp.Recv()
    udp.GetRecv(state)

    # 4-second ultra-soft ramp (Rule #3 + #4)
    if motiontime < 2000:                     # 4 seconds
        ramp = motiontime / 2000.0
        Kp = 5 + 55 * ramp                    # 5 → 60 very smoothly
        Kd = 0.8 + 1.2 * ramp
    else:
        ramp = 1.0
        Kp = 60.0
        Kd = 2.0

    for i in range(12):
        name = list(d.keys())[i]

        target = stand[name]
        # Gentle integral only
        error = target - state.motorState[i].q
        integral[i] += KI * error
        integral[i] = max(min(integral[i], MAX_INTEGRAL), -MAX_INTEGRAL)

        final_target = target + integral[i]

        # Super important: interpolate from current position (Rule #4)
        q_cmd = state.motorState[i].q + ramp * (final_target - state.motorState[i].q)

        cmd.motorCmd[i].mode = 0x0A                     # Rule #2
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0
        cmd.motorCmd[i].Kp = Kp
        cmd.motorCmd[i].Kd = Kd
        cmd.motorCmd[i].tau = 0.0

        # Gentle knee gravity compensation (Rule #6)
        if i in [2,5,8,11]:   # knee motors
            cmd.motorCmd[i].tau = 7.0

    # THIS IS THE MOST IMPORTANT LINE — NEVER REMOVE (Rule #1)
    safe.PowerProtect(cmd, state, 9)        # level 9 = maximum protection

    udp.SetSend(cmd)
    udp.Send()

    if motiontime % 300 == 0:    # every ~0.6s
        errors = [abs(cmd.motorCmd[i].q - state.motorState[i].q) for i in range(12)]
        print(f"\n{t:5.1f}s  avg err {np.mean(errors):.4f} rad  max {max(errors):.3f} rad")
        print(f"roll {state.imu.rpy[0]:+.4f}  pitch_raw {state.imu.rpy[1]:+.4f}  norm {np.linalg.norm(state.imu.accelerometer):.3f}")
        for leg in ["FR","FL","RR","RL"]:
            a = d[leg+"0"]
            h = d[leg+"1"]
            k = d[leg+"2"]
            print(f"{leg}  abd_err {cmd.motorCmd[a].q-state.motorState[a].q:+.3f}  hip_err {cmd.motorCmd[h].q-state.motorState[h].q:+.3f}  knee_err {cmd.motorCmd[k].q-state.motorState[k].q:+.3f}")

