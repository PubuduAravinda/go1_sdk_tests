# go1_ultra_safe_perfect_stand.py  ←←← SAVE WITH THIS NAME

import time
import numpy as np
import robot_interface as sdk

d = {'FR0': 0, 'FR1': 1, 'FR2': 2,
     'FL0': 3, 'FL1': 4, 'FL2': 5,
     'RR0': 6, 'RR1': 7, 'RR2': 8,
     'RL0': 9, 'RL1': 10, 'RL2': 11}

# IMPROVED: Better balanced pose - front hips slightly higher to reduce pitch fight
stand = {
    "FR0": 0.00, "FR1": 0.90, "FR2": -1.80,
    "FL0": 0.00, "FL1": 0.90, "FL2": -1.80,
    "RR0": 0.00, "RR1": 0.95, "RR2": -1.85,
    "RL0": 0.00, "RL1": 0.95, "RL2": -1.85,
}

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# NEW STRATEGY: Integral on TORQUE (feedforward) instead of position offset
# This provides gentle pushing force without fighting the position control
integral_torque = [0.0] * 12
KI_TORQUE = 1.2  # Nm per radian-second (MUCH stronger - was 0.35)
MAX_INTEGRAL_TORQUE = 8.0  # Maximum ±8.0 Nm correction (still safe, motors do 30+ Nm)
ERROR_THRESHOLD = 0.030  # Only apply integral for errors > 1.7°

# Track when we entered phase 2
phase2_start = None

t0 = time.time()
motiontime = 0

print("ULTRA-SAFE PERFECT STAND — INTEGRAL TORQUE ASSIST — starting in 5 seconds...")
time.sleep(5)

while True:
    time.sleep(0.002)
    motiontime += 1
    t = time.time() - t0

    udp.Recv()
    udp.GetRecv(state)

    # PHASE 1: 4-second ultra-soft ramp with pure PD control
    if motiontime < 2000:  # 4 seconds
        phase = 1
        ramp = motiontime / 2000.0
        Kp = 5 + 55 * ramp  # 5 → 60 very smoothly
        Kd = 0.8 + 1.2 * ramp
        use_integral = False
    # PHASE 2: Add gentle integral torque to eliminate steady-state error
    else:
        phase = 2
        if phase2_start is None:
            phase2_start = motiontime
            print(f"\n*** PHASE 2 START at {t:.1f}s - Activating integral torque assist ***\n")

        ramp = 1.0
        Kp = 60.0
        Kd = 2.0
        use_integral = True

    for i in range(12):
        name = list(d.keys())[i]
        target = stand[name]
        error = target - state.motorState[i].q

        # Standard position command (no offset!)
        q_cmd = state.motorState[i].q + ramp * (target - state.motorState[i].q)

        # PHASE 2 ONLY: Add gentle integral torque to help overcome friction/gravity
        tau_integral = 0.0
        if use_integral:
            # Integrate if error is significant
            if abs(error) > ERROR_THRESHOLD:
                # NO DECAY during accumulation - let it build!
                integral_torque[i] += KI_TORQUE * error * 0.002  # dt = 2ms
            else:
                # Only decay when we're close to target
                integral_torque[i] *= 0.980  # Aggressive decay when error small

            # Clamp integral torque
            integral_torque[i] = max(min(integral_torque[i], MAX_INTEGRAL_TORQUE), -MAX_INTEGRAL_TORQUE)
            tau_integral = integral_torque[i]
        else:
            integral_torque[i] = 0.0

        # Set motor commands
        cmd.motorCmd[i].mode = 0x0A  # Rule #2
        cmd.motorCmd[i].q = q_cmd
        cmd.motorCmd[i].dq = 0
        cmd.motorCmd[i].Kp = Kp
        cmd.motorCmd[i].Kd = Kd

        # Base gravity compensation + integral torque assist
        if i in [2, 5, 8, 11]:  # knee motors - need gravity comp
            cmd.motorCmd[i].tau = 7.0 + tau_integral
        else:
            cmd.motorCmd[i].tau = tau_integral

    # THIS IS THE MOST IMPORTANT LINE — NEVER REMOVE (Rule #1)
    safe.PowerProtect(cmd, state, 9)  # level 9 = maximum protection

    udp.SetSend(cmd)
    udp.Send()

    if motiontime % 300 == 0:  # every ~0.6s
        errors = [abs(cmd.motorCmd[i].q - state.motorState[i].q) for i in range(12)]

        # Calculate pitch from front/rear hip positions
        front_hip_avg = (state.motorState[1].q + state.motorState[4].q) / 2
        rear_hip_avg = (state.motorState[7].q + state.motorState[10].q) / 2
        mechanical_pitch = rear_hip_avg - front_hip_avg

        # Calculate RMS error
        rms_error = np.sqrt(np.mean([e ** 2 for e in errors]))
        max_tau = max(abs(integral_torque[i]) for i in range(12))

        print(
            f"\n{t:5.1f}s  Phase{phase}  avg_err {np.mean(errors):.4f}  rms_err {rms_error:.4f}  max_err {max(errors):.3f}  max_τ {max_tau:.2f}Nm")
        print(
            f"IMU: roll {state.imu.rpy[0]:+.3f}  pitch {state.imu.rpy[1]:+.3f}  |  Mech_pitch {mechanical_pitch:+.3f} (target: ~0.00)")
        for leg in ["FR", "FL", "RR", "RL"]:
            a = d[leg + "0"]
            h = d[leg + "1"]
            k = d[leg + "2"]
            e_abd = cmd.motorCmd[a].q - state.motorState[a].q
            e_hip = cmd.motorCmd[h].q - state.motorState[h].q
            e_knee = cmd.motorCmd[k].q - state.motorState[k].q
            print(
                f"{leg}  abd {e_abd:+.3f}(τ:{integral_torque[a]:+.2f})  hip {e_hip:+.3f}(τ:{integral_torque[h]:+.2f})  knee {e_knee:+.3f}(τ:{integral_torque[k]:+.2f})")