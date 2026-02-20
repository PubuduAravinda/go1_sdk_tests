import time
import numpy as np
import robot_interface as sdk
import sys

# ─── CONFIG ─────────────────────────────────────────────────────────────────
KP_START = 5.0
KD_FIXED = 3.0
KP_STEP = 5.0
RAMP_MAX_LEVEL = 10
HOLD_SECONDS_PER_PHASE = 5.0
SETTLE_SECONDS = 3.0

joint_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_th", "FR_th", "RL_th", "RR_th", "FL_cal", "FR_cal", "RL_cal", "RR_cal"]

crouch_q = np.array([0.0, 0.0, 0.0, 0.0, 1.8, 1.8, 1.8, 1.8, -2.8, -2.8, -2.8, -2.8])

sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0]*12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

# ─── ANGLES ─────────────────────────────────────────────────────────────────
th_delta = -np.deg2rad(90)
k_delta  = np.deg2rad(90)
abd_delta = np.deg2rad(20)

# ─── PHASE DELTAS (Isaac order) ─────────────────────────────────────────────

fl_thigh_forward  = np.array([0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, 0.0, -0.15, 0.10, -0.10])
fl_knee_forward   = np.array([0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, k_delta, -0.15, 0.10, -0.10])
fl_abd_forward    = np.array([abd_delta, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, k_delta, -0.15, 0.10, -0.10])
fl_abd_back       = np.array([0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, k_delta, -0.15, 0.10, -0.10])
fl_knee_back      = np.array([0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, 0.0, 0.0, -0.15, 0.10, -0.10])
fl_thigh_back     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.10, -0.10])

fr_thigh_forward  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, 0.0, -0.10, 0.10])
fr_knee_forward   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, k_delta, -0.10, 0.10])
fr_abd_forward    = np.array([0.0, -abd_delta, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, k_delta, -0.10, 0.10])
fr_abd_back       = np.array([0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, k_delta, -0.10, 0.10])
fr_knee_back      = np.array([0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.0, -0.15, 0.0, -0.10, 0.10])
fr_thigh_back     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0, -0.10, 0.10])

rl_thigh_forward  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, 0.0, -0.15])
rl_knee_forward   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, k_delta, -0.15])
rl_abd_forward    = np.array([0.0, 0.0, abd_delta, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, k_delta, -0.15])
rl_abd_back       = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, k_delta, -0.15])
rl_knee_back      = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, 0.0, 0.10, -0.10, 0.0, -0.15])
rl_thigh_back     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, -0.10, 0.0, -0.15])

rr_thigh_forward  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, 0.0])
rr_knee_forward   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, k_delta])
rr_abd_forward    = np.array([0.0, 0.0, 0.0, -abd_delta, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, k_delta])
rr_abd_back       = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, k_delta])
rr_knee_back      = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, th_delta, -0.10, 0.10, -0.15, 0.0])
rr_thigh_back     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10, 0.10, -0.15, 0.0])

null_for_crouch = np.zeros(12)

phases = [
    null_for_crouch,
    fl_thigh_forward, fl_knee_forward, fl_abd_forward,
    fl_abd_back, fl_knee_back, fl_thigh_back,
    fr_thigh_forward, fr_knee_forward, fr_abd_forward,
    fr_abd_back, fr_knee_back, fr_thigh_back,
    rl_thigh_forward, rl_knee_forward, rl_abd_forward,
    rl_abd_back, rl_knee_back, rl_thigh_back,
    rr_thigh_forward, rr_knee_forward, rr_abd_forward,
    rr_abd_back, rr_knee_back, rr_thigh_back,
]

phase_names = [
    "crouch (baseline)",
    "FL thigh forward", "FL knee forward", "FL abductor forward",
    "FL abductor back", "FL knee back", "FL thigh back",
    "FR thigh forward", "FR knee forward", "FR abductor forward",
    "FR abductor back", "FR knee back", "FR thigh back",
    "RL thigh forward", "RL knee forward", "RL abductor forward",
    "RL abductor back", "RL knee back", "RL thigh back",
    "RR thigh forward", "RR knee forward", "RR abductor forward",
    "RR abductor back", "RR knee back", "RR thigh back",
]

# ─── INIT ───────────────────────────────────────────────────────────────────
udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd = sdk.LowCmd()
state = sdk.LowState()

udp.InitCmdData(cmd)

print("\n" + "="*90)
print("ONE LEG AT A TIME – 90° THIGH → 90° KNEE → 10° ABDUCT → REVERSE")
print(f"Hold: {HOLD_SECONDS_PER_PHASE:.1f} s | Settle: {SETTLE_SECONDS:.1f} s | KP start: {KP_START} step {KP_STEP}")
print("Starting in 10 seconds...")
print("="*90 + "\n")

time.sleep(10)

t0 = time.time()
current_phase = 0
phase_start_time = time.time()
current_kp = KP_START
cycle_count = 0

phase_real_list = []
phase_error_list = []

cycle_abs_errors = [[] for _ in range(12)]

while True:
    time.sleep(0.005)

    try:
        udp.Recv()
        udp.GetRecv(state)
    except Exception as e:
        print(f"UDP RECV ERROR: {e}", flush=True)
        break

    real_joint_pos_sdk = np.array([state.motorState[i].q for i in range(12)])
    real_joint_pos = real_joint_pos_sdk[sdk_to_isaac]

    delta_q = phases[current_phase]
    delta_q_sdk = delta_q[isaac_to_sdk]
    target_q_sdk = crouch_q[isaac_to_sdk] + delta_q_sdk

    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = target_q_sdk[i]
        cmd.motorCmd[i].dq = 0.0
        cmd.motorCmd[i].Kp = current_kp
        cmd.motorCmd[i].Kd = KD_FIXED
        cmd.motorCmd[i].tau = 0.0

    time_in_phase = time.time() - phase_start_time
    if time_in_phase >= SETTLE_SECONDS:
        phase_real_list.append(real_joint_pos.copy())
        target_q = target_q_sdk[sdk_to_isaac]
        phase_error_list.append((target_q - real_joint_pos).copy())

    if time_in_phase >= HOLD_SECONDS_PER_PHASE:

        if len(phase_real_list) > 0:
            avg_real = np.mean(phase_real_list, axis=0)
            avg_error = np.mean(phase_error_list, axis=0)

            print("─────────────────────────────────────────────────────────────────────────────────────")
            print(f"KP = {current_kp:.1f} | Phase {current_phase:2d}/{len(phases)-1} | {phase_names[current_phase]}")
            print("Given target pos: " + ", ".join(f"{x:+6.3f}" for x in target_q_sdk[sdk_to_isaac]))
            print("Avg real pos (settled): " + ", ".join(f"{x:+6.3f}" for x in avg_real))
            print("Avg error (given - real): " + ", ".join(f"{x:+6.3f}" for x in avg_error))
            print(f"(based on last {HOLD_SECONDS_PER_PHASE - SETTLE_SECONDS:.1f} s)")
            print("─────────────────────────────────────────────────────────────────────────────────────")

            abs_errors = np.abs(avg_error)
            for j in range(12):
                cycle_abs_errors[j].append(abs_errors[j])

        phase_real_list = []
        phase_error_list = []

        current_phase += 1
        phase_start_time = time.time()

        if current_phase >= len(phases):
            current_phase = 0
            cycle_count += 1

            # Cycle summary table
            print(f"\n" + "="*90)
            print(f" CYCLE {cycle_count} SUMMARY | KP = {current_kp:.1f}")
            print(" Motor | Joint name     | Min |err| | Avg |err| | Max |err| | Notes")
            print("-"*80)
            motor_avg_errors = []
            for j in range(12):
                errs = cycle_abs_errors[j]
                min_e = np.min(errs) if errs else 0.0
                avg_e = np.mean(errs) if errs else 0.0
                max_e = np.max(errs) if errs else 0.0
                note = "HIGH resistance / poor tracking" if avg_e > 0.30 else "moderate tracking" if avg_e > 0.15 else ""
                print(f" {j:2d}   | {joint_names[j]:14} | {min_e:8.3f} | {avg_e:8.3f} | {max_e:8.3f} | {note}")
                motor_avg_errors.append(avg_e)

            # Overall robot stats
            overall_min = min(motor_avg_errors)
            overall_avg = np.mean(motor_avg_errors)
            overall_max = max(motor_avg_errors)
            print("-"*80)
            print(f"Overall robot error stats | Min: {overall_min:5.3f} | Avg: {overall_avg:5.3f} | Max: {overall_max:5.3f}")
            print("="*90 + "\n")

            cycle_abs_errors = [[] for _ in range(12)]

            current_kp += KP_STEP
            if current_kp > KP_START + RAMP_MAX_LEVEL * KP_STEP:
                current_kp = KP_START + RAMP_MAX_LEVEL * KP_STEP
                print("\nMax KP reached — no more increment\n")

            print(f"\n=== New cycle | KP = {current_kp:.1f} ===\n")

        if current_phase < len(phases):
            print(f"→ Next: {phase_names[current_phase]}")

    try:
        udp.SetSend(cmd)
        udp.Send()
    except Exception as e:
        print(f"UDP SEND ERROR: {e}", flush=True)
        break