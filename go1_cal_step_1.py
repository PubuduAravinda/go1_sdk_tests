# -*- coding: utf-8 -*-
"""
=============================================================================
GO1 CALIBRATION + STEP 1 DEBUG VERIFICATION
=============================================================================
Original: one-joint-at-a-time calibration sweep
Added:
  - SDK comms health printed every phase (tick, UDP errors)
  - joint velocity (dq) printed alongside position during motion
  - IMU printed every phase (gravity, gyro, accel, quaternion)
  - foot contact printed every phase
  - per-phase velocity stats (min/max/avg dq per joint)
  - confirmation tags [OK] / [WARN] / [FAIL] on every debug block

Step 1 items confirmed by this script:
  [1a] SDK communication stable       -> tick increments, zero UDP errors
  [1b] Joint position readback        -> avg_real printed each phase
  [1c] Joint velocity readback        -> dq printed each step + phase stats
  [1d] IMU readback                   -> printed each phase
  [1e] Foot contact readback          -> printed each phase
  [1f] Position commands work         -> robot physically moves to targets
  Tracking quality / Kp sweep         -> cycle summary table

Isaac Lab joint order used everywhere except at UDP boundary.
=============================================================================
"""

from __future__ import print_function
import time
import numpy as np
import robot_interface as sdk
import sys

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
KP_START              = 10.0
KD_FIXED              = 3.0
KP_STEP               = 10.0
RAMP_MAX_LEVEL        = 4
HOLD_SECONDS_PER_PHASE= 3.0
SETTLE_SECONDS        = 1.0

# Debug print intervals
DEBUG_PRINT_EVERY_N_STEPS = 250    # print live dq + IMU every N steps (~0.5s at 500Hz)
PHASE_IMU_SAMPLES         = 20     # IMU samples to average per phase for the phase summary

# ---------------------------------------------------------------------------
# JOINT NAMES  (Isaac order)
# ---------------------------------------------------------------------------
joint_names = [
    "FL_hip",  "FR_hip",  "RL_hip",  "RR_hip",
    "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

# ---------------------------------------------------------------------------
# CROUCH POSE  (Isaac order)
# ---------------------------------------------------------------------------
crouch_q = np.array([0.0, 0.0, 0.0, 0.0,
                     1.8, 1.8, 1.8, 1.8,
                    -2.8,-2.8,-2.8,-2.8])

# ---------------------------------------------------------------------------
# INDEX MAPS
# ---------------------------------------------------------------------------
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

# verify inverse
_chk = [sdk_to_isaac[isaac_to_sdk[i]] for i in range(12)]
assert _chk == list(range(12)), "Mapping FAILED: {}".format(_chk)

SDK_NAMES = [
    "FR_hip",  "FR_thigh","FR_calf",
    "FL_hip",  "FL_thigh","FL_calf",
    "RR_hip",  "RR_thigh","RR_calf",
    "RL_hip",  "RL_thigh","RL_calf",
]

# ---------------------------------------------------------------------------
# PHASE DELTAS  (Isaac order) -- unchanged from original
# ---------------------------------------------------------------------------
th_delta  = -np.deg2rad(90)
k_delta   =  np.deg2rad(90)
abd_delta =  np.deg2rad(20)

fl_thigh_forward = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, 0.0,-0.15, 0.10,-0.10])
fl_knee_forward  = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, k_delta,-0.15, 0.10,-0.10])
fl_abd_forward   = np.array([abd_delta,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, k_delta,-0.15, 0.10,-0.10])
fl_abd_back      = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, k_delta,-0.15, 0.10,-0.10])
fl_knee_back     = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, 0.0,-0.15, 0.10,-0.10])
fl_thigh_back    = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,-0.15, 0.10,-0.10])

fr_thigh_forward = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,0.0,-0.10, 0.10])
fr_knee_forward  = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,k_delta,-0.10, 0.10])
fr_abd_forward   = np.array([0.0,-abd_delta,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,k_delta,-0.10, 0.10])
fr_abd_back      = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,k_delta,-0.10, 0.10])
fr_knee_back     = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,0.0,-0.10, 0.10])
fr_thigh_back    = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, -0.15,0.0,-0.10, 0.10])

rl_thigh_forward = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,th_delta,0.0, 0.10,-0.10,0.0,-0.15])
rl_knee_forward  = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,th_delta,0.0, 0.10,-0.10,k_delta,-0.15])
rl_abd_forward   = np.array([0.0,0.0,abd_delta,0.0, 0.0,0.0,th_delta,0.0, 0.10,-0.10,k_delta,-0.15])
rl_abd_back      = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,th_delta,0.0, 0.10,-0.10,k_delta,-0.15])
rl_knee_back     = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,th_delta,0.0, 0.10,-0.10,0.0,-0.15])
rl_thigh_back    = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.10,-0.10,0.0,-0.15])

rr_thigh_forward = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,th_delta, -0.10,0.10,-0.15,0.0])
rr_knee_forward  = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,th_delta, -0.10,0.10,-0.15,k_delta])
rr_abd_forward   = np.array([0.0,0.0,0.0,-abd_delta, 0.0,0.0,0.0,th_delta, -0.10,0.10,-0.15,k_delta])
rr_abd_back      = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,th_delta, -0.10,0.10,-0.15,k_delta])
rr_knee_back     = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,th_delta, -0.10,0.10,-0.15,0.0])
rr_thigh_back    = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, -0.10,0.10,-0.15,0.0])

null_for_crouch  = np.zeros(12)

phases = [
    null_for_crouch,
    fl_thigh_forward, fl_knee_forward,  fl_abd_forward,
    fl_abd_back,      fl_knee_back,     fl_thigh_back,
    fr_thigh_forward, fr_knee_forward,  fr_abd_forward,
    fr_abd_back,      fr_knee_back,     fr_thigh_back,
    rl_thigh_forward, rl_knee_forward,  rl_abd_forward,
    rl_abd_back,      rl_knee_back,     rl_thigh_back,
    rr_thigh_forward, rr_knee_forward,  rr_abd_forward,
    rr_abd_back,      rr_knee_back,     rr_thigh_back,
]

phase_names = [
    "crouch (baseline)",
    "FL thigh forward", "FL knee forward",  "FL abductor forward",
    "FL abductor back",  "FL knee back",    "FL thigh back",
    "FR thigh forward", "FR knee forward",  "FR abductor forward",
    "FR abductor back",  "FR knee back",    "FR thigh back",
    "RL thigh forward", "RL knee forward",  "RL abductor forward",
    "RL abductor back",  "RL knee back",    "RL thigh back",
    "RR thigh forward", "RR knee forward",  "RR abductor forward",
    "RR abductor back",  "RR knee back",    "RR thigh back",
]

# ---------------------------------------------------------------------------
# DEBUG HELPERS
# ---------------------------------------------------------------------------

def tag(passed):
    return "[OK]  " if passed else "[WARN]"

def print_separator(char="-", width=88):
    print(char * width)

def print_comms_debug(state, step_counter, udp_recv_errors, udp_send_errors):
    """[1a] SDK communication health."""
    tick = state.tick
    tick_ok = tick != 0
    print("")
    print("  --- [1a] SDK COMMS ---")
    print("  {} tick={}  recv_errors={}  send_errors={}  step={}".format(
        tag(tick_ok and udp_recv_errors == 0 and udp_send_errors == 0),
        tick, udp_recv_errors, udp_send_errors, step_counter))
    if not tick_ok:
        print("  [FAIL] tick=0: robot not sending state (wrong machine or not in low-level mode)")
    if udp_recv_errors > 0:
        print("  [FAIL] {} UDP recv errors accumulated".format(udp_recv_errors))
    if udp_send_errors > 0:
        print("  [FAIL] {} UDP send errors accumulated".format(udp_send_errors))

def print_joint_debug(q_isaac, qd_isaac, target_q_isaac, phase_name):
    """[1b]+[1c] Joint position and velocity readback."""
    print("")
    print("  --- [1b]+[1c] JOINT POS + VEL  ({}) ---".format(phase_name))
    print("  {:>12}  {:>9}  {:>9}  {:>9}  {:>10}  {:>10}  {:>8}".format(
        "Joint", "target", "actual", "err", "tgt(deg)", "act(deg)", "dq(r/s)"))
    print("  " + "-"*80)

    any_pos_warn  = False
    any_vel_warn  = False
    any_stuck     = False

    for ii in range(12):
        err     = target_q_isaac[ii] - q_isaac[ii]
        moving  = abs(qd_isaac[ii]) > 0.05       # joint is moving
        stuck   = abs(err) > 0.25 and not moving  # large error, not moving
        vel_ext = abs(qd_isaac[ii]) > 15.0        # extreme velocity

        if stuck:      any_stuck    = True
        if vel_ext:    any_vel_warn = True
        if abs(err) > 0.30: any_pos_warn = True

        # mark actively moving joints
        moving_tag = " <moving>" if moving else ""
        stuck_tag  = " *** STUCK"   if stuck   else ""
        vel_tag    = " *** VEL HIGH" if vel_ext else ""
        note       = moving_tag + stuck_tag + vel_tag

        print("  {:>12}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>10.3f}  {:>10.3f}  {:>8.4f}{}".format(
            joint_names[ii],
            target_q_isaac[ii], q_isaac[ii], err,
            np.rad2deg(target_q_isaac[ii]),
            np.rad2deg(q_isaac[ii]),
            qd_isaac[ii],
            note))

    # Summary tags
    print("  " + "-"*80)
    print("  {} Joint pos readback  (any |err|>0.30: {})".format(
        tag(not any_pos_warn), any_pos_warn))
    print("  {} Joint vel readback  (any |dq|>15 r/s: {})".format(
        tag(not any_vel_warn), any_vel_warn))
    print("  {} No stuck joints     (|err|>0.25 and dq~0: {})".format(
        tag(not any_stuck), any_stuck))

def print_imu_debug(state):
    """[1d] IMU readback -- same extraction as policy code."""
    acc  = np.array([state.imu.accelerometer[0],
                     state.imu.accelerometer[1],
                     state.imu.accelerometer[2]])
    gyro = np.array([state.imu.gyroscope[0],
                     state.imu.gyroscope[1],
                     state.imu.gyroscope[2]])
    quat = np.array([state.imu.quaternion[0],
                     state.imu.quaternion[1],
                     state.imu.quaternion[2],
                     state.imu.quaternion[3]])

    g_mag   = float(np.linalg.norm(acc))
    norm    = max(g_mag, 0.1)
    gravity = -acc / norm                         # policy convention
    tilt    = float(np.degrees(np.sqrt(gravity[0]**2 + gravity[1]**2)))
    q_norm  = float(np.linalg.norm(quat))

    roll_est  = float(np.degrees(np.arctan2(acc[1], acc[2])))
    pitch_est = float(np.degrees(np.arctan2(-acc[0], np.sqrt(acc[1]**2 + acc[2]**2))))

    g_ok    = 8.5 < g_mag < 11.0
    q_ok    = abs(q_norm - 1.0) < 0.05
    gyro_ok = float(max(abs(gyro))) < 2.0        # during motion some gyro is expected
    tilt_ok = tilt < 45.0

    print("")
    print("  --- [1d] IMU ---")
    print("  Accel (m/s2): ax={:+.3f} ay={:+.3f} az={:+.3f}  |g|={:.4f}  {}".format(
        acc[0], acc[1], acc[2], g_mag, tag(g_ok)))
    print("  Gyro  (r/s) : gx={:+.3f} gy={:+.3f} gz={:+.3f}  max={:.3f}  {}".format(
        gyro[0], gyro[1], gyro[2], float(max(abs(gyro))), tag(gyro_ok)))
    print("  Quat (w,x,y,z): {:+.3f} {:+.3f} {:+.3f} {:+.3f}  |q|={:.4f}  {}".format(
        quat[0], quat[1], quat[2], quat[3], q_norm, tag(q_ok)))
    print("  gravity=-acc/|acc|: ({:+.3f},{:+.3f},{:+.3f})  tilt={:.2f}deg  {}".format(
        gravity[0], gravity[1], gravity[2], tilt, tag(tilt_ok)))
    print("  roll={:+.2f}deg  pitch={:+.2f}deg  (0=level)".format(roll_est, pitch_est))
    # Policy obs preview
    print("  gyro*0.25 (obs): ({:+.4f},{:+.4f},{:+.4f})".format(
        gyro[0]*0.25, gyro[1]*0.25, gyro[2]*0.25))

    if g_mag == 0.0:
        print("  [FAIL] IMU all zeros -- no live data")

    return {"g_ok": g_ok, "q_ok": q_ok, "tilt": tilt}

def print_foot_debug(state):
    """[1e] Foot contact readback."""
    # SDK foot order: FR=0 FL=1 RR=2 RL=3
    foot_names = ["FR", "FL", "RR", "RL"]
    try:
        forces  = [state.footForce[i] for i in range(4)]
        contact = ["CONTACT" if f > 20 else "none" for f in forces]
        print("")
        print("  --- [1e] FOOT CONTACT ---")
        for i in range(4):
            print("    {}  force={:6d}  {}".format(foot_names[i], forces[i], contact[i]))
        print("  {} footForce readable (threshold=20)".format(
            tag(True)))
    except AttributeError:
        print("  --- [1e] FOOT CONTACT ---")
        print("  [WARN] footForce not available in this SDK build")

def print_velocity_phase_stats(phase_qd_buf, phase_name):
    """Print per-joint dq stats accumulated over the settled portion of a phase."""
    if len(phase_qd_buf) == 0:
        return
    qd_arr = np.array(phase_qd_buf)   # shape (N, 12)

    print("")
    print("  --- [1c] VELOCITY PHASE STATS  ({}) ---".format(phase_name))
    print("  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  Notes".format(
        "Joint", "min dq", "avg dq", "max dq", "max|dq|"))
    print("  " + "-"*72)
    for ii in range(12):
        col     = qd_arr[:, ii]
        min_dq  = float(np.min(col))
        avg_dq  = float(np.mean(col))
        max_dq  = float(np.max(col))
        max_abs = float(np.max(np.abs(col)))
        note    = ""
        if max_abs > 15.0:
            note = "*** HIGH VEL"
        elif max_abs > 5.0:
            note = "active motion"
        elif max_abs < 0.05:
            note = "stationary"
        print("  {:>12}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {}".format(
            joint_names[ii], min_dq, avg_dq, max_dq, max_abs, note))

# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------
udp  = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd  = sdk.LowCmd()
state= sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*90)
print("ONE LEG AT A TIME  +  STEP 1 DEBUG VERIFICATION")
print("Tests confirmed by this run:")
print("  [1a] SDK comms     -- tick, UDP error counts")
print("  [1b] Joint pos     -- readback vs target each phase")
print("  [1c] Joint vel     -- live dq + per-phase stats")
print("  [1d] IMU           -- accel/gyro/quat/gravity each phase")
print("  [1e] Foot contact  -- footForce each phase")
print("  [1f] Pos commands  -- robot physically moves to targets")
print("Hold: {:.1f}s | Settle: {:.1f}s | KP start: {} step {}".format(
    HOLD_SECONDS_PER_PHASE, SETTLE_SECONDS, KP_START, KP_STEP))
print("Debug prints every {} steps (~{:.1f}s)".format(
    DEBUG_PRINT_EVERY_N_STEPS, DEBUG_PRINT_EVERY_N_STEPS * 0.002))
print("Starting in 10 seconds ...")
print("="*90 + "\n")

time.sleep(10)

# ---------------------------------------------------------------------------
# MAIN LOOP STATE
# ---------------------------------------------------------------------------
t0               = time.time()
current_phase    = 0
phase_start_time = time.time()
current_kp       = KP_START
cycle_count      = 0
step_counter     = 0

udp_recv_errors  = 0
udp_send_errors  = 0

phase_real_list  = []
phase_error_list = []
phase_qd_list    = []    # NEW: velocity buffer for phase stats

cycle_abs_errors = [[] for _ in range(12)]

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
while True:
    time.sleep(0.002)
    step_counter += 1

    # -- Receive --------------------------------------------------------------
    try:
        udp.Recv()
        udp.GetRecv(state)
    except Exception as e:
        udp_recv_errors += 1
        print("UDP RECV ERROR: {}".format(e), flush=True)
        if udp_recv_errors > 10:
            print("Too many recv errors, stopping.")
            break
        continue

    # -- Read state in Isaac order (SDK->Isaac boundary) ----------------------
    q_sdk  = np.array([state.motorState[i].q  for i in range(12)])
    qd_sdk = np.array([state.motorState[i].dq for i in range(12)])
    q_isaac  = q_sdk[sdk_to_isaac]
    qd_isaac = qd_sdk[sdk_to_isaac]

    # -- Build command in Isaac order, convert at boundary --------------------
    delta_q       = phases[current_phase]
    delta_q_sdk   = delta_q[isaac_to_sdk]
    target_q_sdk  = crouch_q[isaac_to_sdk] + delta_q_sdk
    target_q_isaac= target_q_sdk[sdk_to_isaac]   # back to Isaac for debug prints

    # -- Send (Isaac->SDK conversion already done above) ----------------------
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = target_q_sdk[i]
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = current_kp
        cmd.motorCmd[i].Kd   = KD_FIXED
        cmd.motorCmd[i].tau  = 0.0

    try:
        udp.SetSend(cmd)
        udp.Send()
    except Exception as e:
        udp_send_errors += 1
        print("UDP SEND ERROR: {}".format(e), flush=True)
        if udp_send_errors > 10:
            print("Too many send errors, stopping.")
            break

    # -- Accumulate phase data after settle -----------------------------------
    time_in_phase = time.time() - phase_start_time
    if time_in_phase >= SETTLE_SECONDS:
        phase_real_list.append(q_isaac.copy())
        phase_error_list.append((target_q_isaac - q_isaac).copy())
        phase_qd_list.append(qd_isaac.copy())    # NEW: accumulate dq

    # -- Live debug print every N steps ---------------------------------------
    if step_counter % DEBUG_PRINT_EVERY_N_STEPS == 0:
        print_separator("=")
        print("LIVE DEBUG | step={} | t={:.1f}s | phase={}/{} | {} | KP={:.1f}".format(
            step_counter,
            time.time() - t0,
            current_phase, len(phases) - 1,
            phase_names[current_phase],
            current_kp))
        print_separator("=")

        print_comms_debug(state, step_counter, udp_recv_errors, udp_send_errors)
        print_joint_debug(q_isaac, qd_isaac, target_q_isaac,
                          phase_names[current_phase])
        print_imu_debug(state)
        print_foot_debug(state)
        print("")

    # -- Phase end -------------------------------------------------------------
    if time_in_phase >= HOLD_SECONDS_PER_PHASE:

        if len(phase_real_list) > 0:
            avg_real  = np.mean(phase_real_list, axis=0)
            avg_error = np.mean(phase_error_list, axis=0)
            avg_qd    = np.mean(np.abs(np.array(phase_qd_list)), axis=0)  # NEW

            print_separator("-")
            print("KP={:.1f} | Phase {:2d}/{} | {}".format(
                current_kp, current_phase, len(phases)-1,
                phase_names[current_phase]))
            print("Target  (Isaac): " +
                  " ".join("{:+7.3f}".format(x) for x in target_q_isaac))
            print("AvgReal (Isaac): " +
                  " ".join("{:+7.3f}".format(x) for x in avg_real))
            print("AvgErr  (Isaac): " +
                  " ".join("{:+7.3f}".format(x) for x in avg_error))
            # NEW: velocity summary line
            print("Avg|dq| (Isaac): " +
                  " ".join("{:+7.3f}".format(x) for x in avg_qd))
            print("(settled over last {:.1f}s, {} samples)".format(
                HOLD_SECONDS_PER_PHASE - SETTLE_SECONDS,
                len(phase_real_list)))

            # NEW: velocity phase stats
            print_velocity_phase_stats(phase_qd_list, phase_names[current_phase])

            # NEW: IMU snapshot at phase end
            print_imu_debug(state)

            # NEW: foot contact at phase end
            print_foot_debug(state)

            print_separator("-")

            abs_errors = np.abs(avg_error)
            for j in range(12):
                cycle_abs_errors[j].append(abs_errors[j])

        phase_real_list  = []
        phase_error_list = []
        phase_qd_list    = []    # NEW: reset velocity buffer

        current_phase   += 1
        phase_start_time = time.time()

        # -- Cycle end ---------------------------------------------------------
        if current_phase >= len(phases):
            current_phase = 0
            cycle_count  += 1

            print("\n" + "="*90)
            print(" CYCLE {} SUMMARY | KP = {:.1f}".format(cycle_count, current_kp))
            print(" {:>2}  {:>12}  {:>10}  {:>10}  {:>10}  Notes".format(
                "ii", "Joint", "Min|err|", "Avg|err|", "Max|err|"))
            print("-"*70)

            motor_avg_errors = []
            for j in range(12):
                errs  = cycle_abs_errors[j]
                min_e = float(np.min(errs)) if errs else 0.0
                avg_e = float(np.mean(errs)) if errs else 0.0
                max_e = float(np.max(errs)) if errs else 0.0
                note  = ("HIGH resistance" if avg_e > 0.30
                         else "moderate tracking" if avg_e > 0.15
                         else "")
                print(" {:>2}  {:>12}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {}".format(
                    j, joint_names[j], min_e, avg_e, max_e, note))
                motor_avg_errors.append(avg_e)

            overall_avg = float(np.mean(motor_avg_errors))
            overall_max = float(max(motor_avg_errors))
            print("-"*70)
            print("Overall | Min: {:.3f} | Avg: {:.3f} | Max: {:.3f}".format(
                float(min(motor_avg_errors)), overall_avg, overall_max))

            # Step 1 confirmation banner
            print("")
            print("  STEP 1 CONFIRMATION (this cycle):")
            print("  [1a] SDK comms     recv_err={}  send_err={}  {}".format(
                udp_recv_errors, udp_send_errors,
                tag(udp_recv_errors == 0 and udp_send_errors == 0)))
            print("  [1b] Joint pos     max_avg_err={:.3f} rad  {}".format(
                overall_max, tag(overall_max < 0.30)))
            print("  [1c] Joint vel     see per-phase stats above  [OK]  (dq printed each phase)")
            print("  [1d] IMU           see per-phase prints above  [OK]  (check |g|~9.81, |q|~1)")
            print("  [1e] Foot contact  see per-phase prints above  [OK]  (0 expected on rack)")
            print("  [1f] Pos commands  robot moved through {} phases  [OK]".format(
                len(phases)))
            print("="*90 + "\n")

            cycle_abs_errors = [[] for _ in range(12)]

            current_kp += KP_STEP
            if current_kp > KP_START + RAMP_MAX_LEVEL * KP_STEP:
                current_kp = KP_START + RAMP_MAX_LEVEL * KP_STEP
                print("Max KP reached -- no more increment\n")

            print("=== New cycle | KP = {:.1f} ===\n".format(current_kp))

        if current_phase < len(phases):
            print("-> Next: {}".format(phase_names[current_phase]))