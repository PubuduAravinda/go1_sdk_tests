# -*- coding: utf-8 -*-
"""
GO1 -- TORQUE FEEDFORWARD + DQ TEST  (fixed)

Control law (all joints, always):
    tau_out = Kp*(q_target - q) + Kd*(-dq) + tau_ff

Position control stays ON the whole time -- joints cannot rotate freely.
tau_ff applied on active joint only, after it settles at target.

Fixes vs previous version:
  1. TAU_HOLD_S = 1.5s + INTER_LEVEL_SETTLE = 0.5s between levels
     -> each level fully settles before next one starts
     -> tau_ff=0 should show near-zero dq
  2. Active joint = joint whose delta CHANGED vs previous phase (not largest delta)
     -> FL_knee phase correctly identifies FL_calf, not FL_thigh
  3. HOLD_SECONDS_PER_PHASE = 8.0s so position stats have plenty of samples
     after tau sweep finishes (tau sweep takes ~10s total, runs in parallel)
  4. Tau sweep runs in a blocking inner loop, position data collection resumes after
"""

from __future__ import print_function
import time
import numpy as np
import robot_interface as sdk

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
KP                     = 20.0
KD                     = 3.0
HOLD_SECONDS_PER_PHASE = 12.0   # long enough for tau sweep + settled position samples
SETTLE_SECONDS         = 1.0    # wait before starting tau sweep
NEAR_THRESH            = 0.15   # rad -- joint must be this close to target

TAU_FF_LEVELS          = [-6.0, -3.0, 0.0, 3.0, 6.0]   # Nm
TAU_HOLD_S             = 1.5    # seconds recording at each level (was 0.4 -- too short)
TAU_SETTLE_S           = 0.5    # seconds to discard at start of each level (rebound settle)
INTER_LEVEL_SETTLE     = 0.5    # seconds of zero tau between levels to damp motion

KP_STEP                = 10.0
RAMP_MAX_LEVEL         = 4

# ---------------------------------------------------------------------------
# JOINT NAMES  (Isaac order)
# ---------------------------------------------------------------------------
joint_names = ["FL_hip","FR_hip","RL_hip","RR_hip",
               "FL_th", "FR_th", "RL_th", "RR_th",
               "FL_cal","FR_cal","RL_cal","RR_cal"]

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

# ---------------------------------------------------------------------------
# PHASE DELTAS  (Isaac order) -- identical to position cal code
# ---------------------------------------------------------------------------
th_delta  = -np.deg2rad(90)
k_delta   =  np.deg2rad(90)
abd_delta =  np.deg2rad(20)

fl_thigh_forward = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, 0.0,-0.15,0.10,-0.10])
fl_knee_forward  = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, k_delta,-0.15,0.10,-0.10])
fl_abd_forward   = np.array([abd_delta,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, k_delta,-0.15,0.10,-0.10])
fl_abd_back      = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, k_delta,-0.15,0.10,-0.10])
fl_knee_back     = np.array([0.0,0.0,0.0,0.0, th_delta,0.0,0.0,0.0, 0.0,-0.15,0.10,-0.10])
fl_thigh_back    = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,-0.15,0.10,-0.10])

fr_thigh_forward = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,0.0,-0.10,0.10])
fr_knee_forward  = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,k_delta,-0.10,0.10])
fr_abd_forward   = np.array([0.0,-abd_delta,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,k_delta,-0.10,0.10])
fr_abd_back      = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,k_delta,-0.10,0.10])
fr_knee_back     = np.array([0.0,0.0,0.0,0.0, 0.0,th_delta,0.0,0.0, -0.15,0.0,-0.10,0.10])
fr_thigh_back    = np.array([0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, -0.15,0.0,-0.10,0.10])

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
    "FL thigh forward","FL knee forward","FL abductor forward",
    "FL abductor back","FL knee back","FL thigh back",
    "FR thigh forward","FR knee forward","FR abductor forward",
    "FR abductor back","FR knee back","FR thigh back",
    "RL thigh forward","RL knee forward","RL abductor forward",
    "RL abductor back","RL knee back","RL thigh back",
    "RR thigh forward","RR knee forward","RR abductor forward",
    "RR abd back","RR knee back","RR thigh back",
]

# ---------------------------------------------------------------------------
# ACTIVE JOINT: joint whose delta CHANGED vs previous phase
# This correctly identifies the NEW joint being tested each phase,
# not just the largest delta (which stays FL_thigh across FL leg phases).
# ---------------------------------------------------------------------------
def changed_joint(phase_idx):
    """
    Return the Isaac index of the joint that changed most between
    phases[phase_idx-1] and phases[phase_idx].
    Falls back to argmax(|delta|) for phase 0 and crouch.
    """
    if phase_idx == 0:
        return int(np.argmax(np.abs(phases[phase_idx])))
    prev  = phases[phase_idx - 1]
    curr  = phases[phase_idx]
    diff  = np.abs(curr - prev)
    if np.max(diff) < 0.01:
        # no change -- fall back to largest delta
        return int(np.argmax(np.abs(curr)))
    return int(np.argmax(diff))

# ---------------------------------------------------------------------------
# SEND HELPER
# ---------------------------------------------------------------------------
def send_tick(udp, cmd, state, target_q_isaac, tau_ff_isaac):
    """One 500Hz tick: recv state, build cmd, send. Returns (q_isaac, qd_isaac)."""
    udp.Recv()
    udp.GetRecv(state)
    q_sdk   = target_q_isaac[isaac_to_sdk]
    tau_sdk = tau_ff_isaac[isaac_to_sdk]
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(q_sdk[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = KP
        cmd.motorCmd[i].Kd   = KD
        cmd.motorCmd[i].tau  = float(tau_sdk[i])
    udp.SetSend(cmd)
    udp.Send()
    q_isaac  = np.array([state.motorState[i].q  for i in range(12)])[sdk_to_isaac]
    qd_isaac = np.array([state.motorState[i].dq for i in range(12)])[sdk_to_isaac]
    return q_isaac, qd_isaac

# ---------------------------------------------------------------------------
# TAU SWEEP  (blocking, called once per phase after joint settles)
# ---------------------------------------------------------------------------
def run_tau_sweep(udp, cmd, state, target_q_isaac, act_j, phase_name):
    """
    For each tau level:
      1. INTER_LEVEL_SETTLE seconds with tau=0 to damp any residual motion
      2. TAU_SETTLE_S seconds with tau applied but data discarded (transient)
      3. TAU_HOLD_S seconds with tau applied and data recorded

    Returns list of (tau_ff, mean_dq, max_dq, mean_qerr, kp_term, std_dq)
    """
    results = []
    tau_zero = np.zeros(12)

    print("")
    print("  [TAU SWEEP] Phase {} | active joint: [{:2d}] {}".format(
        phase_name, act_j, joint_names[act_j]))
    print("  tau_out = Kp*(q_target-q) + Kd*(-dq) + tau_ff")
    print("  q_target={:.4f} rad ({:.2f} deg)".format(
        target_q_isaac[act_j], np.rad2deg(target_q_isaac[act_j])))
    print("  Each level: {:.1f}s settle + {:.1f}s discard + {:.1f}s record".format(
        INTER_LEVEL_SETTLE, TAU_SETTLE_S, TAU_HOLD_S))
    print("  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  Notes".format(
        "tau_ff(Nm)", "mean_dq", "max|dq|", "std_dq", "mean_qerr", "Kp_term"))
    print("  " + "-"*78)

    for tau_val in TAU_FF_LEVELS:
        tau_ff = np.zeros(12)
        tau_ff[act_j] = tau_val

        # --- inter-level settle: tau=0, damp residual motion ---
        t0 = time.time()
        while time.time() - t0 < INTER_LEVEL_SETTLE:
            send_tick(udp, cmd, state, target_q_isaac, tau_zero)
            time.sleep(0.002)

        # --- discard window: tau applied but not recorded ---
        t0 = time.time()
        while time.time() - t0 < TAU_SETTLE_S:
            send_tick(udp, cmd, state, target_q_isaac, tau_ff)
            time.sleep(0.002)

        # --- record window ---
        dq_log = []
        q_log  = []
        t0 = time.time()
        while time.time() - t0 < TAU_HOLD_S:
            q_i, qd_i = send_tick(udp, cmd, state, target_q_isaac, tau_ff)
            dq_log.append(float(qd_i[act_j]))
            q_log.append(float(q_i[act_j]))
            time.sleep(0.002)

        mean_dq  = float(np.mean(dq_log))
        max_dq   = float(np.max(np.abs(dq_log)))
        std_dq   = float(np.std(dq_log))
        mean_qerr= float(np.mean(np.abs(np.array(q_log) - target_q_isaac[act_j])))
        kp_term  = float(KP * (target_q_isaac[act_j] - np.mean(q_log)))

        # sign check only meaningful for nonzero tau
        sign_ok  = (tau_val == 0.0) or (mean_dq * tau_val > 0) or (max_dq < 0.02)
        note = ""
        if not sign_ok:
            note = "WARN: sign mismatch"
        elif tau_val == 0.0 and max_dq > 0.05:
            note = "WARN: dq nonzero at tau=0 (not settled)"
        elif abs(tau_val) > 0 and max_dq > 0.05:
            note = "OK"
        elif abs(tau_val) > 0 and max_dq < 0.02:
            note = "low response"

        print("  {:>10.1f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>8.3f}  {}".format(
            tau_val, mean_dq, max_dq, std_dq, mean_qerr, kp_term, note))

        results.append((tau_val, mean_dq, max_dq, std_dq, mean_qerr, kp_term))

    # final inter-level settle back to zero tau
    t0 = time.time()
    while time.time() - t0 < INTER_LEVEL_SETTLE:
        send_tick(udp, cmd, state, target_q_isaac, tau_zero)
        time.sleep(0.002)

    print("  [TAU SWEEP DONE]")
    return results

# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------
udp  = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd  = sdk.LowCmd()
state= sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*80)
print("GO1 -- TORQUE FEEDFORWARD + DQ TEST")
print("tau_out = Kp*(q_target-q) + Kd*(-dq) + tau_ff")
print("Kp={}  Kd={}  tau_ff levels={}".format(KP, KD, TAU_FF_LEVELS))
print("TAU_HOLD_S={:.1f}  TAU_SETTLE_S={:.1f}  INTER_LEVEL_SETTLE={:.1f}".format(
    TAU_HOLD_S, TAU_SETTLE_S, INTER_LEVEL_SETTLE))
print("Active joint = joint that CHANGED vs previous phase")
print("Starting in 10 seconds...")
print("="*80 + "\n")
time.sleep(10)

# ---------------------------------------------------------------------------
# LOOP STATE
# ---------------------------------------------------------------------------
current_phase    = 0
phase_start_time = time.time()
cycle_count      = 0
current_kp       = KP

tau_test_done    = False
tau_results      = []

phase_real_list  = []
phase_error_list = []
phase_qd_list    = []
cycle_abs_errors = [[] for _ in range(12)]

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
while True:
    time.sleep(0.002)

    try:
        udp.Recv()
        udp.GetRecv(state)
    except Exception as e:
        print("UDP RECV ERROR: {}".format(e), flush=True)
        break

    # read state -- SDK->Isaac boundary
    q_sdk  = np.array([state.motorState[i].q  for i in range(12)])
    qd_sdk = np.array([state.motorState[i].dq for i in range(12)])
    real_q  = q_sdk[sdk_to_isaac]
    real_dq = qd_sdk[sdk_to_isaac]

    delta_q        = phases[current_phase]
    target_q_isaac = crouch_q + delta_q
    tau_ff_isaac   = np.zeros(12)

    time_in_phase  = time.time() - phase_start_time

    # active joint = the one that CHANGED vs previous phase
    act_j          = changed_joint(current_phase)
    q_err_active   = abs(target_q_isaac[act_j] - real_q[act_j])
    joint_settled  = q_err_active < NEAR_THRESH

    # accumulate position data after settle
    if time_in_phase >= SETTLE_SECONDS:
        phase_real_list.append(real_q.copy())
        phase_error_list.append((target_q_isaac - real_q).copy())
        phase_qd_list.append(real_dq.copy())

    # tau sweep: once per phase, after joint settles, skip crouch phase
    phase_has_motion = np.any(np.abs(delta_q) > 0.01)
    if (joint_settled and not tau_test_done
            and time_in_phase >= SETTLE_SECONDS and phase_has_motion):
        tau_test_done = True
        # blocking call -- runs entire sweep then returns
        tau_results = run_tau_sweep(
            udp, cmd, state, target_q_isaac, act_j, phase_names[current_phase])

    # normal position hold tick (tau_ff=0 outside sweep)
    send_tick(udp, cmd, state, target_q_isaac, tau_ff_isaac)

    # phase end
    if time_in_phase >= HOLD_SECONDS_PER_PHASE:

        if len(phase_real_list) > 0:
            avg_real  = np.mean(phase_real_list,  axis=0)
            avg_error = np.mean(phase_error_list, axis=0)
            avg_qd    = np.mean(np.abs(np.array(phase_qd_list)), axis=0)

            print("---------------------------------------------------------------------------------")
            print("Kp={:.0f} | Phase {:2d}/{} | {}".format(
                current_kp, current_phase, len(phases)-1, phase_names[current_phase]))
            print("Target  (Isaac): " + ", ".join("{:+.3f}".format(x) for x in target_q_isaac))
            print("AvgReal (Isaac): " + ", ".join("{:+.3f}".format(x) for x in avg_real))
            print("AvgErr  (Isaac): " + ", ".join("{:+.3f}".format(x) for x in avg_error))
            print("Avg|dq| (Isaac): " + ", ".join("{:+.4f}".format(x) for x in avg_qd))
            print("({} samples, settled {:.1f}s)".format(
                len(phase_real_list), HOLD_SECONDS_PER_PHASE - SETTLE_SECONDS))

            print("  {:>12}  {:>20}  {:>9}  {:>9}  {:>9}".format(
                "Joint","tau_result","avg_q","avg_err","avg|dq|"))
            for j in range(12):
                if j == act_j and tau_results:
                    best = max(tau_results, key=lambda r: abs(r[0]))
                    tau_str = "tau={:+.1f}->dq={:+.4f}".format(best[0], best[1])
                else:
                    tau_str = "--"
                print("  {:>12}  {:>20}  {:>9.4f}  {:>9.4f}  {:>9.4f}".format(
                    joint_names[j], tau_str,
                    float(avg_real[j]), float(avg_error[j]), float(avg_qd[j])))
                cycle_abs_errors[j].append(abs(float(avg_error[j])))

            print("---------------------------------------------------------------------------------")

        phase_real_list  = []
        phase_error_list = []
        phase_qd_list    = []
        tau_test_done    = False
        tau_results      = []
        current_phase   += 1
        phase_start_time = time.time()

        if current_phase >= len(phases):
            current_phase = 0
            cycle_count  += 1

            print("\n" + "="*80)
            print("CYCLE {} SUMMARY | Kp={}".format(cycle_count, current_kp))
            print("  {:>2}  {:>12}  {:>10}  {:>10}  {:>10}  Notes".format(
                "j","Joint","Min|err|","Avg|err|","Max|err|"))
            print("-"*60)
            for j in range(12):
                errs = cycle_abs_errors[j]
                mn=float(np.min(errs)); av=float(np.mean(errs)); mx=float(np.max(errs))
                note = "HIGH" if av > 0.30 else "moderate" if av > 0.15 else ""
                print("  {:>2}  {:>12}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {}".format(
                    j, joint_names[j], mn, av, mx, note))
            print("="*80 + "\n")
            cycle_abs_errors = [[] for _ in range(12)]

            current_kp = min(KP + KP_STEP * cycle_count, KP + KP_STEP * RAMP_MAX_LEVEL)
            print("=== New cycle | Kp={:.1f} ===\n".format(current_kp))

        if current_phase < len(phases):
            print("-> Next: {}".format(phase_names[current_phase]))

    try:
        udp.SetSend(cmd)
        udp.Send()
    except Exception as e:
        print("UDP SEND ERROR: {}".format(e), flush=True)
        break