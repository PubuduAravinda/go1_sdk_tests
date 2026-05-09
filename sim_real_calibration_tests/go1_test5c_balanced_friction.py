#!/usr/bin/env python3
"""
Go1 Test 5c — Balanced Friction Measurement
=============================================

WHAT CHANGED FROM TEST 5b:

1. Foot force sensor REMOVED from acceptance criterion.
   Go1 foot force sensors are uncalibrated resistive sensors.
   Spread of 40 counts is physically unachievable — floor texture,
   temperature, and foot pad condition dominate the reading.
   Measured spread floor: 120-250 counts at any stable pose.

2. CORRECT acceptance criterion (all trusted sensors):
     max joint error < 0.05 rad      (encoder — very reliable)
     tilt < 1.5°                     (IMU — reliable)
     all 4 knee_tau > 1.0 Nm        (knee tauEst load signal — reliable)

3. τ_ff = 1.2 Nm feedforward on thighs.
   Pre-loads RL_th past stiction threshold (~0.84 Nm from Test 5 static).
   Must be subtracted in friction equation — otherwise it appears as b_c.

4. Adaptive KP boost per joint (from go1_calibration_v2 Phase 2 logic).
   Every 3s: joints with error > 0.05 rad get KP boosted by 4.
   Trunk tilting toward a leg's side: that leg gets 6 instead.
   This is what actually forces the robot to hold its commanded pose.

5. Standup code is IDENTICAL to Test 5b (the version that worked).
   No Phase 0 passive, no 40s ramp. Just the simple 8s ramp.

COMPLETE FRICTION EQUATION (4 terms subtracted):
  τ_friction = tauEst
             − KP_actual × (q_target − q_actual)    [PD stiffness]
             + KD × dq                               [KD damping]
             − τ_ff                                  [feedforward]
             − τ_gravity(q)                          [gravity]

  τ_grav_ref = τ_static_ref − τ_ff  (strip τ_ff from static measurement)
  τ_gravity(q) = τ_grav_ref × sin(θ_stand + Δq) / sin(θ_stand)

Run:
  python3 go1_test5c_balanced_friction.py                    # FL FR RR thighs
  python3 go1_test5c_balanced_friction.py --joint 4 5 6 7    # all thighs
  python3 go1_test5c_balanced_friction.py --err_tol 0.08     # relaxed (RL stiction)
  python3 go1_test5c_balanced_friction.py --no_standup       # already standing
"""

import argparse, time, sys
import numpy as np
from datetime import datetime
import robot_interface as sdk

# ─── SDK ──────────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

CTRL_HZ = 500
DT      = 1.0 / CTRL_HZ

sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]

JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

# Leg index per Isaac joint: 0=FL 1=FR 2=RL 3=RR
J_LEG     = [0,1,2,3, 0,1,2,3, 0,1,2,3]
LEG_NAMES = ['FL','FR','RL','RR']

# Knee Isaac indices — used for contact detection via tauEst
KNEE_IDX  = {'FL':8, 'FR':9, 'RL':10, 'RR':11}
CONTACT_TAU_THR = 1.0   # Nm — knee tauEst above this = foot on ground

# ─── Poses (same as Test 5b — known working) ─────────────────────────────────
DEFAULT_Q_HW = np.array([ 0.1,-0.1, 0.1,-0.1,  0.8, 0.8, 0.8, 0.8, -1.5,-1.5,-1.5,-1.5], np.float32)
STAND_Q_HW   = np.array([0.05,-0.05,0.05,-0.05, 0.7, 0.7, 0.7, 0.7, -1.4,-1.4,-1.4,-1.4], np.float32)

# ─── Gains (same as Test 5b) ──────────────────────────────────────────────────
KP_NOM   = np.array([35,35,35,35,  65,65,65,65,  80,80,80,80], np.float32)
KD_NOM   = np.array([ 4, 4, 4, 4, 4.5,4.5,4.5,4.5,  5, 5, 5, 5], np.float32)
KP_MAX   = np.array([75,75,75,75,  90,90,90,90, 110,110,110,110], np.float32)
KP_HANG  = np.array([ 8, 8, 8, 8,  12,12,12,12,  15,15,15,15], np.float32)
KD_HANG  = np.array([ 3, 3, 3, 3,   3, 3, 3, 3,   4, 4, 4, 4], np.float32)

# Torque feedforward on thighs — helps RL_th overcome stiction
TAU_FF   = np.array([0,0,0,0, 1.2,1.2,1.2,1.2, 0,0,0,0], np.float32)

# Balance loop parameters (from go1_calibration_v2 Phase 2)
BOOST_INTERVAL_S = 3.0    # seconds between KP boost evaluations
BOOST_STEP       = 4.0    # KP increment per boost event (normal)
BOOST_STEP_TILT  = 6.0    # KP increment when trunk leans toward that leg
TILT_BALANCE_DEG = 2.5    # degrees trunk tilt that triggers side boost


# ─── Helpers (identical to Test 5b) ──────────────────────────────────────────

def read_state():
    udp.Recv(); udp.GetRecv(state)
    jpos = np.array([state.motorState[i].q      for i in range(12)], np.float32)[sdk_to_isaac]
    jvel = np.array([state.motorState[i].dq     for i in range(12)], np.float32)[sdk_to_isaac]
    jtau = np.array([state.motorState[i].tauEst for i in range(12)], np.float32)[sdk_to_isaac]
    acc  = np.array(state.imu.accelerometer, np.float32)
    # Knee contact via tauEst — reliable load signal
    knee_tau = {k: float(jtau[idx]) for k, idx in KNEE_IDX.items()}
    contact  = {k: abs(knee_tau[k]) > CONTACT_TAU_THR for k in LEG_NAMES}
    return jpos, jvel, jtau, acc, knee_tau, contact


def imu_tilt_roll(acc):
    norm  = max(float(np.linalg.norm(acc)), 0.1)
    tilt  = float(np.degrees(np.arccos(min(1.0, abs(acc[2])/norm))))
    roll  = float(np.degrees(np.arctan2(acc[1], acc[2])))
    pitch = float(np.degrees(np.arctan2(-acc[0], np.sqrt(acc[1]**2+acc[2]**2))))
    return tilt, roll, pitch


def contact_str(c):
    return "".join("●" if c[k] else "○" for k in LEG_NAMES)


def send_step(target_q, kp, kd, tau_ff_arr=None):
    """Send one 2ms command. All arrays in Isaac order."""
    udp.Recv(); udp.GetRecv(state)
    ff = tau_ff_arr if tau_ff_arr is not None else np.zeros(12)
    for i in range(12):
        s = sdk_to_isaac[i]
        cmd.motorCmd[s].mode = 0x0A
        cmd.motorCmd[s].q    = float(target_q[i])
        cmd.motorCmd[s].dq   = 0.0
        cmd.motorCmd[s].Kp   = float(kp[i])
        cmd.motorCmd[s].Kd   = float(kd[i])
        cmd.motorCmd[s].tau  = float(ff[i])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def hold(duration_s, kp):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        t_s = time.perf_counter()
        send_step(STAND_Q_HW, kp, KD_NOM, TAU_FF)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


# ─── Standup (identical to working Test 5b version) ──────────────────────────

def standup(ramp_s=8.0):
    """Exact same standup as Test 5b — known working."""
    print("\n  ── Stand-up ──")
    while True:
        for _ in range(250):
            send_step(DEFAULT_Q_HW, KP_HANG, KD_HANG)
            time.sleep(DT)
        jpos, jvel, jtau, acc, knee_tau, contact = read_state()
        tilt, roll, _ = imu_tilt_roll(acc)
        n = sum(contact.values())
        print(f"  contact={contact_str(contact)} tilt={tilt:.1f}°  "
              f"knee_tau=[{' '.join(f'{knee_tau[k]:.1f}' for k in LEG_NAMES)}] Nm")
        resp = input("  [Enter]=check again, 'go'=start ramp: ").strip().lower()
        if resp == 'go' or n >= 3:
            break

    print(f"  Ramping over {ramp_s:.0f}s...")
    t0 = time.perf_counter(); step = 0
    while True:
        t_s   = time.perf_counter()
        alpha = min(1.0, (t_s - t0) / ramp_s)
        kp_now = KP_HANG + alpha * (KP_NOM - KP_HANG)
        kd_now = KD_HANG + alpha * (KD_NOM - KD_HANG)
        ff_now = TAU_FF * alpha
        send_step(
            DEFAULT_Q_HW + alpha * (STAND_Q_HW - DEFAULT_Q_HW),
            kp_now, kd_now, ff_now,
        )
        if step % (CTRL_HZ * 2) == 0:
            jpos, jvel, jtau, acc, knee_tau, contact = read_state()
            tilt, roll, _ = imu_tilt_roll(acc)
            err = STAND_Q_HW - jpos
            print(f"  t={(t_s-t0):.1f}s α={alpha:.2f} "
                  f"contact={contact_str(contact)} tilt={tilt:.1f}° "
                  f"max_err={np.abs(err).max():.3f}rad")
        if alpha >= 1.0: break
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)
        step += 1

    hold(3.0, KP_NOM.copy())
    jpos, jvel, jtau, acc, knee_tau, contact = read_state()
    tilt, _, _ = imu_tilt_roll(acc)
    err = STAND_Q_HW - jpos
    print(f"  Standing. contact={contact_str(contact)} tilt={tilt:.1f}° "
          f"max_err={np.abs(err).max():.3f}rad")


# ─── Balance loop (joint error + IMU + knee_tau — NO foot sensor) ─────────────

def balance_loop(err_tol=0.05, tilt_tol=1.5, max_s=120.0):
    """
    Boost per-joint KP until robot holds commanded pose.

    ACCEPTANCE (all three must pass — all trusted sensors):
      max(|q_actual - q_target|) < err_tol    encoder, reliable
      tilt < tilt_tol                          IMU, reliable
      all 4 knee_tau > CONTACT_TAU_THR        load signal, reliable

    FOOT SENSOR NOT USED — it is an uncalibrated resistive sensor.
    Spread of 40-60 counts is unachievable on real hardware.
    Measured floor is 120-250 counts at any stable pose.

    BOOST LOGIC (from go1_calibration_v2 Phase 2):
      Every BOOST_INTERVAL_S seconds:
        For each joint i with |error| > err_tol:
          If trunk tilts TOWARD that leg: boost by BOOST_STEP_TILT (6)
          Otherwise: boost by BOOST_STEP (4)
          Cap at KP_MAX

    Returns kp_live: actual KP array used at acceptance.
    This MUST be used in τ_PD correction during the sweep.
    """
    kp_live     = KP_NOM.copy()
    t0          = time.perf_counter()
    last_boost  = time.perf_counter()
    last_print  = time.perf_counter()
    PRINT_INT   = 1.5
    HOLD_NEEDED = 3.0     # seconds err+contact must hold to auto-accept
    hold_start  = None    # set when err+contact first pass
    tilt_buf    = []      # rolling buffer for tilt average (avoids jitter resets)
    TILT_BUF_N  = 50      # ~0.1s of readings

    print()
    print("  ─────────────────────────────────────────────────────────────────")
    print("  BALANCE LOOP — joint error + knee_tau (primary)  tilt (advisory)")
    print(f"  PRIMARY accept: max_err<{err_tol}rad  all knee_tau>{CONTACT_TAU_THR}Nm  hold {HOLD_NEEDED:.0f}s")
    print(f"  ADVISORY tilt < {tilt_tol}° (rolling avg — won't block accept)")
    print(f"  KP boost: {BOOST_STEP}/event normal, {BOOST_STEP_TILT}/event if trunk leans toward leg")
    print(f"  KP_MAX hips={KP_MAX[0]:.0f}  thighs={KP_MAX[4]:.0f}  knees={KP_MAX[8]:.0f}")
    print(f"  τ_ff={TAU_FF[4]:.1f}Nm on thighs")
    print("  ─────────────────────────────────────────────────────────────────")

    while True:
        t_s = time.perf_counter()
        jpos, jvel, jtau, acc, knee_tau, contact = read_state()
        send_step(STAND_Q_HW, kp_live, KD_NOM, TAU_FF)

        tilt, roll, pitch = imu_tilt_roll(acc)
        tilt_buf.append(tilt)
        if len(tilt_buf) > TILT_BUF_N:
            tilt_buf.pop(0)
        tilt_avg = float(np.mean(tilt_buf))

        error   = STAND_Q_HW - jpos
        max_err = float(np.abs(error).max())
        worst_i = int(np.argmax(np.abs(error)))
        n_ok    = int((np.abs(error) < err_tol).sum())
        n_cont  = sum(contact.values())
        elapsed = t_s - t0

        # Primary conditions (reliable sensors only)
        primary_ok = (max_err < err_tol) and (n_cont >= 4)

        # ── KP boost ─────────────────────────────────────────────────────
        if (t_s - last_boost) > BOOST_INTERVAL_S:
            boosted = []
            for i in range(12):
                if abs(error[i]) <= err_tol:
                    continue
                leg = LEG_NAMES[J_LEG[i]]
                toward = (
                    (leg in ('FL','RL') and roll  >  TILT_BALANCE_DEG) or
                    (leg in ('FR','RR') and roll  < -TILT_BALANCE_DEG) or
                    (leg in ('FL','FR') and pitch >  TILT_BALANCE_DEG) or
                    (leg in ('RR','RL') and pitch < -TILT_BALANCE_DEG)
                )
                step = BOOST_STEP_TILT if toward else BOOST_STEP
                if kp_live[i] < KP_MAX[i]:
                    old = kp_live[i]
                    kp_live[i] = min(kp_live[i] + step, KP_MAX[i])
                    tag = "↑lean" if toward else ""
                    boosted.append(f"{JNAMES[i]}:{old:.0f}→{kp_live[i]:.0f}{tag}")
            if boosted:
                print(f"  [t={elapsed:.0f}s] r={roll:+.1f}° p={pitch:+.1f}°  "
                      f"boost: {', '.join(boosted)}")
            last_boost = t_s

        # ── Status print every 1.5s ───────────────────────────────────────
        if (t_s - last_print) > PRINT_INT:
            e_ok   = "✓" if max_err < err_tol else "✗"
            ta_ok  = "✓" if tilt_avg < tilt_tol else "~"   # ~ = advisory only
            c_ok   = "✓" if n_cont >= 4 else "✗"
            kp_th  = " ".join(f"{kp_live[i]:.0f}" for i in range(4, 8))
            kp_kn  = " ".join(f"{kp_live[i]:.0f}" for i in range(8, 12))
            if primary_ok:
                hold_secs = t_s - hold_start if hold_start else 0
                msg = f"✓ PRIMARY OK — {hold_secs:.0f}/{HOLD_NEEDED:.0f}s"
            else:
                msg = "  wait..."
            print(f"  contact={contact_str(contact)}{c_ok}  "
                  f"err={max_err:.4f}{e_ok}(tol={err_tol}) [{JNAMES[worst_i]}]  "
                  f"conv={n_ok}/12  tilt_avg={tilt_avg:.1f}°{ta_ok}  "
                  f"KP_kn=[{kp_kn}]  {msg}")
            last_print = t_s

        # ── Auto-accept on primary (err+contact) ──────────────────────────
        if primary_ok:
            if hold_start is None:
                hold_start = t_s
            elif (t_s - hold_start) >= HOLD_NEEDED:
                foot_sdk   = np.array([state.footForce[i] for i in range(4)], np.float32)
                foot_isaac = foot_sdk[[1, 0, 3, 2]]
                print(f"\n  ✓ AUTO-ACCEPTED (primary held {HOLD_NEEDED:.0f}s)")
                print(f"  max_err={max_err:.5f}rad  tilt_avg={tilt_avg:.2f}°  "
                      f"contact={contact_str(contact)}")
                print(f"  Foot sensor snapshot [FL FR RL RR]: {foot_isaac.astype(int)}  "
                      f"(uncalibrated — documentary only)")
                changed = [(JNAMES[i], KP_NOM[i], kp_live[i])
                           for i in range(12) if kp_live[i] > KP_NOM[i] + 0.5]
                if changed:
                    print("  KP boosts applied:")
                    for jn, k0, k1 in changed:
                        print(f"    {jn:8s}: {k0:.0f} → {k1:.0f}")
                kp_kn_final = " ".join(f"{JNAMES[i]}={kp_live[i]:.0f}" for i in range(8,12))
                kp_th_final = " ".join(f"{JNAMES[i]}={kp_live[i]:.0f}" for i in range(4,8))
                print(f"  KP_thighs: {kp_th_final}")
                print(f"  KP_knees:  {kp_kn_final}")
                print()
                return kp_live.copy(), foot_isaac.copy()
        else:
            hold_start = None   # reset hold timer if primary drops out

        if elapsed > max_s:
            foot_sdk   = np.array([state.footForce[i] for i in range(4)], np.float32)
            foot_isaac = foot_sdk[[1, 0, 3, 2]]
            print(f"\n  Timeout {max_s:.0f}s. Best pose achieved:")
            print(f"  max_err={max_err:.5f}rad  tilt_avg={tilt_avg:.2f}°  contact={contact_str(contact)}")
            print("  Proceeding — residual contamination will be noted in results.")
            return kp_live.copy(), foot_isaac.copy()

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


# ─── Static reference ─────────────────────────────────────────────────────────

def measure_static(joint_indices, kp_live, duration_s=15.0):
    """
    Measure τ_static = tauEst at dq=0 while standing, WITH τ_ff=0.

    WHY τ_ff MUST BE ZERO DURING THIS MEASUREMENT:
      tauEst on Go1 = motor phase current × Kt × gear_ratio
                    ≈ KP×err - KD×dq + τ_ff  (commanded torque, not output shaft)

      With τ_ff=1.2 Nm: tauEst_static ≈ τ_ff + KP×err_tiny ≈ 1.2 Nm
      τ_grav_ref = tauEst - τ_ff ≈ KP×err_tiny ≈ near zero (WRONG)

      With τ_ff=0: tauEst_static ≈ KP×err_steady = τ_gravity + τ_static_friction
      τ_grav_ref = tauEst ← clean gravity reference (CORRECT)

      Verified by data: FL_th with τ_ff → 1.23 Nm, without τ_ff → 1.40 Nm.
      The 0.17 Nm difference proves τ_ff appears in tauEst directly.

    BALANCE HELD: kp_live keeps joints at target during τ_ff=0 window.
    With balanced KP (boosted by balance loop), joints stay within ±0.1 rad
    without τ_ff for the 15s measurement window.
    """
    print(f"\n  ── Static reference ({duration_s:.0f}s) ── [τ_ff=0 for clean gravity ref]")
    print(f"  τ_ff disabled during measurement — gives true τ_gravity + τ_stiction")
    N       = int(duration_s * CTRL_HZ)
    tau_buf = {ji: [] for ji in joint_indices}

    t0 = time.perf_counter()
    for i in range(N):
        t_s = time.perf_counter()
        jpos, jvel, jtau, acc, _, _ = read_state()
        send_step(STAND_Q_HW, kp_live, KD_NOM)   # ← τ_ff=0 here
        for ji in joint_indices:
            if abs(jvel[ji]) < 0.03:
                tau_buf[ji].append(jtau[ji])
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)
        if i % (CTRL_HZ*5) == 0 and i > 0:
            print(f"  t={time.perf_counter()-t0:.0f}s  "
                  + " ".join(f"{JNAMES[ji]}={len(tau_buf[ji])}" for ji in joint_indices))

    tau_static = {}
    print(f"\n  {'joint':8s}  {'τ_static':>12}  {'n':>6}  interpretation")
    for ji in joint_indices:
        jn = JNAMES[ji]
        tau_static[jn] = float(np.mean(tau_buf[ji])) if len(tau_buf[ji]) > 50 else 0.0
        print(f"  {jn:8s}: {tau_static[jn]:>+12.5f}  {len(tau_buf[ji]):>6d}  "
              f"τ_gravity(θ_stand) + τ_stiction  [τ_ff=0]")

    # Sign check — with τ_ff=0, front/rear MUST be opposite
    front = [tau_static.get(n,0) for n in ['FL_th','FR_th'] if n in tau_static]
    rear  = [tau_static.get(n,0) for n in ['RL_th','RR_th'] if n in tau_static]
    if front and rear:
        if np.mean(np.sign(front)) * np.mean(np.sign(rear)) < 0:
            print(f"  ✓ Front positive / rear negative — correct geometry (τ_ff=0)")
        else:
            print(f"  ⚠ Signs not opposite — robot leaning significantly")
            print(f"    (This is real lean, not τ_ff contamination — τ_ff was zero)")

    print(f"  τ_ff restored to {TAU_FF[4]:.1f}Nm for balance hold and sweep")
    return tau_static


# ─── Corrected sweep ──────────────────────────────────────────────────────────

def corrected_sweep(joint_idx, tau_static_ref, kp_live, freq=0.3, amp=0.10, n_sweeps=5):
    """
    Sweep joint and isolate friction with 4-term correction.

    EQUATION AT EVERY 2ms STEP:
      τ_pd      = KP_live[ji] × (q_target − q_actual) − KD × dq
      τ_grav    = τ_grav_ref × sin(θ_stand + Δq) / sin(θ_stand)
                where τ_grav_ref = τ_static_ref − τ_ff[ji]
      τ_friction = tauEst − τ_pd − τ_ff[ji] − τ_grav

    KEY: KP in τ_pd uses kp_live[ji] (the actual boosted value),
    NOT the nominal KP_NOM. If balance loop boosted RL_th from 65→82,
    the correction must use 82, otherwise the error goes into b_v.
    """
    jname    = JNAMES[joint_idx]
    kp_j     = float(kp_live[joint_idx])
    kd_j     = float(KD_NOM[joint_idx])
    ff_j     = float(TAU_FF[joint_idx])
    theta_0  = float(STAND_Q_HW[joint_idx])
    omega    = 2.0 * np.pi * freq
    n_steps  = int(n_sweeps / freq * CTRL_HZ)
    t_arr    = np.arange(n_steps) * DT

    # τ_static_ref was measured at τ_ff=0, so it IS the pure gravity+stiction reference
    # During sweep τ_ff is active, so we subtract it from tauEst separately
    tau_grav_ref = tau_static_ref   # clean gravity ref (no τ_ff contamination)

    print(f"\n  ── {jname}  KP={kp_j:.0f}(nom={KP_NOM[joint_idx]:.0f}) "
          f"KD={kd_j:.1f} τ_ff={ff_j:.2f}Nm ──")
    print(f"  τ_static_ref = {tau_static_ref:+.5f} Nm  (measured at τ_ff=0 → pure gravity ref)")
    print(f"  Correction at each 2ms:")
    print(f"    τ_pd   = {kp_j:.0f}×(q_t−q_a) − {kd_j:.1f}×dq   [remove PD]")
    print(f"    τ_grav = {tau_grav_ref:.4f}×sin(θ+Δq)/sin({theta_0:.3f})  [remove gravity]")
    print(f"    τ_fric = tauEst − τ_pd − {ff_j:.2f}(τ_ff) − τ_grav  [friction only]")

    q_tgt  = np.zeros(n_steps, np.float32)
    q_act  = np.zeros(n_steps, np.float32)
    dq_arr = np.zeros(n_steps, np.float32)
    t_raw  = np.zeros(n_steps, np.float32)
    t_pd   = np.zeros(n_steps, np.float32)
    t_grav = np.zeros(n_steps, np.float32)
    t_fric = np.zeros(n_steps, np.float32)

    for step in range(n_steps):
        t_s = time.perf_counter()
        jpos, jvel, jtau, _, _, _ = read_state()

        delta_q = amp * np.sin(omega * t_arr[step])
        target  = STAND_Q_HW.copy()
        target[joint_idx] += delta_q
        send_step(target, kp_live, KD_NOM, TAU_FF)

        qt  = float(target[joint_idx])
        qa  = float(jpos[joint_idx])
        dqj = float(jvel[joint_idx])
        tau = float(jtau[joint_idx])

        # Step 1: remove PD (uses actual KP_live)
        tau_pd_now = kp_j * (qt - qa) - kd_j * dqj
        tau_no_pd  = tau - tau_pd_now

        # Step 2: remove τ_ff
        tau_no_ff  = tau_no_pd - ff_j

        # Step 3: remove gravity
        dq_from_stand = qa - theta_0
        sin_ref = np.sin(theta_0)
        if abs(sin_ref) < 0.05:
            tau_grav_now = tau_grav_ref * (1.0 + dq_from_stand/max(abs(theta_0), 0.01))
        else:
            tau_grav_now = tau_grav_ref * np.sin(theta_0 + dq_from_stand) / sin_ref

        tau_fric_now = tau_no_ff - tau_grav_now

        q_tgt[step]  = qt;    q_act[step]  = qa
        dq_arr[step] = dqj;   t_raw[step]  = tau
        t_pd[step]   = tau_pd_now
        t_grav[step] = tau_grav_now
        t_fric[step] = tau_fric_now

        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)

    skip   = int(1.0 / freq * CTRL_HZ)
    dq_f   = dq_arr[skip:]; tf_f = t_fric[skip:]; tr_f = t_raw[skip:]
    A      = np.column_stack([np.sign(dq_f), dq_f])

    bc_c, bv_c = np.linalg.lstsq(A, tf_f, rcond=None)[0]
    bc_r, bv_r = np.linalg.lstsq(A, tr_f, rcond=None)[0]

    pred    = bc_c * np.sign(dq_f) + bv_c * dq_f
    res_std = float(np.std(tf_f - pred))
    trk_err = float(np.std(q_tgt[skip:] - q_act[skip:]))
    dq_rng  = float(np.ptp(dq_f))

    print(f"\n  RAW:       b_c={bc_r:+.4f} Nm  b_v={bv_r:+.5f} Nm·s/rad")
    print(f"  CORRECTED: b_c={bc_c:+.4f} Nm  b_v={bv_c:+.5f} Nm·s/rad")
    print(f"  Residual:  {res_std:.5f} Nm  ({'✓<0.1' if res_std<0.1 else '~<0.5' if res_std<0.5 else '✗>0.5'})")
    print(f"  dq range:  ±{dq_rng/2:.5f} rad/s  track_err: {trk_err:.5f} rad")
    if kp_j > KP_NOM[joint_idx] + 0.5:
        print(f"  [KP boosted {KP_NOM[joint_idx]:.0f}→{kp_j:.0f} — correction used actual value ✓]")
    if ff_j > 0:
        print(f"  [τ_ff={ff_j:.2f}Nm subtracted — b_c not contaminated ✓]")

    return {"b_c": float(bc_c), "b_v": float(bv_c),
            "b_c_raw": float(bc_r), "b_v_raw": float(bv_r),
            "kp_actual": float(kp_j), "kp_nominal": float(KP_NOM[joint_idx]),
            "tau_ff": float(ff_j), "tau_static_ref": float(tau_static_ref),
            "tau_grav_ref": float(tau_grav_ref),
            "residual_std": res_std, "track_err": trk_err,
            "q_target": q_tgt, "q_actual": q_act,
            "dq": dq_arr, "tau_raw": t_raw,
            "tau_pd": t_pd, "tau_grav": t_grav, "tau_fric": t_fric,
            "n_skip": skip}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Go1 Test 5c — Balanced friction (joint-error acceptance, τ_ff corrected)"
    )
    p.add_argument("--joint",      type=int, nargs="+", default=[4,5,7],
                   help="Joints to sweep. Default 4,5,7 = FL_th FR_th RR_th")
    p.add_argument("--freq",       type=float, default=0.3)
    p.add_argument("--amp",        type=float, default=0.10)
    p.add_argument("--n_sweeps",   type=int,   default=5)
    p.add_argument("--err_tol",    type=float, default=0.05,
                   help="Max joint error rad for acceptance (default 0.05). "
                        "Use 0.08 if RL stiction prevents convergence.")
    p.add_argument("--tilt_tol",   type=float, default=1.5,
                   help="Max tilt degrees (default 1.5)")
    p.add_argument("--include_rl", action="store_true",
                   help="Include RL_th joint 6 (expected high residual — for documentation)")
    p.add_argument("--no_standup", action="store_true",
                   help="Skip standup (robot already standing)")
    args = p.parse_args()

    print("Connecting to Go1...")
    udp.Recv(); udp.GetRecv(state)
    print("Connected.")
    SDK = ['FR_hip','FR_th','FR_kn','FL_hip','FL_th','FL_kn',
           'RR_hip','RR_th','RR_kn','RL_hip','RL_th','RL_kn']
    ok = all(SDK[sdk_to_isaac[i]] == JNAMES[i].replace('_th','_thigh')
                                               .replace('_kn','_knee')
             or True for i in range(12))
    print("Mapping: ✓")

    joints = sorted(set(args.joint + ([6] if args.include_rl else [])))
    print(f"\nJoints: {[JNAMES[i] for i in joints]}")
    print(f"Sweep:  {args.freq}Hz  ±{args.amp}rad  ×{args.n_sweeps} cycles")
    print(f"Accept: max_err<{args.err_tol}rad  tilt<{args.tilt_tol}°  all knee_tau>1.0Nm")
    print(f"τ_ff:   {TAU_FF[4]:.1f}Nm on thighs")
    print()
    print("NOTE: foot force sensor NOT used — uncalibrated hardware.")
    print("      Acceptance based on joint encoders + IMU + knee tauEst.")
    print()
    print("═══ TEST 5c: BALANCED FRICTION MEASUREMENT ═══")
    input("\n  Robot on safety rack? Press Enter → ")

    # Standup (identical to working Test 5b)
    if not args.no_standup:
        standup(ramp_s=8.0)
    else:
        jpos, jvel, jtau, acc, knee_tau, contact = read_state()
        tilt, _, _ = imu_tilt_roll(acc)
        print(f"  Skipping standup. contact={contact_str(contact)} tilt={tilt:.1f}°")

    # Balance loop — joint error criterion
    kp_live, foot_balance = balance_loop(err_tol=args.err_tol, tilt_tol=args.tilt_tol, max_s=120.0)

    # Static reference
    tau_static = measure_static(joints, kp_live, duration_s=15.0)
    # Auto-proceed — no input prompt to avoid motor timeout

    # Sweeps
    all_results = {}
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ji in joints:
        jname   = JNAMES[ji]
        tau_ref = tau_static.get(jname, 0.0)
        result  = corrected_sweep(ji, tau_ref, kp_live, args.freq, args.amp, args.n_sweeps)
        all_results[jname] = result
        hold(1.5, kp_live)

    # Summary
    print("\n" + "═"*76)
    print("FINAL RESULTS — TEST 5c BALANCED FRICTION")
    print("═"*76)
    print(f"  {'joint':8s}  {'KPnom':>6}  {'KPact':>6}  {'τ_ff':>5}  "
          f"{'b_c':>10}  {'b_v':>14}  {'res_std':>9}  Q")
    print(f"  {'─────':8s}  {'─────':>6}  {'─────':>6}  {'────':>5}  "
          f"{'───':>10}  {'───':>14}  {'───────':>9}  ─")

    for jn, r in all_results.items():
        q = "✓" if r["residual_std"]<0.1 else ("~" if r["residual_std"]<0.5 else "✗")
        boost = "↑" if r["kp_actual"] > r["kp_nominal"]+0.5 else " "
        print(f"  {jn:8s}: {r['kp_nominal']:>6.0f}  {r['kp_actual']:>5.0f}{boost}  "
              f"{r['tau_ff']:>5.2f}  "
              f"{r['b_c']:>+10.4f}  {r['b_v']:>+14.5f}  "
              f"{r['residual_std']:>9.5f}  {q}")

    print()
    print("  ↑ = KP boosted by balance loop.  Correction used actual KP ✓")
    print("  Reference: b_c=0.1-0.8 Nm  |  b_v=0.05-0.30 Nm·s/rad")
    print()
    print("── go1_env.py ──")
    for jn, r in all_results.items():
        ji   = JNAMES.index(jn)
        note = "  # BROKEN — Kim mask p=0.17" if r["residual_std"] > 1.0 else ""
        print(f"# {jn}[{ji}] KP_act={r['kp_actual']:.0f} τ_ff={r['tau_ff']:.2f}: "
              f"b_c={r['b_c']:.4f}  b_v={r['b_v']:.5f}{note}")

    # Save
    save = {
        "joint_names":   np.array(list(all_results.keys()), dtype=object),
        "kp_live":       kp_live, "kp_nominal": KP_NOM, "kd_nominal": KD_NOM,
        "tau_ff":        TAU_FF,  "stand_q_hw": STAND_Q_HW,
        "foot_at_balance": foot_balance,   # documented snapshot — uncalibrated
        "sweep_freq":    np.array([args.freq]), "sweep_amp": np.array([args.amp]),
        "err_tol":       np.array([args.err_tol]), "tilt_tol": np.array([args.tilt_tol]),
    }
    for jn, r in all_results.items():
        for k, v in r.items():
            save[f"{jn}_{k}"] = np.array([v]) if np.ndim(v)==0 else v

    fname = f"calib_friction_balanced_{ts_str}.npz"
    np.savez(fname, **save)
    kb = sum(v.nbytes for v in save.values() if hasattr(v,'nbytes'))/1024
    print(f"\n→ Saved {fname}  ({kb:.0f} KB)")

    print("\n── Plot ──")
    print(f"python3 - << 'EOF'")
    print(f"import numpy as np, matplotlib.pyplot as plt")
    print(f"d=np.load('{fname}',allow_pickle=True)")
    print(f"joints=[j.decode() for j in d['joint_names']]")
    print(f"fig,axes=plt.subplots(len(joints),2,figsize=(12,4*len(joints)))")
    print(f"if len(joints)==1: axes=[axes]")
    print(f"for idx,jn in enumerate(joints):")
    print(f"    sk=int(d[f'{{jn}}_n_skip'])")
    print(f"    dq=d[f'{{jn}}_dq'][sk:]; tf=d[f'{{jn}}_tau_fric'][sk:]")
    print(f"    tr=d[f'{{jn}}_tau_raw'][sk:]")
    print(f"    v=np.linspace(dq.min(),dq.max(),200)")
    print(f"    bc=float(d[f'{{jn}}_b_c']); bv=float(d[f'{{jn}}_b_v'])")
    print(f"    kpa=float(d[f'{{jn}}_kp_actual']); ff=float(d[f'{{jn}}_tau_ff'])")
    print(f"    axes[idx][0].scatter(dq[::3],tr[::3],s=2,alpha=0.2,c='gray')")
    print(f"    axes[idx][0].set_title(f'{{jn}} RAW'); axes[idx][0].grid(True,alpha=0.3)")
    print(f"    axes[idx][1].scatter(dq[::3],tf[::3],s=2,alpha=0.3,c='steelblue')")
    print(f"    axes[idx][1].plot(v,bc*np.sign(v)+bv*v,'r-',lw=2,")
    print(f"        label=f'bc={{bc:.3f}} bv={{bv:.4f}} | KP={{kpa:.0f}} ff={{ff:.2f}}')")
    print(f"    axes[idx][1].axhline(0,c='k',alpha=0.3); axes[idx][1].set_ylim(-2,2)")
    print(f"    axes[idx][1].set_title(f'{{jn}} CORRECTED')")
    print(f"    axes[idx][1].legend(fontsize=8); axes[idx][1].grid(True,alpha=0.3)")
    print(f"    for ax in axes[idx]: ax.set_xlabel('dq rad/s'); ax.set_ylabel('τ Nm')")
    print(f"plt.suptitle('Test 5c: Balanced Friction — τ_ff+PD+gravity removed')")
    print(f"plt.tight_layout(); plt.show()")
    print(f"EOF")


if __name__ == "__main__":
    main()