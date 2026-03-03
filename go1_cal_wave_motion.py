# -*- coding: utf-8 -*-
"""
GO1 WAVE - 9 SNAPSHOT DEBUG PER CYCLE
Kp starts at 10, increases by 10 each full cycle: cycle1=10, cycle2=20, cycle3=30, cycle4=40 (fixed)

9 snapshots per cycle triggered by wave position:
  snap 1: center (start, wave=0 rising)
  snap 2: 45deg  (wave=+0.5*AMP, going fwd)
  snap 3: fwd peak (wave=+AMP)
  snap 4: 135deg (wave=+0.5*AMP, coming back)
  snap 5: center (wave=0 falling)
  snap 6: 225deg (wave=-0.5*AMP, going bwd)
  snap 7: bwd peak (wave=-AMP)
  snap 8: 315deg (wave=-0.5*AMP, returning)
  snap 9: center (wave=0 rising = cycle complete, Kp bumps here)

Control law unchanged: tau_out = Kp*(q_target-q) + Kd*(-dq) + tau_ff
"""

import time
import numpy as np
import robot_interface as sdk
import math

# =============================================================================
# CONFIG
# =============================================================================
WAVE_AMP   = 0.18
WAVE_FREQ  = 0.25
WAVE_T     = 1.0 / WAVE_FREQ

KP_START   = 10.0
KP_STEP    = 10.0
KP_MAX     = 50.0
KD         = 3.5

# 9 snap points as fraction of wave period (0..1)
# wave(t) = AMP * sin(2*pi*t/T)
# sin=0 rising at phase=0, peak at 0.25, sin=0 falling at 0.5, trough at 0.75
SNAP_FRACS = [0.0,   0.125, 0.25,  0.375,
              0.5,   0.625, 0.75,  0.875,
              1.0]
SNAP_LABELS = [
    "1: center (start)",
    "2: center->peak (+0.5A)",
    "3: fwd peak (+A)",
    "4: peak->center (+0.5A)",
    "5: center (mid)",
    "6: center->trough (-0.5A)",
    "7: bwd peak (-A)",
    "8: trough->center (-0.5A)",
    "9: center (cycle end)",
]

# window around each snap point to average (seconds)
SNAP_WINDOW_S = 0.04   # ~20 ticks either side

KI_TORQUE           = 1.2
MAX_INTEGRAL_TORQUE = 8.0
ERROR_THRESHOLD     = 0.030
RAMP1_TICKS         = 2000   # 4s initial ramp to stand at Kp=10

# =============================================================================
# JOINT MAPPING
# =============================================================================
d = {'FR0':0,'FR1':1,'FR2':2,'FL0':3,'FL1':4,'FL2':5,
     'RR0':6,'RR1':7,'RR2':8,'RL0':9,'RL1':10,'RL2':11}

stand = {
    "FR0": 0.00, "FR1": 0.90, "FR2": -1.80,
    "FL0": 0.00, "FL1": 0.90, "FL2": -1.80,
    "RR0": 0.00, "RR1": 0.95, "RR2": -1.85,
    "RL0": 0.00, "RL1": 0.95, "RL2": -1.85,
}
default_joint_pos = np.array([stand[k] for k in sorted(d.keys())])

sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_names  = ["FL_hip","FR_hip","RL_hip","RR_hip",
                "FL_th", "FR_th", "RL_th", "RR_th",
                "FL_cal","FR_cal","RL_cal","RR_cal"]

# =============================================================================
# SENSOR STATE
# =============================================================================
contact_state  = np.ones(4)
force_filter   = None
gravity_filter = None

def update_sensors(state_obj):
    global contact_state, force_filter, gravity_filter
    raw = np.array(state_obj.footForce, dtype=float)
    if force_filter is None:
        force_filter = raw.copy()
    force_filter = 0.95*force_filter + 0.05*raw
    avg   = float(np.mean(force_filter))
    off_t = 0.35*avg;  on_t = 0.65*avg
    prev  = contact_state.copy()
    nc    = np.where(force_filter > on_t, 1.0, 0.0)
    nc[( prev==1.0)&(force_filter<off_t)] = 0.0
    nc    = np.where(prev==0, np.where(force_filter>on_t,1.0,0.0), nc)
    contact_state[:] = nc

    acc = np.array(state_obj.imu.accelerometer)
    n   = float(np.linalg.norm(acc))
    gr  = -acc/n if n>0.1 else np.array([0.,0.,-1.])
    if gravity_filter is None:
        gravity_filter = gr.copy()
    gravity_filter = 0.95*gravity_filter + 0.05*gr
    return contact_state.copy(), avg, off_t, on_t, gravity_filter.copy()

# =============================================================================
# PRINT SNAPSHOT
# =============================================================================
def print_snap(snap_idx, cycle, wave_val, Kp, t,
               q_buf, qd_buf, target_sdk,
               foot_c, avg_f, off_t, on_t, grav, gyro, rpy):

    avg_q_sdk  = np.mean(q_buf,  axis=0)
    avg_qd_sdk = np.mean(np.abs(np.array(qd_buf)), axis=0)
    tgt_isaac  = target_sdk[sdk_to_isaac]
    avg_q_i    = avg_q_sdk[sdk_to_isaac]
    avg_qd_i   = avg_qd_sdk[sdk_to_isaac]
    err_i      = tgt_isaac - avg_q_i

    rp = math.sqrt(float(grav[0])**2 + float(grav[1])**2)
    fp = (force_filter[0]+force_filter[1])/np.sum(force_filter)*100 \
         if force_filter is not None and np.sum(force_filter)>0 else 0.0

    print("\n" + "="*88)
    print("CYCLE {:2d} | Snap {:d}/9: {}".format(cycle, snap_idx+1, SNAP_LABELS[snap_idx]))
    print("t={:.2f}s  wave={:+.4f}rad  Kp={:.0f}  Kd={:.1f}".format(t, wave_val, Kp, KD))
    print("Contacts:{}  FR:{:.0f} FL:{:.0f} RR:{:.0f} RL:{:.0f}  Front%:{:.1f}%  avg={:.0f}N".format(
        foot_c,
        force_filter[0] if force_filter is not None else 0,
        force_filter[1] if force_filter is not None else 0,
        force_filter[2] if force_filter is not None else 0,
        force_filter[3] if force_filter is not None else 0,
        fp, avg_f))
    print("Thresh OFF<{:.0f}N  ON>{:.0f}N".format(off_t, on_t))
    print("Gravity X:{:+.4f} Y:{:+.4f} Z:{:+.4f}  |tilt|={:.4f}".format(
        float(grav[0]),float(grav[1]),float(grav[2]),rp))
    print("AngVel  r:{:+.3f} p:{:+.3f} y:{:+.3f}".format(
        float(gyro[0]),float(gyro[1]),float(gyro[2])))
    print("IMU rpy roll:{:+.3f} pitch:{:+.3f} yaw:{:+.3f}".format(
        float(rpy[0]),float(rpy[1]),float(rpy[2])))
    print("")
    print("  {:>12}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "Joint","target","avg_q","avg_err","avg|dq|"))
    print("  " + "-"*52)
    for j in range(12):
        print("  {:>12}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}".format(
            isaac_names[j],
            float(tgt_isaac[j]), float(avg_q_i[j]),
            float(err_i[j]),     float(avg_qd_i[j])))
    print("AvgErr  (Isaac): " + " ".join("{:+.4f}".format(float(x)) for x in err_i))
    print("Avg|dq| (Isaac): " + " ".join("{:+.4f}".format(float(x)) for x in avg_qd_i))
    print("="*88)

# =============================================================================
# INIT
# =============================================================================
udp  = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd  = sdk.LowCmd()
state= sdk.LowState()
udp.InitCmdData(cmd)

integral_torque = [0.0]*12
phase2_start    = None
wave_start_time = None
t0              = time.time()
motiontime      = 0

# wave cycle tracking
cycle           = 0         # completed cycles
current_snap    = 0         # which of the 9 snaps we're waiting for
snap_buf_q      = []        # accumulate q in snap window
snap_buf_qd     = []
in_snap_window  = False
snap_fired      = [False]*9 # which snaps have fired this cycle
Kp_wave         = KP_START  # current wave-phase Kp

print("\n" + "="*88)
print("GO1 WAVE - 9 SNAPSHOT DEBUG PER CYCLE")
print("Kp: starts={:.0f}, +{:.0f} each cycle, max={:.0f}".format(
    KP_START, KP_STEP, KP_MAX))
print("Wave amp={} rad  freq={} Hz  period={:.3f}s".format(WAVE_AMP, WAVE_FREQ, WAVE_T))
print("9 snaps: center | +45 | +peak | +45 | center | -45 | -peak | -45 | center")
print("Starting in 8 seconds...")
print("="*88)
time.sleep(8)

# =============================================================================
# MAIN LOOP
# =============================================================================
while True:
    time.sleep(0.002)
    motiontime += 1
    t = time.time() - t0

    udp.Recv()
    udp.GetRecv(state)

    foot_c, avg_f, off_t, on_t, grav = update_sensors(state)
    gyro = np.array(state.imu.gyroscope)
    rpy  = state.imu.rpy

    # ---- Kp/Kd: ramp1 to stand, then wave phase uses Kp_wave --------------
    if motiontime < RAMP1_TICKS:
        ramp  = motiontime / float(RAMP1_TICKS)
        Kp    = 5.0 + 5.0 * ramp     # 5 -> 10
        Kd_   = 0.8 + 1.2 * ramp     # 0.8 -> 2.0
        use_integral = False
        wave_offset  = 0.0
    else:
        if phase2_start is None:
            phase2_start    = motiontime
            wave_start_time = t
            print("\n*** STAND REACHED  t={:.1f}s  Kp={:.0f} -> wave starts ***\n".format(
                t, KP_START))
        Kp   = Kp_wave
        Kd_  = KD
        use_integral = True
        wave_t      = t - wave_start_time
        wave_offset = WAVE_AMP * math.sin(2 * math.pi * WAVE_FREQ * wave_t)

    # ---- Build target + control (wave code unchanged) ---------------------
    target_sdk = np.zeros(12)
    for i in range(12):
        name   = list(d.keys())[i]
        target = stand[name] + (wave_offset if i in [1,4,7,10] else 0.0)
        target_sdk[i] = target
        error  = target - state.motorState[i].q
        q_cmd  = state.motorState[i].q + (1.0 if motiontime>=RAMP1_TICKS else
                 motiontime/float(RAMP1_TICKS)) * error

        tau_integral = 0.0
        if use_integral:
            if abs(error) > ERROR_THRESHOLD:
                integral_torque[i] += KI_TORQUE * error * 0.002
            else:
                integral_torque[i] *= 0.98
            integral_torque[i] = float(np.clip(
                integral_torque[i], -MAX_INTEGRAL_TORQUE, MAX_INTEGRAL_TORQUE))
            tau_integral = integral_torque[i]

        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = q_cmd
        cmd.motorCmd[i].dq   = 0
        cmd.motorCmd[i].Kp   = Kp
        cmd.motorCmd[i].Kd   = Kd_
        cmd.motorCmd[i].tau  = (7.0 + tau_integral) if i in [2,5,8,11] else tau_integral

    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()

    # ---- Snapshot detection (only after wave starts, only at Kp>=10) ------
    if wave_start_time is not None:
        wave_t   = t - wave_start_time
        cycle_t  = math.fmod(wave_t, WAVE_T)        # position in current cycle 0..T
        cycle_n  = int(wave_t / WAVE_T)             # which cycle we're in

        # detect new cycle
        if cycle_n > cycle:
            # cycle just completed -- bump Kp
            cycle = cycle_n
            Kp_wave = min(KP_START + KP_STEP * cycle, KP_MAX)
            snap_fired = [False]*9
            print("\n--- Cycle {} complete -> Kp={:.0f} ---".format(cycle, Kp_wave))

        # check each snap point
        q_now  = np.array([state.motorState[i].q  for i in range(12)])
        qd_now = np.array([state.motorState[i].dq for i in range(12)])

        for si, frac in enumerate(SNAP_FRACS):
            if snap_fired[si]:
                continue
            snap_t    = frac * WAVE_T             # time in cycle when snap should fire
            half_win  = SNAP_WINDOW_S / 2.0
            in_window = abs(cycle_t - snap_t) < half_win

            # handle wrap-around for snap 9 (frac=1.0) near cycle boundary
            if frac == 1.0:
                in_window = (cycle_t > WAVE_T - half_win) or (cycle_t < half_win)

            if in_window:
                snap_buf_q.append(q_now.copy())
                snap_buf_qd.append(qd_now.copy())
            elif len(snap_buf_q) > 0 and not snap_fired[si]:
                # window just closed -- print snapshot
                print_snap(si, cycle+1, wave_offset, Kp, t,
                           snap_buf_q, snap_buf_qd, target_sdk,
                           foot_c, avg_f, off_t, on_t, grav, gyro, rpy)
                snap_fired[si] = True
                snap_buf_q  = []
                snap_buf_qd = []
                break   # one snap per tick