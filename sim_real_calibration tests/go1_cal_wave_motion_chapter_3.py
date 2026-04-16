#!/usr/bin/env python3
"""
GO1 WAVE v3 — 3-POINT TRACKING MONITOR
=======================================
Prints ONLY at three diagnostic snap points per cycle:
  Snap 3: forward peak     (wave = +A,   thigh most flexed forward)
  Snap 5: centre falling   (wave =  0,   max velocity, mid-swing)
  Snap 7: backward peak    (wave = -A,   thigh most extended back)

These three points capture:
  - Peak position tracking accuracy (snaps 3 and 7)
  - Mid-swing velocity and body sway (snap 5)

KP starts at 40 and increases by KP_STEP each cycle.
KD starts at KD_START and increases by KD_STEP each cycle.
Both are applied to ALL joints (not thigh-only) so you see the
full gain effect on hips and calves too.

After each cycle a compact summary table is printed showing
tracking error at all three snap points side by side.

SDK joint order (motorState[i]):
  0: FR_hip   1: FR_thigh   2: FR_calf
  3: FL_hip   4: FL_thigh   5: FL_calf
  6: RR_hip   7: RR_thigh   8: RR_calf
  9: RL_hip  10: RL_thigh  11: RL_calf
"""

import time
import math
import numpy as np
import robot_interface as sdk

# =============================================================================
# JOINT NAMES
# =============================================================================
SDK_NAMES = [
    "FR_hip",  "FR_thigh", "FR_calf",
    "FL_hip",  "FL_thigh", "FL_calf",
    "RR_hip",  "RR_thigh", "RR_calf",
    "RL_hip",  "RL_thigh", "RL_calf",
]
THIGH_SDK = [1, 4, 7, 10]

# =============================================================================
# DEFAULT JOINT POSITIONS — SDK ORDER
# =============================================================================
DEFAULT_Q = np.array([
     0.0,  0.9, -1.80,   # FR
     0.0,  0.9, -1.80,   # FL
     0.0,  0.95,-1.85,   # RR
     0.0,  0.95,-1.85,   # RL
], dtype=np.float32)

# =============================================================================
# GAINS — SDK ORDER
# =============================================================================
KP_MULT = np.array([
    1.000, 1.857, 2.286,
    1.000, 1.857, 2.286,
    1.000, 1.857, 2.286,
    1.000, 1.857, 2.286,
], dtype=np.float32)

# Base KD per joint type — scaled per cycle
KD_BASE = np.array([
    4.0, 4.5, 5.0,
    4.0, 4.5, 5.0,
    4.0, 4.5, 5.0,
    4.0, 4.5, 5.0,
], dtype=np.float32)

TAU_FF = np.zeros(12, np.float32)
TAU_FF[THIGH_SDK] = 1.2

# Stand ramp
KP_STAND     = 5.0
KP_STAND_MAX = 35.0
HOLD_RAMP_S  = 4.0
HOLD_FULL_S  = 3.0
CONTROL_HZ   = 500
TILT_STOP_DEG = 55.0   # hold stage only

# =============================================================================
# WAVE + GAIN SCHEDULE
# =============================================================================
WAVE_AMP   = 0.18    # radians
WAVE_FREQ  = 0.25    # Hz
WAVE_T     = 1.0 / WAVE_FREQ

# KP and KD increase every cycle — applied to ALL joints via KP_MULT / KD_BASE
KP_WAVE_START = 30.0
KP_WAVE_STEP  = 10.0
KP_WAVE_MAX   = 70.0

KD_WAVE_START = 1.0    # multiplier on KD_BASE (1.0 = base values)
KD_WAVE_STEP  = 0.2    # adds 20% of base KD each cycle
KD_WAVE_MAX   = 3.0    # caps at 3x base KD

# Only these three snap fractions are printed
# Index 2 = snap3 (frac 0.25) = forward peak
# Index 4 = snap5 (frac 0.50) = centre falling
# Index 6 = snap7 (frac 0.75) = backward peak
ALL_SNAP_FRACS  = [0.0, 0.125, 0.25, 0.375, 0.5,
                   0.625, 0.75, 0.875, 1.0]
PRINT_SNAP_IDX  = [2, 4, 6]    # indices into ALL_SNAP_FRACS to print
SNAP_LABELS_ALL = [
    "1: centre start   (wave= 0 rising )",
    "2: centre->peak   (wave=+0.5A     )",
    "3: FORWARD PEAK   (wave=+A        )",   # printed
    "4: peak->centre   (wave=+0.5A     )",
    "5: CENTRE FALLING (wave= 0 falling)",   # printed
    "6: centre->trough (wave=-0.5A     )",
    "7: BACKWARD PEAK  (wave=-A        )",   # printed
    "8: trough->centre (wave=-0.5A     )",
    "9: centre end     (wave= 0 rising )",
]
SNAP_WINDOW_S = 0.04

# =============================================================================
# SDK INIT
# =============================================================================
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# =============================================================================
# UTILITIES
# =============================================================================
def read_state(state_obj):
    jpos = np.array([state_obj.motorState[i].q      for i in range(12)], np.float32)
    jvel = np.array([state_obj.motorState[i].dq     for i in range(12)], np.float32)
    tau  = np.array([state_obj.motorState[i].tauEst for i in range(12)], np.float32)
    acc  = np.array(state_obj.imu.accelerometer,    np.float32)
    gyro = np.array(state_obj.imu.gyroscope,        np.float32)
    rpy  = np.array(state_obj.imu.rpy,              np.float32)
    foot = np.array(state_obj.footForce,            np.float32)
    norm_a    = max(float(np.linalg.norm(acc)), 0.1)
    proj_grav = -acc / norm_a
    tilt_deg  = float(np.degrees(
                np.sqrt(proj_grav[0]**2 + proj_grav[1]**2)))
    return dict(jpos=jpos, jvel=jvel, tau=tau,
                acc=acc, gyro=gyro, rpy=rpy, foot=foot,
                proj_grav=proj_grav, tilt_deg=tilt_deg)


def send_cmd(target_q, kp_base, kd_mult):
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(target_q[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(kp_base  * KP_MULT[i])
        cmd.motorCmd[i].Kd   = float(kd_mult  * KD_BASE[i])
        cmd.motorCmd[i].tau  = float(TAU_FF[i])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()


def print_snap(snap_idx_all, cycle, wave_val, kp, kd_mult,
               t_elapsed, buf, target_q):
    """Print detailed snapshot for one of the three diagnostic points."""
    jpos_avg = np.mean([s["jpos"]      for s in buf], axis=0)
    jvel_avg = np.mean([s["jvel"]      for s in buf], axis=0)
    grav_avg = np.mean([s["proj_grav"] for s in buf], axis=0)
    gyro_avg = np.mean([s["gyro"]      for s in buf], axis=0)
    foot_avg = np.mean([s["foot"]      for s in buf], axis=0)
    tilt_avg = float(np.mean([s["tilt_deg"]  for s in buf]))
    rpy_last = buf[-1]["rpy"]
    err      = target_q - jpos_avg
    jdelta   = jpos_avg - DEFAULT_Q

    label = SNAP_LABELS_ALL[snap_idx_all]
    kp_eff_thigh = kp * KP_MULT[1]
    kp_eff_hip   = kp * KP_MULT[0]
    kp_eff_calf  = kp * KP_MULT[2]
    kd_eff_thigh = kd_mult * KD_BASE[1]

    print(f"\n{'='*86}")
    print(f"  Cycle {cycle:2d}  |  {label}")
    print(f"  t={t_elapsed:.2f}s   wave={wave_val:+.4f} rad   "
          f"Kp={kp:.0f}  "
          f"(hip={kp_eff_hip:.0f} thigh={kp_eff_thigh:.0f} calf={kp_eff_calf:.0f})  "
          f"Kd_thigh={kd_eff_thigh:.2f}   samples={len(buf)}")

    # IMU
    print(f"\n  IMU:")
    print(f"    proj_grav  [{grav_avg[0]:+.4f}  {grav_avg[1]:+.4f}  "
          f"{grav_avg[2]:+.4f}]   tilt={tilt_avg:.2f} deg")
    print(f"    gyro       [{gyro_avg[0]:+.4f}  {gyro_avg[1]:+.4f}  "
          f"{gyro_avg[2]:+.4f}]   rpy=[{rpy_last[0]:+.3f} "
          f"{rpy_last[1]:+.3f} {rpy_last[2]:+.3f}]")

    # Foot contact
    contact = "  ".join(
        f"{'●' if foot_avg[i]>20 else '○'}{foot_avg[i]:4.0f}N"
        for i in range(4))
    print(f"    foot [FR FL RR RL]: {contact}")

    # Joint table
    print(f"\n  {'Joint':<12} {'target':>8} {'actual':>8} "
          f"{'err':>8} {'q-def':>8} {'vel':>8}  flag")
    print(f"  {'─'*66}")
    for i in range(12):
        wave_flag = " ← wave" if i in THIGH_SDK else ""
        warn_flag = " *** " if abs(err[i]) > 0.05 else ""
        print(f"  {SDK_NAMES[i]:<12} "
              f"{float(target_q[i]):>8.4f} "
              f"{float(jpos_avg[i]):>8.4f} "
              f"{float(err[i]):>8.4f} "
              f"{float(jdelta[i]):>8.4f} "
              f"{float(jvel_avg[i]):>8.4f}"
              f"{wave_flag}{warn_flag}")
    print(f"{'='*86}")


def print_cycle_summary(cycle, kp, kd_mult, snap_data):
    """
    After all three snaps collected for one cycle, print a compact
    side-by-side comparison table.
    snap_data: dict keyed by snap label short name ->
               {"err": array(12), "tilt": float, "gyro_mag": float}
    """
    keys   = ["fwd_peak", "centre", "bwd_peak"]
    labels = ["fwd peak (+A)", "centre (0↓)", "bwd peak (-A)"]

    print(f"\n{'━'*86}")
    print(f"  CYCLE {cycle:2d} SUMMARY  |  "
          f"Kp={kp:.0f}  "
          f"(thigh={kp*KP_MULT[1]:.0f})  "
          f"Kd_mult={kd_mult:.1f}  "
          f"(thigh={kd_mult*KD_BASE[1]:.2f})")
    print(f"  {'Joint':<12}  "
          + "  ".join(f"{'err@'+l:>16}" for l in labels))
    print(f"  {'─'*78}")

    for i in range(12):
        wave_mark = " ←" if i in THIGH_SDK else "  "
        errs = []
        for k in keys:
            if k in snap_data:
                errs.append(f"{float(snap_data[k]['err'][i]):>+8.4f}")
            else:
                errs.append(f"{'---':>8}")
        print(f"  {SDK_NAMES[i]:<12}{wave_mark}  "
              + "        ".join(errs))

    print(f"\n  Tilt (deg):  "
          + "   ".join(
              f"{k}={snap_data[k]['tilt']:.2f}"
              if k in snap_data else f"{k}=---"
              for k in keys))
    print(f"  Gyro mag:    "
          + "   ".join(
              f"{k}={snap_data[k]['gyro_mag']:.4f}"
              if k in snap_data else f"{k}=---"
              for k in keys))
    print(f"{'━'*86}\n")


# =============================================================================
# STAGE 1 — HOLD: policy stand ramp + equilibrium measurement
# =============================================================================
print("\n" + "=" * 70)
print("GO1 WAVE v3 — 3-POINT TRACKING MONITOR")
print(f"  Snap points: forward peak (+A) | centre falling (0) | backward peak (-A)")
print(f"  KP start={KP_WAVE_START:.0f}  step=+{KP_WAVE_STEP:.0f}/cycle  "
      f"max={KP_WAVE_MAX:.0f}")
print(f"  KD start=base×{KD_WAVE_START:.1f}  step=+{KD_WAVE_STEP:.1f}×base/cycle  "
      f"max=base×{KD_WAVE_MAX:.1f}")
print(f"  Wave: amp={WAVE_AMP} rad  freq={WAVE_FREQ} Hz  "
      f"period={WAVE_T:.1f}s")
print(f"  Stand ramp: {HOLD_RAMP_S:.0f}s + {HOLD_FULL_S:.0f}s hold")
print("  Place robot on ground. Starting in 8s. Ctrl+C to stop.")
print("=" * 70 + "\n")
time.sleep(8)

# --- KP ramp to stand --------------------------------------------------------
hold_t0   = time.perf_counter()
hold_step = 0
print("[HOLD] Ramping KP to stand values...")

while True:
    t_loop = time.perf_counter()
    dt     = t_loop - hold_t0
    if dt >= HOLD_RAMP_S + HOLD_FULL_S:
        break

    udp.Recv()
    udp.GetRecv(state)
    s = read_state(state)

    if s["tilt_deg"] > TILT_STOP_DEG:
        print(f"[SAFETY] tilt={s['tilt_deg']:.1f} deg during hold. Aborting.")
        raise SystemExit

    alpha   = min(1.0, dt / HOLD_RAMP_S)
    kp_base = KP_STAND + alpha * (KP_STAND_MAX - KP_STAND)
    send_cmd(DEFAULT_Q, kp_base=kp_base, kd_mult=1.0)

    if hold_step % (CONTROL_HZ // 2) == 0:
        max_err = float(np.max(np.abs(s["jpos"] - DEFAULT_Q)))
        print(f"[HOLD t={dt:4.1f}s]  "
              f"KP_base={kp_base:.1f}  "
              f"thigh={kp_base*KP_MULT[1]:.0f}  "
              f"max_err={max_err:.4f}  "
              f"tilt={s['tilt_deg']:.1f}deg", flush=True)
    hold_step += 1
    sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - t_loop)
    if sl > 0:
        time.sleep(sl)

# --- Measure equilibrium offset ----------------------------------------------
print("\n[HOLD] Measuring equilibrium over 20 samples...")
eq_buf = []
for _ in range(20):
    udp.Recv()
    udp.GetRecv(state)
    eq_buf.append(
        np.array([state.motorState[i].q for i in range(12)], np.float32))
    time.sleep(0.005)

jpos_eq       = np.mean(eq_buf, axis=0)
jdelta_offset = jpos_eq - DEFAULT_Q
max_off       = float(np.max(np.abs(jdelta_offset)))

print(f"\n[HOLD COMPLETE]  max_offset={max_off:.4f} rad")
print("  jdelta_offset (SDK order):")
for i in range(12):
    print(f"    {SDK_NAMES[i]:<12}: {jdelta_offset[i]:+.5f} rad")

if max_off > 0.20:
    print("  *** WARNING: offset > 0.20 rad. Ctrl+C within 5s to abort. ***")
    time.sleep(5.0)
else:
    print("  OK. Proceeding to wave.\n")

# =============================================================================
# STAGE 2 — WAVE: 3-point observation monitor
# =============================================================================
print("\n" + "=" * 70)
print("Stage 2: Wave — printing snaps 3, 5, 7 only")
print("─" * 70)
print(f"  Snap 3: forward peak   (thigh = default + {WAVE_AMP:.2f} rad)")
print(f"  Snap 5: centre falling (thigh = default,  max velocity)")
print(f"  Snap 7: backward peak  (thigh = default - {WAVE_AMP:.2f} rad)")
print("─" * 70)
print("  Ctrl+C to stop cleanly at any time.")
print("=" * 70 + "\n")

wave_start = time.perf_counter()
t0_global  = time.perf_counter()
Kp_wave    = KP_WAVE_START
Kd_mult    = KD_WAVE_START
cycle      = 0
snap_fired = [False] * 9
snap_buf   = []
active_snap_idx = None      # which of ALL_SNAP_FRACS we are currently buffering
cycle_snaps = {}            # accumulate snap results for cycle summary

try:
    while True:
        t_loop  = time.perf_counter()
        t_total = t_loop - t0_global
        wave_t  = t_loop - wave_start

        cycle_t     = math.fmod(wave_t, WAVE_T)
        cycle_n     = int(wave_t / WAVE_T)
        wave_offset = WAVE_AMP * math.sin(2.0 * math.pi * WAVE_FREQ * wave_t)

        # New cycle: print previous cycle summary, update gains
        if cycle_n > cycle:
            # Print summary for completed cycle before bumping gains
            if cycle_snaps:
                print_cycle_summary(cycle + 1, Kp_wave, Kd_mult, cycle_snaps)

            cycle   = cycle_n
            Kp_wave = min(KP_WAVE_START + KP_WAVE_STEP * cycle, KP_WAVE_MAX)
            Kd_mult = min(KD_WAVE_START + KD_WAVE_STEP * cycle, KD_WAVE_MAX)
            snap_fired  = [False] * 9
            cycle_snaps = {}
            print(f"--- Cycle {cycle} start  |  "
                  f"Kp={Kp_wave:.0f}  "
                  f"(thigh={Kp_wave*KP_MULT[1]:.0f})  "
                  f"Kd_mult={Kd_mult:.1f}  "
                  f"(thigh={Kd_mult*KD_BASE[1]:.2f}) ---")

        # Read sensors
        udp.Recv()
        udp.GetRecv(state)
        s = read_state(state)

        # Build target — wave on thighs only
        target = DEFAULT_Q.copy()
        for ti in THIGH_SDK:
            target[ti] += wave_offset

        # Send at current cycle gains (all joints)
        send_cmd(target, kp_base=Kp_wave, kd_mult=Kd_mult)

        # --- Snapshot detection: only for print indices 2, 4, 6 ---
        for si in PRINT_SNAP_IDX:
            if snap_fired[si]:
                continue

            frac      = ALL_SNAP_FRACS[si]
            half_win  = SNAP_WINDOW_S / 2.0
            snap_time = frac * WAVE_T

            if frac == 1.0:
                in_window = (cycle_t > WAVE_T - half_win) or \
                            (cycle_t < half_win)
            else:
                in_window = abs(cycle_t - snap_time) < half_win

            if in_window:
                if active_snap_idx != si:
                    snap_buf        = []
                    active_snap_idx = si
                snap_buf.append(s)

            elif len(snap_buf) > 0 and active_snap_idx == si \
                    and not snap_fired[si]:
                # Window just closed — print and store for summary
                print_snap(si, cycle + 1, wave_offset,
                           Kp_wave, Kd_mult,
                           t_total, snap_buf, target)

                err_avg = target - np.mean([b["jpos"] for b in snap_buf],
                                           axis=0)
                gyro_avg = np.mean([b["gyro"] for b in snap_buf], axis=0)
                tilt_avg = float(np.mean([b["tilt_deg"] for b in snap_buf]))

                snap_key = {2: "fwd_peak", 4: "centre", 6: "bwd_peak"}[si]
                cycle_snaps[snap_key] = {
                    "err":      err_avg,
                    "tilt":     tilt_avg,
                    "gyro_mag": float(np.linalg.norm(gyro_avg)),
                }

                snap_fired[si]  = True
                snap_buf        = []
                active_snap_idx = None
                break

        # Status line every 1s
        step_n = int(wave_t * CONTROL_HZ)
        if step_n % 500 < 2:
            gv = s["proj_grav"]
            ft = "".join("●" if s["foot"][i] > 20 else "○"
                         for i in range(4))
            print(f"t={t_total:5.1f}s | wave={wave_offset:+.3f} | "
                  f"grav({gv[0]:+.2f},{gv[1]:+.2f},{gv[2]:+.2f}) | "
                  f"tilt={s['tilt_deg']:.1f}deg | feet={ft} | "
                  f"Kp={Kp_wave:.0f} Kd×{Kd_mult:.1f}",
                  flush=True)

        sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - t_loop)
        if sl > 0:
            time.sleep(sl)

except KeyboardInterrupt:
    print("\n[WAVE] Stopped by user.", flush=True)
    if cycle_snaps:
        print_cycle_summary(cycle + 1, Kp_wave, Kd_mult, cycle_snaps)

finally:
    print("[FINAL] Returning to stand pose...", flush=True)
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(DEFAULT_Q[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(KP_STAND_MAX * KP_MULT[i])
        cmd.motorCmd[i].Kd   = float(KD_BASE[i])
        cmd.motorCmd[i].tau  = float(TAU_FF[i])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()
    print("[FINAL] Done.", flush=True)