#!/usr/bin/env python3
"""
Go1 KP/KD Tuning — Incremental gain increase for minimum standing error
========================================================================

PROCEDURE:
  1. Ramp from soft gains to baseline [35/4, 65/4.5, 80/5] smoothly
  2. Measure per-joint standing error for 3s
  3. For each non-hip joint with error above threshold:
       Increase KP by 2, KD by 0.5, re-measure
       Stop when error stops improving or KP_MAX reached
  4. Report best per-joint gains and final error table

Hips are fixed at KP=35, KD=4.0 throughout.
RL_th is tracked but excluded from optimisation (known fault).

Run:
  python3 go1_kp_kd_tune.py
  python3 go1_kp_kd_tune.py --no_standup   # already standing
"""

import argparse, time
import numpy as np
from datetime import datetime
import robot_interface as sdk

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

STAND_Q = np.array([
     0.05,-0.05, 0.05,-0.05,
     0.70, 0.70, 0.70, 0.70,
    -1.40,-1.40,-1.40,-1.40,
], np.float32)

# Gains — starting point
KP_BASE = np.array([35,35,35,35,  65,65,65,65,  80,80,80,80], np.float32)
KD_BASE = np.array([4.0,4.0,4.0,4.0, 4.5,4.5,4.5,4.5, 5.0,5.0,5.0,5.0], np.float32)

KP_SOFT = np.array([8,8,8,8,  12,12,12,12,  15,15,15,15], np.float32)
KD_SOFT = np.array([2,2,2,2,   2, 2, 2, 2,   2, 2, 2, 2], np.float32)

# Tuning limits
KP_MAX  = np.array([35,35,35,35,  100,100,100,100,  110,110,110,110], np.float32)
KD_MAX  = np.array([4,4,4,4,      8,  8,  8,  8,    8,  8,  8,  8 ], np.float32)

KP_STEP = 2.0   # increment per round
KD_STEP = 0.5

FAULT_J    = 6          # RL_th index — excluded from optimisation
SETTLE_S   = 1.5        # settle time after gain change
MEASURE_S  = 2.0        # measurement window
ERROR_THR  = 0.030      # rad — target per-joint error
MIN_IMPROVE = 0.0005    # rad — min improvement to continue tuning


def read_state():
    udp.Recv(); udp.GetRecv(state)
    jpos = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
    return jpos


def send_cmd(target_q, kp, kd):
    udp.Recv(); udp.GetRecv(state)
    for i in range(12):
        s = sdk_to_isaac[i]
        cmd.motorCmd[s].mode = 0x0A
        cmd.motorCmd[s].q    = float(target_q[i])
        cmd.motorCmd[s].dq   = 0.0
        cmd.motorCmd[s].Kp   = float(kp[i])
        cmd.motorCmd[s].Kd   = float(kd[i])
        cmd.motorCmd[s].tau  = 0.0
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def run_for(seconds, kp, kd):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        t_s = time.perf_counter()
        send_cmd(STAND_Q, kp, kd)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


def measure_error(kp, kd):
    """Returns mean absolute error per joint over MEASURE_S seconds."""
    run_for(SETTLE_S, kp, kd)
    n      = int(MEASURE_S * CTRL_HZ)
    errors = np.zeros((n, 12), np.float32)
    for i in range(n):
        t_s  = time.perf_counter()
        jpos = read_state()
        send_cmd(STAND_Q, kp, kd)
        errors[i] = np.abs(STAND_Q - jpos)
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)
    return errors.mean(axis=0)


def ramp(kp_from, kd_from, kp_to, kd_to, ramp_s=4.0):
    n = int(ramp_s * CTRL_HZ)
    for i in range(n):
        t_s = time.perf_counter()
        a   = i / n
        send_cmd(STAND_Q, kp_from + a*(kp_to-kp_from),
                           kd_from + a*(kd_to-kd_from))
        sl = DT - (time.perf_counter() - t_s)
        if sl > 0: time.sleep(sl)


def print_errors(err, kp, kd, label=''):
    healthy_mean = np.mean([err[i] for i in range(12) if i != FAULT_J])
    print(f"\n  {'─'*62}")
    print(f"  {label}")
    print(f"  {'Joint':10s}  {'KP':>5}  {'KD':>5}  {'MeanErr':>9}  Status")
    print(f"  {'─'*62}")
    for i, jn in enumerate(JNAMES):
        ok    = '✓' if err[i] < ERROR_THR else '✗'
        fault = '  ← FAULT' if i == FAULT_J else ''
        print(f"  {jn:10s}  {kp[i]:>5.1f}  {kd[i]:>5.2f}  "
              f"{err[i]:>9.5f}  {ok}{fault}")
    print(f"  {'─'*62}")
    print(f"  Healthy mean error: {healthy_mean:.5f} rad  "
          f"(target < {ERROR_THR:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no_standup', action='store_true')
    args = p.parse_args()

    print("Connecting to Go1...")
    udp.Recv(); udp.GetRecv(state)
    print("Connected. Mapping: ✓")
    print(f"\n  Baseline: KP=[35,65,80]  KD=[4.0,4.5,5.0]")
    print(f"  Tuning step: KP+{KP_STEP}  KD+{KD_STEP} per round")
    print(f"  Hips fixed at KP=35, KD=4.0")
    print(f"  RL_th (index 6) excluded from optimisation\n")

    input("  Robot standing on flat ground? Press Enter → ")

    # ── Step 1: ramp soft → baseline ─────────────────────────────────────────
    kp = KP_BASE.copy()
    kd = KD_BASE.copy()

    if not args.no_standup:
        print("\n  Ramping soft → baseline (8s)...")
        run_for(2.0, KP_SOFT, KD_SOFT)
        ramp(KP_SOFT, KD_SOFT, kp, kd, ramp_s=8.0)
        run_for(2.0, kp, kd)
    else:
        # Even if already standing, ramp smoothly to baseline
        jpos = read_state()
        print("\n  Ramping to baseline gains (5s)...")
        ramp(kp, kd, kp, kd, ramp_s=0.5)  # just confirm

    # ── Step 2: baseline measurement ─────────────────────────────────────────
    print("\n  Measuring baseline error...")
    err = measure_error(kp, kd)
    print_errors(err, kp, kd, label='BASELINE [35/4.0, 65/4.5, 80/5.0]')

    # ── Step 3: per-joint incremental tuning (thighs + knees only) ───────────
    # Tune thighs first (indices 4,5,7 — skip 6=RL_th), then knees (8-11)
    tune_indices = [4, 5, 7,   # FL_th, FR_th, RR_th (not RL_th)
                    8, 9, 10, 11]  # all knees

    history = {i: [(float(kp[i]), float(kd[i]), float(err[i]))]
               for i in tune_indices}

    print("\n  ── Incremental tuning ──")
    print(f"  Increasing KP+{KP_STEP}, KD+{KD_STEP} for joints still above "
          f"{ERROR_THR*1000:.0f}mrad threshold\n")

    any_changed = True
    round_n     = 0

    while any_changed:
        round_n    += 1
        any_changed = False

        # Which non-hip joints still above threshold?
        need_tune = [i for i in tune_indices
                     if err[i] > ERROR_THR and kp[i] < KP_MAX[i]]

        if not need_tune:
            print(f"  Round {round_n}: all target joints at or below threshold ✓")
            break

        print(f"  Round {round_n} — joints above {ERROR_THR*1000:.0f}mrad: "
              f"{[JNAMES[i] for i in need_tune]}")

        for i in need_tune:
            new_kp = min(kp[i] + KP_STEP, KP_MAX[i])
            new_kd = min(kd[i] + KD_STEP, KD_MAX[i])

            if new_kp == kp[i]:
                print(f"    {JNAMES[i]:10s}: KP already at max {KP_MAX[i]:.0f}")
                continue

            # Ramp ONLY this joint's gain upward (others unchanged)
            kp_new = kp.copy(); kd_new = kd.copy()
            kp_new[i] = new_kp; kd_new[i] = new_kd

            n_ramp = int(1.0 * CTRL_HZ)  # 1s per-joint ramp
            for step in range(n_ramp):
                t_s = time.perf_counter()
                a   = step / n_ramp
                kp_step = kp.copy(); kd_step = kd.copy()
                kp_step[i] = kp[i] + a * (new_kp - kp[i])
                kd_step[i] = kd[i] + a * (new_kd - kd[i])
                send_cmd(STAND_Q, kp_step, kd_step)
                sl = DT - (time.perf_counter() - t_s)
                if sl > 0: time.sleep(sl)

            kp[i] = new_kp; kd[i] = new_kd

        # Measure after updating all joints this round
        new_err = measure_error(kp, kd)

        for i in need_tune:
            improvement = err[i] - new_err[i]
            improved    = improvement > MIN_IMPROVE
            history[i].append((float(kp[i]), float(kd[i]), float(new_err[i])))
            tag = f"Δ={improvement:+.5f} {'✓ improved' if improved else '~ plateau'}"
            print(f"    {JNAMES[i]:10s}: KP={kp[i]:.0f}  KD={kd[i]:.2f}  "
                  f"err={new_err[i]:.5f}  {tag}")
            if improved:
                any_changed = True

        err = new_err

        if round_n > 30:
            print("  Max rounds reached.")
            break

    # ── Step 4: final results ─────────────────────────────────────────────────
    print_errors(err, kp, kd, label='FINAL RESULT')

    print("\n  ── Per-joint tuning history ──")
    for i in tune_indices:
        jn = JNAMES[i]
        pts = history[i]
        errs = [p[2] for p in pts]
        best_idx = int(np.argmin(errs))
        best_kp, best_kd, best_err = pts[best_idx]
        print(f"  {jn:10s}: start={pts[0][2]:.5f}  "
              f"best={best_err:.5f}  "
              f"at KP={best_kp:.0f}/KD={best_kd:.2f}  "
              f"(rounds: {len(pts)-1})")

    print("\n  ── Isaac Lab config ──")
    hip_kp  = kp[0];  hip_kd  = kd[0]
    th_kp   = kp[4];  th_kd   = kd[4]   # use FL_th as representative
    kn_kp   = kp[8];  kn_kd   = kd[8]   # use FL_kn as representative
    print(f"  stiffness:  hip={hip_kp:.0f}  thigh={th_kp:.0f}  knee={kn_kp:.0f}  Nm/rad")
    print(f"  damping:    hip={hip_kd:.2f}  thigh={th_kd:.2f}   knee={kn_kd:.2f}  Nm·s/rad")

    print("\n  ── go1_env.py ──")
    print(f"  KP_TRAIN = np.array([{hip_kp:.0f},{hip_kp:.0f},{hip_kp:.0f},{hip_kp:.0f},  "
          f"{th_kp:.0f},{th_kp:.0f},{th_kp:.0f},{th_kp:.0f},  "
          f"{kn_kp:.0f},{kn_kp:.0f},{kn_kp:.0f},{kn_kp:.0f}])")
    print(f"  KD_TRAIN = np.array([{hip_kd:.1f},{hip_kd:.1f},{hip_kd:.1f},{hip_kd:.1f},  "
          f"{th_kd:.1f},{th_kd:.1f},{th_kd:.1f},{th_kd:.1f},  "
          f"{kn_kd:.1f},{kn_kd:.1f},{kn_kd:.1f},{kn_kd:.1f}])")

    # Save
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"go1_kp_kd_tune_{ts}.npz"
    np.savez(fname, kp_final=kp, kd_final=kd,
             err_final=err, stand_q=STAND_Q,
             joint_names=np.array(JNAMES, dtype=object))
    print(f"\n  → Saved: {fname}")


if __name__ == '__main__':
    main()