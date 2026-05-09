#!/usr/bin/env python3
"""
=============================================================================
GO1 PACE DATA COLLECTION  v12  — HIPS FIXED, SAFE AMPLITUDES
=============================================================================
Fixes from v11:

PROBLEM: ±0.20 rad thigh at resonance (fn≈1.1Hz) → actual ≈ ±0.50 rad
         Torque = 65 × 0.50 = 32.5 Nm > 23.7 Nm limit → PowerProtect fires
         After protection: joints limp, hips dragged 0.45 rad → data invalid
         Loop ran 2382Hz in 6.3s (not 30s) — all data unusable

FIX: Amplitudes chosen so torque stays safe EVEN AT RESONANCE PEAK:
         τ_max = KP_eff × resonance_gain × amplitude < 15 Nm (conservative)
         resonance_gain ≈ 2.5× for typical Go1 joint damping
         thigh: 65 × 2.5 × 0.07 = 11.4 Nm ✓
         calf:  80 × 2.5 × 0.06 = 12.0 Nm ✓
         hip:   zero chirp → hip held firmly by KP=35

NOTE: Small amplitude is fine for PACE. The RESONANCE PEAK SHAPE contains
      all the information CMA-ES needs:
      - Peak frequency → identifies Ia (shifts fn = √(KP/Ia)/2π)
      - Peak height    → identifies d  (controls resonance gain Q=1/2ζ)
      - Phase slope    → identifies Td (constant phase offset at all frequencies)

Wall-time check: if loop completes in < 80% of expected time → abort.
This catches the 2382Hz bug from power protection events.

Run:
    python3 go1_pace_data_collection_v12.py
    python3 go1_pace_data_collection_v12.py --duration 40 --max_freq 5.0
=============================================================================
"""

import argparse, time, sys
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    print("[ERROR] pip install torch"); sys.exit(1)
try:
    import robot_interface as sdk
except ImportError:
    print("[ERROR] robot_interface not found."); sys.exit(1)

# =============================================================================
# SDK SETUP
# =============================================================================
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

CTRL_HZ = 500
DT      = 1.0 / CTRL_HZ

# =============================================================================
# JOINT MAPPINGS
# =============================================================================
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i
SDK_TO_ISAAC = np.array(sdk_to_isaac, dtype=np.int32)
ISAAC_TO_SDK = np.array(isaac_to_sdk, dtype=np.int32)

ISAAC_NAMES = [
    "FL_hip",   "FR_hip",   "RL_hip",   "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf",  "FR_calf",  "RL_calf",  "RR_calf",
]

# =============================================================================
# ROBOT CONSTANTS
# =============================================================================
DEFAULT_JOINT_POS = np.array([
     0.1,  0.1,  0.1,  0.1,
     0.8,  0.8,  0.8,  0.8,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float64)

KP_MULTIPLIER = np.array([
    1.000, 1.000, 1.000, 1.000,
    1.857, 1.857, 1.857, 1.857,
    2.286, 2.286, 2.286, 2.286,
], dtype=np.float64)

KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,
    4.5, 4.5, 4.5, 4.5,
    5.0, 5.0, 5.0, 5.0,
], dtype=np.float64)

TAU_FF_ISAAC = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.2, 1.2, 1.2, 1.2,
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)

KP_START = 5.0
KP_FULL  = 35.0

Q_LIM_LO = np.array([-0.80,-0.80,-0.80,-0.80,
                      -0.50,-0.50,-0.50,-0.50,
                      -2.70,-2.70,-2.70,-2.70], np.float64)
Q_LIM_HI = np.array([ 0.80, 0.80, 0.80, 0.80,
                       4.40, 4.40, 4.40, 4.40,
                      -0.95,-0.95,-0.95,-0.95], np.float64)

# ── SAFE AMPLITUDES — torque safe even at resonance peak ─────────────────────
# Derivation:
#   τ_max = KP_eff × resonance_gain × amplitude < 15 Nm
#   resonance_gain ≈ 2.5 (empirical from v10/v11 data showing ratio 1.6-2.5)
#   thigh: KP_eff = 35 × 1.857 = 65 Nm/rad
#          A < 15 / (65 × 2.5) = 0.092 rad → use 0.07 rad (safety margin)
#   calf:  KP_eff = 35 × 2.286 = 80 Nm/rad
#          A < 15 / (80 × 2.5) = 0.075 rad → use 0.06 rad
CHIRP_AMP = np.array([
    0.00, 0.00, 0.00, 0.00,   # HIPS: zero — fixed at default, no chirp
    0.07, 0.07, 0.04, 0.07,   # THIGHS: ±0.07 rad (RL_thigh=0.04 — stiction)
    0.06, 0.06, 0.06, 0.06,   # CALVES: ±0.06 rad
], dtype=np.float64)

# ── SYMMETRIC FORE-AFT CHIRP ──────────────────────────────────────────────────
# FL+, FR+ thighs forward | RL-, RR- thighs backward → net pitch moment ≈ 0
# FL-, FR- calves extend  | RL+, RR+ calves flex     → matches thigh direction
CHIRP_SIGN = np.array([
     1,  1,  1,  1,   # hips: irrelevant (amp=0)
     1,  1, -1, -1,   # thighs: FL+ FR+ RL- RR-
    -1, -1,  1,  1,   # calves: FL- FR- RL+ RR+
], dtype=np.float64)

# Safety clamp — tight enough to prevent torque spikes
# At full KP: τ = KP_eff × clamp_width
# thigh: 65 × 0.10 = 6.5 Nm ✓   calf: 80 × 0.08 = 6.4 Nm ✓
MAX_CMD_ERR = np.array([
    0.05, 0.05, 0.05, 0.05,   # hips: very tight
    0.10, 0.10, 0.10, 0.10,   # thighs
    0.08, 0.08, 0.08, 0.08,   # calves
], dtype=np.float64)

# Pre-computed
kp_sdk_full = np.array([
    KP_FULL * KP_MULTIPLIER[isaac_to_sdk[s]] for s in range(12)
], dtype=np.float64)
kd_sdk = np.array([
    KD_PER_JOINT[isaac_to_sdk[s]] for s in range(12)
], dtype=np.float64)
tau_ff_sdk = np.array([
    TAU_FF_ISAAC[isaac_to_sdk[s]] for s in range(12)
], dtype=np.float64)

_q_sdk_buf = np.zeros(12, dtype=np.float64)


# =============================================================================
# HOT LOOP HELPERS
# =============================================================================
def read_isaac_fast():
    udp.Recv(); udp.GetRecv(state)
    ms = state.motorState
    _q_sdk_buf[0] =ms[0].q;  _q_sdk_buf[1] =ms[1].q
    _q_sdk_buf[2] =ms[2].q;  _q_sdk_buf[3] =ms[3].q
    _q_sdk_buf[4] =ms[4].q;  _q_sdk_buf[5] =ms[5].q
    _q_sdk_buf[6] =ms[6].q;  _q_sdk_buf[7] =ms[7].q
    _q_sdk_buf[8] =ms[8].q;  _q_sdk_buf[9] =ms[9].q
    _q_sdk_buf[10]=ms[10].q; _q_sdk_buf[11]=ms[11].q
    return _q_sdk_buf[SDK_TO_ISAAC]


def send_cmd_fast(q_target_isaac, kp_sdk, use_tau_ff=False):
    q_sdk = q_target_isaac[ISAAC_TO_SDK]
    mc = cmd.motorCmd
    mc[0].q=q_sdk[0];  mc[0].Kp=kp_sdk[0];  mc[0].Kd=kd_sdk[0]
    mc[1].q=q_sdk[1];  mc[1].Kp=kp_sdk[1];  mc[1].Kd=kd_sdk[1]
    mc[2].q=q_sdk[2];  mc[2].Kp=kp_sdk[2];  mc[2].Kd=kd_sdk[2]
    mc[3].q=q_sdk[3];  mc[3].Kp=kp_sdk[3];  mc[3].Kd=kd_sdk[3]
    mc[4].q=q_sdk[4];  mc[4].Kp=kp_sdk[4];  mc[4].Kd=kd_sdk[4]
    mc[5].q=q_sdk[5];  mc[5].Kp=kp_sdk[5];  mc[5].Kd=kd_sdk[5]
    mc[6].q=q_sdk[6];  mc[6].Kp=kp_sdk[6];  mc[6].Kd=kd_sdk[6]
    mc[7].q=q_sdk[7];  mc[7].Kp=kp_sdk[7];  mc[7].Kd=kd_sdk[7]
    mc[8].q=q_sdk[8];  mc[8].Kp=kp_sdk[8];  mc[8].Kd=kd_sdk[8]
    mc[9].q=q_sdk[9];  mc[9].Kp=kp_sdk[9];  mc[9].Kd=kd_sdk[9]
    mc[10].q=q_sdk[10]; mc[10].Kp=kp_sdk[10]; mc[10].Kd=kd_sdk[10]
    mc[11].q=q_sdk[11]; mc[11].Kp=kp_sdk[11]; mc[11].Kd=kd_sdk[11]
    if use_tau_ff:
        mc[0].tau=tau_ff_sdk[0];   mc[1].tau=tau_ff_sdk[1]
        mc[2].tau=tau_ff_sdk[2];   mc[3].tau=tau_ff_sdk[3]
        mc[4].tau=tau_ff_sdk[4];   mc[5].tau=tau_ff_sdk[5]
        mc[6].tau=tau_ff_sdk[6];   mc[7].tau=tau_ff_sdk[7]
        mc[8].tau=tau_ff_sdk[8];   mc[9].tau=tau_ff_sdk[9]
        mc[10].tau=tau_ff_sdk[10]; mc[11].tau=tau_ff_sdk[11]
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()


def deadline_sleep(t0):
    r = DT - (time.perf_counter() - t0)
    if r > 0.0001: time.sleep(r)


def dual_ramp(total_s=10.0):
    q_start = read_isaac_fast().copy()
    n = int(total_s * CTRL_HZ)
    print(f"  KP {KP_START:.0f}→{KP_FULL:.0f}  max_err={np.max(np.abs(q_start-DEFAULT_JOINT_POS)):.3f}")
    for i in range(n):
        t0 = time.perf_counter()
        a = (i/n)**2 * (3.0-2.0*(i/n))
        kp = KP_START + a*(KP_FULL-KP_START)
        q  = q_start + a*(DEFAULT_JOINT_POS-q_start)
        kp_sdk = np.array([kp*KP_MULTIPLIER[isaac_to_sdk[s]]
                           for s in range(12)], dtype=np.float64)
        send_cmd_fast(q, kp_sdk=kp_sdk, use_tau_ff=True)
        if i % (2*CTRL_HZ) == 0:
            q_now = read_isaac_fast()
            print(f"  t={i*DT:.0f}s  KP={kp:.0f}  "
                  f"max_err={np.max(np.abs(q_now-DEFAULT_JOINT_POS)):.3f}")
        deadline_sleep(t0)
    print("  Done.")


def hold_default(duration_s=2.0):
    for _ in range(int(duration_s*CTRL_HZ)):
        t0 = time.perf_counter()
        send_cmd_fast(DEFAULT_JOINT_POS, kp_sdk=kp_sdk_full, use_tau_ff=True)
        deadline_sleep(t0)


def release():
    q_s = read_isaac_fast().copy()
    n = int(2.0*CTRL_HZ)
    for i in range(n):
        t0 = time.perf_counter()
        send_cmd_fast(q_s+(i/n)**2*(DEFAULT_JOINT_POS-q_s),
                      kp_sdk=kp_sdk_full, use_tau_ff=True)
        deadline_sleep(t0)
    for _ in range(int(0.5*CTRL_HZ)):
        for s in range(12):
            cmd.motorCmd[s].q=0.0; cmd.motorCmd[s].Kp=0.0
            cmd.motorCmd[s].Kd=2.0; cmd.motorCmd[s].tau=0.0
        udp.SetSend(cmd); udp.Send()
        time.sleep(DT)


# =============================================================================
# MAIN
# =============================================================================
def main(args):
    duration   = args.duration
    f0         = args.min_freq
    f1         = min(args.max_freq, 5.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(duration * CTRL_HZ)
    t_lin   = np.linspace(0.0, duration, n_steps, dtype=np.float64)

    # Chirp — hips zero amplitude, thigh+calf symmetric
    phase = 2*np.pi*(f0*t_lin + ((f1-f0)/(2*duration))*t_lin**2)
    chirp = np.sin(phase)

    q_cmd = np.tile(DEFAULT_JOINT_POS, (n_steps, 1))
    for j in range(12):
        if CHIRP_AMP[j] > 0:
            q_cmd[:, j] = DEFAULT_JOINT_POS[j] + CHIRP_AMP[j]*CHIRP_SIGN[j]*chirp
    q_cmd = np.clip(q_cmd, Q_LIM_LO, Q_LIM_HI)

    # Verify max torque estimate
    max_tau_thigh = KP_FULL * KP_MULTIPLIER[4] * CHIRP_AMP[4] * 2.5
    max_tau_calf  = KP_FULL * KP_MULTIPLIER[8] * CHIRP_AMP[8] * 2.5

    print("\n" + "="*65)
    print("  GO1 PACE DATA COLLECTION  v12  — SAFE AMPLITUDES")
    print("="*65)
    print(f"  Freq      : {f0} → {f1} Hz  ({duration}s, {n_steps} steps)")
    print()
    print(f"  HIPS:   FIXED (zero amplitude)")
    print(f"  THIGHS: ±{CHIRP_AMP[4]:.2f} rad  "
          f"max_τ_at_resonance = {max_tau_thigh:.1f} Nm  "
          f"{'✓ safe' if max_tau_thigh < 18 else '⚠ borderline'}")
    print(f"  CALVES: ±{CHIRP_AMP[8]:.2f} rad  "
          f"max_τ_at_resonance = {max_tau_calf:.1f} Nm  "
          f"{'✓ safe' if max_tau_calf < 18 else '⚠ borderline'}")
    print(f"  RL_thigh: ±{CHIRP_AMP[6]:.2f} rad (stiction fault)")
    print()
    print(f"  Safety clamp: thigh ±0.10  calf ±0.08  hip ±0.05 rad")
    print(f"  Output: {output_dir / 'chirp_data.pt'}")
    print()
    print("  ✓ Base RIGIDLY clamped  ✓ Legs FREE  ✓ Kill switch ready")
    print()
    input("  Press Enter → ")

    # Connect + ramp
    udp.Recv(); udp.GetRecv(state)
    print("\n── Startup ramp (10s) ──")
    dual_ramp(total_s=10.0)
    hold_default(duration_s=2.0)

    q_check = read_isaac_fast()
    errs = np.abs(q_check - DEFAULT_JOINT_POS)
    print(f"\n  Hold check (thighs/calves must be <0.10):")
    for j in range(12):
        if errs[j] > 0.08:
            flag = ("⚠ RL_thigh stiction (expected)" if j==6
                    else "⚠ HIGH — check joint")
        else:
            flag = "✓"
        print(f"  {ISAAC_NAMES[j]:12s}  {q_check[j]:+.3f}  err={errs[j]:.3f}  {flag}")

    if float(np.max(np.delete(errs, 6))) > 0.12:
        if input("\n  Non-RL error > 0.12. Continue? [y/N] → ").lower() != 'y':
            release(); return

    # Chirp sweep
    print(f"\n── Chirp {f0}→{f1}Hz × {duration}s ──")
    print(f"  Small amplitude — safe through resonance peak (~1.1Hz)")
    print(f"  Watch: ratio should PEAK at ~1.1Hz then DROP below 1.0 at higher freq")
    print(f"  hip_err should stay < 0.10 rad throughout")
    print()

    q_actual_buf = np.zeros((n_steps, 12), np.float32)
    q_target_buf = np.zeros((n_steps, 12), np.float32)
    t_actual_buf = np.zeros(n_steps,       np.float32)
    n_clamped    = np.zeros(12, np.int32)
    power_prot_count = 0

    prog = 5 * CTRL_HZ
    t0_wall = time.perf_counter()
    expected_end = t0_wall + duration

    for step in range(n_steps):
        t_step = time.perf_counter()

        q_now = read_isaac_fast()

        # ── WALL TIME SANITY CHECK ───────────────────────────────────────
        # If loop is running much faster than expected → power protection
        # fired and SDK is returning without blocking. Abort cleanly.
        elapsed = t_step - t0_wall
        expected = step * DT
        if step > 500 and elapsed < expected * 0.5:
            power_prot_count += 1
            if power_prot_count > 50:
                print(f"\n  ⚠ ABORT: Loop running {expected/elapsed:.1f}× too fast")
                print(f"  Power protection likely fired. Robot may have gone limp.")
                print(f"  Saving {step} steps collected before abort.")
                q_actual_buf = q_actual_buf[:step]
                q_target_buf = q_target_buf[:step]
                t_actual_buf = t_actual_buf[:step]
                n_steps_saved = step
                break
        else:
            power_prot_count = 0

        # Safety clamp
        q_des = q_cmd[step]
        q_cl  = np.clip(q_des, q_now - MAX_CMD_ERR, q_now + MAX_CMD_ERR)
        n_clamped += (np.abs(q_des - q_cl) > 0.001).astype(np.int32)

        q_actual_buf[step] = q_now
        q_target_buf[step] = q_cl
        t_actual_buf[step] = elapsed

        send_cmd_fast(q_cl, kp_sdk=kp_sdk_full, use_tau_ff=True)

        if step > 0 and step % prog == 0:
            freq_now = f0 + (f1-f0)*step/n_steps
            w = min(step, CTRL_HZ)
            # Thigh tracking
            fl_act = float(np.max(np.abs(q_actual_buf[step-w:step,4]-DEFAULT_JOINT_POS[4])))
            fl_cmd = float(np.max(np.abs(q_target_buf[step-w:step,4]-DEFAULT_JOINT_POS[4])))
            fl_r   = fl_act / max(fl_cmd, 1e-3)
            # Hip stability
            hip_err = float(np.max(np.abs(q_now[0:4] - DEFAULT_JOINT_POS[0:4])))
            # Phase check at this frequency
            phase_deg = 360.0 * 0.016 * freq_now  # expected phase from 16ms delay
            status = ("⚠ RESONANCE — stop if robot vibrating" if fl_r > 1.2
                      else "✓ clean")
            print(f"  t={elapsed:5.1f}s  f={freq_now:.2f}Hz  "
                  f"FL_th_ratio={fl_r:.2f}  hip_err={hip_err:.3f}rad  "
                  f"exp_phase={phase_deg:.0f}°  {status}")

        deadline_sleep(t_step)
    else:
        n_steps_saved = n_steps

    total_wall = time.perf_counter() - t0_wall

    # Release
    print(f"\n── Release ──")
    release()

    # Loop rate check
    actual_dur = float(t_actual_buf[:n_steps_saved][-1] -
                       t_actual_buf[:n_steps_saved][0])
    actual_hz  = (n_steps_saved-1) / max(actual_dur, 1e-3)

    print(f"\n── Loop rate: {actual_hz:.0f}Hz  wall={total_wall:.1f}s  "
          f"steps={n_steps_saved}/{n_steps} ──")

    if actual_hz > 1000:
        print(f"  ⚠ {actual_hz:.0f}Hz — power protection fired, data invalid.")
        print(f"  Reduce amplitude further or check robot state.")
        if input("  Save anyway? [y/N] → ").lower() != 'y':
            return
    elif actual_hz < 350:
        print(f"  ⚠ {actual_hz:.0f}Hz too slow for delay identification.")
        if input("  Save anyway? [y/N] → ").lower() != 'y':
            return
    else:
        print(f"  ✓ {actual_hz:.0f}Hz OK")

    # Save
    out = output_dir / "chirp_data.pt"
    torch.save({
        "time":        torch.tensor(t_actual_buf[:n_steps_saved], dtype=torch.float32),
        "dof_pos":     torch.tensor(q_actual_buf[:n_steps_saved], dtype=torch.float32),
        "des_dof_pos": torch.tensor(q_target_buf[:n_steps_saved], dtype=torch.float32),
    }, out)
    print(f"  ✓ {out}  ({n_steps_saved} steps, {actual_dur:.1f}s)")

    # Motion ranges
    qa = q_actual_buf[:n_steps_saved]
    qt = q_target_buf[:n_steps_saved]
    print(f"\n── Motion ranges ──")
    print(f"  {'Joint':12s}  {'Role':>6s}  {'Actual':>8s}  "
          f"{'Cmd':>8s}  {'Ratio':>6s}  Note")
    print("  " + "-"*62)
    for j in range(12):
        a_r   = float(qa[:,j].max()-qa[:,j].min())
        c_r   = float(qt[:,j].max()-qt[:,j].min())
        ratio = a_r / max(c_r, 1e-3)
        role  = "fixed" if j < 4 else "active"
        if j < 4:
            note = ("hip stable ✓" if a_r < 0.08
                    else "⚠ hip moved (reaction force?)")
        elif j == 6:
            note = "RL_thigh stiction"
        elif ratio > 1.20:
            note = "⚠ resonance — amplitude too high"
        elif ratio > 0.80:
            note = "good tracking ✓"
        elif ratio > 0.40:
            note = "partial (inertia+delay signal) ✓"
        else:
            note = "⚠ barely moved"
        print(f"  {ISAAC_NAMES[j]:12s}  {role:>6s}  {a_r:8.3f}rad  "
              f"{c_r:8.3f}rad  {ratio:6.2f}  {note}")

    print(f"\n── Next steps ──")
    print(f"  scp {out} <sim>:~/pace-sim2real/data/go1/chirp_data.pt")
    print(f"  ~/IsaacLab/isaaclab.sh -p ~/pace-sim2real/scripts/pace/fit.py \\")
    print(f"    --task Isaac-Pace-Go1-v0 --num_envs 4096 --headless")
    print(f"\n  Target: Td→7-10 steps, thigh ratio 0.5-1.2, hip_err<0.08, score<0.025")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Go1 PACE v12 — safe small amplitudes, power-protection detection")
    p.add_argument("--duration",   type=float, default=30.0)
    p.add_argument("--min_freq",   type=float, default=0.1)
    p.add_argument("--max_freq",   type=float, default=5.0,
                   help="Max Hz (default 5.0, cap 5.0)")
    p.add_argument("--output_dir", type=str,   default="./pace_data/go1")
    main(p.parse_args())