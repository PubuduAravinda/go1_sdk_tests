#!/usr/bin/env python3
"""
Go1 Enhanced Calibration + Leg Independence Profiler  v2
==========================================================

Phase 0 (  0- 10s)  Passive natural rest — measure gravity drop
Phase 1 ( 10- 50s)  Uniform KP ramp 5→35
Phase 2 ( 50-120s)  Balance tuning:
                      - Per-joint KP boost driven by joint error AND trunk roll/pitch
                      - Roll/pitch tells us WHICH side needs more force
                      - Contact binary (tauEst) shows when all 4 feet grounded
Phase 3 (120-135s)  Final calibrated measurement + contact baseline
Phase 4 (135-215s)  Incremental liftoff profiling per leg (FR→FL→RR→RL):
                      - Ramp thigh up 0.01 rad/step until tauEst < threshold
                      - Record: liftoff delta, trunk roll/pitch change,
                                peer leg load redistribution
                      - 5s settle between legs

Output:
  KP_MULTIPLIER         per-joint KP multipliers for deploy.py
  DEFAULT_JOINT_POS     measured real standing pose
  KNEE_TAU_THRESHOLD    validated contact threshold
  Leg asymmetry profile weak / dominant / normal classification
"""

import time
import numpy as np
import robot_interface as sdk

# ─── TARGET (sim training defaults) ──────────────────────────────────────────
TARGET_DEFAULT = np.array([
    0.1,  0.1,  0.1,  0.1,
    0.8,  0.8,  0.8,  0.8,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

# ─── JOINT MAPPING ───────────────────────────────────────────────────────────
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

SDK_NAMES = ["FR_hip","FR_thigh","FR_knee",
             "FL_hip","FL_thigh","FL_knee",
             "RR_hip","RR_thigh","RR_knee",
             "RL_hip","RL_thigh","RL_knee"]
ISAAC_NAMES = [None] * 12
for si, ii in enumerate(sdk_to_isaac):
    ISAAC_NAMES[ii] = SDK_NAMES[si]

THIGH_I = {n.split("_")[0]: i for i, n in enumerate(ISAAC_NAMES) if n and "thigh" in n}
KNEE_I  = {n.split("_")[0]: i for i, n in enumerate(ISAAC_NAMES) if n and "knee"  in n}

KNEE_SDK  = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}
LEG_ORDER = ["FR", "FL", "RR", "RL"]

# ─── CONFIG ──────────────────────────────────────────────────────────────────
KP_BASE_MAX      = 35.0
KP_RAMP_TIME     = 40.0
KP_MAX_PER_JOINT = np.array([40,40,40,40, 70,70,70,70, 80,80,80,80], dtype=np.float32)
KD_PER_JOINT     = np.array([4.0]*4 + [4.5]*4 + [4.5]*4, dtype=np.float32)
THIGH_TAU_FF     = 1.2

BOOST_INTERVAL   = 3.0
BOOST_STEP_KP    = 3.0
ERROR_THRESH     = 0.05

TILT_BALANCE_DEG = 3.0   # trunk tilt deg that triggers side-specific boost
TILT_SAFETY_DEG  = 35.0

CONTACT_THRESH   = 1.0   # Nm knee tauEst threshold

LIFT_STEP_RAD    = 0.01  # rad per sweep step  (~2mm foot rise)
LIFT_STEP_DT     = 0.15  # s between steps
LIFT_MAX_RAD     = 0.40  # rad ceiling (was 0.25 — FR/RR/RL need >0.35 to liftoff when loaded)
SETTLE_TIME      = 5.0   # s settle between legs

# ─── SDK INIT ────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

kp_mult = np.ones(12, dtype=np.float32)
kp_mult[4:8] = 1.4   # thigh baseline

TARGET_SDK = TARGET_DEFAULT[isaac_to_sdk]
KD_SDK     = KD_PER_JOINT[isaac_to_sdk]


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def read_state():
    udp.Recv(); udp.GetRecv(state)
    q_sdk   = np.array([state.motorState[i].q for i in range(12)])
    q_isaac = q_sdk[sdk_to_isaac]
    tau_k   = {k: abs(state.motorState[idx].tauEst) for k, idx in KNEE_SDK.items()}
    contact = {k: tau_k[k] > CONTACT_THRESH for k in LEG_ORDER}
    gx = state.imu.accelerometer[0]
    gy = state.imu.accelerometer[1]
    gz = state.imu.accelerometer[2]
    gmag  = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-9
    tilt  = np.degrees(np.arccos(min(1.0, abs(gz) / gmag)))
    roll  = np.degrees(np.arctan2(gy, gz))
    pitch = np.degrees(np.arctan2(-gx, np.sqrt(gy**2 + gz**2)))
    return q_isaac, tau_k, contact, tilt, roll, pitch


def send_joints(kp_arr, target_arr, tau_ff_arr=None):
    kp_s  = kp_arr[isaac_to_sdk]
    tg_s  = target_arr[isaac_to_sdk]
    tf_s  = tau_ff_arr[isaac_to_sdk] if tau_ff_arr is not None else np.zeros(12)
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(tg_s[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(kp_s[i])
        cmd.motorCmd[i].Kd   = float(KD_SDK[i])
        cmd.motorCmd[i].tau  = float(tf_s[i])
    try:
        safe.PowerProtect(cmd, state, 9)
        udp.SetSend(cmd)
        udp.Send()
    except Exception:
        pass


def tau_ff():
    t = np.zeros(12, dtype=np.float32)
    t[4:8] = THIGH_TAU_FF
    return t


def feet_str(c):
    return "".join("●" if c[k] else "○" for k in LEG_ORDER)


def tau_row(tk):
    return "  ".join(f"{k}:{tk[k]:4.1f}" for k in LEG_ORDER)


# ─── BANNER ──────────────────────────────────────────────────────────────────
print("\n" + "="*74)
print("  Go1 Enhanced Calibration + Leg Independence Profiler  v2")
print("="*74)
print("  Ph0 (  0- 10s)  Passive rest measurement")
print("  Ph1 ( 10- 50s)  KP uniform ramp 5→35")
print("  Ph2 ( 50-120s)  Per-joint boost  (roll/pitch guided balance)")
print("  Ph3 (120-135s)  Final calibrated snapshot")
print("  Ph4 (135-215s)  Incremental liftoff: FR → FL → RR → RL")
print()
print("  Status columns: tilt  roll  pitch  feet●  tauEst[Nm]  conv/12")
print("="*74 + "\n")

# ─── DATA STORES ─────────────────────────────────────────────────────────────
passive_samples  = []
final_samples    = []
final_mean       = TARGET_DEFAULT.copy()   # live running mean, updated each phase-3 sample
contact_baseline = {k: [] for k in LEG_ORDER}
liftoff_results  = {}

phase        = 0
current_kp   = 5.0
ramp_t0      = None
last_boost_t = 0.0
t0           = time.time()

p4_leg_idx      = 0
p4_lift_delta   = 0.0
p4_liftoff_done = False
p4_stage        = "settle"
p4_stage_t      = None
p4_baseline_tau = {}
p4_baseline_rp  = (0.0, 0.0)

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
try:
    while True:
        time.sleep(0.002)
        t = time.time() - t0

        try:
            q_isaac, tau_k, contact, tilt, roll, pitch = read_state()
        except Exception:
            continue

        error = TARGET_DEFAULT - q_isaac

        # Phase transitions
        if   t < 10.0:  np_ = 0
        elif t < 50.0:  np_ = 1
        elif t < 120.0: np_ = 2
        elif t < 135.0: np_ = 3
        else:           np_ = 4

        if np_ > phase:
            phase = np_
            if phase == 1:
                ramp_t0 = time.time()
                print(f"\n[t={t:.0f}s] ── PHASE 1: KP ramp 5→35 ──")
            elif phase == 2:
                print(f"\n[t={t:.0f}s] ── PHASE 2: Per-joint boost + balance ──")
                print(f"  kp_mult: {np.round(kp_mult, 2)}")
            elif phase == 3:
                print(f"\n[t={t:.0f}s] ── PHASE 3: Final measurement ──")
            elif phase == 4:
                print(f"\n[t={t:.0f}s] ── PHASE 4: Liftoff profiling ──")
                p4_stage   = "settle"
                p4_stage_t = time.time()

        # Active KP
        if phase == 1:
            frac = min((time.time() - ramp_t0) / KP_RAMP_TIME, 1.0)
            current_kp = 5.0 + frac * (KP_BASE_MAX - 5.0)
        elif phase >= 2:
            current_kp = KP_BASE_MAX

        kp_arr = np.clip(current_kp * kp_mult, 0, KP_MAX_PER_JOINT)

        # ── Phase 0 ──────────────────────────────────────────────────────────
        if phase == 0:
            for i in range(12):
                cmd.motorCmd[i].mode = 0x0A; cmd.motorCmd[i].q   = 0.0
                cmd.motorCmd[i].dq   = 0.0;  cmd.motorCmd[i].Kp  = 0.0
                cmd.motorCmd[i].Kd   = 2.0;  cmd.motorCmd[i].tau = 0.0
            try:
                safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
            except Exception:
                pass
            passive_samples.append(q_isaac.copy())

        # ── Phase 1 + 2 ──────────────────────────────────────────────────────
        elif phase in (1, 2):
            if phase == 2 and (t - last_boost_t) > BOOST_INTERVAL and tilt < TILT_SAFETY_DEG:
                boosted = []
                for i in range(12):
                    if abs(error[i]) <= ERROR_THRESH:
                        continue
                    leg = ISAAC_NAMES[i].split("_")[0]
                    # Side-specific trunk bias
                    roll_bias  = (leg in ("FL","RL") and roll  >  TILT_BALANCE_DEG) or \
                                 (leg in ("FR","RR") and roll  < -TILT_BALANCE_DEG)
                    pitch_bias = (leg in ("FR","FL") and pitch >  TILT_BALANCE_DEG) or \
                                 (leg in ("RR","RL") and pitch < -TILT_BALANCE_DEG)
                    step = (BOOST_STEP_KP * 1.5 if (roll_bias or pitch_bias) else BOOST_STEP_KP)
                    max_m = KP_MAX_PER_JOINT[i] / current_kp
                    if kp_mult[i] < max_m:
                        kp_mult[i] = min(kp_mult[i] + step / current_kp, max_m)
                        tag = "↑BAL" if (roll_bias or pitch_bias) else ""
                        boosted.append(f"{ISAAC_NAMES[i]}(e={error[i]:+.2f}"
                                       f"→KP={kp_mult[i]*current_kp:.0f}{tag})")
                if boosted:
                    print(f"  [t={t:.0f}s] r={roll:+.1f}° p={pitch:+.1f}°  "
                          f"Boost: {', '.join(boosted)}")
                last_boost_t = t

            kp_arr = np.clip(current_kp * kp_mult, 0, KP_MAX_PER_JOINT)
            send_joints(kp_arr, TARGET_DEFAULT.copy(), tau_ff())

        # ── Phase 3 ──────────────────────────────────────────────────────────
        elif phase == 3:
            send_joints(kp_arr, TARGET_DEFAULT.copy(), tau_ff())
            final_samples.append(q_isaac.copy())
            final_mean = np.mean(final_samples, axis=0)   # update live — available to phase 4
            for k in LEG_ORDER:
                contact_baseline[k].append(tau_k[k])

        # ── Phase 4 ──────────────────────────────────────────────────────────
        elif phase == 4:
            if p4_leg_idx >= len(LEG_ORDER):
                break

            leg   = LEG_ORDER[p4_leg_idx]
            th_i  = THIGH_I[leg]
            kn_i  = KNEE_I[leg]
            now   = time.time()
            target_now = TARGET_DEFAULT.copy()

            if p4_stage == "settle":
                # Use calibrated DEFAULT (not sim TARGET) so feet start grounded
                calibrated_default = final_mean if len(final_samples) > 0 else TARGET_DEFAULT
                target_now = calibrated_default.copy()
                send_joints(kp_arr, target_now, tau_ff())
                if now - p4_stage_t > SETTLE_TIME:
                    p4_baseline_tau = {k: tau_k[k] for k in LEG_ORDER}
                    p4_baseline_rp  = (roll, pitch)
                    p4_lift_delta   = 0.0
                    p4_liftoff_done = False
                    liftoff_results[leg] = {
                        "liftoff_delta": None, "liftoff_tau": None,
                        "droll": None, "dpitch": None,
                        "peer_delta": {},
                        "force_at_max": None,   # tau at max delta if no liftoff (dominance proxy)
                        "droll_at_max": None, "dpitch_at_max": None,
                        "sweep": [],   # (delta, tau, roll, pitch, contact_bool)
                    }
                    p4_stage   = "sweep"
                    p4_stage_t = now
                    print(f"\n  ── {leg}: sweep start (max δ={LIFT_MAX_RAD:.2f} rad) ──")
                    print(f"  Baseline tau: {tau_row(p4_baseline_tau)}  "
                          f"r={roll:+.1f}° p={pitch:+.1f}°")

            elif p4_stage == "sweep":
                target_now[th_i] = TARGET_DEFAULT[th_i] - p4_lift_delta
                target_now[kn_i] = TARGET_DEFAULT[kn_i] + p4_lift_delta * 0.5
                send_joints(kp_arr, target_now, tau_ff())

                r = liftoff_results[leg]
                r["sweep"].append((
                    round(p4_lift_delta, 3),
                    round(tau_k[leg], 3),
                    round(roll, 2),
                    round(pitch, 2),
                    bool(contact[leg]),
                ))

                # Liftoff event
                if not contact[leg] and not p4_liftoff_done:
                    p4_liftoff_done = True
                    r["liftoff_delta"] = round(p4_lift_delta, 3)
                    r["liftoff_tau"]   = round(tau_k[leg], 3)
                    r["droll"]         = round(roll  - p4_baseline_rp[0], 2)
                    r["dpitch"]        = round(pitch - p4_baseline_rp[1], 2)
                    r["peer_delta"]    = {k: round(tau_k[k] - p4_baseline_tau[k], 2)
                                          for k in LEG_ORDER if k != leg}
                    print(f"  ★ {leg} LIFTOFF  delta={r['liftoff_delta']:.3f}rad  "
                          f"tau={r['liftoff_tau']:.2f}Nm  "
                          f"Δroll={r['droll']:+.2f}°  Δpitch={r['dpitch']:+.2f}°")
                    print(f"    Peer Δ: "
                          + "  ".join(f"{k}:{v:+.1f}Nm" for k, v in r["peer_delta"].items()))

                # Step advance
                if now - p4_stage_t > LIFT_STEP_DT:
                    if not p4_liftoff_done:
                        p4_lift_delta += LIFT_STEP_RAD
                    if p4_lift_delta > LIFT_MAX_RAD or p4_liftoff_done:
                        # Record force at max delta for stuck legs
                        if not p4_liftoff_done:
                            r["force_at_max"]   = round(tau_k[leg], 3)
                            r["droll_at_max"]   = round(roll  - p4_baseline_rp[0], 2)
                            r["dpitch_at_max"]  = round(pitch - p4_baseline_rp[1], 2)
                            peer_d = {k: round(tau_k[k]-p4_baseline_tau[k],2)
                                      for k in LEG_ORDER if k!=leg}
                            print(f"  ✗ {leg} MAX DELTA reached — no liftoff. "
                                  f"tau@max={r['force_at_max']:.2f}Nm  "
                                  f"Δroll={r['droll_at_max']:+.2f}°  Δpitch={r['dpitch_at_max']:+.2f}°")
                            print(f"    Peer Δ at max: "
                                  + "  ".join(f"{k}:{v:+.1f}Nm" for k,v in peer_d.items()))
                        p4_stage   = "hold"
                        p4_stage_t = now
                    else:
                        p4_stage_t = now

            elif p4_stage == "hold":
                target_now[th_i] = TARGET_DEFAULT[th_i] - p4_lift_delta
                target_now[kn_i] = TARGET_DEFAULT[kn_i] + p4_lift_delta * 0.5
                send_joints(kp_arr, target_now, tau_ff())
                if now - p4_stage_t > 2.0:
                    print(f"  {leg} complete. Next leg settling...")
                    p4_leg_idx += 1
                    p4_stage    = "settle"
                    p4_stage_t  = now

        # ── Status every 2s ──────────────────────────────────────────────────
        if int(t * 0.5) != int((t - 0.002) * 0.5):
            conv = np.sum(np.abs(error) < ERROR_THRESH)
            wi   = np.argmax(np.abs(error))
            if phase == 4 and p4_leg_idx < len(LEG_ORDER):
                ph_lbl = f"LIFT={LEG_ORDER[p4_leg_idx]}({p4_stage}) δ={p4_lift_delta:.2f}"
            else:
                ph_lbl = f"ph={phase}"
            print(f"t={t:6.1f}s | {ph_lbl} | KP={current_kp:.0f} | "
                  f"tilt={tilt:.1f}° r={roll:+.1f}° p={pitch:+.1f}° | "
                  f"feet {feet_str(contact)} [{tau_row(tau_k)} Nm] | "
                  f"conv={conv}/12 | worst={ISAAC_NAMES[wi]}={error[wi]:+.3f}")

except KeyboardInterrupt:
    print("\nAborted — printing partial results.")

# ─── FINAL RESULTS ───────────────────────────────────────────────────────────
print("\n" + "="*74)
print("CALIBRATION RESULTS")
print("="*74)

passive_mean = np.mean(passive_samples, axis=0) if passive_samples else np.zeros(12)
final_mean   = np.mean(final_samples,   axis=0) if final_samples   else np.zeros(12)
final_std    = np.std(final_samples,    axis=0) if final_samples   else np.zeros(12)
cb_mean      = {k: float(np.mean(v)) if v else 0.0 for k, v in contact_baseline.items()}

print(f"\n{'#':>2}  {'Joint':>14}  {'Target':>7}  {'Passive':>8}  {'Final±Std':>12}"
      f"  {'Error':>7}  {'KP_mult':>8}  {'EffKP':>6}")
for i in range(12):
    err  = TARGET_DEFAULT[i] - final_mean[i]
    flag = "✓" if abs(err) < ERROR_THRESH else ("~" if abs(err) < 0.15 else "⚠")
    print(f"{i:>2}  {ISAAC_NAMES[i]:>14}  {TARGET_DEFAULT[i]:>7.3f}  "
          f"{passive_mean[i]:>8.3f}  "
          f"{final_mean[i]:>6.3f}±{final_std[i]:.3f}  "
          f"{err:>+7.3f}  {kp_mult[i]:>8.2f}  "
          f"{kp_mult[i]*KP_BASE_MAX:>6.1f}  {flag}")

print(f"\nStanding contact baseline (tauEst at KP=35, threshold={CONTACT_THRESH:.1f} Nm):")
max_t = max(cb_mean.values()) + 0.1
for k in LEG_ORDER:
    bar = "█" * int(cb_mean[k])
    pct = cb_mean[k] / max_t * 100
    print(f"  {k}: {cb_mean[k]:5.2f} Nm  {bar}  ({pct:.0f}% of max)")

# ─── PHASE 4 ─────────────────────────────────────────────────────────────────
if liftoff_results:
    print(f"\n{'='*74}")
    print("LEG INDEPENDENCE / ASYMMETRY PROFILE")
    print(f"{'='*74}\n")

    print(f"{'Leg':>4}  {'δ_liftoff':>10}  {'tau@off':>8}  "
          f"{'Δroll':>7}  {'Δpitch':>8}  {'Peer Δ (Nm)'}")
    print("─"*74)
    for leg in LEG_ORDER:
        r = liftoff_results.get(leg)
        if not r:
            print(f"{leg:>4}  (no data)"); continue
        if r["liftoff_delta"] is not None:
            peers = "  ".join(f"{k}:{v:+.1f}" for k, v in r["peer_delta"].items())
            print(f"{leg:>4}  {r['liftoff_delta']:>10.3f}  {r['liftoff_tau']:>8.3f}  "
                  f"{r['droll']:>+7.2f}°  {r['dpitch']:>+8.2f}°  {peers}")
        elif r.get("force_at_max") is not None:
            print(f"{leg:>4}  {'STUCK':>10}  "
                  f"tau@δ={LIFT_MAX_RAD:.2f}={r['force_at_max']:5.2f}Nm  "
                  f"Δr={r['droll_at_max']:>+.1f}°  Δp={r['dpitch_at_max']:>+.1f}°"
                  f"  (dominant leg — needs >{LIFT_MAX_RAD:.2f}rad)")
        else:
            print(f"{leg:>4}  (incomplete)")

    print("""
Key features:
  δ_liftoff     small (<0.04 rad) = barely grounded → weak/short leg
                large (>0.10 rad) = well grounded → healthy
  tau@off       tauEst at liftoff (should be near threshold=1.0 Nm)
  Δroll/pitch   trunk tilt change when this leg unweights
                large = PRIMARY support leg  |  small = low contributor
  Peer Δ        load shift to other legs at liftoff
  STUCK legs    tau@max = residual load at δ=0.40 rad — high = DOMINANT
""")

    print("── Leg classification ──")
    for leg in LEG_ORDER:
        r = liftoff_results.get(leg)
        if not r:
            print(f"  {leg}: (no data)"); continue
        if r["liftoff_delta"] is not None:
            tilt_response = abs(r.get("droll") or 0) + abs(r.get("dpitch") or 0)
            delta = r["liftoff_delta"]
            if delta < 0.04:
                label = "WEAK/SHORT  — barely grounded at rest"
            elif tilt_response > 8.0:
                label = "DOMINANT    — primary support leg"
            elif tilt_response < 2.0 and delta < 0.08:
                label = "MARGINAL    — low contribution, asymmetric"
            else:
                label = "NORMAL      — healthy contribution"
            print(f"  {leg}: {label}  (δ={delta:.3f}rad  |Δtilt|={tilt_response:.1f}°)")
        elif r.get("force_at_max") is not None:
            tilt_r = abs(r.get("droll_at_max") or 0) + abs(r.get("dpitch_at_max") or 0)
            print(f"  {leg}: DOMINANT/STUCK  "
                  f"tau@max={r['force_at_max']:.2f}Nm  |Δtilt|={tilt_r:.1f}°  "
                  f"(needs thigh scale >0.40 or unloadable)")
        else:
            print(f"  {leg}: STUCK — foot never lifted")

# ─── DEPLOY PASTE ─────────────────────────────────────────────────────────────
print(f"\n{'='*74}")
print(">>> Paste into go1_deploy.py <<<")
print(f"{'='*74}")
h  = final_mean[0:4]
th = final_mean[4:8]
kn = final_mean[8:12]
print(f"\nKP_MULTIPLIER = np.array([")
print(f"    {kp_mult[0]:.2f}, {kp_mult[1]:.2f}, {kp_mult[2]:.2f}, {kp_mult[3]:.2f},   # hips")
print(f"    {kp_mult[4]:.2f}, {kp_mult[5]:.2f}, {kp_mult[6]:.2f}, {kp_mult[7]:.2f},  # thighs")
print(f"    {kp_mult[8]:.2f}, {kp_mult[9]:.2f}, {kp_mult[10]:.2f}, {kp_mult[11]:.2f},  # knees")
print(f"], dtype=np.float32)\n")
print(f"DEFAULT_JOINT_POS = np.array([")
print(f"    {h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}, {h[3]:.3f},   # hips")
print(f"    {th[0]:.3f}, {th[1]:.3f}, {th[2]:.3f}, {th[3]:.3f},  # thighs")
print(f"    {kn[0]:.3f}, {kn[1]:.3f}, {kn[2]:.3f}, {kn[3]:.3f},  # knees")
print(f"], dtype=np.float32)\n")
print(f"KNEE_TAU_THRESHOLD = {CONTACT_THRESH:.1f}  # Nm (validated this session)")
print("="*74)