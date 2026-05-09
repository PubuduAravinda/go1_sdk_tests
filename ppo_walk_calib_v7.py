#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — model_15000 (v8 go1_env.py — air-time trot reward)
policy.pt interface: raw_obs(45) → delta(12)  [normalise + MLP + tanh baked in]
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

PURPOSE: Confirm air-time trot reward (v8) produces real knee lift on hardware.
  Expected visible behaviour:
    FR_kn: lifts ~0.15-0.22 rad above default (46% of swing time in sim)
    RL_kn: lifts ~0.10-0.16 rad above default (75% of swing time in sim)
    RR_kn: lifts ~0.13-0.25 rad above default (55% of swing time in sim)
    FL_kn: minimal lift — FL in swing only 24% (asymmetric gait, expected)
  Gait: 2.68 Hz in sim → expect ~2.0-2.5 Hz on real hardware (delay adds ~15%)
  Tilt: 7.5° mean sim → expect 10-14° on real hardware

WHY model_15000 OVER model_24999:
  model_24999: lower tilt (6.7°), truer 2Hz gait, but roll=0.779 rad/s mean
               → high lateral oscillation risk on hardware
  model_15000: pitch=0.433 (safest), RL_kn lift 75%, zero FR_th binding events,
               59 RL_th stiction spikes (manageable vs 7500's 138)
  model_7500:  DISQUALIFIED — 138 RL_th spikes + FR_th >0.820 on 42 steps

CHANGES FROM model_24000 DEPLOY SCRIPT:
  1. Policy path → go1_deploy_model15000/policy.pt
  2. MAX_DELTA knees: 0.08 → 0.10  (sim FR_kn p99=0.396, RR_kn p99=0.387)
     Allows full knee lift to execute. At 500Hz: 0.10×10steps=1.0 rad max/policy-step
  3. Console: knee lift printed for all 4 legs (was FL+RR only)
  4. Log summary: reports which legs are lifting (confirms reward working)

UNCHANGED FROM model_24000 DEPLOY SCRIPT:
  ✓ DELTA_LO/HI: hip ±0.08, thigh/knee ±0.35
  ✓ KD: hip=4.0  FR_th=8.0  other_th=4.5  knee=5.0  (training nominal)
  ✓ FR_TH_BINDING_MAX = 0.800 rad (deploy cap, train cap=0.820)
  ✓ TAU feedforward: RL_th only = 1.2 Nm (stiction pre-load)
  ✓ HIP_SCALE=1.00  THIGH_SCALE=1.00  KNEE_SCALE=1.00
  ✓ sdk_to_isaac / isaac_to_sdk remapping
  ✓ Hip sign flips: obs[4]=-obs[4], obs[6]=-obs[6]
  ✓ jdelta_offset from HOLD stage
  ✓ prev_delta updated whenever ready (not % 10)

Observation layout (45D, Isaac order):
  [0:3]   cmd [vx, vy, wz]
  [3:15]  jpos - default_q - jdelta_offset    (hw sign-flipped FR/RR hip)
  [15:27] jvel clipped ±5
  [27:30] gyro clipped ±5
  [30:33] proj_gravity = -acc/|acc|
  [33:45] prev_delta  (policy output t-1, pre hw-flip)

SDK motor order: SDK[0]=FR_hip [1]=FR_th [2]=FR_kn
                  SDK[3]=FL_hip [4]=FL_th [5]=FL_kn
                  SDK[6]=RR_hip [7]=RR_th [8]=RR_kn
                  SDK[9]=RL_hip [10]=RL_th [11]=RL_kn
  sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]
"""

import time
import threading
from datetime import datetime
import numpy as np
import torch
import robot_interface as sdk

# ── tanh bounds — MUST match go1_env.py v8 _delta_soft_lo / _delta_soft_hi ──
DELTA_LO = np.array([
    -0.08, -0.08, -0.08, -0.08,
    -0.35, -0.35, -0.35, -0.35,
    -0.35, -0.35, -0.35, -0.35,
], dtype=np.float32)
DELTA_HI = np.array([
     0.08,  0.08,  0.08,  0.08,
     0.35,  0.35,  0.35,  0.35,
     0.35,  0.35,  0.35,  0.35,
], dtype=np.float32)

_T_MID  = (DELTA_HI + DELTA_LO) * 0.5
_T_HALF = (DELTA_HI - DELTA_LO) * 0.5

def delta_to_raw_net(delta):
    ratio = np.clip((delta - _T_MID) / _T_HALF, -0.9999, 0.9999)
    return np.arctanh(ratio)

# ── Robot constants ───────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,
    0.8,  0.8,  0.8,  0.8,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

# FR_th binding cap — same as model_24000 deploy
# Train cap: 0.820, deploy cap 0.800 = extra 0.020 margin for overshoot
FR_TH_BINDING_MAX = 0.800

JVEL_CLIP   = 5.0
ANGVEL_CLIP = 5.0

# ── KP / KD ───────────────────────────────────────────────────────────────────
KP_START       = 5.0
KP_STEP        = 3.0
RAMP_MAX_LEVEL = 10         # KP_BASE = 35 Nm/rad
KP_MULTIPLIER  = np.array([
    1.000, 1.000, 1.000, 1.000,   # hips   → KP = 35
    1.857, 1.857, 1.857, 1.857,   # thighs → KP = 65
    2.286, 2.286, 2.286, 2.286,   # knees  → KP = 80
], dtype=np.float32)

# KD in Isaac order — training nominal values
# FR_th[5]=8.0: intentional mismatch — slows approach to binding zone
KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,
    4.5, 8.0, 4.5, 4.5,
    5.0, 5.0, 5.0, 5.0,
], dtype=np.float32)

# Feedforward torque — RL_th only (stiction pre-load)
# RL_th τf = 4.944 Nm fault. 1.2 Nm partial pre-load to help swing initiation.
TAU_PER_JOINT_ISAAC = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.2, 0.0,   # only RL_th[idx6]
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

HIP_SCALE   = 1.00
THIGH_SCALE = 1.00
KNEE_SCALE  = 1.00
HIP_OBS_CORRECTION = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# ── Velocity command ──────────────────────────────────────────────────────────
VX_TARGET   = 0.5
VX_KICK     = 0.6
VX_RAMP_S   = 0.3
VX_KICK_S   = 1.0
VX_SETTLE_S = 1.0

# ── Safety ────────────────────────────────────────────────────────────────────
TILT_THRESHOLD = 20.0
TILT_STOP_DEG  = 30.0
HOLD_RAMP_S    = 4.0
HOLD_FULL_S    = 3.0
INFERENCE_HZ   = 50
CONTROL_HZ     = 500

# Per-joint rate limiter at 500Hz
# Calibrated from sim p99 per-policy-step delta changes (model_15000):
#   hips:   p99 ≤ 0.083  → 0.04/500Hz-step (allows 0.40 over one policy period)
#   thighs: p99 ≤ 0.171  → 0.08/step       (allows 0.80 over one policy period)
#   knees:  p99 ≤ 0.396  → 0.10/step       (allows 1.00 → full knee lift OK)
# Knee limit widened from 0.08 to 0.10 vs model_24000:
#   Needed for FR_kn (p99=0.396) and RR_kn (p99=0.387) to execute full lift.
#   With old 0.08: knee lift would be clipped to ~80% of commanded range.
MAX_DELTA_PER_JOINT = np.array([
    0.04, 0.04, 0.04, 0.04,   # hips
    0.08, 0.08, 0.08, 0.08,   # thighs (RL_th stiction spikes handled here)
    0.10, 0.10, 0.10, 0.10,   # knees — wider for air-time knee lift
], dtype=np.float32)

# ── Joint remapping ───────────────────────────────────────────────────────────
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i

_JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
           'FL_th', 'FR_th', 'RL_th', 'RR_th',
           'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

# ── Load policy ───────────────────────────────────────────────────────────────
device = torch.device("cpu")
policy = torch.jit.load("go1_deploy_re_calib_v6/policy.pt").to(device).eval()
print("[POLICY] Loaded go1_deploy_re_calib_v5/policy.pt  (v8 air-time trot)")

with torch.no_grad():
    _test  = policy(torch.zeros(1, 45))
    _hip_r = _test[0, :4].numpy()
    _fr_r  = _test[0, 5].item()
    _kn_r  = _test[0, 8:12].numpy()

print(f"[POLICY] Hip delta at rest (expect ≈0.0): {_hip_r.round(4)}")
print(f"[POLICY] FR_th delta at rest: {_fr_r:.4f} → target "
      f"{DEFAULT_JOINT_POS[5]+_fr_r:.4f} rad  cap={FR_TH_BINDING_MAX:.3f}")
print(f"[POLICY] Knee deltas at rest: {_kn_r.round(4)}")
print(f"         FL_kn={_kn_r[0]:+.4f}  FR_kn={_kn_r[1]:+.4f}  "
      f"RL_kn={_kn_r[2]:+.4f}  RR_kn={_kn_r[3]:+.4f}")
if np.any(np.abs(_hip_r) > 0.06):
    print("[POLICY] WARNING: large hip delta at rest — check DELTA_LO/HI!")
if abs(_fr_r) > 0.20:
    print("[POLICY] WARNING: large FR_th delta at rest — check normalizer!")

# ── Shared state ──────────────────────────────────────────────────────────────
obs_lock         = threading.Lock()
action_lock      = threading.Lock()
shutdown_event   = threading.Event()
_shared_obs      = np.zeros(45, dtype=np.float32)
_shared_delta    = np.zeros(12, dtype=np.float32)
_inference_ready = False


def inference_thread_fn():
    global _shared_delta, _inference_ready
    period = 1.0 / INFERENCE_HZ
    while not shutdown_event.is_set():
        t0 = time.time()
        with obs_lock:
            obs_snap = _shared_obs.copy()
        try:
            obs_t = torch.from_numpy(obs_snap).float().unsqueeze(0)
            with torch.no_grad():
                delta_out = policy(obs_t).squeeze(0).cpu().numpy()
            with action_lock:
                _shared_delta[:] = delta_out
                _inference_ready  = True
        except Exception as e:
            print(f"[INFERENCE ERROR] {e}", flush=True)
        sl = period - (time.time() - t0)
        if sl > 0:
            time.sleep(sl)


# ── SDK setup ─────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*72)
print("Go1 model_15000 | v8 air-time trot | knee lift test")
print(f"  KP: hip=35  thigh=65  knee=80  Nm/rad")
print(f"  KD: hip=4.0  FR_th=8.0  other_th=4.5  knee=5.0")
print(f"  FR_th cap: ≤{FR_TH_BINDING_MAX:.3f} rad")
print(f"  TAU feedforward: RL_th = 1.2 Nm only")
print(f"  MAX_DELTA knees: 0.10 rad/500Hz-step (wider for lift)")
print(f"  Sim expected: FR_kn lift 46%, RL_kn 75%, RR_kn 55%")
print(f"  Watch for visible knee clearance during swing phase")
print("  Place on flat ground. Starting in 10s.")
print("="*72 + "\n")
time.sleep(10)

# ════════════════════════════════════════════════════════════════════════════
# HOLD STAGE
# ════════════════════════════════════════════════════════════════════════════
print("[HOLD] Ramping to training gains...")
_hold_t0   = time.perf_counter()
_hold_step = 0
KP_FULL    = KP_START + RAMP_MAX_LEVEL * KP_STEP   # 35.0

while True:
    _t  = time.perf_counter()
    _dt = _t - _hold_t0
    if _dt >= HOLD_RAMP_S + HOLD_FULL_S:
        break
    udp.Recv(); udp.GetRecv(state)
    _alpha  = min(1.0, _dt / HOLD_RAMP_S)
    _kp_now = KP_START + _alpha * (KP_FULL - KP_START)
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(_kp_now * KP_MULTIPLIER[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd   = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau  = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()
    if _hold_step % (CONTROL_HZ // 2) == 0:
        _jp  = np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac]
        _err = float(np.max(np.abs(_jp - DEFAULT_JOINT_POS)))
        print(f"[HOLD t={_dt:4.1f}s] KP_base={_kp_now:.1f}  max_err={_err:.3f} rad", flush=True)
    _hold_step += 1
    _sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - _t)
    if _sl > 0: time.sleep(_sl)

# Measure equilibrium offset
_eq = []
for _ in range(20):
    udp.Recv(); udp.GetRecv(state)
    _eq.append(np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac])
    time.sleep(0.005)
jpos_eq       = np.mean(_eq, axis=0)
jdelta_offset = jpos_eq - DEFAULT_JOINT_POS
max_err       = float(np.max(np.abs(jdelta_offset)))

print(f"\n[HOLD COMPLETE] max_err={max_err:.3f} rad")
print("  jdelta_offset:")
for ji, jn in enumerate(_JNAMES):
    print(f"    {jn:8s}: {jdelta_offset[ji]:+.4f}")
print(f"\n  FR_th offset: {jdelta_offset[5]:+.4f}  RL_th offset: {jdelta_offset[6]:+.4f}")
if max_err > 0.20:
    print("  WARNING: err > 0.20 rad — Ctrl+C in 3s to abort.")
    time.sleep(3.0)
else:
    print("  OK — proceeding.\n")

# ════════════════════════════════════════════════════════════════════════════
# Start inference thread
# ════════════════════════════════════════════════════════════════════════════
inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
inf_thread.start()

_t_wait = time.perf_counter()
while not _inference_ready:
    udp.Recv(); udp.GetRecv(state)
    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = float(KP_FULL * KP_MULTIPLIER[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
    if time.perf_counter() - _t_wait > 0.5:
        print("[WARN] inference not ready after 0.5s"); break
    time.sleep(0.001)

# ════════════════════════════════════════════════════════════════════════════
# WARMUP: 50 silent steps to seed prev_delta
# ════════════════════════════════════════════════════════════════════════════
prev_delta = np.zeros(12, np.float32)

for _ws in range(50):
    udp.Recv(); udp.GetRecv(state)
    _jp   = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
    _jv   = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
    _acc  = np.array(state.imu.accelerometer, np.float32)
    _gyro = np.array(state.imu.gyroscope,     np.float32)
    _na   = max(float(np.linalg.norm(_acc)), 0.1)
    _obs = np.zeros(45, np.float32)
    _obs[0:3]   = [0.0, 0.0, 0.0]
    _obs[3:15]  = (_jp - DEFAULT_JOINT_POS) - jdelta_offset
    _obs[4]     = -_obs[4]
    _obs[6]     = -_obs[6]
    _obs[3:7]  -= HIP_OBS_CORRECTION
    _obs[15:27] = np.clip(_jv,   -JVEL_CLIP,   JVEL_CLIP)
    _obs[27:30] = np.clip(_gyro, -ANGVEL_CLIP, ANGVEL_CLIP)
    _obs[30:33] = -_acc / _na
    _obs[33:45] = prev_delta
    with obs_lock: _shared_obs[:] = _obs
    time.sleep(1.0 / INFERENCE_HZ)
    with action_lock:
        if _inference_ready: prev_delta[:] = _shared_delta.copy()
    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = float(KP_FULL * KP_MULTIPLIER[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()

print(f"[WARMUP] Done. prev_delta knees: "
      f"FL={prev_delta[8]:+.4f}  FR={prev_delta[9]:+.4f}  "
      f"RL={prev_delta[10]:+.4f}  RR={prev_delta[11]:+.4f}")
print(f"[WARMUP] Hip prev_delta: {prev_delta[:4].round(4)}\n", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Logging setup
# ════════════════════════════════════════════════════════════════════════════
_LOG_STEPS     = 1500
_log_step      = 0
_log_obs       = np.zeros((_LOG_STEPS, 45),  np.float32)
_log_raw_net   = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_tanh      = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_target    = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_actual    = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_actual_qd = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_grav      = np.zeros((_LOG_STEPS, 3),   np.float32)
_log_angvel    = np.zeros((_LOG_STEPS, 3),   np.float32)
_log_tilt      = np.zeros(_LOG_STEPS,        np.float32)
_log_contact   = np.zeros((_LOG_STEPS, 4),   np.float32)
_last_log_ctrl = -10
_log_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

KNEE_IDX_SDK       = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}
KNEE_TAU_THRESHOLD = 1.0

# ════════════════════════════════════════════════════════════════════════════
# Control loop — 500Hz
# ════════════════════════════════════════════════════════════════════════════
step_counter       = 0
t0_global          = time.time()
tilt_exceeded_t    = None
prev_target_q      = DEFAULT_JOINT_POS.copy()
current_kp         = KP_FULL
_fr_th_clamp_count = 0
_track_err_sum     = np.zeros(12, np.float32)
_track_err_n       = 0

print("[CONTROL] 500Hz loop started. Watch for knee lift during swing.\n")

try:
    while True:
        t_loop    = time.time()
        step_counter += 1
        t_elapsed = t_loop - t0_global

        try:
            udp.Recv(); udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP ERROR] {e}", flush=True); break

        joint_pos = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
        joint_vel = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]

        acc          = np.array(state.imu.accelerometer, np.float32)
        gyro         = np.array(state.imu.gyroscope,     np.float32)
        norm_a       = max(float(np.linalg.norm(acc)), 0.1)
        proj_gravity = -acc / norm_a
        tilt_deg     = float(np.degrees(np.sqrt(proj_gravity[0]**2 + proj_gravity[1]**2)))
        tilt_scale   = 0.5 if tilt_deg > TILT_THRESHOLD else 1.0

        if tilt_deg > TILT_STOP_DEG:
            if tilt_exceeded_t is None: tilt_exceeded_t = time.time()
            elif time.time() - tilt_exceeded_t > 0.3:
                print(f"[SAFETY STOP] Tilt {tilt_deg:.1f}°", flush=True); break
        else:
            tilt_exceeded_t = None

        # ── Build observation ──────────────────────────────────────────────
        if   t_elapsed < VX_RAMP_S:
            _cmd_vx = VX_KICK * (t_elapsed / VX_RAMP_S)
        elif t_elapsed < VX_RAMP_S + VX_KICK_S:
            _cmd_vx = VX_KICK
        elif t_elapsed < VX_RAMP_S + VX_KICK_S + VX_SETTLE_S:
            _cmd_vx = VX_KICK + (VX_TARGET - VX_KICK) * (
                t_elapsed - (VX_RAMP_S + VX_KICK_S)) / VX_SETTLE_S
        else:
            _cmd_vx = VX_TARGET

        obs = np.zeros(45, dtype=np.float32)
        obs[0:3]   = [_cmd_vx, 0.0, 0.0]
        obs[3:15]  = (joint_pos - DEFAULT_JOINT_POS) - jdelta_offset
        obs[4]     = -obs[4]
        obs[6]     = -obs[6]
        obs[3:7]  -= HIP_OBS_CORRECTION
        obs[15:27] = np.clip(joint_vel, -JVEL_CLIP,   JVEL_CLIP)
        obs[27:30] = np.clip(gyro,      -ANGVEL_CLIP, ANGVEL_CLIP)
        obs[30:33] = proj_gravity
        obs[33:45] = prev_delta
        _obs_snapshot = obs.copy()

        with obs_lock: _shared_obs[:] = obs
        with action_lock:
            ready       = _inference_ready
            delta_isaac = _shared_delta.copy() if ready else np.zeros(12, np.float32)

        if ready: prev_delta[:] = delta_isaac.copy()

        if ready:
            delta_hw      = delta_isaac.copy()
            delta_hw     *= tilt_scale
            delta_hw[:4] *= HIP_SCALE
            delta_hw[4:8]*= THIGH_SCALE
            delta_hw[8:] *= KNEE_SCALE
            delta_hw[1]   = -delta_hw[1]
            delta_hw[3]   = -delta_hw[3]
            raw_target    = DEFAULT_JOINT_POS + delta_hw
        else:
            raw_target = DEFAULT_JOINT_POS.copy()

        # FR_th binding cap
        if raw_target[5] > FR_TH_BINDING_MAX:
            raw_target[5] = FR_TH_BINDING_MAX
            _fr_th_clamp_count += 1

        # Rate limiter
        target_q = np.clip(raw_target,
                           prev_target_q - MAX_DELTA_PER_JOINT,
                           prev_target_q + MAX_DELTA_PER_JOINT)
        prev_target_q[:] = target_q

        if ready:
            _track_err_sum += np.abs(joint_pos - target_q)
            _track_err_n   += 1

        # Send
        target_q_sdk = target_q[isaac_to_sdk]
        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q    = float(target_q_sdk[i])
            cmd.motorCmd[i].dq   = 0.0
            cmd.motorCmd[i].Kp   = float(current_kp * KP_MULTIPLIER[isaac_to_sdk[i]])
            cmd.motorCmd[i].Kd   = float(KD_PER_JOINT[isaac_to_sdk[i]])
            cmd.motorCmd[i].tau  = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
        try:
            safe.PowerProtect(cmd, state, 9)
            udp.SetSend(cmd); udp.Send()
        except Exception as e:
            print(f"[UDP SEND ERROR] {e}", flush=True); break

        # Log at 50Hz
        if ready and (step_counter - _last_log_ctrl) >= 10 and _log_step < _LOG_STEPS:
            _last_log_ctrl = step_counter
            _qd = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
            _kn = [abs(state.motorState[KNEE_IDX_SDK[k]].tauEst)
                   for k in ["FL", "FR", "RL", "RR"]]
            _log_obs[_log_step]       = _obs_snapshot
            _log_tanh[_log_step]      = delta_isaac
            _log_raw_net[_log_step]   = delta_to_raw_net(delta_isaac)
            _log_target[_log_step]    = target_q
            _log_actual[_log_step]    = joint_pos
            _log_actual_qd[_log_step] = _qd
            _log_grav[_log_step]      = proj_gravity
            _log_angvel[_log_step]    = gyro
            _log_tilt[_log_step]      = tilt_deg
            _log_contact[_log_step]   = _kn
            _log_step += 1

        # ── Console every 0.2s (100 control steps) ────────────────────────
        if step_counter % 100 == 1:
            _knt  = {k: abs(state.motorState[idx].tauEst)
                     for k, idx in KNEE_IDX_SDK.items()}
            _feet = "".join("●" if _knt[k] > KNEE_TAU_THRESHOLD else "○"
                            for k in ["FR", "FL", "RR", "RL"])
            # Knee lift relative to default — confirms air-time reward effect
            _fl_lift = joint_pos[8]  - DEFAULT_JOINT_POS[8]
            _fr_lift = joint_pos[9]  - DEFAULT_JOINT_POS[9]
            _rl_lift = joint_pos[10] - DEFAULT_JOINT_POS[10]
            _rr_lift = joint_pos[11] - DEFAULT_JOINT_POS[11]
            print(f"t={t_elapsed:5.1f}s | "
                  f"tilt={tilt_deg:.1f}° | feet(FR FL RR RL)={_feet} | "
                  f"cmd={_cmd_vx:.2f} | "
                  f"kn_lift FL={_fl_lift:+.3f} FR={_fr_lift:+.3f} "
                  f"RL={_rl_lift:+.3f} RR={_rr_lift:+.3f}", flush=True)

        # ── Detailed debug every 1s (500 control steps) ───────────────────
        if step_counter % 500 == 0 and ready:
            print(f"\n[DEBUG step {step_counter}]", flush=True)
            print(f"  cmd_vx={_cmd_vx:.3f}  tilt={tilt_deg:.1f}°", flush=True)
            print(f"  delta: {delta_isaac.round(3)}", flush=True)
            print(f"  target: {target_q.round(3)}", flush=True)
            print(f"  actual: {joint_pos.round(3)}", flush=True)
            print(f"  FAULT JOINTS:", flush=True)
            print(f"    FR_th: tgt={target_q[5]:.3f}  act={joint_pos[5]:.3f}  "
                  f"cap={FR_TH_BINDING_MAX:.3f}  clamp_count={_fr_th_clamp_count}", flush=True)
            print(f"    RL_th: tgt={target_q[6]:.3f}  act={joint_pos[6]:.3f}  "
                  f"err={joint_pos[6]-target_q[6]:+.3f}", flush=True)
            print(f"  KNEE LIFT (vs default -1.5):", flush=True)
            for ki, kn in enumerate(['FL_kn','FR_kn','RL_kn','RR_kn']):
                idx  = ki + 8
                lift = joint_pos[idx] - DEFAULT_JOINT_POS[idx]
                flag = " *** LIFTING" if lift > 0.08 else ""
                print(f"    {kn}: act={joint_pos[idx]:.3f}  "
                      f"lift={lift:+.3f}  vel={joint_vel[idx]:+.3f}{flag}", flush=True)

        _sl = (1.0 / CONTROL_HZ) - (time.time() - t_loop)
        if _sl > 0: time.sleep(_sl)

except KeyboardInterrupt:
    print("\n[STOP] KeyboardInterrupt.", flush=True)

finally:
    shutdown_event.set()
    inf_thread.join(timeout=2.0)
    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = 20.0
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = 0.0
    udp.SetSend(cmd); udp.Send()
    print("[FINAL] Stand pose sent.", flush=True)

    if _log_step > 10:
        _rpath = f"real_log_model15000_{_log_ts}.npz"
        np.savez(_rpath,
            obs_raw       = _log_obs[:_log_step],
            raw_net       = _log_raw_net[:_log_step],
            tanh_delta    = _log_tanh[:_log_step],
            target_q      = _log_target[:_log_step],
            actual_q      = _log_actual[:_log_step],
            actual_qd     = _log_actual_qd[:_log_step],
            proj_grav     = _log_grav[:_log_step],
            ang_vel       = _log_angvel[:_log_step],
            cmd           = _log_obs[:_log_step, 0:3],
            contact       = _log_contact[:_log_step],
            tilt_deg      = _log_tilt[:_log_step],
            default_q     = DEFAULT_JOINT_POS,
            delta_lo      = DELTA_LO,
            delta_hi      = DELTA_HI,
            jdelta_offset = jdelta_offset,
            step_dt       = np.array([1.0 / INFERENCE_HZ]),
            src           = np.array(["real"], dtype=object),
        )
        print(f"[LOG] Saved {_log_step} steps → {_rpath}", flush=True)

        _aa   = _log_actual[:_log_step]
        _tilt = _log_tilt[:_log_step]
        print(f"\n  TILT: mean={_tilt.mean():.1f}°  max={_tilt.max():.1f}°")
        print(f"  FR_th: max={_aa[:,5].max():.3f}  binding>0.900={(_aa[:,5]>0.900).sum()} steps")
        print(f"  RL_th: range=[{_aa[:,6].min():.3f},{_aa[:,6].max():.3f}]")
        print(f"\n  KNEE LIFT SUMMARY (vs default -1.5, positive=lifted):")
        sim_expected = [0.023, 0.215, 0.155, 0.245]
        for ki, kn in enumerate(['FL_kn','FR_kn','RL_kn','RR_kn']):
            idx  = ki + 8
            lift = _aa[:, idx] - DEFAULT_JOINT_POS[idx]
            sim  = sim_expected[ki]
            print(f"    {kn}: max_lift={lift.max():+.3f}  "
                  f"mean={lift.mean():+.3f}  "
                  f">0.05m={(lift>0.05).mean()*100:.0f}%  "
                  f"[sim_max={sim:+.3f}]")

        if _track_err_n > 0:
            mean_err = _track_err_sum / _track_err_n
            worst = np.argmax(mean_err)
            print(f"\n  Worst tracking: {_JNAMES[worst]} = {mean_err[worst]:.4f} rad")

        print(f"\n  → python compare_sim_real.py "
              f"<sim_log_model_15000_*.npz> {_rpath}")
    else:
        print("[LOG] Too few steps.", flush=True)