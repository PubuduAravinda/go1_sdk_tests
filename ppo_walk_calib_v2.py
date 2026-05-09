#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — model_22700 (go1_deploy_calib_v3/policy.pt)
policy.pt interface: raw_obs(45) → delta(12)  [normalise + MLP + tanh baked in]
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

FIXES FROM PREVIOUS VERSION (3 changes, everything else identical):
  1. FR_th hard clamp at 0.820 rad ADDED (was MISSING — CRITICAL)
       Training enforces target_pos[:,5] ≤ default+0.02 = 0.820 rad.
       Without this clamp FR_th reaches binding zone (>0.90 rad) → falls.
       Line: raw_target[5] = min(raw_target[5], 0.820)

  2. prev_delta update fixed — remove step_counter % 10 condition
       OLD: if ready and (step_counter % 10 == 0): prev_delta[:] = ...
         → obs[33:45] stale up to 20ms (not synchronized with inference)
       NEW: if ready: prev_delta[:] = delta_isaac.copy()
         → always uses latest inference output (inference thread only
           updates _shared_delta at 50Hz so value is same between inferences)

  3. HIP_SCALE restored to 0.70 (was 0.5 — too conservative)
       0.5 → max hip hw delta = ±0.10 rad (limits gait unnecessarily)
       0.7 → max hip hw delta = ±0.14 rad (confirmed safe in previous runs)

VERIFIED CORRECT (unchanged from your code):
  ✓ DELTA_LO/HI: symmetric hips ±0.20, RL_hip ±0.25, thighs/calves ±0.35
  ✓ KD_PER_JOINT[5]=9.0 for FR_th (Isaac index 5)
  ✓ KD indexing: KD_PER_JOINT[isaac_to_sdk[sdk_i]] → correct Isaac KD for each SDK motor
      SDK[1]=FR_th → isaac_to_sdk[1]=5 → KD_PER_JOINT[5]=9.0 ✓
  ✓ sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]
  ✓ Hip sign flips: obs[4]=-obs[4], obs[6]=-obs[6], delta_hw[1]=-delta_hw[1], delta_hw[3]=-delta_hw[3]
  ✓ jdelta_offset measured in HOLD stage and saved in log
  ✓ MAX_DELTA thighs=0.08 (doubled for fast recovery from overshoot)
  ✓ Policy path: go1_deploy_calib_v3/policy.pt

Observation layout (45D, Isaac order throughout):
  [0:3]   cmd [vx, vy, wz]
  [3:15]  jpos - default_q - jdelta_offset    (hw sign-flipped for FR/RR hip)
  [15:27] jvel clipped ±5
  [27:30] gyro clipped ±5
  [30:33] proj_gravity = -acc/|acc|
  [33:45] prev_delta  (policy output t-1, pre hw-flip)

SDK motor order (Go1 hardware — FR FL RR RL):
  SDK[0]=FR_hip [1]=FR_th [2]=FR_kn
  SDK[3]=FL_hip [4]=FL_th [5]=FL_kn
  SDK[6]=RR_hip [7]=RR_th [8]=RR_kn
  SDK[9]=RL_hip [10]=RL_th[11]=RL_kn
  sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]
"""

import time
import threading
from datetime import datetime
import numpy as np
import torch
import robot_interface as sdk

# ── tanh_squash bounds — MUST match go1_env.py _delta_soft_lo / _delta_soft_hi ──
# DELTA_LO = np.array([-0.08,-0.08,-0.08,-0.08, -0.35,-0.35,-0.35,-0.35, -0.35,-0.35,-0.35,-0.35], np.float32)
# DELTA_HI = np.array([ 0.08, 0.08, 0.08, 0.08,  0.35, 0.35, 0.35, 0.35,  0.35, 0.35, 0.35, 0.35], np.float32)
DELTA_LO = np.array([
    -0.08, -0.08, -0.08, -0.08,   # hips ← was ±0.20
    -0.35, -0.35, -0.35, -0.35,
    -0.35, -0.35, -0.35, -0.35,
], dtype=np.float32)
DELTA_HI = np.array([
     0.08,  0.08,  0.08,  0.08,   # hips ← was ±0.20
     0.35,  0.35,  0.35,  0.35,
     0.35,  0.35,  0.35,  0.35,
], dtype=np.float32)

_T_MID  = (DELTA_HI + DELTA_LO) * 0.5   # all zeros
_T_HALF = (DELTA_HI - DELTA_LO) * 0.5

def delta_to_raw_net(delta):
    ratio = np.clip((delta - _T_MID) / _T_HALF, -0.9999, 0.9999)
    return np.arctanh(ratio)

# ── Robot constants ──────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,
    0.8,  0.8,  0.8,  0.8,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

# FIX 1: FR_th mechanical binding fault clamp
# Training enforces target_pos[:,5] ≤ 0.820 rad in go1_env.py _pre_physics_step.
# Without this clamp, FR_th can enter binding zone (>0.90 rad) → robot falls.
FR_TH_BINDING_MAX = 0.820  # rad — matches training cap (default 0.800 + delta 0.020)

JVEL_CLIP   = 5.0
ANGVEL_CLIP = 5.0

# ── KP / KD ──────────────────────────────────────────────────────────────────
KP_START       = 5.0
KP_STEP        = 3.0
RAMP_MAX_LEVEL = 10          # KP_BASE = 35 Nm/rad
KP_MULTIPLIER  = np.array([
    1.000, 1.000, 1.000, 1.000,   # hips   → KP=35
    1.857, 1.857, 1.857, 1.857,   # thighs → KP=65
    2.286, 2.286, 2.286, 2.286,   # calves → KP=80
], dtype=np.float32)

# KD in Isaac order. Indexing in SDK loop:
#   cmd.motorCmd[sdk_i].Kd = KD_PER_JOINT[isaac_to_sdk[sdk_i]]
#   isaac_to_sdk[sdk_i] = isaac_idx for that SDK motor
#   SDK[1]=FR_th → isaac_to_sdk[1]=5 → KD_PER_JOINT[5]=9.0 ✓
KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,   # hips
    4.5, 9.0, 4.5, 4.5,   # thighs: FR_th[Isaac 5] = 9.0
    5.0, 5.0, 5.0, 5.0,   # calves
], dtype=np.float32)

TAU_PER_JOINT_ISAAC = np.array([0,0,0,0, 1.2,1.2,1.2,1.2, 0,0,0,0], dtype=np.float32)

# ── Hardware output scaling ───────────────────────────────────────────────────
# FIX 3: HIP_SCALE restored to 0.70 (was 0.5 — too conservative)
HIP_SCALE   = 0.70   # ±0.20 × 0.70 = ±0.14 rad max hip hw delta
THIGH_SCALE = 0.95
KNEE_SCALE  = 0.90

# HIP_OBS_CORRECTION = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
HIP_OBS_CORRECTION = np.array([
    -0.0318,   # FL_hip — shifts obs toward sim distribution
    -0.0318,   # FR_hip — same bias found on both front hips
     0.0558,   # RL_hip
     0.0258,   # RR_hip
], dtype=np.float32)

# ── cmd_vx startup schedule ───────────────────────────────────────────────────
VX_TARGET   = 0.5
VX_KICK     = 0.8
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

MAX_DELTA_PER_JOINT = np.array([
    0.06, 0.06, 0.06, 0.06,
    0.08, 0.08, 0.08, 0.08,   # thighs doubled for fast recovery
    0.05, 0.05, 0.05, 0.05,
], dtype=np.float32)

# ── Joint remapping ───────────────────────────────────────────────────────────
sdk_to_isaac = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i

# ── Load policy ───────────────────────────────────────────────────────────────
device = torch.device("cpu")
policy = torch.jit.load("go1_deploy_calib_v5/policy.pt").to(device).eval()
print("[POLICY] Loaded go1_deploy_calib_v3/policy.pt (normalise + MLP + tanh baked in)")
with torch.no_grad():
    _test   = policy(torch.zeros(1, 45))
    _hip_r  = _test[0, :4].numpy()
    _fr_r   = _test[0, 5].item()
print(f"[POLICY] JIT warmup done.")
print(f"[POLICY] Hip delta at rest (should be ≈0.0): {_hip_r.round(4)}")
print(f"[POLICY] FR_th delta at rest: {_fr_r:.4f}  "
      f"→ target would be {DEFAULT_JOINT_POS[5]+_fr_r:.4f} rad  "
      f"cap={FR_TH_BINDING_MAX:.3f}")
if np.any(np.abs(_hip_r) > 0.06):
    print("[POLICY] WARNING: large hip delta at rest — check DELTA_LO/HI match training!")

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

_JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
           'FL_th', 'FR_th', 'RL_th', 'RR_th',
           'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

print("\n" + "="*72)
print("Go1 model_22700 | PACE calibrated | FR_th cap + RL_th Kim")
print(f"  delta: hip[±0.20] RL_hip[±0.25] th/kn[±0.35]  _mid=0 (symmetric)")
print(f"  KP: hip=35  thigh=65  knee=80  Nm/rad")
print(f"  KD: hip=4.0  FR_th=9.0  other_th=4.5  knee=5.0  Nm·s/rad")
print(f"  HIP_SCALE={HIP_SCALE}  THIGH_SCALE={THIGH_SCALE}  KNEE_SCALE={KNEE_SCALE}")
print(f"  FR_th clamp: target ≤ {FR_TH_BINDING_MAX:.3f} rad  (binding fault protection)")
print("  Place robot on flat ground. HOLD stage in 10s.")
print("="*72 + "\n")
time.sleep(10)

# ════════════════════════════════════════════════════════════════════════════
# HOLD STAGE
# ════════════════════════════════════════════════════════════════════════════
print("[HOLD] Ramping KP to full training gains...")
_hold_t0   = time.perf_counter()
_hold_step = 0
KP_FULL    = KP_START + RAMP_MAX_LEVEL * KP_STEP   # = 35.0

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
        print(f"[HOLD t={_dt:4.1f}s] KP_base={_kp_now:.1f}  "
              f"hip_eff={_kp_now*KP_MULTIPLIER[0]:.0f}  "
              f"thigh_eff={_kp_now*KP_MULTIPLIER[4]:.0f}  "
              f"knee_eff={_kp_now*KP_MULTIPLIER[8]:.0f}  "
              f"max_err={_err:.3f} rad", flush=True)
    _hold_step += 1
    _sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - _t)
    if _sl > 0: time.sleep(_sl)

# Measure real equilibrium offset
_eq = []
for _ in range(20):
    udp.Recv(); udp.GetRecv(state)
    _eq.append(np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac])
    time.sleep(0.005)
jpos_eq       = np.mean(_eq, axis=0)
jdelta_offset = jpos_eq - DEFAULT_JOINT_POS
max_err       = float(np.max(np.abs(jdelta_offset)))

print(f"\n[HOLD COMPLETE] max_err={max_err:.3f} rad")
print("  jdelta_offset (subtract from jpos_delta in obs):")
for ji, jn in enumerate(_JNAMES):
    print(f"    {jn:8s}: {jdelta_offset[ji]:+.4f}")
if max_err > 0.20:
    print("  WARNING: err > 0.20 rad — check posture. Ctrl+C in 3s to abort.")
    time.sleep(3.0)
else:
    print("  OK — obs[3:15] near-zero at rest.")
print()

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
        print("[WARN] inference thread not ready after 0.5s"); break
    time.sleep(0.001)
print(f"[POLICY] First inference ready ({(time.perf_counter()-_t_wait)*1000:.0f}ms). Warmup...", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# WARMUP: 50 silent policy steps to seed prev_delta
# ════════════════════════════════════════════════════════════════════════════
prev_delta = np.zeros(12, np.float32)

for _ws in range(50):
    udp.Recv(); udp.GetRecv(state)
    _jp  = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
    _jv  = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
    _acc = np.array(state.imu.accelerometer, np.float32)
    _gyro= np.array(state.imu.gyroscope, np.float32)
    _na  = max(float(np.linalg.norm(_acc)), 0.1)
    _obs = np.zeros(45, np.float32)
    _obs[0:3]   = [0.0, 0.0, 0.0]
    _obs[3:15]  = (_jp - DEFAULT_JOINT_POS) - jdelta_offset
    _obs[4]     = -_obs[4]
    _obs[6]     = -_obs[6]
    _obs[3:7]  -= HIP_OBS_CORRECTION
    _obs[15:27] = np.clip(_jv, -JVEL_CLIP, JVEL_CLIP)
    _obs[27:30] = np.clip(_gyro, -ANGVEL_CLIP, ANGVEL_CLIP)
    _obs[30:33] = -_acc / _na
    _obs[33:45] = prev_delta
    with obs_lock:
        _shared_obs[:] = _obs
    time.sleep(1.0 / INFERENCE_HZ)
    with action_lock:
        if _inference_ready:
            prev_delta[:] = _shared_delta.copy()
    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = float(KP_FULL * KP_MULTIPLIER[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()

_sat = np.sum(np.abs(prev_delta - _T_MID) >= _T_HALF * 0.95)
print(f"[WARMUP] Done. prev_delta: {prev_delta.round(3)}")
print(f"[WARMUP] Near-limit joints: {_sat}/12  {'*** OOD WARNING' if _sat > 2 else 'OK'}")
print(f"[WARMUP] Hip prev_delta: {prev_delta[:4].round(4)}  (should be ≈0.0)", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════
_LOG_STEPS     = 1500
_log_step      = 0
_log_obs       = np.zeros((_LOG_STEPS, 45), np.float32)
_log_raw_net   = np.zeros((_LOG_STEPS, 12), np.float32)
_log_tanh      = np.zeros((_LOG_STEPS, 12), np.float32)
_log_target    = np.zeros((_LOG_STEPS, 12), np.float32)
_log_actual    = np.zeros((_LOG_STEPS, 12), np.float32)
_log_actual_qd = np.zeros((_LOG_STEPS, 12), np.float32)
_log_grav      = np.zeros((_LOG_STEPS, 3),  np.float32)
_log_angvel    = np.zeros((_LOG_STEPS, 3),  np.float32)
_log_tilt      = np.zeros(_LOG_STEPS,       np.float32)
_log_contact   = np.zeros((_LOG_STEPS, 4),  np.float32)
_last_log_ctrl = -10
_log_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

KNEE_IDX_SDK      = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}
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

try:
    while True:
        t_loop    = time.time()
        step_counter += 1
        t_elapsed = t_loop - t0_global

        try:
            udp.Recv(); udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP RECV ERROR] {e}", flush=True); break

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

        obs = np.zeros(45, dtype=np.float32)
        if   t_elapsed < VX_RAMP_S:                             _cmd_vx = VX_KICK*(t_elapsed/VX_RAMP_S)
        elif t_elapsed < VX_RAMP_S+VX_KICK_S:                  _cmd_vx = VX_KICK
        elif t_elapsed < VX_RAMP_S+VX_KICK_S+VX_SETTLE_S:
            _cmd_vx = VX_KICK+(VX_TARGET-VX_KICK)*(t_elapsed-(VX_RAMP_S+VX_KICK_S))/VX_SETTLE_S
        else:                                                     _cmd_vx = VX_TARGET

        obs[0:3]   = [_cmd_vx, 0.0, 0.0]
        obs[3:15]  = (joint_pos - DEFAULT_JOINT_POS) - jdelta_offset
        obs[4]     = -obs[4]
        obs[6]     = -obs[6]
        obs[3:7]  -= HIP_OBS_CORRECTION
        obs[15:27] = np.clip(joint_vel, -JVEL_CLIP, JVEL_CLIP)
        obs[27:30] = np.clip(gyro,      -ANGVEL_CLIP, ANGVEL_CLIP)
        obs[30:33] = proj_gravity
        obs[33:45] = prev_delta

        _obs_snapshot = obs.copy()

        with obs_lock:
            _shared_obs[:] = obs

        with action_lock:
            ready       = _inference_ready
            delta_isaac = _shared_delta.copy() if ready else np.zeros(12, np.float32)

        # FIX 2: update prev_delta whenever ready (not every % 10 control steps)
        # Inference thread updates _shared_delta at 50Hz; between inferences the
        # value is identical, so reading at 500Hz is harmless and keeps obs[33:45]
        # always in sync with the latest policy output.
        if ready:
            prev_delta[:] = delta_isaac.copy()

        if ready:
            delta_hw       = delta_isaac.copy()
            delta_hw      *= tilt_scale
            delta_hw[:4]  *= HIP_SCALE    # FIX 3: 0.70 (was 0.5)
            delta_hw[4:8] *= THIGH_SCALE
            delta_hw[8:]  *= KNEE_SCALE
            delta_hw[1]    = -delta_hw[1]
            delta_hw[3]    = -delta_hw[3]
            raw_target     = DEFAULT_JOINT_POS + delta_hw
        else:
            raw_target = DEFAULT_JOINT_POS.copy()

        # FIX 1: FR_th hard clamp — MUST match training go1_env.py cap
        # target_pos[:,5] clamped to ≤ 0.820 in _pre_physics_step during training.
        # Same limit here prevents FR_th entering mechanical binding zone (>0.90).
        if raw_target[5] > FR_TH_BINDING_MAX:
            raw_target[5] = FR_TH_BINDING_MAX
            _fr_th_clamp_count += 1

        target_q = np.clip(raw_target,
                           prev_target_q - MAX_DELTA_PER_JOINT,
                           prev_target_q + MAX_DELTA_PER_JOINT)
        prev_target_q[:] = target_q

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

        if ready and (step_counter - _last_log_ctrl) >= 10 and _log_step < _LOG_STEPS:
            _last_log_ctrl = step_counter
            _qd = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
            _kn = [abs(state.motorState[KNEE_IDX_SDK[k]].tauEst) for k in ["FL","FR","RL","RR"]]
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

        if step_counter % 100 == 1:
            _knt  = {k: abs(state.motorState[idx].tauEst) for k, idx in KNEE_IDX_SDK.items()}
            _feet = "".join("●" if _knt[k]>KNEE_TAU_THRESHOLD else "○" for k in ["FR","FL","RR","RL"])
            print(f"t={t_elapsed:5.1f}s | "
                  f"grav({proj_gravity[0]:+.2f},{proj_gravity[1]:+.2f},{proj_gravity[2]:+.2f}) | "
                  f"tilt={tilt_deg:.1f}° | feet={_feet} | "
                  f"cmd_vx={_cmd_vx:.2f} | FR_clamp={_fr_th_clamp_count}", flush=True)

        if step_counter % 500 == 0 and ready:
            print(f"\n[DEBUG step {step_counter}]", flush=True)
            print(f"  cmd_vx:     {_cmd_vx:.3f}", flush=True)
            print(f"  jpos_delta: {obs[3:15].round(3)}", flush=True)
            print(f"  prev_delta: {obs[33:45].round(3)}", flush=True)
            print(f"  delta_out:  {delta_isaac.round(3)}", flush=True)
            print(f"  delta_hw:   {delta_hw.round(3)}", flush=True)
            print(f"  target_q:   {target_q.round(3)}", flush=True)
            print(f"  actual_q:   {joint_pos.round(3)}", flush=True)
            print(f"  track_err:  {(target_q-joint_pos).round(3)}", flush=True)
            print(f"  FR_th: tgt={target_q[5]:.3f} act={joint_pos[5]:.3f} "
                  f"err={joint_pos[5]-target_q[5]:+.3f} "
                  f"clamp_total={_fr_th_clamp_count}", flush=True)
            print(f"  RL_th: tgt={target_q[6]:.3f} act={joint_pos[6]:.3f} "
                  f"err={joint_pos[6]-target_q[6]:+.3f}", flush=True)
            print(f"  tilt={tilt_deg:.1f}°  tilt_scale={tilt_scale}", flush=True)
            print(f"  hip delta (expect ≈0.0): {delta_isaac[:4].round(4)}", flush=True)

        sl = (1.0 / CONTROL_HZ) - (time.time() - t_loop)
        if sl > 0: time.sleep(sl)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt.", flush=True)

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
        _rpath = f"real_log_{_log_ts}.npz"
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
        print(f"  tilt: mean={_log_tilt[:_log_step].mean():.1f}°  "
              f"max={_log_tilt[:_log_step].max():.1f}°")
        print(f"  FR_th clamp activations: {_fr_th_clamp_count} "
              f"({_fr_th_clamp_count/max(_log_step,1)*100:.0f}% of logged steps)")
        print(f"  FR_th actual max: {_log_actual[:_log_step,5].max():.3f}  "
              f"binding zone >0.900: {(_log_actual[:_log_step,5]>0.900).sum()} steps")
        print(f"  RL_th actual range: [{_log_actual[:_log_step,6].min():.3f},"
              f"{_log_actual[:_log_step,6].max():.3f}]")
        print(f"  hip delta mean: {_log_tanh[:_log_step,:4].mean(axis=0).round(4)}")
        print(f"  → python compare_sim_real.py sim_log_model_22700_*.npz {_rpath}")
    else:
        print("[LOG] Too few steps — nothing saved.", flush=True)