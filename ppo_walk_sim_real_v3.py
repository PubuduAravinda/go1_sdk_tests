#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — policy.pt (normalizer + MLP + tanh_squash baked in)
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

policy.pt interface  (from export_policy.py):
  INPUT:  obs (1, 45) — raw unnormalised observation
  OUTPUT: delta (1, 12) — joint delta in Isaac order, ALREADY tanh-squashed
          guaranteed in [DELTA_LO, DELTA_HI], no tanh_squash needed after this

  target_q_isaac = delta + DEFAULT_JOINT_POS
  (apply HW sign flip and joint scaling ONLY for hardware send, not for obs)

Observation layout (45D, Isaac joint order throughout):
  [0:3]   cmd [vx, vy, wz]
  [3:15]  jpos - default_q - jdelta_offset   (zero-centred at real equilibrium)
  [15:27] jvel clipped to +-5
  [27:30] ang_vel (gyro) clipped to +-5
  [30:33] proj_gravity  (-acc/|acc|)
  [33:45] prev_delta    (delta output from policy.pt at t-1, pre-hw-flip)

Isaac joint order: [FL_hip FR_hip RL_hip RR_hip  FL_th FR_th RL_th RR_th  FL_kn FR_kn RL_kn RR_kn]
SDK joint order:   per-leg FL(h,t,k) FR(h,t,k) RL(h,t,k) RR(h,t,k)
sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]

Hip sign convention:
  Hardware FR_hip+ = inward (adduction)
  Isaac    FR_hip+ = outward (abduction)
  flip obs[4] (FR_hip jdelta) and obs[6] (RR_hip jdelta) when reading hardware
  flip delta_hw[1] (FR_hip) and delta_hw[3] (RR_hip) when sending to hardware
  NEVER flip prev_delta stored in obs[33:45]
"""

import time
import threading
from datetime import datetime
import numpy as np
import torch
import robot_interface as sdk

# ---- tanh_squash parameters (verified from sim_log delta_lo/delta_hi) -------
# ONLY used for: prev_delta init, and reconstructing synthetic raw_net for logging
# NOT used to process policy output -- policy.pt already applies tanh internally
DELTA_LO = np.array([-0.15,-0.15,-0.15,-0.15, -0.35,-0.35,-0.35,-0.35, -0.35,-0.35,-0.35,-0.35], np.float32)
DELTA_HI = np.array([ 0.25, 0.25, 0.25, 0.25,  0.35, 0.35, 0.35, 0.35,  0.35, 0.35, 0.35, 0.35], np.float32)
_T_MID   = (DELTA_HI + DELTA_LO) * 0.5   # [+0.05x4, 0x8]
_T_HALF  = (DELTA_HI - DELTA_LO) * 0.5   # [0.20x4, 0.35x8]

def delta_to_raw_net(delta):
    """Reconstruct equivalent raw_net from delta for logging/compare_sim_real.py.
    Inverse of tanh_squash. Clipped to avoid arctanh blowup at limits."""
    ratio = np.clip((delta - _T_MID) / _T_HALF, -0.9999, 0.9999)
    return np.arctanh(ratio)

# ---- Robot constants ---------------------------------------------------------
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,
    0.8,  0.8,  0.8,  0.8,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

KNEE_TAU_THRESHOLD = 1.0
KNEE_IDX_SDK = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}

MAX_DELTA_PER_JOINT = np.array([
    0.06, 0.06, 0.06, 0.06,
    0.04, 0.04, 0.04, 0.04,
    0.05, 0.05, 0.05, 0.05,
], dtype=np.float32)

JVEL_CLIP   = 5.0
ANGVEL_CLIP = 5.0

# ---- KP/KD (match go1_env_cfg.py training values) ---------------------------
KP_START       = 5.0
KP_STEP        = 3.0
RAMP_MAX_LEVEL = 10
KP_MULTIPLIER  = np.array([1.000,1.000,1.000,1.000,
                             1.857,1.857,1.857,1.857,
                             2.286,2.286,2.286,2.286], dtype=np.float32)
KD_PER_JOINT   = np.array([4.0,4.0,4.0,4.0,
                             4.5,4.5,4.5,4.5,  # RL_thigh reverted 5.5->4.5: higher KD amplified oscillation
                             5.0,5.0,5.0,5.0], dtype=np.float32)
TAU_PER_JOINT_ISAAC = np.array([0,0,0,0, 1.2,1.2,1.2,1.2, 0,0,0,0], dtype=np.float32)

# Hardware output scaling -- delta_hw only, NEVER stored in obs
HIP_SCALE   = 0.7   # 0.5 too conservative, 1.0 splays front legs (+6°). 0.7 balanced.
THIGH_SCALE = 0.85  # was 0.7 — jvel std too low, robot shuffled
KNEE_SCALE  = 0.90  # was 0.8

# cmd_vx startup schedule — 3 phases to break standing attractor:
#   Phase 1 RAMP:  0 → VX_KICK over VX_RAMP_S   (0.0 → 0.3s)
#   Phase 2 KICK:  hold VX_KICK for VX_KICK_S    (0.3 → 1.3s)
#   Phase 3 SETTLE: VX_KICK → VX_TARGET linear   (1.3 → 2.3s)
#
# Why kick to 0.8: training cmd range is [0.3, 0.6]. cmd=0.8 is OOD-ish
# but in the direction of "stride harder" — FL/FR thighs must lift to track it.
# Once striding, settle to 0.5 for stable locomotion.
# Data showed 2.5s ramp re-locks into standing attractor at 0.5.
VX_TARGET  = 0.5   # m/s — steady-state forward velocity
VX_KICK    = 0.80   # m/s — startup kick to break standing attractor
VX_RAMP_S  = 0.3   # s   — 0 → VX_KICK ramp duration
VX_KICK_S  = 0.3   # s   — hold VX_KICK duration
VX_SETTLE_S= 1.0   # s   — VX_KICK → VX_TARGET settle duration

# Hip abduction correction applied to obs[3:6] (front hips only).
# Real data: FL/FR hips persistently +0.08 delta even with jdelta_offset.
# Cause: real urdf compliance pushes front hips outward vs sim default.
# This extra offset shifts policy's hip perception toward sim distribution.
# Values measured as mean hip delta from 3 real runs: FL=+0.082 FR=+0.073
# Apply 60% correction (not 100% — avoid overcorrection on first run).
HIP_OBS_CORRECTION = np.array([0.049, 0.044, -0.048, 0.024], dtype=np.float32)

# ---- Safety ------------------------------------------------------------------
TILT_THRESHOLD = 20.0
TILT_STOP_DEG  = 30.0
HOLD_RAMP_S    = 4.0
HOLD_FULL_S    = 3.0
INFERENCE_HZ   = 50
CONTROL_HZ     = 500

# ---- Joint remapping ---------------------------------------------------------
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i

# ---- Load policy -------------------------------------------------------------
# policy.pt: raw_obs(45) -> delta(12)   [normalise + MLP + tanh_squash baked in]
# Output is already delta -- NO tanh_squash call needed after this
device = torch.device("cpu")
policy = torch.jit.load("go1_deploy_v3/policy.pt").to(device).eval()
print("[POLICY] Loaded policy.pt -- output is delta (tanh baked in)")
with torch.no_grad():
    policy(torch.zeros(1, 45))
print("[POLICY] JIT warmup done.")

# ---- Shared state ------------------------------------------------------------
# _shared_delta: policy.pt output = delta (tanh already applied) -- NOT raw_net
obs_lock         = threading.Lock()
action_lock      = threading.Lock()
shutdown_event   = threading.Event()
_shared_obs      = np.zeros(45, dtype=np.float32)
_shared_delta    = np.zeros(12, dtype=np.float32)
_inference_ready = False


def inference_thread_fn():
    """50Hz. policy.pt returns delta directly -- store as-is, no tanh needed."""
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
                _shared_delta[:] = delta_out   # already delta
                _inference_ready  = True
        except Exception as e:
            print(f"[INFERENCE ERROR] {e}", flush=True)
        sl = period - (time.time() - t0)
        if sl > 0:
            time.sleep(sl)


# ---- SDK setup ---------------------------------------------------------------
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "="*70)
print("Go1 PPO | policy.pt (delta output) | 500Hz ctrl | 50Hz policy")
print(f"  delta range: hip [{DELTA_LO[0]:.2f},{DELTA_HI[0]:.2f}]  "
      f"thigh [{DELTA_LO[4]:.2f},{DELTA_HI[4]:.2f}]  "
      f"knee [{DELTA_LO[8]:.2f},{DELTA_HI[8]:.2f}]")
print(f"  KP at max: hip={35*KP_MULTIPLIER[0]:.0f}  "
      f"thigh={35*KP_MULTIPLIER[4]:.0f}  knee={35*KP_MULTIPLIER[8]:.0f}")
print("  Place robot on flat ground. Starting in 10s.")
print("="*70 + "\n")
time.sleep(10)

# ==== HOLD STAGE: ramp KP, measure real equilibrium ==========================
print("[HOLD] Ramping KP to full training values...")
_hold_t0 = time.perf_counter()
_hold_step = 0
while True:
    _t  = time.perf_counter()
    _dt = _t - _hold_t0
    if _dt >= HOLD_RAMP_S + HOLD_FULL_S:
        break
    udp.Recv(); udp.GetRecv(state)
    _alpha   = min(1.0, _dt / HOLD_RAMP_S)
    _kp_base = KP_START + _alpha * (KP_START + RAMP_MAX_LEVEL * KP_STEP - KP_START)
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(_kp_base * KP_MULTIPLIER[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd   = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau  = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()
    if _hold_step % (CONTROL_HZ // 2) == 0:
        _jp   = np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac]
        _err  = float(np.max(np.abs(_jp - DEFAULT_JOINT_POS)))
        _knt  = {k: abs(state.motorState[idx].tauEst) for k,idx in KNEE_IDX_SDK.items()}
        _feet = "".join("●" if _knt[k]>KNEE_TAU_THRESHOLD else "○" for k in ["FR","FL","RR","RL"])
        print(f"[HOLD t={_dt:4.1f}s] KP h={_kp_base*KP_MULTIPLIER[0]:.0f} "
              f"t={_kp_base*KP_MULTIPLIER[4]:.0f} k={_kp_base*KP_MULTIPLIER[8]:.0f}"
              f"  err={_err:.3f}  feet={_feet}", flush=True)
    _hold_step += 1
    _sl = (1.0/CONTROL_HZ) - (time.perf_counter() - _t)
    if _sl > 0: time.sleep(_sl)

# Measure real equilibrium offset
_eq = []
for _ in range(20):
    udp.Recv(); udp.GetRecv(state)
    _eq.append(np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac])
    time.sleep(0.005)
_jpos_eq      = np.mean(_eq, axis=0)
_err_final    = float(np.max(np.abs(_jpos_eq - DEFAULT_JOINT_POS)))
jdelta_offset = _jpos_eq - DEFAULT_JOINT_POS

_JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip','FL_th','FR_th','RL_th','RR_th','FL_kn','FR_kn','RL_kn','RR_kn']
print(f"\n[HOLD COMPLETE] max_err={_err_final:.3f} rad")
print("  jdelta_offset:")
for _ji, _jn in enumerate(_JNAMES):
    print(f"    {_jn:8s}: {jdelta_offset[_ji]:+.4f}")
if _err_final > 0.20:
    print("  WARNING: err > 0.20 -- check robot posture. Ctrl+C in 3s to abort.")
    time.sleep(3.0)
else:
    print("  OK -- obs[3:15] near-zero at rest.")
print()

# ==== Start inference thread with real sensor data ===========================
inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
inf_thread.start()
_t_wait = time.perf_counter()
while not _inference_ready:
    udp.Recv(); udp.GetRecv(state)
    for i in range(12):
        cmd.motorCmd[i].q  = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp = float((KP_MULTIPLIER*(KP_START+RAMP_MAX_LEVEL*KP_STEP))[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau= float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
    if time.perf_counter() - _t_wait > 0.5:
        print("[WARN] inference thread not ready after 0.5s"); break
    time.sleep(0.001)
print(f"[POLICY] First inference ready ({(time.perf_counter()-_t_wait)*1000:.0f}ms). Warmup...", flush=True)

# ==== WARMUP: 50 silent policy steps to seed prev_delta ======================
prev_delta = np.zeros(12, np.float32)   # policy.pt output at t-1, Isaac order, pre-hw-flip

for _ws in range(50):
    udp.Recv(); udp.GetRecv(state)
    _jpos_w = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
    _jvel_w = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
    _acc_w  = np.array(state.imu.accelerometer, np.float32)
    _gyro_w = np.array(state.imu.gyroscope,     np.float32)
    _norm_a = max(float(np.linalg.norm(_acc_w)), 0.1)
    _obs_w  = np.zeros(45, np.float32)
    _obs_w[0:3]   = [0.0, 0.0, 0.0]   # warmup at cmd_vx=0 — standing obs
    _obs_w[3:15]  = (_jpos_w - DEFAULT_JOINT_POS) - jdelta_offset
    _obs_w[15:27] = np.clip(_jvel_w, -JVEL_CLIP, JVEL_CLIP)
    _obs_w[27:30] = np.clip(_gyro_w, -ANGVEL_CLIP, ANGVEL_CLIP)
    _obs_w[30:33] = -_acc_w / _norm_a
    _obs_w[33:45] = prev_delta
    _obs_w[4] = -_obs_w[4]   # FR_hip
    _obs_w[6] = -_obs_w[6]   # RR_hip
    with obs_lock:
        _shared_obs[:] = _obs_w
    time.sleep(1.0 / INFERENCE_HZ)
    with action_lock:
        if _inference_ready:
            prev_delta[:] = _shared_delta.copy()   # already delta -- no tanh needed
    for i in range(12):
        cmd.motorCmd[i].q  = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp = float((KP_MULTIPLIER*(KP_START+RAMP_MAX_LEVEL*KP_STEP))[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau= float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()

_sat = np.sum(np.abs(prev_delta - _T_MID) >= _T_HALF * 0.95)
print(f"[WARMUP] Done. prev_delta: {prev_delta.round(3)}")
print(f"[WARMUP] Near-limit joints: {_sat}/12  {'*** OOD WARNING' if _sat > 2 else 'OK'}", flush=True)

# ==== Logging arrays ==========================================================
_LOG_STEPS = 1500
_log_step  = 0
_log_obs     = np.zeros((_LOG_STEPS,45), np.float32)
_log_raw_net = np.zeros((_LOG_STEPS,12), np.float32)
_log_tanh    = np.zeros((_LOG_STEPS,12), np.float32)
_log_target  = np.zeros((_LOG_STEPS,12), np.float32)
_log_actual  = np.zeros((_LOG_STEPS,12), np.float32)
_log_actual_qd = np.zeros((_LOG_STEPS,12), np.float32)
_log_grav    = np.zeros((_LOG_STEPS,3),  np.float32)
_log_angvel  = np.zeros((_LOG_STEPS,3),  np.float32)
_log_tilt    = np.zeros(_LOG_STEPS,      np.float32)
_log_contact = np.zeros((_LOG_STEPS,4),  np.float32)
_last_log_ctrl_step = -10
_log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==== Control loop ============================================================
step_counter    = 0
t0_global       = time.time()
tilt_exceeded_t = None
prev_target_q   = DEFAULT_JOINT_POS.copy()
current_kp      = KP_START + RAMP_MAX_LEVEL * KP_STEP   # already ramped in HOLD

try:
    while True:
        t_loop = time.time()
        step_counter += 1
        t_elapsed = t_loop - t0_global

        # 1. Receive state
        try:
            udp.Recv(); udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP RECV ERROR] {e}", flush=True); break

        # 2. Joints SDK -> Isaac
        joint_pos = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
        joint_vel = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]

        # 3. IMU
        acc          = np.array(state.imu.accelerometer, np.float32)
        gyro         = np.array(state.imu.gyroscope,     np.float32)
        norm_a       = max(float(np.linalg.norm(acc)), 0.1)
        proj_gravity = -acc / norm_a
        tilt_deg     = float(np.degrees(np.sqrt(proj_gravity[0]**2 + proj_gravity[1]**2)))
        tilt_scale   = 0.5 if tilt_deg > TILT_THRESHOLD else 1.0
        if tilt_deg > TILT_STOP_DEG:
            if tilt_exceeded_t is None: tilt_exceeded_t = time.time()
            elif time.time() - tilt_exceeded_t > 0.3:
                print(f"[SAFETY STOP] Tilt {tilt_deg:.1f}deg", flush=True); break
        else:
            tilt_exceeded_t = None

        # 4. Build 45D obs
        obs = np.zeros(45, dtype=np.float32)
        # cmd_vx startup schedule: ramp → kick → settle → steady
        _t_kick_end   = VX_RAMP_S
        _t_settle_end = VX_RAMP_S + VX_KICK_S
        _t_steady_end = VX_RAMP_S + VX_KICK_S + VX_SETTLE_S
        if t_elapsed < _t_kick_end:
            _cmd_vx = VX_KICK * (t_elapsed / VX_RAMP_S)
        elif t_elapsed < _t_settle_end:
            _cmd_vx = VX_KICK
        elif t_elapsed < _t_steady_end:
            _frac   = (t_elapsed - _t_settle_end) / VX_SETTLE_S
            _cmd_vx = VX_KICK + (VX_TARGET - VX_KICK) * _frac
        else:
            _cmd_vx = VX_TARGET
        obs[0:3]   = [_cmd_vx, 0.0, 0.0]
        obs[3:15]  = (joint_pos - DEFAULT_JOINT_POS) - jdelta_offset
        obs[15:27] = np.clip(joint_vel,  -JVEL_CLIP,   JVEL_CLIP)
        obs[27:30] = np.clip(gyro,       -ANGVEL_CLIP, ANGVEL_CLIP)
        obs[30:33] = proj_gravity
        obs[33:45] = prev_delta   # policy.pt output at t-1, pre-hw-flip
        obs[4] = -obs[4]          # FR_hip hw->Isaac
        obs[6] = -obs[6]          # RR_hip hw->Isaac
        # Hip abduction correction: front hips persistently +0.08 outward in real
        # hardware vs sim. Subtract learned bias so policy sees near-sim hip state.
        obs[3:7] -= HIP_OBS_CORRECTION

        # 5. Snapshot obs BEFORE updating prev_delta
        #    obs_snapshot[33:45] = delta[t-1]  (what policy received at step t)
        #    compare_sim_real.py: obs_log[t+1,33:45] == tanh_log[t]  ->  correct
        _obs_snapshot = obs.copy()

        # 6. Push obs to inference thread
        with obs_lock:
            _shared_obs[:] = obs

        # 7. Read latest delta (policy.pt output -- already tanh-squashed)
        with action_lock:
            ready       = _inference_ready
            delta_isaac = _shared_delta.copy() if ready else np.zeros(12, np.float32)

        # 8. Update prev_delta AFTER snapshot, BEFORE hardware scaling
        if ready:
            prev_delta[:] = delta_isaac.copy()   # pure policy output, pre-hw-flip

        # 9. Hardware delta -- apply scales HERE ONLY, never in obs
        if ready:
            delta_hw       = delta_isaac.copy()
            delta_hw      *= tilt_scale
            delta_hw[:4]  *= HIP_SCALE
            delta_hw[4:8] *= THIGH_SCALE
            delta_hw[8:]  *= KNEE_SCALE
            delta_hw[1]    = -delta_hw[1]   # FR_hip Isaac->hardware
            delta_hw[3]    = -delta_hw[3]   # RR_hip Isaac->hardware
            raw_target = DEFAULT_JOINT_POS + delta_hw
        else:
            raw_target = DEFAULT_JOINT_POS.copy()

        # 10. Rate limiter
        target_q = np.clip(raw_target,
                           prev_target_q - MAX_DELTA_PER_JOINT,
                           prev_target_q + MAX_DELTA_PER_JOINT)
        prev_target_q[:] = target_q

        # 11. Send to hardware
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

        # 12. Logger @ 50Hz
        #     Uses _obs_snapshot (before prev_delta update):
        #       obs_log[t, 33:45] = delta[t-1]  -- what policy received at step t
        #     tanh_log[t] = delta[t]
        #     So obs_log[t+1, 33:45] == tanh_log[t]  ✓
        if ready and (step_counter - _last_log_ctrl_step) >= 10 and _log_step < _LOG_STEPS:
            _last_log_ctrl_step = step_counter
            _qd  = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
            _kn  = [abs(state.motorState[KNEE_IDX_SDK[k]].tauEst) for k in ["FL","FR","RL","RR"]]
            _log_obs[_log_step]       = _obs_snapshot           # pre-update snapshot
            _log_tanh[_log_step]      = delta_isaac              # direct policy.pt output
            _log_raw_net[_log_step]   = delta_to_raw_net(delta_isaac)  # reconstructed for comparison
            _log_target[_log_step]    = target_q
            _log_actual[_log_step]    = joint_pos
            _log_actual_qd[_log_step] = _qd
            _log_grav[_log_step]      = proj_gravity
            _log_angvel[_log_step]    = gyro
            _log_tilt[_log_step]      = tilt_deg
            _log_contact[_log_step]   = _kn
            _log_step += 1

        # 13. Status @ 100 ctrl steps = 0.2s
        if step_counter % 100 == 1:
            _knt  = {k: abs(state.motorState[idx].tauEst) for k,idx in KNEE_IDX_SDK.items()}
            _feet = "".join("●" if _knt[k]>KNEE_TAU_THRESHOLD else "○" for k in ["FR","FL","RR","RL"])
            print(f"t={time.time()-t0_global:5.1f}s | "
                  f"grav({proj_gravity[0]:+.2f},{proj_gravity[1]:+.2f},{proj_gravity[2]:+.2f}) | "
                  f"tilt {tilt_deg:.1f}deg | feet {_feet} | ready={ready}", flush=True)

        if step_counter % 500 == 0 and ready:
            print(f"\n[DEBUG step {step_counter}]", flush=True)
            print(f"  jdelta:     {obs[3:15].round(3)}", flush=True)
            print(f"  prev_delta: {obs[33:45].round(3)}", flush=True)
            print(f"  delta_out:  {delta_isaac.round(3)}", flush=True)
            print(f"  raw_net_eq: {delta_to_raw_net(delta_isaac).round(2)}", flush=True)
            print(f"  target_q:   {target_q.round(3)}", flush=True)
            print(f"  actual_q:   {joint_pos.round(3)}", flush=True)
            print(f"  track_err:  {(target_q-joint_pos).round(3)}", flush=True)
            print(f"  tilt {tilt_deg:.1f}deg  tilt_scale={tilt_scale}", flush=True)

        # 14. 500Hz timing
        sl = (1.0/CONTROL_HZ) - (time.time() - t_loop)
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
            obs_raw    = _log_obs[:_log_step],
            raw_net    = _log_raw_net[:_log_step],
            tanh_delta = _log_tanh[:_log_step],
            target_q   = _log_target[:_log_step],
            actual_q   = _log_actual[:_log_step],
            actual_qd  = _log_actual_qd[:_log_step],
            proj_grav  = _log_grav[:_log_step],
            ang_vel    = _log_angvel[:_log_step],
            cmd        = _log_obs[:_log_step, 0:3],
            contact    = _log_contact[:_log_step],
            tilt_deg   = _log_tilt[:_log_step],
            default_q  = DEFAULT_JOINT_POS,
            delta_lo   = DELTA_LO,
            delta_hi   = DELTA_HI,
            step_dt    = np.array([1.0/INFERENCE_HZ]),
            src        = np.array(["real"], dtype=object),
        )
        print(f"[LOG] Saved {_log_step} steps -> {_rpath}", flush=True)
        print(f"  tilt mean={_log_tilt[:_log_step].mean():.1f}deg  "
              f"max={_log_tilt[:_log_step].max():.1f}deg")
        print(f"  delta max abs={np.abs(_log_tanh[:_log_step]).max():.3f}")
        print(f"  Compare: python compare_sim_real.py sim_log_*.npz {_rpath}")
    else:
        print("[LOG] Too few steps to save.", flush=True)