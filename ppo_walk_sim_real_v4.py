#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — model_3500 (v3 retrain, iter 3500)
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

Policy trained with (go1_env.py v3):
  - Symmetric hip limits [-0.20,+0.20], RL [-0.25,+0.25]
  - Per-joint KP DR: hips/thighs [0.40,1.20], knees [0.60,1.20]
  - Foot friction DR [0.5,1.1] per episode
  - r_action_rate=-0.5, r_action_jerk=-0.3, r_hip_reg=-0.8
  - Rudin reward set

Obs corrections (measured from real_log_20260319_112153 vs sim_log_model_3500):
  HIP_OBS_SIM_OFFSET  — sim3500 has 3.8° lean, hips sit off-neutral during gait
  KNEE_OBS_OFFSET     — knee obs gap +0.19-0.26 rad (sim knees lag more than real)
  GRAV_X_OFFSET       — sim3500 grav_x=-0.103, real raw=-0.009, gap=+0.094
"""

import time
import threading
from datetime import datetime
import numpy as np
import torch
import robot_interface as sdk

# ── tanh_squash limits — must match go1_env.py exactly ───────────────────────
DELTA_LO = np.array([-0.20,-0.20,-0.25,-0.20, -0.35,-0.35,-0.35,-0.35, -0.35,-0.35,-0.35,-0.35], np.float32)
DELTA_HI = np.array([ 0.20, 0.20, 0.25, 0.20,  0.35, 0.35, 0.35, 0.35,  0.35, 0.35, 0.35, 0.35], np.float32)
_T_MID   = (DELTA_HI + DELTA_LO) * 0.5
_T_HALF  = (DELTA_HI - DELTA_LO) * 0.5


def delta_to_raw_net(delta):
    return np.arctanh(np.clip((delta - _T_MID) / _T_HALF, -0.9999, 0.9999))


# ── Robot constants ───────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,
    0.8,  0.8,  0.8,  0.8,
   -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

KNEE_TAU_THRESHOLD = 1.0
KNEE_IDX_SDK = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}

MAX_DELTA_PER_JOINT = np.array([
    0.10, 0.10, 0.10, 0.10,   # hips
    0.15, 0.15, 0.15, 0.15,   # thighs
    0.12, 0.12, 0.12, 0.12,   # knees
], dtype=np.float32)

JVEL_CLIP   = 5.0
ANGVEL_CLIP = 5.0

# ── KP/KD — match go1_env_cfg.py nominal values ──────────────────────────────
KP_START       = 5.0
KP_STEP        = 3.0
RAMP_MAX_LEVEL = 10
KP_MULTIPLIER  = np.array([1.000, 1.000, 1.000, 1.000,
                            1.857, 1.857, 1.857, 1.857,
                            2.286, 2.286, 2.286, 2.286], dtype=np.float32)
# Resulting KP at max: hip=35, thigh=65, knee=80
KD_PER_JOINT   = np.array([4.0, 4.0, 4.0, 4.0,
                            4.5, 4.5, 4.5, 4.5,
                            5.0, 5.0, 5.0, 5.0], dtype=np.float32)
TAU_PER_JOINT_ISAAC = np.array([0, 0, 0, 0, 1.2, 1.2, 1.2, 1.2, 0, 0, 0, 0], dtype=np.float32)

HIP_SCALE   = 0.7
THIGH_SCALE = 1.0
KNEE_SCALE  = 1.0

# ── cmd: ramp 0 → VX_TARGET, no kick ─────────────────────────────────────────
VX_TARGET = 0.4   # m/s — trained at 0.3–0.9, 0.4 is stable mid-range
VX_RAMP_S = 1.0   # s

# ── Obs corrections (all data-driven from real_log_20260319_112153 vs sim3500) ─
#
# HIP: sim3500 walks with 3.8° lean → hips sit off-neutral during gait.
#   Policy trained on: FL=-0.097  FR=+0.040  RL=-0.017  RR=-0.056
#   Real hardware at:  FL=+0.000  FR=+0.029  RL=+0.063  RR=+0.126
#   Gap = sim - real:  FL=-0.097  FR=+0.011  RL=-0.080  RR=-0.182
#   Apply 85% to avoid overcorrection.
#   Applied AFTER FR/RR sign flips.
HIP_OBS_SIM_OFFSET = np.array([
    -0.082,   # FL_hip: gap=-0.097, 85%=-0.082
    +0.009,   # FR_hip: gap=+0.011, small — minor correction
    -0.068,   # RL_hip: gap=-0.080, 85%=-0.068
    -0.155,   # RR_hip: gap=-0.182, 85%=-0.155
], dtype=np.float32)

# KNEE: sim3500 knees lag far behind targets (low KP DR creates large tracking err).
#   sim actual knee much more bent than real → sim knee obs much lower than real.
#   Measured gap (real_raw - sim3500): FL=+0.255  FR=+0.257  RL=+0.090  RR=+0.265
#   Apply 75% to avoid overcorrection.
#   obs[11:15] = FL FR RL RR knee within obs[3:15] block
KNEE_OBS_OFFSET = np.array([
    -0.191,   # FL_kn: gap=+0.255, 75%=-0.191
    -0.193,   # FR_kn: gap=+0.257, 75%=-0.193
    -0.067,   # RL_kn: gap=+0.090, 75%=-0.067
    -0.199,   # RR_kn: gap=+0.265, 75%=-0.199
], dtype=np.float32)

# GRAV_X: sim3500 grav_x=-0.103 (leans backward from friction DR).
#   Real grav_x raw ≈ -0.009. Gap = real - sim = +0.094.
#   Subtract 80% to shift real toward sim distribution.
GRAV_X_OFFSET = -0.075   # was -0.04 (too small for sim3500 which sits at -0.103)

# ── Safety ────────────────────────────────────────────────────────────────────
TILT_STOP_DEG = 30.0
HOLD_RAMP_S   = 4.0
HOLD_FULL_S   = 3.0
INFERENCE_HZ  = 50
CONTROL_HZ    = 500

# ── Joint remapping ───────────────────────────────────────────────────────────
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for _i in range(12):
    isaac_to_sdk[sdk_to_isaac[_i]] = _i

# ── Load policy ───────────────────────────────────────────────────────────────
device = torch.device("cpu")
policy = torch.jit.load("go1_deploy_v4/policy.pt").to(device).eval()
print("[POLICY] Loaded policy.pt")
with torch.no_grad():
    policy(torch.zeros(1, 45))
print("[POLICY] JIT warmup done.")

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

print("\n" + "=" * 70)
print("Go1 PPO model_3500 | 500Hz ctrl | 50Hz policy")
print(f"  hip limits:    [{DELTA_LO[0]:.2f},{DELTA_HI[0]:.2f}]  RL:[{DELTA_LO[2]:.2f},{DELTA_HI[2]:.2f}]")
print(f"  max_delta:     hips={MAX_DELTA_PER_JOINT[0]:.2f}  thighs={MAX_DELTA_PER_JOINT[4]:.2f}  knees={MAX_DELTA_PER_JOINT[8]:.2f}")
print(f"  scales:        HIP={HIP_SCALE}  THIGH={THIGH_SCALE}  KNEE={KNEE_SCALE}")
print(f"  cmd:           0→{VX_TARGET}m/s over {VX_RAMP_S}s")
print(f"  hip_offset:    {HIP_OBS_SIM_OFFSET.round(3)}")
print(f"  knee_offset:   {KNEE_OBS_OFFSET.round(3)}")
print(f"  grav_x_offset: {GRAV_X_OFFSET:.3f}")
print(f"  tilt_kill:     {TILT_STOP_DEG}°")
print("  Place robot on flat ground. Starting in 10s.")
print("=" * 70 + "\n")
time.sleep(10)

# ── HOLD: ramp KP, measure equilibrium ───────────────────────────────────────
print("[HOLD] Ramping KP...")
_hold_t0 = time.perf_counter(); _hold_step = 0
while True:
    _t = time.perf_counter(); _dt = _t - _hold_t0
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
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
    if _hold_step % (CONTROL_HZ // 2) == 0:
        _jp   = np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac]
        _err  = float(np.max(np.abs(_jp - DEFAULT_JOINT_POS)))
        _knt  = {k: abs(state.motorState[idx].tauEst) for k, idx in KNEE_IDX_SDK.items()}
        _feet = "".join("●" if _knt[k] > KNEE_TAU_THRESHOLD else "○" for k in ["FR","FL","RR","RL"])
        print(f"[HOLD t={_dt:4.1f}s] KP h={_kp_base*KP_MULTIPLIER[0]:.0f} "
              f"t={_kp_base*KP_MULTIPLIER[4]:.0f} k={_kp_base*KP_MULTIPLIER[8]:.0f}"
              f"  err={_err:.3f}  feet={_feet}", flush=True)
    _hold_step += 1
    _sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - _t)
    if _sl > 0:
        time.sleep(_sl)

# Measure hardware equilibrium
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
    print("  WARNING: err > 0.20 — robot not settled. Ctrl+C in 3s to abort.")
    time.sleep(3.0)
else:
    print("  OK — obs[3:15] near-zero at rest.")
print()

# ── Start inference thread ────────────────────────────────────────────────────
inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
inf_thread.start()
_t_wait = time.perf_counter()
while not _inference_ready:
    udp.Recv(); udp.GetRecv(state)
    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = float((KP_MULTIPLIER * (KP_START + RAMP_MAX_LEVEL * KP_STEP))[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
    if time.perf_counter() - _t_wait > 0.5:
        print("[WARN] inference thread slow"); break
    time.sleep(0.001)
print(f"[POLICY] Ready ({(time.perf_counter()-_t_wait)*1000:.0f}ms). Warmup...", flush=True)

# ── Warmup: 50 steps to seed prev_delta ──────────────────────────────────────
prev_delta = np.zeros(12, np.float32)

for _ws in range(50):
    udp.Recv(); udp.GetRecv(state)
    _jpos_w = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
    _jvel_w = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
    _acc_w  = np.array(state.imu.accelerometer, np.float32)
    _gyro_w = np.array(state.imu.gyroscope,     np.float32)
    _norm_a = max(float(np.linalg.norm(_acc_w)), 0.1)

    # Read delta FIRST, then update prev_delta, then fill obs[33:45]
    with action_lock:
        if _inference_ready:
            prev_delta[:] = _shared_delta.copy()

    _obs_w         = np.zeros(45, np.float32)
    _obs_w[0:3]    = [0.0, 0.0, 0.0]
    _obs_w[3:15]   = (_jpos_w - DEFAULT_JOINT_POS) - jdelta_offset
    _obs_w[11:15] += KNEE_OBS_OFFSET          # knee obs correction
    _obs_w[15:27]  = np.clip(_jvel_w, -JVEL_CLIP, JVEL_CLIP)
    _obs_w[27:30]  = np.clip(_gyro_w, -ANGVEL_CLIP, ANGVEL_CLIP)
    _obs_w[30:33]  = -_acc_w / _norm_a
    _obs_w[30]    += GRAV_X_OFFSET            # forward pitch correction
    _obs_w[4]      = -_obs_w[4]              # FR_hip hw→Isaac sign flip
    _obs_w[6]      = -_obs_w[6]              # RR_hip hw→Isaac sign flip
    _obs_w[3:7]   += HIP_OBS_SIM_OFFSET      # hip obs correction (after flips)
    _obs_w[33:45]  = prev_delta

    with obs_lock:
        _shared_obs[:] = _obs_w
    time.sleep(1.0 / INFERENCE_HZ)

    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = float((KP_MULTIPLIER * (KP_START + RAMP_MAX_LEVEL * KP_STEP))[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()

_sat = np.sum(np.abs(prev_delta - _T_MID) >= _T_HALF * 0.95)
print(f"[WARMUP] Done. prev_delta: {prev_delta.round(3)}")
print(f"[WARMUP] Near-limit joints: {_sat}/12  {'*** WARNING' if _sat > 2 else 'OK'}", flush=True)

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_STEPS     = 1500; _log_step = 0
_log_obs       = np.zeros((_LOG_STEPS, 45), np.float32)
_log_raw_net   = np.zeros((_LOG_STEPS, 12), np.float32)
_log_tanh      = np.zeros((_LOG_STEPS, 12), np.float32)
_log_target    = np.zeros((_LOG_STEPS, 12), np.float32)
_log_actual    = np.zeros((_LOG_STEPS, 12), np.float32)
_log_actual_qd = np.zeros((_LOG_STEPS, 12), np.float32)
_log_grav      = np.zeros((_LOG_STEPS,  3), np.float32)
_log_angvel    = np.zeros((_LOG_STEPS,  3), np.float32)
_log_tilt      = np.zeros(_LOG_STEPS,       np.float32)
_log_contact   = np.zeros((_LOG_STEPS,  4), np.float32)
_last_log_ctrl = -10
_log_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Control loop ──────────────────────────────────────────────────────────────
step_counter    = 0
t0_global       = time.time()
tilt_exceeded_t = None
prev_target_q   = DEFAULT_JOINT_POS.copy()
current_kp      = KP_START + RAMP_MAX_LEVEL * KP_STEP

try:
    while True:
        t_loop = time.time(); step_counter += 1; t_elapsed = t_loop - t0_global

        # 1. Receive
        try:
            udp.Recv(); udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP RECV ERROR] {e}", flush=True); break

        # 2. Joints SDK → Isaac order
        joint_pos = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
        joint_vel = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]

        # 3. IMU
        acc          = np.array(state.imu.accelerometer, np.float32)
        gyro         = np.array(state.imu.gyroscope,     np.float32)
        norm_a       = max(float(np.linalg.norm(acc)), 0.1)
        proj_gravity = -acc / norm_a
        tilt_deg     = float(np.degrees(np.sqrt(proj_gravity[0]**2 + proj_gravity[1]**2)))

        # Hard kill only — no soft tilt_scale
        if tilt_deg > TILT_STOP_DEG:
            if tilt_exceeded_t is None:
                tilt_exceeded_t = time.time()
            elif time.time() - tilt_exceeded_t > 0.3:
                print(f"[SAFETY STOP] Tilt {tilt_deg:.1f}°", flush=True); break
        else:
            tilt_exceeded_t = None

        # 4. Read delta, update prev_delta FIRST (correct ordering)
        with action_lock:
            ready       = _inference_ready
            delta_isaac = _shared_delta.copy() if ready else np.zeros(12, np.float32)
        if ready:
            prev_delta[:] = delta_isaac.copy()

        # 5. Build 45D obs — all corrections in correct order
        obs = np.zeros(45, dtype=np.float32)
        _cmd_vx    = min(VX_TARGET, VX_TARGET * (t_elapsed / VX_RAMP_S))

        obs[0:3]   = [_cmd_vx, 0.0, 0.0]
        obs[3:15]  = (joint_pos - DEFAULT_JOINT_POS) - jdelta_offset

        obs[11:15] += KNEE_OBS_OFFSET          # ← knee correction (before sign flips, indices unaffected)

        obs[15:27] = np.clip(joint_vel,  -JVEL_CLIP,   JVEL_CLIP)
        obs[27:30] = np.clip(gyro,       -ANGVEL_CLIP, ANGVEL_CLIP)
        obs[30:33] = proj_gravity
        obs[30]   += GRAV_X_OFFSET             # ← grav_x correction

        obs[4]     = -obs[4]                   # FR_hip hw→Isaac sign flip
        obs[6]     = -obs[6]                   # RR_hip hw→Isaac sign flip

        obs[3:7]  += HIP_OBS_SIM_OFFSET        # ← hip correction (after sign flips)

        obs[33:45] = prev_delta                # freshly updated above

        # 6. Snapshot for logging
        _obs_snapshot = obs.copy()

        # 7. Push to inference thread
        with obs_lock:
            _shared_obs[:] = obs

        # 8. Hardware delta
        if ready:
            delta_hw       = delta_isaac.copy()
            delta_hw[:4]  *= HIP_SCALE     # 0.7
            delta_hw[4:8] *= THIGH_SCALE   # 1.0
            delta_hw[8:]  *= KNEE_SCALE    # 1.0
            delta_hw[1]    = -delta_hw[1]  # FR_hip Isaac→hw
            delta_hw[3]    = -delta_hw[3]  # RR_hip Isaac→hw
            raw_target = DEFAULT_JOINT_POS + delta_hw
        else:
            raw_target = DEFAULT_JOINT_POS.copy()

        # 9. Rate limiter
        target_q = np.clip(raw_target,
                           prev_target_q - MAX_DELTA_PER_JOINT,
                           prev_target_q + MAX_DELTA_PER_JOINT)
        prev_target_q[:] = target_q

        # 10. Send to hardware
        target_q_sdk = target_q[isaac_to_sdk]
        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q    = float(target_q_sdk[i])
            cmd.motorCmd[i].dq   = 0.0
            cmd.motorCmd[i].Kp   = float(current_kp * KP_MULTIPLIER[isaac_to_sdk[i]])
            cmd.motorCmd[i].Kd   = float(KD_PER_JOINT[isaac_to_sdk[i]])
            cmd.motorCmd[i].tau  = float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
        try:
            safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
        except Exception as e:
            print(f"[UDP SEND ERROR] {e}", flush=True); break

        # 11. Log @ 50Hz
        if ready and (step_counter - _last_log_ctrl) >= 10 and _log_step < _LOG_STEPS:
            _last_log_ctrl = step_counter
            _qd  = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
            _kn  = [abs(state.motorState[KNEE_IDX_SDK[k]].tauEst) for k in ["FL","FR","RL","RR"]]
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

        # 12. Status @ 0.2s
        if step_counter % 100 == 1:
            _knt  = {k: abs(state.motorState[idx].tauEst) for k, idx in KNEE_IDX_SDK.items()}
            _feet = "".join("●" if _knt[k] > KNEE_TAU_THRESHOLD else "○" for k in ["FR","FL","RR","RL"])
            print(f"t={time.time()-t0_global:5.1f}s | "
                  f"grav({proj_gravity[0]:+.2f},{proj_gravity[1]:+.2f},{proj_gravity[2]:+.2f}) | "
                  f"tilt {tilt_deg:.1f}° | feet {_feet} | cmd={_cmd_vx:.2f}", flush=True)

        # Debug @ 500 ctrl steps (~10s)
        if step_counter % 500 == 0 and ready:
            print(f"\n[DEBUG step {step_counter}]", flush=True)
            raw_jd = (joint_pos - DEFAULT_JOINT_POS) - jdelta_offset
            print(f"  hip raw jdelta:  {raw_jd[:4].round(3)}", flush=True)
            print(f"  obs hips:        {obs[3:7].round(3)}  (after flips+hip_offset)", flush=True)
            print(f"  obs knees:       {obs[11:15].round(3)}  (after knee_offset)", flush=True)
            print(f"  obs grav_x:      {obs[30]:.4f}  (raw={proj_gravity[0]:.4f}  offset={GRAV_X_OFFSET:.3f})", flush=True)
            print(f"  prev_delta:      {obs[33:45].round(3)}", flush=True)
            print(f"  delta_out:       {delta_isaac.round(3)}", flush=True)
            print(f"  target_q:        {target_q.round(3)}", flush=True)
            print(f"  actual_q:        {joint_pos.round(3)}", flush=True)
            print(f"  track_err:       {(target_q - joint_pos).round(3)}", flush=True)
            print(f"  tilt {tilt_deg:.1f}°  cmd={_cmd_vx:.2f}m/s", flush=True)

        # 13. 500Hz timing
        sl = (1.0 / CONTROL_HZ) - (time.time() - t_loop)
        if sl > 0:
            time.sleep(sl)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt.", flush=True)

finally:
    shutdown_event.set(); inf_thread.join(timeout=2.0)
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
                 step_dt    = np.array([1.0 / INFERENCE_HZ]),
                 src        = np.array(["real"], dtype=object),
                 )
        print(f"[LOG] Saved {_log_step} steps → {_rpath}", flush=True)
        print(f"  tilt mean={_log_tilt[:_log_step].mean():.1f}°  max={_log_tilt[:_log_step].max():.1f}°")
        print(f"  Contact: check FL vs FR balance — should be >40% each for trot")
        print(f"  Compare: python compare_sim_real.py sim_log_model_3500*.npz {_rpath}")
    else:
        print("[LOG] Too few steps to save.", flush=True)