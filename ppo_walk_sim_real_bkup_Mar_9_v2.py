#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — PPO policy (45D obs, normalizer baked in)
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

Base: working HIMLoco SDK script.
4 changes only — everything else identical:

  CHANGE 1: ACTION_SCALE removed → tanh_squash() matches go1_env.py exactly
              delta = mid + half*tanh(raw), mid/half from DELTA_LO/HI per joint

  CHANGE 2: KP_MULTIPLIER updated to hit exact training KP values at ramp max
              KP_base_max = 5 + 10*3 = 35
              hip:   35 * 1.000 = 35  ✓ (training hip   KP = 35)
              thigh: 35 * 1.857 = 65  ✓ (training thigh KP = 65)
              knee:  35 * 2.286 = 80  ✓ (training knee  KP = 80)

  CHANGE 3: KD_PER_JOINT knee 4.5 → 5.0 to match go1_env_cfg.py exactly

  CHANGE 4: prev_actions = tanh_squash(raw) in Isaac convention, pre-mirror-flip
              go1_env.py _pre_physics_step: self._prev_actions = tanh_squash(raw)
              NOT rate-limited target, NOT scaled delta — raw tanh output.
              Smoothing removed — tanh already hard-bounds outputs.
"""

import time
import threading
import numpy as np
import torch
import robot_interface as sdk

# ── CHANGE 1: tanh squash replaces ACTION_SCALE ───────────────────────────────
# Exact match to go1_env.py _pre_physics_step:
#   _mid  = (hi + lo) / 2
#   _half = (hi - lo) / 2
#   a = _mid + _half * tanh(raw)
DELTA_LO = np.array([-0.15,-0.15,-0.15,-0.15, -0.35,-0.35,-0.35,-0.35, -0.35,-0.35,-0.35,-0.35], np.float32)
DELTA_HI = np.array([ 0.25, 0.25, 0.25, 0.25,  0.35, 0.35, 0.35, 0.35,  0.35, 0.35, 0.35, 0.35], np.float32)
_T_MID   = (DELTA_HI + DELTA_LO) * 0.5
_T_HALF  = (DELTA_HI - DELTA_LO) * 0.5

def tanh_squash(raw):
    return _T_MID + _T_HALF * np.tanh(np.clip(raw, -10.0, 10.0))

# ── Standing pose — unchanged ─────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,    # hips   FL FR RL RR  (Isaac idx 0-3)
    0.8,  0.8,  0.8,  0.8,    # thighs FL FR RL RR  (Isaac idx 4-7)
   -1.5, -1.5, -1.5, -1.5,    # knees  FL FR RL RR  (Isaac idx 8-11)
], dtype=np.float32)

# ── Contact detection — unchanged ─────────────────────────────────────────────
KNEE_TAU_THRESHOLD = 1.0
KNEE_IDX_SDK = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}

# ── Rate limiter — unchanged ──────────────────────────────────────────────────
MAX_DELTA_PER_JOINT = np.array([
    0.06, 0.06, 0.06, 0.06,
    0.04, 0.04, 0.04, 0.04,
    0.05, 0.05, 0.05, 0.05,
], dtype=np.float32)

# ── Obs clipping — unchanged ──────────────────────────────────────────────────
JVEL_CLIP   = 5.0
ANGVEL_CLIP = 5.0

# ── KP ramp — unchanged ───────────────────────────────────────────────────────
KP_START        = 5.0
KP_STEP         = 3.0
RAMP_MAX_LEVEL  = 10       # max base KP = 5 + 10*3 = 35
RAMP_INTERVAL_S = 7.0

# ── CHANGE 2 & 3: KP_MULTIPLIER and KD updated to match training exactly ──────
# CHANGE 2 — KP_MULTIPLIER
KP_MULTIPLIER = np.array([
    1.000, 1.000, 1.000, 1.000,    # hips:   35 * 1.000 = 35
    1.857, 1.857, 1.857, 1.857,    # thighs: 35 * 1.857 = 65
    2.286, 2.286, 2.286, 2.286,    # knees:  35 * 2.286 = 80
], dtype=np.float32)

# CHANGE 3 — KD knee 4.5 → 5.0
KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,    # hips
    4.5, 4.5, 4.5, 4.5,    # thighs
    5.0, 5.0, 5.0, 5.0,    # knees  (was 4.5 in base script)
], dtype=np.float32)

# ── Feedforward tau — unchanged ───────────────────────────────────────────────
TAU_PER_JOINT_ISAAC = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.2, 1.2, 1.2, 1.2,
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

# ── Safety — unchanged ────────────────────────────────────────────────────────
TILT_THRESHOLD = 20.0
TILT_STOP_DEG  = 30.0

# ── Timing — unchanged ────────────────────────────────────────────────────────
INFERENCE_HZ = 50
CONTROL_HZ   = 500

# ── SDK remapping — unchanged (confirmed working) ─────────────────────────────
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

# ── Load policy — unchanged ───────────────────────────────────────────────────
device = torch.device("cpu")
policy = torch.jit.load("go1_deploy/policy.pt").to(device).eval()
print("Loaded policy.pt (normalizer + actor, 45D input)")

# ── Shared state — unchanged ──────────────────────────────────────────────────
obs_lock       = threading.Lock()
action_lock    = threading.Lock()
shutdown_event = threading.Event()

_shared_obs     = np.zeros(45, dtype=np.float32)
_shared_raw_act = np.zeros(12, dtype=np.float32)   # raw network output (pre-tanh)
_inference_ready = False

# ── Inference thread — CHANGE 4 partial ───────────────────────────────────────
# Writes RAW network output. tanh_squash applied in control loop (not here).
# Smoothing removed — tanh already bounds outputs to (DELTA_LO, DELTA_HI).
def inference_thread_fn():
    global _shared_raw_act, _inference_ready
    period = 1.0 / INFERENCE_HZ
    while not shutdown_event.is_set():
        t0 = time.time()
        with obs_lock:
            obs_snap = _shared_obs.copy()
        try:
            obs_t = torch.from_numpy(obs_snap).float().unsqueeze(0)
            with torch.no_grad():
                raw_out = policy(obs_t).squeeze(0).cpu().numpy()
            # No ACTION_SCALE, no smoothing — raw output stored as-is
            with action_lock:
                _shared_raw_act[:] = raw_out
                _inference_ready   = True
        except Exception as e:
            print(f"[INFERENCE ERROR] {e}", flush=True)
        sl = period - (time.time() - t0)
        if sl > 0:
            time.sleep(sl)

# ── Setup — unchanged ─────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("Warming up policy...", flush=True)
with torch.no_grad():
    policy(torch.zeros(1, 45))
print("Warmup done.", flush=True)

inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
# FIX 1: NOT started here — started AFTER hold so first inference sees REAL obs.
# If started here: 17s of zeros → normalizer proj_grav[2]≈+9.5 → "upside-down panic"
# → prev_tanh_delta poisoned → OOD spiral from step 1.

print("\n" + "=" * 90)
print("Go1 PPO Policy — tanh squash | 45D obs | 500Hz ctrl | 50Hz policy")
print(f"DEFAULT: {DEFAULT_JOINT_POS}")
print(f"delta range: hip [{DELTA_LO[0]:.2f},{DELTA_HI[0]:.2f}] "
      f"thigh [{DELTA_LO[4]:.2f},{DELTA_HI[4]:.2f}] "
      f"knee [{DELTA_LO[8]:.2f},{DELTA_HI[8]:.2f}]")
print(f"KP at max ramp: hip={35*KP_MULTIPLIER[0]:.0f} "
      f"thigh={35*KP_MULTIPLIER[4]:.0f} knee={35*KP_MULTIPLIER[8]:.0f}")
print("Place robot on flat ground standing. Starting in 10 seconds.")
print("=" * 90 + "\n")
time.sleep(10)

# ══════════════════════════════════════════════════════════════════════════════
# PRE-POLICY: hold DEFAULT at full KP before handing to policy
#
# Why: at t=0 KP=5/9/11 — robot is too floppy for policy to stabilise.
# This stage runs pure position control, KP ramps 5→35/65/80 over HOLD_RAMP_S,
# then holds at full KP for HOLD_FULL_S so robot is stiff and upright
# before a single policy inference runs.
#
# Expected terminal during this stage:
#   [HOLD t= 0.0s] KP h=5  t=9  k=11  err=0.003  feet=●●●●
#   [HOLD t= 2.0s] KP h=18 t=33 k=41  err=0.002  feet=●●●●
#   [HOLD t= 4.0s] KP h=35 t=65 k=80  err=0.001  feet=●●●●  ← full KP
#   [HOLD t= 6.0s] KP h=35 t=65 k=80  err=0.001  feet=●●●●
#   [HOLD COMPLETE] handing to policy
#
# If err > 0.10 at full KP: robot is not at DEFAULT — check physical setup.
# If feet ≠ ●●●●: one foot not in contact — reposition before policy starts.
# ══════════════════════════════════════════════════════════════════════════════
HOLD_RAMP_S = 4.0    # seconds to ramp KP from KP_START → full training values
HOLD_FULL_S = 3.0    # seconds to hold at full KP before policy starts

print("[HOLD] Ramping KP to full training values before policy starts...")
_hold_t0 = time.perf_counter()
_hold_total = HOLD_RAMP_S + HOLD_FULL_S
_hold_step  = 0

while True:
    _t  = time.perf_counter()
    _dt = _t - _hold_t0
    if _dt >= _hold_total:
        break

    udp.Recv(); udp.GetRecv(state)

    # KP ramp: 0→HOLD_RAMP_S ramps from KP_START to full, then holds
    _alpha   = min(1.0, _dt / HOLD_RAMP_S)
    _kp_base = KP_START + _alpha * (KP_START + RAMP_MAX_LEVEL * KP_STEP - KP_START)
    _kp_sdk  = (KP_MULTIPLIER * _kp_base)[isaac_to_sdk]
    _kd_sdk  = KD_PER_JOINT[isaac_to_sdk]
    _tau_sdk = TAU_PER_JOINT_ISAAC[isaac_to_sdk]

    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].Kp   = float(_kp_sdk[i])
        cmd.motorCmd[i].Kd   = float(_kd_sdk[i])
        cmd.motorCmd[i].tau  = float(_tau_sdk[i])
    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd); udp.Send()

    # Print every 0.5s
    if _hold_step % (CONTROL_HZ // 2) == 0:
        _jpos = np.array([state.motorState[i].q for i in range(12)],
                         np.float32)[sdk_to_isaac]
        _err  = float(np.max(np.abs(_jpos - DEFAULT_JOINT_POS)))
        _knee = {k: abs(state.motorState[idx].tauEst)
                 for k, idx in KNEE_IDX_SDK.items()}
        _cont = {k: _knee[k] > KNEE_TAU_THRESHOLD for k in ["FR","FL","RR","RL"]}
        _feet = ("●" if _cont["FR"] else "○") + ("●" if _cont["FL"] else "○") + \
                ("●" if _cont["RR"] else "○") + ("●" if _cont["RL"] else "○")
        _kph  = _kp_base * KP_MULTIPLIER[0]
        _kpt  = _kp_base * KP_MULTIPLIER[4]
        _kpk  = _kp_base * KP_MULTIPLIER[8]
        print(f"[HOLD t={_dt:4.1f}s] KP h={_kph:.0f} t={_kpt:.0f} k={_kpk:.0f}"
              f"  err={_err:.3f}  feet={_feet}", flush=True)

    _hold_step += 1
    _sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - _t)
    if _sl > 0:
        time.sleep(_sl)

# Final check before handing to policy
# Average 20 readings at end of HOLD for stable equilibrium estimate
_eq_readings = []
for _ in range(20):
    udp.Recv(); udp.GetRecv(state)
    _eq_readings.append(np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac])
    time.sleep(0.005)
_jpos_final   = np.mean(_eq_readings, axis=0)
_err_final    = float(np.max(np.abs(_jpos_final - DEFAULT_JOINT_POS)))

# jdelta_offset: real hardware equilibrium minus sim DEFAULT.
# Applied to obs[3:15] so the policy sees near-zero jdelta when standing still.
# WHY: sim trains with DEFAULT as equilibrium → obs[3:15]≈0 at rest.
# Real hardware at KP=35/65/80 settles 0.04–0.12 rad from sim DEFAULT due to
# gravity + cable compliance. Without correction obs[3:15]≠0 → policy panics.
# offset is measured fresh each run — no hardcoded constants needed.
jdelta_offset = _jpos_final - DEFAULT_JOINT_POS
print(f"\n[HOLD COMPLETE] err_max={_err_final:.3f} rad from DEFAULT")
print(f"  jdelta_offset (real − sim_default):")
_JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip','FL_th','FR_th','RL_th','RR_th','FL_kn','FR_kn','RL_kn','RR_kn']
for _ji, _jn in enumerate(_JNAMES):
    print(f"    {_jn:8s}: {jdelta_offset[_ji]:+.4f}")

if _err_final > 0.20:
    print(f"  WARNING: err_max={_err_final:.3f} > 0.20 — robot may not be upright.")
    print(f"  Ctrl+C within 3s to abort, otherwise continuing...")
    time.sleep(3.0)
else:
    print(f"  Robot near DEFAULT ✓ — obs offset correction applied")
print()

# FIX 1: Start inference thread NOW — first obs is real sensor data, not zeros.
# proj_grav = [-0.02, +0.04, -1.00] (actual upright robot) → normalizer sees valid input.
inf_thread.start()
# Warm the thread: wait for first real inference before entering loop
# so _inference_ready=True from step 1 (no "else: DEFAULT" branch at all).
_t_wait = time.perf_counter()
while not _inference_ready:
    udp.Recv(); udp.GetRecv(state)   # keep sending hold cmd during wait
    for i in range(12):
        cmd.motorCmd[i].q  = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp = float((KP_MULTIPLIER * (KP_START + RAMP_MAX_LEVEL*KP_STEP))[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau= float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()
    if time.perf_counter() - _t_wait > 0.5:
        print("[WARN] inference thread not ready after 0.5s — check policy.pt", flush=True)
        break
    time.sleep(0.001)
print(f"[POLICY] First inference ready ({(time.perf_counter()-_t_wait)*1000:.0f}ms). Starting loop.", flush=True)

# prev_tanh_delta init = zeros.
# With jdelta_offset applied, obs[3:15]≈0 at real equilibrium → matches training.
# Policy natural output for obs≈0 is near zero → zeros is the correct seed.
# (Previously used _T_MID but that made raw_net WORSE: -345 vs -168)
prev_tanh_delta = np.zeros(12, np.float32)

# ── Policy warm-start: run 50 silent inferences at DEFAULT pose ───────────────
# Why: even with prev_tanh_delta=_T_MID, the FULL obs (jvel, ang_vel, etc.)
# from the real robot is slightly different from sim at step 0. After 50 steps
# (~1s) prev_tanh_delta converges to the policy's natural output for the
# DEFAULT-standing state, so the main loop starts with a valid history.
# The robot stays at DEFAULT during this phase (targets not applied).
#
# Expected: prev_tanh_delta hips converge from +0.05 toward ~-0.05 to -0.10
# (policy's natural standing output). If any joint hits ±limit during warmup,
# print a warning — means obs is still OOD at DEFAULT.
print("[WARMUP] Running 50 silent policy steps to seed prev_actions...", flush=True)
_warmup_steps = 50
for _ws in range(_warmup_steps):
    udp.Recv(); udp.GetRecv(state)

    # Build obs identical to main loop
    _jpos_w = np.array([state.motorState[i].q  for i in range(12)], np.float32)[sdk_to_isaac]
    _jvel_w = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
    _grav_w = np.array(state.imu.accelerometer,  np.float32)
    _gyro_w = np.array(state.imu.gyroscope,       np.float32)
    _grav_norm = np.linalg.norm(_grav_w)
    if _grav_norm > 0.1:
        _grav_w /= _grav_norm
    _obs_w = np.zeros(45, np.float32)
    _obs_w[0:3]   = [0.5, 0.0, 0.0]
    _obs_w[3:15]  = (_jpos_w - DEFAULT_JOINT_POS) - jdelta_offset  # zero-centred at real equilibrium
    _obs_w[15:27] = np.clip(_jvel_w, -JVEL_CLIP, JVEL_CLIP)
    _obs_w[27:30] = np.clip(_gyro_w, -ANGVEL_CLIP, ANGVEL_CLIP)
    _obs_w[30:33] = _grav_w
    _obs_w[33:45] = prev_tanh_delta
    _obs_w[4] = -_obs_w[4]
    _obs_w[6] = -_obs_w[6]

    with obs_lock:
        _shared_obs[:] = _obs_w

    time.sleep(1.0 / INFERENCE_HZ)   # wait for inference thread to consume obs

    with action_lock:
        _raw_w  = _shared_raw_act.copy()
        _ready_w = _inference_ready

    if _ready_w:
        _delta_w = tanh_squash(_raw_w)
        prev_tanh_delta[:] = _delta_w.copy()

    # Hold robot at DEFAULT throughout (DO NOT apply policy targets)
    for i in range(12):
        cmd.motorCmd[i].q  = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp = float((KP_MULTIPLIER*(KP_START+RAMP_MAX_LEVEL*KP_STEP))[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kd = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau= float(TAU_PER_JOINT_ISAAC[isaac_to_sdk[i]])
    safe.PowerProtect(cmd, state, 9); udp.SetSend(cmd); udp.Send()

_sat = np.sum(np.abs(prev_tanh_delta) >= np.array([0.14]*4 + [0.34]*8))
print(f"[WARMUP] Done. prev_tanh_delta after warmup: {prev_tanh_delta.round(3)}")
print(f"[WARMUP] Saturated joints: {_sat}/12  {'*** WARNING: OOD obs at DEFAULT' if _sat > 2 else '✓ normal'}", flush=True)


# ── Real-robot logger — same format as sim_log from play_log.py ──────────────
# Saves every policy step (50Hz) to real_log_<ts>.npz for sim-real comparison.
# Only logs at policy rate (every 10 control steps = 50Hz) to match sim log rate.
_LOG_STEPS   = 1500    # 30 seconds at 50Hz
_log_step    = 0
_log_obs     = np.zeros((_LOG_STEPS, 45),  np.float32)
_log_raw_net = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_tanh    = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_target  = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_actual  = np.zeros((_LOG_STEPS, 12),  np.float32)
_log_actual_qd = np.zeros((_LOG_STEPS, 12), np.float32)
_log_grav    = np.zeros((_LOG_STEPS, 3),   np.float32)
_log_angvel  = np.zeros((_LOG_STEPS, 3),   np.float32)
_log_tilt    = np.zeros(_LOG_STEPS,        np.float32)
_log_contact = np.zeros((_LOG_STEPS, 4),   np.float32)   # FL FR RL RR knee tauEst
_last_log_ctrl_step = -10   # log every 10 ctrl steps = 50Hz

from datetime import datetime as _dt
_log_ts = _dt.now().strftime("%Y%m%d_%H%M%S")

# ── Control loop state ────────────────────────────────────────────────────────
step_counter   = 0
t0_global      = time.time()
current_kp     = KP_START + RAMP_MAX_LEVEL * KP_STEP   # start at FULL KP — hold already ramped
tilt_exceeded_t = None
prev_target_q  = DEFAULT_JOINT_POS.copy()

# CHANGE 4: prev_actions = tanh_squash(raw), Isaac convention, pre-mirror-flip
# go1_env.py: self._prev_actions = _mid + _half * tanh(raw_network_out)
#
# CRITICAL INIT: must be _T_MID not zeros.
# _T_MID = tanh_squash(0) = [+0.05, +0.05, +0.05, +0.05, 0, 0, 0, 0, 0, 0, 0, 0]
# In training _prev_actions is NEVER zeros after episode reset step 0 — the policy
# always outputs something non-zero. Normalizer running mean/std for obs[33:45]
# was learned on values near _T_MID for hips, never on zeros.
# Feeding zeros → normalizer sees hip prev_action = (0 - mean)/std ≈ -5 → raw_net=-168
# Confirmed from real_log step 0: prev_actions=[0,...0], raw_net FL=-168.7 RL=-179.3
# at tilt=3.5° (perfectly upright) — pure normalizer OOD from bad init.
# prev_tanh_delta initialised above (before warmup loop) — do not re-init here.

try:
    while True:
        t_loop = time.time()
        step_counter += 1
        t_elapsed = t_loop - t0_global

        # 1. Receive state — unchanged
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP RECV ERROR] {e}", flush=True)
            break

        # 2. Read joints SDK → Isaac — unchanged
        joint_pos = np.array([state.motorState[i].q  for i in range(12)],
                              np.float32)[sdk_to_isaac]
        joint_vel = np.array([state.motorState[i].dq for i in range(12)],
                              np.float32)[sdk_to_isaac]

        # 3. IMU — unchanged
        acc          = np.array(state.imu.accelerometer, np.float32)
        gyro         = np.array(state.imu.gyroscope, np.float32)
        norm_a       = max(float(np.linalg.norm(acc)), 0.1)
        proj_gravity = -acc / norm_a
        tilt_deg     = float(np.degrees(np.sqrt(proj_gravity[0]**2 + proj_gravity[1]**2)))
        tilt_scale   = 0.5 if tilt_deg > TILT_THRESHOLD else 1.0

        if tilt_deg > TILT_STOP_DEG:
            if tilt_exceeded_t is None:
                tilt_exceeded_t = time.time()
            elif time.time() - tilt_exceeded_t > 0.3:
                print(f"[SAFETY STOP] Tilt {tilt_deg:.1f}deg — stopping!", flush=True)
                break
        else:
            tilt_exceeded_t = None

        # 4. Build 45D obs — unchanged except prev_actions source (CHANGE 4)
        obs = np.zeros(45, dtype=np.float32)
        obs[0:3]   = [0.5, 0.0, 0.0]   # FIX 2: 0.5=centre of training range [0.3,0.6]. 0.3 was -1.5σ OOD
        obs[3:15]  = (joint_pos - DEFAULT_JOINT_POS) - jdelta_offset  # zero-centred at real equilibrium
        obs[15:27] = np.clip(joint_vel, -JVEL_CLIP,   JVEL_CLIP)     # joint vel
        obs[27:30] = np.clip(gyro,      -ANGVEL_CLIP, ANGVEL_CLIP)   # ang vel
        obs[30:33] = proj_gravity                                      # projected gravity
        obs[33:45] = prev_tanh_delta   # CHANGE 4: tanh delta, Isaac, pre-flip

        # Hip sign convention flip — hardware Go1 vs Isaac Lab URDF:
        # Hardware FR_hip+ = INWARD (adduction) — confirmed by calibration
        # Isaac   FR_hip+ = OUTWARD (abduction)  — URDF definition
        # Without flip: obs reads FR inward as "outward" → wrong sign loop:
        #   policy sees "FR outward" → commands inward → no flip → hardware goes outward
        #   → policy sees "even more outward" → death spiral
        # doc5 (WITH flip) lasted 14s no safety stop vs doc7 (NO flip) 8s + safety stop.
        obs[4] = -obs[4]    # FR_hip: hardware → Isaac sign
        obs[6] = -obs[6]    # RR_hip: hardware → Isaac sign

        with obs_lock:
            _shared_obs[:] = obs

        # 6. Read latest policy output — raw (pre-tanh)
        with action_lock:
            ready   = _inference_ready
            raw_out = _shared_raw_act.copy() if ready else np.zeros(12, np.float32)

        # CHANGE 1 & 4: apply tanh_squash here, store pre-flip for prev_actions
        # HIP_SCALE: reduce hip delta sent to HARDWARE only.
        # CRITICAL: prev_tanh_delta must store UNSCALED tanh output.
        # Training stored self._prev_actions = full tanh_squash(raw) — no scaling.
        # If we store scaled hip (-0.06 instead of -0.15), policy sees "hip barely
        # moved" → outputs -45 next step → stores -0.06 again → escalates to -46, -47...
        # Confirmed in your data: step500=-14.7, step1000=-15.0, step1500=-45.6 GROWING.
        # Fix: scale hips ONLY for delta_hw (hardware send), NOT for prev_tanh_delta.
        # Action scales — applied to delta_hw ONLY, never stored in prev_tanh_delta.
        # doc5 thighs maxed at ±0.35 every step → large oscillation amplitude.
        # Reduce all joints to damp real-hardware oscillation without affecting obs.
        HIP_SCALE   = 0.5    # hip:   [-0.075, +0.125]
        THIGH_SCALE = 0.7    # thigh: [-0.245, +0.245]  was ±0.35, maxed causing oscillation
        KNEE_SCALE  = 0.8    # knee:  [-0.280, +0.280]
        if ready:
            delta_isaac = tanh_squash(raw_out)     # pure tanh, Isaac convention, no scaling

            # Store PURE tanh for obs — matches training _prev_actions exactly.
            # NEVER apply tilt_scale or any HW scale here.
            prev_tanh_delta[:] = delta_isaac.copy()

            # Hardware delta — all scaling here ONLY, never stored in obs
            delta_hw       = delta_isaac.copy()
            delta_hw      *= tilt_scale             # safety scale
            delta_hw[:4]  *= HIP_SCALE
            delta_hw[4:8] *= THIGH_SCALE
            delta_hw[8:]  *= KNEE_SCALE
            delta_hw[1]    = -delta_hw[1]    # FR_hip: Isaac → hardware sign
            delta_hw[3]    = -delta_hw[3]    # RR_hip: Isaac → hardware sign

            raw_target = DEFAULT_JOINT_POS + delta_hw
        else:
            raw_target = DEFAULT_JOINT_POS.copy()

        # 8. Rate limiter — unchanged
        target_q = np.clip(raw_target,
                           prev_target_q - MAX_DELTA_PER_JOINT,
                           prev_target_q + MAX_DELTA_PER_JOINT)
        prev_target_q[:] = target_q

        # 9. Send commands — unchanged
        target_q_sdk = target_q[isaac_to_sdk]
        kd_sdk       = KD_PER_JOINT[isaac_to_sdk]
        kp_mult_sdk  = KP_MULTIPLIER[isaac_to_sdk]
        tau_sdk      = TAU_PER_JOINT_ISAAC[isaac_to_sdk]

        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q    = float(target_q_sdk[i])
            cmd.motorCmd[i].dq   = 0.0
            cmd.motorCmd[i].Kp   = float(current_kp * kp_mult_sdk[i])
            cmd.motorCmd[i].Kd   = float(kd_sdk[i])
            cmd.motorCmd[i].tau  = float(tau_sdk[i])

        try:
            safe.PowerProtect(cmd, state, 9)
            udp.SetSend(cmd)
            udp.Send()
        except Exception as e:
            print(f"[UDP SEND ERROR] {e}", flush=True)
            break

        # 10. KP — full training values always (hold stage already ramped up)
        current_kp = KP_START + RAMP_MAX_LEVEL * KP_STEP   # = 35 base → h=35 t=65 k=80

        # 10b. Logger — write at 50Hz (every 10 ctrl steps) matching sim log rate
        # IMPORTANT: log obs AFTER prev_tanh_delta update so obs[33:45]=tanh[t-1]
        # which is what compare_sim_real.py checks.
        if ready and (step_counter - _last_log_ctrl_step) >= 10 and _log_step < _LOG_STEPS:
            _last_log_ctrl_step = step_counter
            _kn = [abs(state.motorState[KNEE_IDX_SDK[k]].tauEst) for k in ["FL","FR","RL","RR"]]
            _qd = np.array([state.motorState[i].dq for i in range(12)], np.float32)[sdk_to_isaac]
            # Rebuild obs[33:45] with the NOW-updated prev_tanh_delta for correct logging
            _log_obs_this           = obs.copy()
            _log_obs_this[33:45]    = prev_tanh_delta   # updated this step
            _log_obs[_log_step]     = _log_obs_this
            _log_raw_net[_log_step] = raw_out
            _log_tanh[_log_step]    = tanh_squash(raw_out)   # pure tanh = what was stored
            _log_target[_log_step]  = target_q
            _log_actual[_log_step]  = joint_pos
            _log_actual_qd[_log_step]= _qd
            _log_grav[_log_step]    = proj_gravity
            _log_angvel[_log_step]  = gyro
            _log_tilt[_log_step]    = tilt_deg
            _log_contact[_log_step] = _kn
            _log_step += 1

        # 11. Status — unchanged
        if step_counter % 100 == 1:
            knee_tau = {k: abs(state.motorState[idx].tauEst)
                        for k, idx in KNEE_IDX_SDK.items()}
            contact  = {k: knee_tau[k] > KNEE_TAU_THRESHOLD for k in ["FR","FL","RR","RL"]}
            feet = ("●" if contact["FR"] else "○") + ("●" if contact["FL"] else "○") + \
                   ("●" if contact["RR"] else "○") + ("●" if contact["RL"] else "○")
            kp_h = current_kp * KP_MULTIPLIER[0]
            kp_t = current_kp * KP_MULTIPLIER[4]
            kp_k = current_kp * KP_MULTIPLIER[8]
            print(f"t={t_elapsed:5.1f}s | "
                  f"grav({proj_gravity[0]:+.2f},{proj_gravity[1]:+.2f},{proj_gravity[2]:+.2f}) | "
                  f"tilt {tilt_deg:.1f}deg | feet {feet} | "
                  f"kp h={kp_h:.0f} t={kp_t:.0f} k={kp_k:.0f} | ready={ready}",
                  flush=True)

        if step_counter % 500 == 0:
            print(f"\n[DEBUG] step {step_counter}", flush=True)
            print(f"  jdelta:      {obs[3:15].round(3)}", flush=True)
            print(f"  jvel:        {obs[15:27].round(3)}", flush=True)
            print(f"  ang_vel:     {obs[27:30].round(3)}", flush=True)
            print(f"  proj_grav:   {obs[30:33].round(3)}", flush=True)
            print(f"  prev_action: {obs[33:45].round(3)}", flush=True)
            if ready:
                print(f"  raw_net_out: {raw_out.round(3)}", flush=True)
                print(f"  tanh_delta:  {tanh_squash(raw_out).round(3)}", flush=True)
                print(f"  target_q:    {target_q.round(3)}", flush=True)
                print(f"  actual_q:    {joint_pos.round(3)}", flush=True)
                print(f"  err:         {(target_q-joint_pos).round(3)}", flush=True)
                print(f"  knee tau: FL={knee_tau['FL']:.1f} FR={knee_tau['FR']:.1f} "
                      f"RL={knee_tau['RL']:.1f} RR={knee_tau['RR']:.1f} Nm", flush=True)

        # 12. Maintain 500Hz — unchanged
        sl = (1.0 / CONTROL_HZ) - (time.time() - t_loop)
        if sl > 0:
            time.sleep(sl)

except KeyboardInterrupt:
    print("\nShutdown.", flush=True)

finally:
    shutdown_event.set()
    inf_thread.join(timeout=2.0)
    for i in range(12):
        cmd.motorCmd[i].q    = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp   = 20.0
        cmd.motorCmd[i].Kd   = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau  = 0.0
    udp.SetSend(cmd)
    udp.Send()
    print("Stand pose. Done.", flush=True)

    # Save real robot log
    if _log_step > 10:
        _rpath = f"real_log_{_log_ts}.npz"
        # cmd is already in obs_raw[:,0:3] but saved separately for easy access
        # in compare_sim_real.py (matches sim log structure exactly)
        _log_cmd_arr = _log_obs[:_log_step, 0:3]
        np.savez(_rpath,
            obs_raw    = _log_obs[:_log_step],
            raw_net    = _log_raw_net[:_log_step],
            tanh_delta = _log_tanh[:_log_step],
            target_q   = _log_target[:_log_step],
            actual_q   = _log_actual[:_log_step],
            actual_qd  = _log_actual_qd[:_log_step],
            proj_grav  = _log_grav[:_log_step],
            ang_vel    = _log_angvel[:_log_step],
            cmd        = _log_cmd_arr,           # [vx,vy,wz] = obs[:,0:3] — matches sim log
            contact    = _log_contact[:_log_step],
            tilt_deg   = _log_tilt[:_log_step],
            default_q  = DEFAULT_JOINT_POS,
            delta_lo   = DELTA_LO,
            delta_hi   = DELTA_HI,
            step_dt    = np.array([1.0/INFERENCE_HZ]),
            src        = np.array(["real"], dtype=object),
        )
        print(f"[LOG] Saved {_log_step} steps → {_rpath}")
        print(f"  Copy to your laptop and run: python compare_sim_real.py sim_log_*.npz {_rpath}")
    else:
        print("[LOG] Too few steps to save.")