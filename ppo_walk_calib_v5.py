#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — model_24000 (go1_env.py v5 rewards)
policy.pt interface: raw_obs(45) → delta(12)  [normalise + MLP + tanh baked in]
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

PURPOSE: Confirm v5 gait rewards work on real hardware.
  Expected: FL and RR legs visibly lift during swing (knee clearance reward active)
  Expected: FR+RL remain mostly in stance (contact asymmetry from sim carries over)
  Expected: RL_th will jerk/drag — KNOWN fault, robot should still walk

CHANGES FROM PREVIOUS DEPLOY SCRIPT (go1_deploy_calib_v3):
───────────────────────────────────────────────────────────
  FIX 1: FR_TH_BINDING_MAX: 0.730 → 0.800
    Old value 0.730 was WRONG (comment said 0.820 but code had 0.730).
    Training cap: default(0.800) + delta(0.020) = 0.820.
    Deploy cap 0.800: gives 0.100 rad safety margin before binding zone (>0.900).
    Model_24000 FR_th sim actual max=0.818 — with 0.800 cap and real hardware
    overshoot, actual_q should peak ~0.850-0.870, safely below 0.900.

  FIX 2: KD thighs: 3.0 → 4.5 (training nominal)
    Training KD nominal thigh = 4.5, DR range [3.375, 5.625].
    Old 3.0 was BELOW training minimum — caused 6.6 Hz RR_th resonance in
    real log (confirmed: 51 sign changes in 3.8s, max vel 5.037 rad/s).
    FR_th exception: keeps 8.0 for binding zone protection (intentional mismatch).

  FIX 3: KD knees: 4.0 → 5.0 (training nominal)
    Training KD nominal knee = 5.0, DR range [4.0, 6.0].
    Old 4.0 was at the minimum — knees slightly underdamped on real hardware.

  FIX 4: TAU feedforward: all thighs 1.2 Nm → RL_th only 1.2 Nm
    1.2 Nm is for RL_th stiction pre-loading (τf=4.944 Nm fault).
    Applying to FL/FR/RR thighs (τf≈0.007 Nm healthy) was causing systematic
    over-flexion — healthy thighs would flex more than trained.

UNCHANGED FROM PREVIOUS:
  ✓ DELTA_LO/HI: hip ±0.08, thigh/knee ±0.35 — matches go1_env.py v5
  ✓ sdk_to_isaac / isaac_to_sdk remapping
  ✓ Hip sign flips: obs[4]=-obs[4], obs[6]=-obs[6], delta_hw[1]=-delta_hw[1], delta_hw[3]=-delta_hw[3]
  ✓ jdelta_offset measured in HOLD stage
  ✓ prev_delta update: always updated when ready (not % 10)
  ✓ HIP_SCALE = 1.00 (delta already bounded at ±0.08 by DELTA_LO/HI)
  ✓ KP: hip=35, thigh=65, knee=80 Nm/rad

HOW TO DEPLOY:
  1. Copy exported policy:
       cp logs/rsl_rl/go1_himloco/2026-05-06_19-57-43/exported/policy.pt \
          go1_deploy_model24000/policy.pt
  2. Run:
       python3 go1_deploy_model24000.py
  3. Watch for FL/RR knee lift during swing phase — confirms clearance reward.
     RL_th will jerk — expected, not a bug.

Observation layout (45D, Isaac order throughout):
  [0:3]   cmd [vx, vy, wz]
  [3:15]  jpos - default_q - jdelta_offset    (hw sign-flipped for FR/RR hip)
  [15:27] jvel clipped ±5
  [27:30] gyro clipped ±5
  [30:33] proj_gravity = -acc/|acc|
  [33:45] prev_delta  (policy output t-1, pre hw-flip)

SDK motor order (Go1 hardware — FR FL RR RL):
  SDK[0]=FR_hip  [1]=FR_th  [2]=FR_kn
  SDK[3]=FL_hip  [4]=FL_th  [5]=FL_kn
  SDK[6]=RR_hip  [7]=RR_th  [8]=RR_kn
  SDK[9]=RL_hip [10]=RL_th [11]=RL_kn
  sdk_to_isaac = [3,0,9,6, 4,1,10,7, 5,2,11,8]
"""

import time
import threading
from datetime import datetime
import numpy as np
import torch
import robot_interface as sdk

# ── tanh_squash bounds — MUST match go1_env.py v5 _delta_soft_lo / _delta_soft_hi
# go1_env.py v5: hip ±0.08, thigh ±0.35, knee ±0.35
DELTA_LO = np.array([
    -0.08, -0.08, -0.08, -0.08,   # hips ±0.08 (flat terrain, v3 real-robot fix)
    -0.35, -0.35, -0.35, -0.35,   # thighs ±0.35
    -0.35, -0.35, -0.35, -0.35,   # knees ±0.35
], dtype=np.float32)
DELTA_HI = np.array([
     0.08,  0.08,  0.08,  0.08,
     0.35,  0.35,  0.35,  0.35,
     0.35,  0.35,  0.35,  0.35,
], dtype=np.float32)

_T_MID  = (DELTA_HI + DELTA_LO) * 0.5   # all zeros (symmetric bounds)
_T_HALF = (DELTA_HI - DELTA_LO) * 0.5   # [0.08, 0.08, 0.08, 0.08, 0.35...]

def delta_to_raw_net(delta):
    """Invert tanh squash for logging raw network output."""
    ratio = np.clip((delta - _T_MID) / _T_HALF, -0.9999, 0.9999)
    return np.arctanh(ratio)

# ── Robot constants ──────────────────────────────────────────────────────────
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,    # hips
    0.8,  0.8,  0.8,  0.8,    # thighs
   -1.5, -1.5, -1.5, -1.5,   # knees
], dtype=np.float32)

# FIX 1: FR_th binding cap corrected from 0.730 → 0.800
# Training cap: default(0.800) + max_delta(0.020) = 0.820 rad
# Deploy cap 0.800: 0.100 rad margin before binding zone (>0.900)
# Model_24000 sim: FR_th actual max = 0.818 → deploy cap 0.800 clips rarely
# but protects against real-hardware overshoot (~0.05-0.08 rad in this model)
FR_TH_BINDING_MAX = 0.800   # rad

JVEL_CLIP   = 5.0
ANGVEL_CLIP = 5.0

# ── KP / KD ──────────────────────────────────────────────────────────────────
KP_START       = 5.0
KP_STEP        = 3.0
RAMP_MAX_LEVEL = 10          # KP_BASE = 5.0 + 10 × 3.0 = 35.0 Nm/rad
KP_MULTIPLIER  = np.array([
    1.000, 1.000, 1.000, 1.000,   # hips   → KP = 35 × 1.000 = 35 Nm/rad
    1.857, 1.857, 1.857, 1.857,   # thighs → KP = 35 × 1.857 = 65 Nm/rad
    2.286, 2.286, 2.286, 2.286,   # knees  → KP = 35 × 2.286 = 80 Nm/rad
], dtype=np.float32)

# KD in Isaac joint order.
# FIX 2: thighs 3.0 → 4.5 (training nominal, DR range [3.375, 5.625])
#         Old 3.0 was below minimum → RR_th oscillated at 6.6 Hz in real log
# FIX 3: knees 4.0 → 5.0 (training nominal, DR range [4.0, 6.0])
# FR_th (idx 5): keeps 8.0 intentionally — higher damping to protect binding fault
#   Not in training range but mechanically justified: slows FR_th before 0.900 binding zone
KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,   # hips   — training nominal = 4.0 ✓
    4.5, 8.0, 4.5, 4.5,   # thighs — nominal 4.5; FR_th[idx5]=8.0 (binding protection)
    5.0, 5.0, 5.0, 5.0,   # knees  — training nominal = 5.0 ✓
], dtype=np.float32)

# FIX 4: feedforward torque — RL_th only (Isaac index 6)
# RL_th stiction τf = 4.944 Nm. 1.2 Nm partial pre-load to help initiate motion.
# Previous script applied 1.2 Nm to ALL thighs — wrong.
# Healthy thigh τf ≈ 0.007 Nm (PACE). Applying 1.2 Nm caused over-flexion.
TAU_PER_JOINT_ISAAC = np.array([
    0.0, 0.0, 0.0, 0.0,   # hips
    0.0, 0.0, 1.2, 0.0,   # thighs: only RL_th[idx6] gets feedforward
    0.0, 0.0, 0.0, 0.0,   # knees
], dtype=np.float32)

# ── Hardware output scaling ───────────────────────────────────────────────────
# DELTA_LO/HI already at ±0.08 for hips — no additional scale needed
HIP_SCALE   = 1.00
THIGH_SCALE = 1.00
KNEE_SCALE  = 1.00

HIP_OBS_CORRECTION = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# ── Velocity command schedule ─────────────────────────────────────────────────
# Start slower than sim for initial test — confirms gait before pushing speed
VX_TARGET   = 0.5   # m/s steady state (sim trained at 0.3–0.9 range)
VX_KICK     = 0.6   # brief higher command to encourage locomotion onset
VX_RAMP_S   = 0.3   # 0–VX_KICK ramp over 0.3s
VX_KICK_S   = 1.0   # hold VX_KICK for 1.0s
VX_SETTLE_S = 1.0   # ramp VX_KICK → VX_TARGET over 1.0s

# ── Safety ────────────────────────────────────────────────────────────────────
TILT_THRESHOLD = 20.0   # deg — reduce output scale above this
TILT_STOP_DEG  = 30.0   # deg — emergency stop after 0.3s above this
HOLD_RAMP_S    = 4.0    # s — KP ramp duration
HOLD_FULL_S    = 3.0    # s — hold at full KP before inference
INFERENCE_HZ   = 50
CONTROL_HZ     = 500

# Per-step rate limiter — applied in 500Hz loop but policy updates at 50Hz
# Clips sudden large delta changes from stiction-release events (RL_th)
# and protects FR_th from rapid commands near binding zone
# Model_24000 FR_th delta p99 = 0.419, RR_kn p99 = 0.623 — these get clipped
MAX_DELTA_PER_JOINT = np.array([
    0.04, 0.04, 0.04, 0.04,   # hips: ±0.08 range, 0.04/step = 2 rad/s max
    0.08, 0.08, 0.08, 0.08,   # thighs: FR_th and RL_th need fast recovery
    0.08, 0.08, 0.08, 0.08,   # knees: RR_kn p99=0.623, clips to 0.08
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
# Export from: play.py --checkpoint model_24000.pt --log
# Saved to: logs/rsl_rl/go1_himloco/2026-05-06_19-57-43/exported/policy.pt
device = torch.device("cpu")
policy = torch.jit.load("go1_deploy_re_calib_v4/policy.pt").to(device).eval()
print("[POLICY] Loaded go1_deploy_model24000/policy.pt")

with torch.no_grad():
    _test  = policy(torch.zeros(1, 45))
    _hip_r = _test[0, :4].numpy()
    _fr_r  = _test[0, 5].item()
    _fl_kn = _test[0, 8].item()
    _rr_kn = _test[0, 11].item()

print(f"[POLICY] Hip delta at rest (expect ≈0.0): {_hip_r.round(4)}")
print(f"[POLICY] FR_th delta at rest: {_fr_r:.4f} → target "
      f"{DEFAULT_JOINT_POS[5]+_fr_r:.4f} rad  cap={FR_TH_BINDING_MAX:.3f}")
print(f"[POLICY] FL_kn delta at rest: {_fl_kn:.4f}  RR_kn: {_rr_kn:.4f}")
print(f"         (non-zero knees at rest = clearance reward shaping ✓ if ≈+0.13)")

if np.any(np.abs(_hip_r) > 0.06):
    print("[POLICY] WARNING: large hip delta at rest — check DELTA_LO/HI match training!")
if abs(_fr_r) > 0.15:
    print("[POLICY] WARNING: large FR_th delta at rest — check normalizer loaded correctly!")

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
print("Go1 model_24000 | v5 gait rewards | FL+RR knee lift expected")
print(f"  KP: hip=35  thigh=65  knee=80  Nm/rad")
print(f"  KD: hip=4.0  FR_th=8.0  other_th=4.5  knee=5.0  Nm·s/rad")
print(f"  FR_th cap: ≤{FR_TH_BINDING_MAX:.3f} rad (binding fault, training cap=0.820)")
print(f"  TAU feedforward: RL_th only = 1.2 Nm (stiction pre-load)")
print(f"  Expected: FL+RR lift knees clearly during swing")
print(f"  Expected: RL_th will jerk — KNOWN fault, confirm robot still walks")
print("  Place robot on flat ground. HOLD stage in 10s.")
print("="*72 + "\n")
time.sleep(10)

# ════════════════════════════════════════════════════════════════════════════
# HOLD STAGE — ramp KP to training gains, measure equilibrium offset
# ════════════════════════════════════════════════════════════════════════════
print("[HOLD] Ramping KP to training gains...")
_hold_t0   = time.perf_counter()
_hold_step = 0
KP_FULL    = KP_START + RAMP_MAX_LEVEL * KP_STEP   # 35.0 Nm/rad base

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
              f"hip={_kp_now*KP_MULTIPLIER[0]:.0f}  "
              f"thigh={_kp_now*KP_MULTIPLIER[4]:.0f}  "
              f"knee={_kp_now*KP_MULTIPLIER[8]:.0f}  "
              f"max_err={_err:.3f} rad", flush=True)
    _hold_step += 1
    _sl = (1.0 / CONTROL_HZ) - (time.perf_counter() - _t)
    if _sl > 0: time.sleep(_sl)

# Measure real equilibrium offset (jdelta_offset)
_eq = []
for _ in range(20):
    udp.Recv(); udp.GetRecv(state)
    _eq.append(np.array([state.motorState[i].q for i in range(12)], np.float32)[sdk_to_isaac])
    time.sleep(0.005)

jpos_eq       = np.mean(_eq, axis=0)
jdelta_offset = jpos_eq - DEFAULT_JOINT_POS
max_err       = float(np.max(np.abs(jdelta_offset)))

print(f"\n[HOLD COMPLETE] max_err={max_err:.3f} rad")
print("  jdelta_offset (subtract from jpos in obs[3:15]):")
for ji, jn in enumerate(_JNAMES):
    print(f"    {jn:8s}: {jdelta_offset[ji]:+.4f}")

# Specific diagnostics for fault joints
print(f"\n  FR_th offset: {jdelta_offset[5]:+.4f}  "
      f"(>+0.05 means FR_th already near binding — watch closely)")
print(f"  RL_th offset: {jdelta_offset[6]:+.4f}  "
      f"(expect negative = stiction holding short of target)")

if max_err > 0.20:
    print("  WARNING: err > 0.20 rad — check posture. Ctrl+C in 3s to abort.")
    time.sleep(3.0)
else:
    print("  OK — proceeding to inference.")
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
print(f"[POLICY] First inference ready ({(time.perf_counter()-_t_wait)*1000:.0f}ms).", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# WARMUP: 50 silent policy steps to seed prev_delta correctly
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
    _obs[4]     = -_obs[4]   # FR_hip sign flip
    _obs[6]     = -_obs[6]   # RR_hip sign flip
    _obs[3:7]  -= HIP_OBS_CORRECTION
    _obs[15:27] = np.clip(_jv,   -JVEL_CLIP,   JVEL_CLIP)
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
print(f"[WARMUP] Hip prev_delta: {prev_delta[:4].round(4)}  (should be ≈0.0)")
print(f"[WARMUP] Knee prev_delta (confirm lift bias): "
      f"FL={prev_delta[8]:+.4f}  RR={prev_delta[11]:+.4f}  "
      f"(expect ≈+0.13 from foot_clear reward)", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════
_LOG_STEPS     = 1500   # 30s at 50Hz
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

# Knee contact detection via tauEst (same as before)
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

# Per-joint tracking error accumulators (for final report)
_track_err_sum = np.zeros(12, np.float32)
_track_err_n   = 0

print("\n[CONTROL] Starting 500Hz loop. Ctrl+C to stop.")
print("  Watch for FL+RR knee lift during swing phase.\n")

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

        # Safety stop
        if tilt_deg > TILT_STOP_DEG:
            if tilt_exceeded_t is None:
                tilt_exceeded_t = time.time()
            elif time.time() - tilt_exceeded_t > 0.3:
                print(f"[SAFETY STOP] Tilt {tilt_deg:.1f}° > {TILT_STOP_DEG}°", flush=True)
                break
        else:
            tilt_exceeded_t = None

        # ── Build observation ─────────────────────────────────────────────
        # Velocity command schedule: ramp → kick → settle → steady
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
        obs[4]     = -obs[4]   # FR_hip sign flip
        obs[6]     = -obs[6]   # RR_hip sign flip
        obs[3:7]  -= HIP_OBS_CORRECTION
        obs[15:27] = np.clip(joint_vel, -JVEL_CLIP,   JVEL_CLIP)
        obs[27:30] = np.clip(gyro,      -ANGVEL_CLIP, ANGVEL_CLIP)
        obs[30:33] = proj_gravity
        obs[33:45] = prev_delta
        _obs_snapshot = obs.copy()

        with obs_lock:
            _shared_obs[:] = obs

        with action_lock:
            ready       = _inference_ready
            delta_isaac = _shared_delta.copy() if ready else np.zeros(12, np.float32)

        # Update prev_delta at 500Hz (value identical between 50Hz inferences)
        if ready:
            prev_delta[:] = delta_isaac.copy()

        if ready:
            delta_hw      = delta_isaac.copy()
            delta_hw     *= tilt_scale
            delta_hw[:4] *= HIP_SCALE
            delta_hw[4:8]*= THIGH_SCALE
            delta_hw[8:] *= KNEE_SCALE
            delta_hw[1]   = -delta_hw[1]   # FR_hip sign flip back to hw convention
            delta_hw[3]   = -delta_hw[3]   # RR_hip sign flip
            raw_target    = DEFAULT_JOINT_POS + delta_hw
        else:
            raw_target = DEFAULT_JOINT_POS.copy()

        # FIX 1: FR_th cap — corrected to 0.800 (was wrong 0.730)
        if raw_target[5] > FR_TH_BINDING_MAX:
            raw_target[5] = FR_TH_BINDING_MAX
            _fr_th_clamp_count += 1

        # Per-step rate limiter
        target_q = np.clip(raw_target,
                           prev_target_q - MAX_DELTA_PER_JOINT,
                           prev_target_q + MAX_DELTA_PER_JOINT)
        prev_target_q[:] = target_q

        # Tracking error accumulation
        if ready:
            _track_err_sum += np.abs(joint_pos - target_q)
            _track_err_n   += 1

        # Send to motors
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

        # ── Log at 50Hz ───────────────────────────────────────────────────
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

        # ── Console status every 100 control steps (0.2s) ─────────────────
        if step_counter % 100 == 1:
            _knt  = {k: abs(state.motorState[idx].tauEst)
                     for k, idx in KNEE_IDX_SDK.items()}
            _feet = "".join("●" if _knt[k] > KNEE_TAU_THRESHOLD else "○"
                            for k in ["FR", "FL", "RR", "RL"])
            # Knee positions relative to default — shows lift
            _fl_kn_lift = joint_pos[8]  - DEFAULT_JOINT_POS[8]   # positive = lifted
            _rr_kn_lift = joint_pos[11] - DEFAULT_JOINT_POS[11]  # positive = lifted
            print(f"t={t_elapsed:5.1f}s | "
                  f"tilt={tilt_deg:.1f}° | feet(FR FL RR RL)={_feet} | "
                  f"cmd={_cmd_vx:.2f} | "
                  f"FL_kn_lift={_fl_kn_lift:+.3f} RR_kn_lift={_rr_kn_lift:+.3f} | "
                  f"FR_clamp={_fr_th_clamp_count}", flush=True)

        # ── Detailed debug every 500 control steps (1s) ───────────────────
        if step_counter % 500 == 0 and ready:
            print(f"\n[DEBUG step {step_counter}]", flush=True)
            print(f"  cmd_vx:    {_cmd_vx:.3f}", flush=True)
            print(f"  jpos_delta:{obs[3:15].round(3)}", flush=True)
            print(f"  delta_out: {delta_isaac.round(3)}", flush=True)
            print(f"  target_q:  {target_q.round(3)}", flush=True)
            print(f"  actual_q:  {joint_pos.round(3)}", flush=True)
            print(f"  track_err: {(target_q - joint_pos).round(3)}", flush=True)
            print(f"  FAULT JOINTS:", flush=True)
            print(f"    FR_th: tgt={target_q[5]:.3f}  act={joint_pos[5]:.3f}  "
                  f"err={joint_pos[5]-target_q[5]:+.3f}  "
                  f"clamp_count={_fr_th_clamp_count}", flush=True)
            print(f"    RL_th: tgt={target_q[6]:.3f}  act={joint_pos[6]:.3f}  "
                  f"err={joint_pos[6]-target_q[6]:+.3f}", flush=True)
            print(f"  KNEE LIFT (vs default -1.5):", flush=True)
            for ki, kn in enumerate(['FL_kn','FR_kn','RL_kn','RR_kn']):
                idx  = ki + 8
                lift = joint_pos[idx] - DEFAULT_JOINT_POS[idx]
                print(f"    {kn}: act={joint_pos[idx]:.3f}  "
                      f"lift={lift:+.3f}  vel={joint_vel[idx]:+.3f}", flush=True)
            print(f"  tilt={tilt_deg:.1f}°  tilt_scale={tilt_scale}", flush=True)
            print(f"  hip delta: {delta_isaac[:4].round(4)}", flush=True)

        _sl = (1.0 / CONTROL_HZ) - (time.time() - t_loop)
        if _sl > 0:
            time.sleep(_sl)

except KeyboardInterrupt:
    print("\n[STOP] KeyboardInterrupt.", flush=True)

finally:
    shutdown_event.set()
    inf_thread.join(timeout=2.0)

    # Soft stand
    for i in range(12):
        cmd.motorCmd[i].q   = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp  = 20.0
        cmd.motorCmd[i].Kd  = float(KD_PER_JOINT[isaac_to_sdk[i]])
        cmd.motorCmd[i].tau = 0.0
    udp.SetSend(cmd); udp.Send()
    print("[FINAL] Stand pose sent.", flush=True)

    # ── Save log ──────────────────────────────────────────────────────────
    if _log_step > 10:
        _rpath = f"real_log_model24000_{_log_ts}.npz"
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

        # ── Summary ───────────────────────────────────────────────────────
        _ta   = _log_tanh[:_log_step]
        _aa   = _log_actual[:_log_step]
        _tilt = _log_tilt[:_log_step]
        print(f"\n  TILT: mean={_tilt.mean():.1f}°  max={_tilt.max():.1f}°  "
              f">15°={((_tilt>15).sum())} steps")
        print(f"  FR_th clamp activations: {_fr_th_clamp_count} "
              f"({_fr_th_clamp_count/max(_log_step,1)*100:.0f}% of logged steps)")
        print(f"  FR_th actual: mean={_aa[:,5].mean():.3f}  "
              f"max={_aa[:,5].max():.3f}  binding>0.900={((_aa[:,5]>0.900).sum())} steps")
        print(f"  RL_th actual: range=[{_aa[:,6].min():.3f},{_aa[:,6].max():.3f}]")

        # Knee lift summary — confirm gait reward effect
        print(f"\n  KNEE LIFT SUMMARY (vs default -1.5, positive = lifted):")
        for ki, kn in enumerate(['FL_kn','FR_kn','RL_kn','RR_kn']):
            idx  = ki + 8
            lift = _aa[:, idx] - DEFAULT_JOINT_POS[idx]
            print(f"    {kn}: max_lift={lift.max():+.3f}  mean_lift={lift.mean():+.3f}  "
                  f"lifted>0.05m_pct={(lift>0.05).mean()*100:.0f}%")
        print(f"  [Expected from sim: FL_kn max_lift≈+0.27, RR_kn≈+0.28, "
              f"FR_kn≈+0.04, RL_kn≈0.00]")

        # Mean tracking error per joint
        if _track_err_n > 0:
            mean_err = _track_err_sum / _track_err_n
            print(f"\n  MEAN TRACKING ERROR per joint:")
            for i, jn in enumerate(_JNAMES):
                flag = " *** HIGH" if mean_err[i] > 0.15 else ""
                print(f"    {jn:8s}: {mean_err[i]:.4f}{flag}")

        print(f"\n  → python compare_sim_real.py "
              f"<sim_log_model_24000_*.npz> {_rpath}")
    else:
        print("[LOG] Too few steps — nothing saved.", flush=True)