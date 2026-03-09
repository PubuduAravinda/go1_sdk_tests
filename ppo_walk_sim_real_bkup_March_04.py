#!/usr/bin/env python3
"""
Go1 Real Robot Deployment — PPO policy (45D obs, normalizer baked in)
Threaded: control loop @ 500Hz, policy inference @ 50Hz.

Changes from HIMLoco SDK template:
  - Loads policy.pt (normalizer + actor fused) — no encoder
  - Correct action: policy_out * ACTION_SCALE + DEFAULT_JOINT_POS
  - Rate limiter on joint targets (hardware safety)
  - prev_actions in obs = policy_out * ACTION_SCALE (matches training env)
  - joint_vel and ang_vel clipped ±5 to match training
  - CLAMP_MIN/MAX removed — ACTION_SCALE handles bounds naturally
"""

import time
import threading
import numpy as np
import torch
import robot_interface as sdk

# ─── VERIFY THIS BEFORE FIRST RUN ────────────────────────────────────────────
# Print this from IsaacLab to confirm joint order:
#   print(env.unwrapped._robot.data.joint_names)
# Expected: FL_hip, FL_thigh, FL_calf, FR_hip, ... (per-leg order)
# If your URDF gives different order, update DEFAULT_JOINT_POS and ACTION_SCALE below.

# ─── POLICY CONFIG (must match go1_env.py training exactly) ──────────────────
# Action scale — balanced for foot clearance vs backdrive resistance
# scale=0.20 → range [0.60,1.00] only 6cm vertical → foot never unweights → hips stuck
# scale=0.30 → range [0.50,1.10] ~9cm clearance → foot lifts → hip can swing forward
# KP_thigh=40 (multiplier 1.4) now resists GRF backdrive at stance position
ACTION_SCALE = np.array([
    0.25, 0.25, 0.25, 0.25,   # hips   FL FR RL RR
    0.30, 0.30, 0.30, 0.30,   # thighs FL FR RL RR  (0.20→0.30, restore foot clearance)
    0.40, 0.40, 0.40, 0.40,   # knees  FL FR RL RR
], dtype=np.float32)

# CONFIRMED: Isaac joint_names order is GROUPED BY TYPE:
# [0]FL_hip [1]FR_hip [2]RL_hip [3]RR_hip
# [4]FL_thigh [5]FR_thigh [6]RL_thigh [7]RR_thigh
# [8]FL_calf  [9]FR_calf [10]RL_calf [11]RR_calf
DEFAULT_JOINT_POS = np.array([
    0.1,  0.1,  0.1,  0.1,   # hips   FL FR RL RR  (Isaac idx 0-3)
    0.8,  0.8,  0.8,  0.8,   # thighs FL FR RL RR  (Isaac idx 4-7)
   -1.5, -1.5, -1.5, -1.5,  # knees  FL FR RL RR  (Isaac idx 8-11)
], dtype=np.float32)

# Contact detection via knee tauEst — footForce is UNRELIABLE on Go1
# footForce reads the same value hanging vs on ground (raw ADC bias, not force)
# tauEst (estimated joint torque) rises sharply when leg bears body weight
# Run foot_force_calibrate.py to measure your robot's exact thresholds
# SDK knee indices: FR=2, FL=5, RR=8, RL=11
KNEE_TAU_THRESHOLD = 1.0   # Nm — hang noise=0.37 Nm, stand min=1.82 Nm → 1.0 safe midpoint
KNEE_IDX_SDK = {"FR": 2, "FL": 5, "RR": 8, "RL": 11}

# Per-joint rate limiter — max change per 2ms control step
# Hip needs 0.5 rad per 20ms policy step = 25 rad/s. Old 0.04 limit = 20 rad/s → 80% only
# Per-joint: hips faster (propulsion), thighs/knees slower (stability)
MAX_DELTA_PER_JOINT = np.array([
    0.06, 0.06, 0.06, 0.06,   # hips   — 30 rad/s, allows full ±0.25 swing
    0.04, 0.04, 0.04, 0.04,   # thighs — 20 rad/s, already stable
    0.05, 0.05, 0.05, 0.05,   # knees  — 25 rad/s
], dtype=np.float32)

# Obs clipping — must match go1_env.py _get_observations
JVEL_CLIP  = 5.0   # rad/s — matches torch.clamp(..., -5.0, 5.0)
ANGVEL_CLIP = 5.0  # rad/s — matches torch.clamp(root_ang_vel_b, -5.0, 5.0)

# ─── CONTROL CONFIG ───────────────────────────────────────────────────────────
KP_START        = 5.0
KP_STEP         = 3.0
RAMP_MAX_LEVEL  = 10    # max base KP = 5 + 10×3 = 35  (thigh KP = 35*1.4 = 49 ≈ sim KP=50)
RAMP_INTERVAL_S = 7.0   # was 5.0 — calibration showed KP=23 step caused fall, more time to settle

# Per-joint KD — Isaac grouped order [hips×4, thighs×4, knees×4]
KD_PER_JOINT = np.array([
    4.0, 4.0, 4.0, 4.0,    # hips   FL FR RL RR
    4.5, 4.5, 4.5, 4.5,    # thighs — KD reduced (5.5→4.5): KP=49 handles stiffness, less damping = faster lift
    4.5, 4.5, 4.5, 4.5,    # knees
], dtype=np.float32)

# Per-joint KP MULTIPLIER — thighs need higher stiffness to resist ground backdrive
# FR_thigh was backdriven from 0.8 to -0.695 absolute (1.3 rad error) under GRF
# At base KP=29: restoring torque = 29*1.3 = 37.7 Nm > Go1 thigh max (23.7 Nm) → saturates
# Boost thighs 40%: base KP=29 → effective 40. At small errors (<0.3 rad) well within torque
# KP_MULTIPLIER applied every step: thigh_kp = current_kp * 1.4
KP_MULTIPLIER = np.array([
    1.0, 1.0, 1.0, 1.0,    # hips   — base KP
    1.4, 1.4, 1.4, 1.4,    # thighs — 40% boost to resist backdrive
    1.0, 1.0, 1.0, 1.0,    # knees  — base KP
], dtype=np.float32)
TILT_THRESHOLD  = 20.0  # degrees — scale actions down above this (tighter for ground)
INFERENCE_HZ    = 50
CONTROL_HZ      = 500

# ─── SDK remapping (verified working for standing) ────────────────────────────
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

# ─── LOAD POLICY ─────────────────────────────────────────────────────────────
device = torch.device("cpu")

# policy.pt = EmpiricalNormalization + Actor MLP (45D → 12D)
# Exported by export_go1_policy.py from model_7000.pt
policy = torch.jit.load("policy.pt").to(device).eval()
print("Loaded policy.pt (normalizer + actor, 45D input)")

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
obs_lock     = threading.Lock()
action_lock  = threading.Lock()
shutdown_event = threading.Event()

# Policy runs on scaled delta actions — this is what prev_actions obs expects
# (training env: self._prev_actions = a = raw_out * action_scale)
_shared_obs    = np.zeros(45, dtype=np.float32)
_shared_action = np.zeros(12, dtype=np.float32)  # policy_out * ACTION_SCALE
_inference_ready = False

# Per-joint smoothing — hips need no smoothing for full stride, thighs/knees moderated
# Hip smoothing=1.0: full policy output, no lag. Rate limiter handles safety.
# Thigh/knee smoothing=0.7/0.8: reduces abrupt changes during stance phase
ACTION_SMOOTH_ALPHA = np.array([
    1.0, 1.0, 1.0, 1.0,   # hips   — no smoothing, full stride swing
    0.7, 0.7, 0.7, 0.7,   # thighs — moderate smoothing
    0.8, 0.8, 0.8, 0.8,   # knees  — moderate smoothing
], dtype=np.float32)
_prev_smooth_action = np.zeros(12, dtype=np.float32)


# ─── INFERENCE THREAD ─────────────────────────────────────────────────────────
def inference_thread_fn():
    """Policy inference at 50Hz — completely decoupled from 500Hz control loop."""
    global _shared_action, _inference_ready
    period = 1.0 / INFERENCE_HZ

    while not shutdown_event.is_set():
        t0 = time.time()

        # Snapshot obs (fast copy)
        with obs_lock:
            obs_snap = _shared_obs.copy()

        try:
            obs_t = torch.from_numpy(obs_snap).float().unsqueeze(0)  # (1, 45)

            with torch.no_grad():
                # policy.pt: normalizes internally, outputs raw network values
                raw_out = policy(obs_t).squeeze(0).cpu().numpy()     # (12,)

            # Clip raw network output to [-1, 1] before scaling
            # Policy was trained with action space Box(-1,1). On real hardware
            # OOD obs (e.g. rack hanging) can cause outputs like ±3 → ±1.5 rad
            # after scaling — dangerous. Clip here as first safety layer.
            raw_out = np.clip(raw_out, -1.0, 1.0)

            # Scale to physical joint deltas — same as training env
            scaled = raw_out * ACTION_SCALE   # (12,) in Isaac order

            # Per-joint EMA smoothing
            # Hips: alpha=1.0 (raw policy), Thighs: 0.7, Knees: 0.8
            global _prev_smooth_action
            smoothed = ACTION_SMOOTH_ALPHA * scaled + (1 - ACTION_SMOOTH_ALPHA) * _prev_smooth_action
            _prev_smooth_action = smoothed.copy()

            with action_lock:
                _shared_action[:] = smoothed
                _inference_ready  = True

        except Exception as e:
            print(f"[INFERENCE ERROR] {e}", flush=True)

        elapsed = time.time() - t0
        sleep_t = period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ─── SETUP ────────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# Warm up JIT model (avoids first-call delay on RPi)
print("Warming up policy...", flush=True)
with torch.no_grad():
    policy(torch.zeros(1, 45))
print("Warmup done.", flush=True)

inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
inf_thread.start()

print("\n" + "=" * 90)
print("Go1 PPO Policy Deployment  —  45D obs  |  control 500Hz  |  policy 50Hz")
print(f"DEFAULT_JOINT_POS: {DEFAULT_JOINT_POS}")
print(f"ACTION_SCALE:      {ACTION_SCALE}")
print(f"MAX_DELTA_PER_JOINT: hips={MAX_DELTA_PER_JOINT[0]:.2f} thighs={MAX_DELTA_PER_JOINT[4]:.2f} knees={MAX_DELTA_PER_JOINT[8]:.2f} rad/2ms")
print("Starting in 10 seconds — Ctrl+C to abort")
print("=" * 90 + "\n")
time.sleep(10)

# ─── MAIN CONTROL LOOP ────────────────────────────────────────────────────────
step_counter      = 0
t0_global         = time.time()
current_kp        = KP_START
tilt_exceeded_t   = None   # tracks when tilt first exceeded danger threshold
TILT_STOP_DEG     = 30.0   # auto-stop if tilted this much for > 0.3s

# Rate limiter state — tracks previous joint targets in Isaac order
prev_target_q = DEFAULT_JOINT_POS.copy()

# prev_actions for obs — scaled delta (matches self._prev_actions in training)
prev_scaled_actions = np.zeros(12, dtype=np.float32)

try:
    while True:
        t_loop = time.time()
        step_counter += 1
        t_elapsed = t_loop - t0_global

        # ── 1. Receive state ───────────────────────────────────────────────
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP RECV ERROR] {e}", flush=True)
            break

        # ── 2. Read joints — SDK → Isaac order ────────────────────────────
        joint_pos_sdk = np.array([state.motorState[i].q  for i in range(12)],
                                  dtype=np.float32)
        joint_vel_sdk = np.array([state.motorState[i].dq for i in range(12)],
                                  dtype=np.float32)
        joint_pos = joint_pos_sdk[sdk_to_isaac]   # Isaac order
        joint_vel = joint_vel_sdk[sdk_to_isaac]   # Isaac order

        # ── 3. IMU ────────────────────────────────────────────────────────
        acc   = np.array(state.imu.accelerometer, dtype=np.float32)
        norm  = max(float(np.linalg.norm(acc)), 0.1)
        proj_gravity = -acc / norm               # [0,0,-1] when upright
        tilt_deg     = float(np.degrees(np.sqrt(proj_gravity[0]**2 + proj_gravity[1]**2)))
        tilt_scale   = 0.5 if tilt_deg > TILT_THRESHOLD else 1.0

        # Auto-stop: if tilt > 30° for more than 0.3s → break loop (catching fall)
        if tilt_deg > TILT_STOP_DEG:
            if tilt_exceeded_t is None:
                tilt_exceeded_t = time.time()
            elif time.time() - tilt_exceeded_t > 0.3:
                print(f"[SAFETY STOP] Tilt {tilt_deg:.1f}° > {TILT_STOP_DEG}° for >0.3s — stopping!", flush=True)
                break
        else:
            tilt_exceeded_t = None

        gyro = np.array(state.imu.gyroscope, dtype=np.float32)

        # ── 4. Build 45D observation (must match _get_observations exactly) ─
        obs = np.zeros(45, dtype=np.float32)
        # [0:3]  velocity commands — TODO: replace with joystick input
        # 0.5 m/s = midpoint of trained range [0.3, 0.6] — strongest forward signal
        # At 0.3 m/s the gait was only bouncing vertically with no forward motion
        obs[0:3]   = [0.5, 0.0, 0.0]
        # [3:15] joint pos delta from default standing pose
        obs[3:15]  = joint_pos - DEFAULT_JOINT_POS
        # [15:27] joint velocity — clipped to match training
        obs[15:27] = np.clip(joint_vel, -JVEL_CLIP, JVEL_CLIP)
        # [27:30] base angular velocity — clipped to match training
        obs[27:30] = np.clip(gyro, -ANGVEL_CLIP, ANGVEL_CLIP)
        # [30:33] projected gravity
        obs[30:33] = proj_gravity
        # [33:45] previous actions — MUST be scaled delta (policy_out * ACTION_SCALE)
        #         Training env: self._prev_actions = a = network_out * action_scale
        obs[33:45] = prev_scaled_actions

        # ── Hip sign convention fix ────────────────────────────────────────
        # Isaac Lab URDF: FR_hip+ = outward (same as FL)
        # Real Go1 hardware: FR_hip+ = INWARD (mirrored)
        # Verified by hip_convention_test.py Phase 5 vs Phase 7
        # Must flip FR and RR hip in obs so policy sees correct sign
        # obs[3:7] = [FL_hip, FR_hip, RL_hip, RR_hip] deltas
        obs[4] = -obs[4]   # FR_hip delta: flip real→sim convention
        obs[6] = -obs[6]   # RR_hip delta: flip real→sim convention

        # ── 5. Push obs to inference thread ───────────────────────────────
        with obs_lock:
            _shared_obs[:] = obs

        # ── 6. Read latest policy output ──────────────────────────────────
        with action_lock:
            ready          = _inference_ready
            scaled_actions = _shared_action.copy() if ready else np.zeros(12, dtype=np.float32)

        # ── Hip sign convention fix (inverse of obs flip) ─────────────────
        # Policy outputs FR/RR hip in Isaac convention (positive = outward)
        # Must flip back to real hardware convention (positive = inward for FR/RR)
        # scaled_actions order: [0]FL_hip [1]FR_hip [2]RL_hip [3]RR_hip ...
        scaled_actions[1] = -scaled_actions[1]   # FR_hip: sim→real convention
        scaled_actions[3] = -scaled_actions[3]   # RR_hip: sim→real convention

        # Apply tilt safety scale
        scaled_actions = scaled_actions * tilt_scale

        # ── 7. Compute target joint positions ─────────────────────────────
        if ready:
            # target = network_out * action_scale + default_pos
            # (action_scale already applied in inference thread)
            raw_target = scaled_actions + DEFAULT_JOINT_POS
        else:
            # First step — hold standing pose until policy produces output
            raw_target = DEFAULT_JOINT_POS.copy()

        # ── 8. Per-joint rate limiter ────────────────────────────────────
        # Hips: 0.06 rad/2ms allows full ±0.25 swing in one policy step
        # Thighs/knees: tighter — already stable, don't need fast changes
        delta_limit = MAX_DELTA_PER_JOINT[isaac_to_sdk]
        target_q = np.clip(
            raw_target,
            prev_target_q - delta_limit,
            prev_target_q + delta_limit
        )
        prev_target_q[:] = target_q

        # Update prev_actions for obs — scaled delta from default (Isaac order)
        # This matches what training env stores in self._prev_actions
        prev_scaled_actions[:] = target_q - DEFAULT_JOINT_POS

        # ── 9. Apply commands — Isaac → SDK order ─────────────────────────
        target_q_sdk = target_q[isaac_to_sdk]

        # Per-joint KP and KD — remap Isaac → SDK order
        kd_sdk  = KD_PER_JOINT[isaac_to_sdk]
        kp_mult = KP_MULTIPLIER[isaac_to_sdk]

        # Per-joint feedforward tau
        # Thighs: 1.2 Nm gravity compensation so KP only handles tracking error
        # Hips/knees: 0.0 — minimal feedforward, KP sufficient
        # Sign: positive tau = flexion direction (lifting the leg)
        # In SDK order, thighs are indices determined by isaac_to_sdk mapping
        TAU_PER_JOINT_ISAAC = np.array([
            0.0, 0.0, 0.0, 0.0,    # hips
            1.2, 1.2, 1.2, 1.2,    # thighs — gravity compensation ~0.7-1.9 Nm at swing pos
            0.0, 0.0, 0.0, 0.0,    # knees
        ], dtype=np.float32)
        tau_sdk = TAU_PER_JOINT_ISAAC[isaac_to_sdk]

        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q    = float(target_q_sdk[i])
            cmd.motorCmd[i].dq   = 0.0
            cmd.motorCmd[i].Kp   = float(current_kp * kp_mult[i])  # per-joint KP
            cmd.motorCmd[i].Kd   = float(kd_sdk[i])                 # per-joint KD
            cmd.motorCmd[i].tau  = float(tau_sdk[i])                 # per-joint feedforward

        try:
            safe.PowerProtect(cmd, state, 9)   # safety wrapper
            udp.SetSend(cmd)
            udp.Send()
        except Exception as e:
            print(f"[UDP SEND ERROR] {e}", flush=True)
            break

        # ── 10. KP ramp ───────────────────────────────────────────────────
        ramp_level = min(RAMP_MAX_LEVEL, int(t_elapsed // RAMP_INTERVAL_S))
        current_kp = KP_START + ramp_level * KP_STEP

        # ── 11. Periodic status ───────────────────────────────────────────
        if step_counter % 100 == 1:
            # Contact via knee tauEst — reliable on Go1 unlike footForce
            knee_tau = {k: abs(state.motorState[idx].tauEst)
                        for k, idx in KNEE_IDX_SDK.items()}
            contact  = {k: knee_tau[k] > KNEE_TAU_THRESHOLD for k in ["FR","FL","RR","RL"]}
            feet = ("●" if contact["FR"] else "○") + ("●" if contact["FL"] else "○") +                    ("●" if contact["RR"] else "○") + ("●" if contact["RL"] else "○")
            print(
                f"t={t_elapsed:5.1f}s | "
                f"grav({proj_gravity[0]:+.2f},{proj_gravity[1]:+.2f},{proj_gravity[2]:+.2f}) | "
                f"tilt {tilt_deg:.1f}° | feet {feet} | kp {current_kp:.1f} | ready={ready}",
                flush=True
            )

        if step_counter % 500 == 0:
            print(f"\n[OBS DEBUG] step {step_counter}", flush=True)
            print(f"  joint_pos - default: {obs[3:15].round(3)}", flush=True)
            print(f"  joint_vel (clipped): {obs[15:27].round(3)}", flush=True)
            print(f"  ang_vel  (clipped):  {obs[27:30].round(3)}", flush=True)
            print(f"  proj_gravity:        {obs[30:33].round(3)}", flush=True)
            print(f"  prev_scaled_actions: {obs[33:45].round(3)}", flush=True)
            if ready:
                print(f"  scaled_actions:      {scaled_actions.round(3)}", flush=True)
                print(f"  target_q (Isaac):    {target_q.round(3)}", flush=True)
                print(f"  actual_q (Isaac):    {joint_pos.round(3)}", flush=True)
                print(f"  delta (tgt-actual):  {(target_q - joint_pos).round(3)}", flush=True)
                print(f"  rate_limit active:   {np.any(np.abs(raw_target - prev_target_q) > MAX_DELTA_PER_JOINT)}", flush=True)
                print(f"  knee_tauEst: FR={knee_tau['FR']:.2f} FL={knee_tau['FL']:.2f} RR={knee_tau['RR']:.2f} RL={knee_tau['RL']:.2f} Nm (threshold={KNEE_TAU_THRESHOLD})", flush=True)

        # ── 12. Maintain 500Hz ────────────────────────────────────────────
        elapsed = time.time() - t_loop
        sleep_t = (1.0 / CONTROL_HZ) - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

except KeyboardInterrupt:
    print("\nShutdown requested.", flush=True)

finally:
    shutdown_event.set()
    inf_thread.join(timeout=2.0)
    # Return to stand pose before exit
    for i in range(12):
        cmd.motorCmd[i].q  = float(DEFAULT_JOINT_POS[isaac_to_sdk[i]])
        cmd.motorCmd[i].Kp = 20.0
        cmd.motorCmd[i].Kd = float(KD_PER_JOINT[isaac_to_sdk[i]])
    udp.SetSend(cmd)
    udp.Send()
    print("Returned to stand pose. Loop ended.", flush=True)