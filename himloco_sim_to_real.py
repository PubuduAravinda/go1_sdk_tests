import time
import threading
import numpy as np
import torch
import robot_interface as sdk

# ─── CONFIG ─────────────────────────────────────────────────────────────────
# !! CRITICAL: KP/KD MUST MATCH your Isaac Lab actuator config exactly !!
# Check your go1_env_cfg.py ActuatorNetCfg or DCMotorCfg stiffness/damping.
# Isaac default Go1: stiffness=20, damping=0.5
# Using higher KP than sim → joints 3x stiffer than trained → jumps/vibration
KP_START              = 5.0
KP_MAX                = 35.0      # ← SET TO YOUR TRAINING KP (check go1_env_cfg.py!)
KD_FIXED              = 0.5       # ← SET TO YOUR TRAINING KD (check go1_env_cfg.py!)
KP_STEP               = 5       # smaller steps now that KP_MAX is lower
RAMP_MAX_LEVEL        = 6         # 6 steps × 2.5 + 5.0 = 20.0 max
TILT_THRESHOLD        = 45.0      # degrees — halve action scale above this
RAMP_INTERVAL_SECONDS = 5.0
INFERENCE_HZ          = 50        # policy thread rate
CONTROL_HZ            = 500       # UDP loop target

# ── Velocity command sent to policy (MUST match training) ───────────────────
CMD_VX  = 0.3   # forward velocity m/s — start low, raise to walk
CMD_VY  = 0.0
CMD_YAW = 0.0

# ── Action smoothing — prevents violent jumps between policy steps ──────────
# α=1.0 → no smoothing (raw policy output).  α=0.2 → heavy smoothing.
# Start with 0.4. If still jumpy, lower to 0.2. If too sluggish, raise to 0.6.
ACTION_SMOOTH_ALPHA   = 0.4

# Isaac Lab standing pose
STAND_Q_ISAAC = np.array([ 0.1,  0.1,  0.1,  0.1,
                            0.8,  0.8,  0.8,  0.8,
                           -1.5, -1.5, -1.5, -1.5], dtype=np.float64)

# Action clamping limits
CLAMP_MIN = np.array([-0.8,-0.8,-0.8,-0.8, -1.2,-1.2,-1.2,-1.2, -1.6,-1.6,-1.6,-1.6])
CLAMP_MAX = np.array([ 0.8, 0.8, 0.8, 0.8,  1.2, 1.2, 1.2, 1.2,  1.6, 1.6, 1.6, 1.6])

# ── Observation scales — MUST match go1_env_cfg.py ObservationsCfg ──────────
# Verify against your training config before deploying!
OBS_SCALE_JOINT_POS = 1.0    # joint_pos - stand_q  (usually 1.0)
OBS_SCALE_JOINT_VEL = 0.05   # joint velocities      (raw is 20x too large without this!)
OBS_SCALE_ANG_VEL   = 0.25   # IMU gyroscope         (raw is 4x too large without this!)
OBS_SCALE_GRAVITY   = 1.0    # projected gravity
OBS_SCALE_ACTIONS   = 1.0    # previous actions

# Joint names in Isaac order (for debug prints)
joint_names = ["FL_hip","FR_hip","RL_hip","RR_hip",
               "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
               "FL_calf","FR_calf","RL_calf","RR_calf"]

# ── SDK <-> Isaac joint remapping ────────────────────────────────────────────
#
# Unitree Go1 SDK motor order:
#   SDK[ 0]=FR_hip,  SDK[ 1]=FR_thigh, SDK[ 2]=FR_calf,
#   SDK[ 3]=FL_hip,  SDK[ 4]=FL_thigh, SDK[ 5]=FL_calf,
#   SDK[ 6]=RR_hip,  SDK[ 7]=RR_thigh, SDK[ 8]=RR_calf,
#   SDK[ 9]=RL_hip,  SDK[10]=RL_thigh, SDK[11]=RL_calf
#
# Isaac Lab order:
#   Isaac[0]=FL_hip,  Isaac[1]=FR_hip,  Isaac[2]=RL_hip,  Isaac[3]=RR_hip,
#   Isaac[4]=FL_thigh,Isaac[5]=FR_thigh,Isaac[6]=RL_thigh,Isaac[7]=RR_thigh,
#   Isaac[8]=FL_calf, Isaac[9]=FR_calf, Isaac[10]=RL_calf,Isaac[11]=RR_calf
#
# sdk_to_isaac[sdk_idx] = isaac_idx
#   Use for READING state:  joint_pos_isaac = joint_pos_sdk[sdk_to_isaac]
#
# isaac_to_sdk[isaac_idx] = sdk_idx
#   Use for WRITING cmds:   target_q_sdk = target_q_isaac[isaac_to_sdk]
#
# sdk_to_isaac = the ORIGINAL values from calibration code (which were correct all along)
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7,  5, 2, 11,  8]
isaac_to_sdk = [1, 5, 9, 0, 4, 8,  3, 7, 11, 2,  6, 10]

# Compile-time sanity check
_check = [sdk_to_isaac[isaac_to_sdk[i]] for i in range(12)]
assert _check == list(range(12)), f"Mapping inverse check FAILED: {_check}"

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
actor   = torch.jit.load("actor.pt").to(device).eval()
encoder = torch.jit.load("encoder.pt").to(device).eval()
print("Loaded actor.pt and encoder.pt successfully")

# ─── SHARED STATE ────────────────────────────────────────────────────────────
obs_lock             = threading.Lock()
shared_obs_history   = torch.zeros(5, 45, device=device)
shared_current_obs_t = torch.zeros(45, device=device)

action_lock    = threading.Lock()
latest_clamped = np.zeros(12)
inference_ready = False
shutdown_event  = threading.Event()

# ─── INFERENCE THREAD ────────────────────────────────────────────────────────
def inference_thread_fn():
    """Runs encoder+actor at INFERENCE_HZ, completely independent of UDP loop."""
    global latest_clamped, inference_ready
    period = 1.0 / INFERENCE_HZ

    while not shutdown_event.is_set():
        t_start = time.time()

        # Snapshot shared obs without holding lock during slow inference
        with obs_lock:
            hist_snap = shared_obs_history.clone()
            obs_snap  = shared_current_obs_t.clone()

        try:
            with torch.no_grad():
                flat_hist   = hist_snap.flatten().unsqueeze(0)
                latent      = encoder(flat_hist)
                policy_obs  = torch.cat([obs_snap.unsqueeze(0), latent], dim=1)
                raw_actions = actor(policy_obs).squeeze(0).cpu().numpy()

            clamped = np.clip(raw_actions, CLAMP_MIN, CLAMP_MAX)

            with action_lock:
                latest_clamped[:] = clamped
                inference_ready   = True

        except Exception as e:
            print(f"[INFERENCE THREAD ERROR] {e}", flush=True)

        elapsed = time.time() - t_start
        sleep_t = period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

# ─── SETUP ──────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

# Model warmup — eliminates first-call JIT compilation spike on RPi
print("Warming up models...", flush=True)
with torch.no_grad():
    _dh = torch.zeros(1, 5*45, device=device)
    _dl = encoder(_dh)
    _do = torch.zeros(1, 45 + _dl.shape[1], device=device)
    actor(_do)
print("Warmup done.", flush=True)

# Print joint mapping for visual verification
unitree_sdk_names = ["FR_hip","FR_thigh","FR_calf","FL_hip","FL_thigh","FL_calf",
                     "RR_hip","RR_thigh","RR_calf","RL_hip","RL_thigh","RL_calf"]
print("\nJoint mapping (SDK -> Isaac):")
for s, i in enumerate(sdk_to_isaac):
    match = "OK" if unitree_sdk_names[s] == joint_names[i] else "WRONG!"
    print(f"  SDK[{s:2d}] {unitree_sdk_names[s]:<12} -> Isaac[{i:2d}] {joint_names[i]:<12}  {match}")

inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
inf_thread.start()

print("\n" + "="*90)
print("REAL GO1 POLICY INFERENCE  --  threaded (control @ 500Hz / policy @ 50Hz)")
print(f"STAND_Q: {STAND_Q_ISAAC}")
print(f"OBS scales: joint_vel x{OBS_SCALE_JOINT_VEL} | ang_vel x{OBS_SCALE_ANG_VEL}")
print(f"Clamp: hips +/-0.8 | thighs +/-1.2 | calves +/-1.6")
print("Starting in 10 seconds ...")
print("="*90 + "\n")
time.sleep(10)

# ─── MAIN CONTROL LOOP ───────────────────────────────────────────────────────
step_counter = 0
t0           = time.time()
current_kp   = KP_START
prev_actions    = np.zeros(12)
smoothed_actions = np.zeros(12)   # exponential moving average of policy output

try:
    while True:
        t_loop_start = time.time()
        step_counter += 1
        t = t_loop_start - t0

        # ── 1. Receive state ─────────────────────────────────────────────────
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"UDP RECV ERROR: {e}", flush=True)
            break

        # ── 2. Joint pos/vel (SDK -> Isaac order) ────────────────────────────
        joint_pos_sdk = np.array([state.motorState[i].q  for i in range(12)])
        joint_vel_sdk = np.array([state.motorState[i].dq for i in range(12)])
        joint_pos = joint_pos_sdk[sdk_to_isaac]   # now in Isaac order
        joint_vel = joint_vel_sdk[sdk_to_isaac]   # now in Isaac order

        # ── 3. IMU ───────────────────────────────────────────────────────────
        acc        = np.array(state.imu.accelerometer)
        norm       = max(np.linalg.norm(acc), 0.1)
        gravity    = -acc / norm
        tilt_angle = np.degrees(np.sqrt(gravity[0]**2 + gravity[1]**2))
        scale      = 0.5 if tilt_angle > TILT_THRESHOLD else 1.0

        # ── 4. Build base obs (45D) — scales MUST match go1_env_cfg.py ───────
        base_obs = np.zeros(45, dtype=np.float32)
        base_obs[0:3]  = [CMD_VX, CMD_VY, CMD_YAW]                                     # commands vx/vy/yaw
        base_obs[3:15] = (joint_pos - STAND_Q_ISAAC) * OBS_SCALE_JOINT_POS   # joint pos offset
        base_obs[15:27]= joint_vel   * OBS_SCALE_JOINT_VEL                   # joint vel  (scaled!)
        base_obs[27:30]= np.array(state.imu.gyroscope) * OBS_SCALE_ANG_VEL  # ang vel    (scaled!)
        base_obs[30:33]= gravity     * OBS_SCALE_GRAVITY                     # proj gravity
        base_obs[33:45]= prev_actions * OBS_SCALE_ACTIONS                    # prev actions

        # ── 5. Push obs to shared state (fast tensor copy, brief lock) ───────
        current_obs_t = torch.from_numpy(base_obs).float().to(device)
        with obs_lock:
            shared_obs_history = torch.roll(shared_obs_history, shifts=-1, dims=0)
            shared_obs_history[-1] = current_obs_t
            shared_current_obs_t.copy_(current_obs_t)

        # ── 6. Read latest policy output (completely non-blocking) ───────────
        with action_lock:
            ready           = inference_ready
            clamped_actions = latest_clamped.copy() if ready else np.zeros(12)

        clamped_actions = clamped_actions * scale   # tilt safety scaling

        # ── 7. Smooth actions + compute target ───────────────────────────────
        # Low-pass filter prevents violent torque spikes between policy steps.
        # smoothed = α × new + (1-α) × prev  (exponential moving average)
        if ready:
            smoothed_actions = (ACTION_SMOOTH_ALPHA * clamped_actions
                               + (1.0 - ACTION_SMOOTH_ALPHA) * smoothed_actions)
            target_q = STAND_Q_ISAAC + smoothed_actions
        else:
            target_q = STAND_Q_ISAAC.copy()

        prev_actions = smoothed_actions   # feed smoothed back as prev_actions obs

        # ── 8. Send commands (Isaac -> SDK order) ─────────────────────────────
        target_q_sdk = target_q[isaac_to_sdk]

        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q    = float(target_q_sdk[i])
            cmd.motorCmd[i].dq   = 0.0
            cmd.motorCmd[i].Kp   = float(current_kp)
            cmd.motorCmd[i].Kd   = float(KD_FIXED)
            cmd.motorCmd[i].tau  = 0.3

        try:
            udp.SetSend(cmd)
            udp.Send()
        except Exception as e:
            print(f"UDP SEND ERROR: {e}", flush=True)
            break

        # ── 9. KP ramp ───────────────────────────────────────────────────────
        current_level = int(t // RAMP_INTERVAL_SECONDS)
        ramp_level    = min(RAMP_MAX_LEVEL, current_level)
        current_kp    = min(KP_MAX, KP_START + ramp_level * KP_STEP)

        # ── 10. Status print ~every 0.2s ─────────────────────────────────────
        if step_counter % 100 == 1:
            contact_str = "".join("●" if state.footForce[i] > 20 else "○" for i in range(4))
            foot_forces = [state.footForce[i] for i in range(4)]
            print(f"\n{'='*90}", flush=True)
            print(f"t={t:5.1f}s | g=({gravity[0]:+.3f},{gravity[1]:+.3f},{gravity[2]:+.3f}) "
                  f"| tilt {tilt_angle:.1f}deg | feet {contact_str} {foot_forces} "
                  f"| kp {current_kp:.1f} | ready={ready}", flush=True)
            print(f"{'='*90}", flush=True)

        # ── 11. Full debug ~every 1s ──────────────────────────────────────────
        if step_counter % 500 == 0:
            # ── A. RAW SDK SPACE (ground truth — no remapping) ───────────────
            sdk_unitree_names = ["FR_hip","FR_thigh","FR_calf",
                                 "FL_hip","FL_thigh","FL_calf",
                                 "RR_hip","RR_thigh","RR_calf",
                                 "RL_hip","RL_thigh","RL_calf"]
            target_q_sdk = target_q[isaac_to_sdk]   # what we actually commanded
            print(f"\n{'='*90}", flush=True)
            print(f"[RAW SDK DEBUG] step={step_counter}  t={t:.1f}s  kp={current_kp:.1f}  tilt={tilt_angle:.1f}deg  scale={scale}", flush=True)
            print(f"  {'SDK[i]':<7} {'Unitree':<13} {'Commanded':>10} {'Actual':>10} {'TrackErr':>10} {'Vel':>8}", flush=True)
            print(f"  {'-'*65}", flush=True)
            for i in range(12):
                cmd_val = float(target_q_sdk[i])
                act_val = float(joint_pos_sdk[i])
                vel_val = float(joint_vel_sdk[i])
                err     = act_val - cmd_val
                flag    = "  *** NOT TRACKING" if abs(err) > 0.20 else ""
                print(f"  SDK[{i:2d}]  {sdk_unitree_names[i]:<13} {cmd_val:>+10.4f} {act_val:>+10.4f} {err:>+10.4f} {vel_val:>+8.3f}{flag}", flush=True)

            # ── B. ISAAC-SPACE TRACKING ───────────────────────────────────────
            print(f"\n  [ISAAC JOINT TRACKING]", flush=True)
            print(f"  {'Joint':<12} {'Action':>8} {'Target':>8} {'Actual':>8} {'Err':>8} {'|Err|':>6}", flush=True)
            print(f"  {'-'*60}", flush=True)
            for j in range(12):
                act = float(joint_pos[j])
                tgt = float(target_q[j])
                act_j = float(clamped_actions[j]) if ready else 0.0
                err = act - tgt
                flag = "  *** STUCK" if abs(err) > 0.25 and abs(float(joint_vel[j])) < 0.05 else \
                       "  *** LARGE" if abs(err) > 0.25 else ""
                print(f"  {joint_names[j]:<12} {act_j:>+8.4f} {tgt:>8.4f} {act:>8.4f} {err:>+8.4f} {abs(err):>6.3f}{flag}", flush=True)

            # ── C. POLICY INPUT/OUTPUT ANALYSIS ──────────────────────────────
            print(f"\n  [POLICY OBS — what network sees]", flush=True)
            print(f"  commands (vx,vy,yaw)  : {base_obs[0:3].round(4)}", flush=True)
            print(f"  joint_pos_offset      : {base_obs[3:15].round(4)}", flush=True)
            print(f"  joint_vel_scaled      : {base_obs[15:27].round(4)}", flush=True)
            print(f"  ang_vel_scaled        : {base_obs[27:30].round(4)}", flush=True)
            print(f"  proj_gravity          : {base_obs[30:33].round(4)}", flush=True)
            print(f"  prev_actions          : {base_obs[33:45].round(4)}", flush=True)

            # Gravity interpretation
            gx, gy, gz = base_obs[30], base_obs[31], base_obs[32]
            pitch_deg = float(np.degrees(np.arctan2(abs(gx), abs(gz))))
            roll_deg  = float(np.degrees(np.arctan2(abs(gy), abs(gz))))
            print(f"  → body pitch={pitch_deg:.1f}deg  roll={roll_deg:.1f}deg  (0°=level)", flush=True)

            if ready:
                print(f"\n  [POLICY OUTPUT]", flush=True)
                raw  = latest_clamped
                clmp = clamped_actions
                print(f"  {'Joint':<12} {'raw_action':>11} {'clamped':>9} {'scaled':>9} {'at_limit?':>10}", flush=True)
                print(f"  {'-'*58}", flush=True)
                for j in range(12):
                    at_min = abs(raw[j] - CLAMP_MIN[j]) < 0.01
                    at_max = abs(raw[j] - CLAMP_MAX[j]) < 0.01
                    limit_str = "AT_MIN" if at_min else ("AT_MAX" if at_max else "")
                    print(f"  {joint_names[j]:<12} {raw[j]:>+11.4f} {clmp[j]:>+9.4f} {clamped_actions[j]:>+9.4f}  {limit_str}", flush=True)

            # ── D. FOOT FORCES ────────────────────────────────────────────────
            forces = [state.footForce[i] for i in range(4)]
            print(f"\n  [FOOT FORCES] FL={forces[0]:.0f}  FR={forces[1]:.0f}  RL={forces[2]:.0f}  RR={forces[3]:.0f}", flush=True)
            print(f"{'='*90}", flush=True)

        # ── 12. Maintain ~500Hz loop rate ────────────────────────────────────
        elapsed = time.time() - t_loop_start
        sleep_t = (1.0 / CONTROL_HZ) - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

except KeyboardInterrupt:
    print("\nKeyboard interrupt — shutting down.", flush=True)

finally:
    shutdown_event.set()
    inf_thread.join(timeout=2.0)
    print("Loop ended.", flush=True)