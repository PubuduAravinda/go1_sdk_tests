import time
import threading
import numpy as np
import torch
import robot_interface as sdk

# ─── CONFIG ─────────────────────────────────────────────────────────────────
KP_START        = 5.0
KD_FIXED        = 4.0
KP_STEP         = 5.0
RAMP_MAX_LEVEL  = 7
TILT_THRESHOLD  = 45.0   # degrees — scale actions down above this
RAMP_INTERVAL_SECONDS = 5.0

INFERENCE_HZ    = 50     # policy runs at 50 Hz (every 20ms)
CONTROL_HZ      = 500    # UDP loop target (every 2ms)

# Isaac Lab standing pose
STAND_Q_ISAAC = np.array([0.1, 0.1, 0.1, 0.1,
                           0.8, 0.8, 0.8, 0.8,
                          -1.5,-1.5,-1.5,-1.5], dtype=np.float64)

# Action clamping limits
CLAMP_MIN = np.array([-0.8,-0.8,-0.8,-0.8, -1.2,-1.2,-1.2,-1.2, -1.6,-1.6,-1.6,-1.6])
CLAMP_MAX = np.array([ 0.8, 0.8, 0.8, 0.8,  1.2, 1.2, 1.2, 1.2,  1.6, 1.6, 1.6, 1.6])

# Joint names (Isaac order)
joint_names = ["FL_hip","FR_hip","RL_hip","RR_hip",
               "FL_thigh","FR_thigh","RL_thigh","RR_thigh",
               "FL_calf","FR_calf","RL_calf","RR_calf"]

# SDK ↔ Isaac remapping
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
actor   = torch.jit.load("actor.pt").to(device).eval()
encoder = torch.jit.load("encoder.pt").to(device).eval()
print("Loaded actor.pt and encoder.pt successfully")

# ─── SHARED STATE (main → inference thread) ──────────────────────────────────
# Protected by obs_lock; inference thread reads, main thread writes
obs_lock      = threading.Lock()
shared_obs_history   = torch.zeros(5, 45, device=device)   # rolling history
shared_current_obs_t = torch.zeros(45, device=device)       # latest single obs

# Protected by action_lock; inference thread writes, main thread reads
action_lock      = threading.Lock()
latest_clamped   = np.zeros(12)          # result consumed by main loop
inference_ready  = False                 # True once first result is available

# ─── INFERENCE THREAD ────────────────────────────────────────────────────────
def inference_thread_fn():
    """Runs policy at INFERENCE_HZ independently of the UDP control loop."""
    global latest_clamped, inference_ready
    period = 1.0 / INFERENCE_HZ

    while not shutdown_event.is_set():
        t_start = time.time()

        # 1. Snapshot shared obs (fast copy, don't hold lock during inference)
        with obs_lock:
            hist_snap = shared_obs_history.clone()
            obs_snap  = shared_current_obs_t.clone()

        # 2. Run encoder + actor (this is the slow part — ~50-150ms on RPi4)
        try:
            with torch.no_grad():
                flat_hist  = hist_snap.flatten().unsqueeze(0)
                latent     = encoder(flat_hist)
                policy_obs = torch.cat([obs_snap.unsqueeze(0), latent], dim=1)
                raw_actions = actor(policy_obs).squeeze(0).cpu().numpy()

            clamped = np.clip(raw_actions, CLAMP_MIN, CLAMP_MAX)
            # print("clamped inference_thread_fn--->", clamped)

            # 3. Publish result for main loop to consume
            with action_lock:
                latest_clamped[:] = clamped
                inference_ready   = True

        except Exception as e:
            print(f"[INFERENCE THREAD ERROR] {e}", flush=True)

        # 4. Sleep remainder of period (inference may already have taken longer)
        elapsed = time.time() - t_start
        sleep_t = period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        # If inference took longer than period, next iteration starts immediately

# ─── SETUP ──────────────────────────────────────────────────────────────────
udp  = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)
cmd  = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

shutdown_event = threading.Event()

# Warm up models once before starting (avoids first-call JIT delay on RPi)
print("Warming up models...", flush=True)
with torch.no_grad():
    _dummy_hist = torch.zeros(1, 5*45, device=device)
    _dummy_lat  = encoder(_dummy_hist)
    _dummy_obs  = torch.zeros(1, 45 + _dummy_lat.shape[1], device=device)
    actor(_dummy_obs)
print("Warmup done.", flush=True)

# Start inference thread BEFORE the countdown so it's already running
inf_thread = threading.Thread(target=inference_thread_fn, daemon=True, name="inference")
inf_thread.start()

print("\n" + "="*90)
print("REAL GO1 POLICY INFERENCE  —  threaded (control @ 500Hz, policy @ 50Hz)")
print(f"STAND_Q: {STAND_Q_ISAAC}")
print(f"Clamp:  hips ±0.8 | thighs ±1.2 | calves ±1.6")
print("Starting in 10 seconds...")
print("="*90 + "\n")
time.sleep(10)

# ─── MAIN CONTROL LOOP ───────────────────────────────────────────────────────
step_counter = 0
t0           = time.time()
current_kp   = KP_START
prev_actions = np.zeros(12)

try:
    while True:
        t_loop_start = time.time()
        step_counter += 1
        t = t_loop_start - t0

        # ── 1. Receive state ────────────────────────────────────────────────
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"UDP RECV ERROR: {e}", flush=True)
            break

        # ── 2. Read joints (SDK → Isaac) ────────────────────────────────────
        joint_pos_sdk = np.array([state.motorState[i].q  for i in range(12)])
        joint_vel_sdk = np.array([state.motorState[i].dq for i in range(12)])
        joint_pos = joint_pos_sdk[sdk_to_isaac]
        joint_vel = joint_vel_sdk[sdk_to_isaac]

        # ── 3. IMU ──────────────────────────────────────────────────────────
        acc   = np.array(state.imu.accelerometer)
        norm  = max(np.linalg.norm(acc), 0.1)
        gravity    = -acc / norm
        tilt_angle = np.degrees(np.sqrt(gravity[0]**2 + gravity[1]**2))
        scale      = 0.5 if tilt_angle > TILT_THRESHOLD else 1.0

        # ── 4. Build base obs (45D) ─────────────────────────────────────────
        base_obs = np.zeros(45, dtype=np.float32)
        base_obs[0:3]  = [0.5, 0.0, 0.0]                   # commands vx/vy/yaw
        base_obs[3:15] = joint_pos - STAND_Q_ISAAC          # joint pos offset
        base_obs[15:27]= joint_vel                          # joint vel
        base_obs[27:30]= state.imu.gyroscope               # ang vel
        base_obs[30:33]= gravity                            # projected gravity
        base_obs[33:45]= prev_actions                       # previous actions

        # ── 5. Push obs to shared state (fast — just a tensor copy) ─────────
        current_obs_t = torch.from_numpy(base_obs).float().to(device)
        with obs_lock:
            shared_obs_history[-1] = current_obs_t              # overwrite latest
            shared_obs_history = torch.roll(shared_obs_history, shifts=-1, dims=0)
            shared_obs_history[-1] = current_obs_t
            shared_current_obs_t.copy_(current_obs_t)

        # ── 6. Read latest policy output (non-blocking) ──────────────────────
        with action_lock:
            ready          = inference_ready
            clamped_actions = latest_clamped.copy() if ready else np.zeros(12)

        # Apply tilt scale safety
        clamped_actions = clamped_actions * scale

        # ── 7. Compute target ───────────────────────────────────────────────
        if ready:
            # print('clamped_actions main-->',clamped_actions)
            target_q = STAND_Q_ISAAC + clamped_actions
        else:
            # Inference thread hasn't produced output yet — hold stand pose
            target_q = STAND_Q_ISAAC.copy()

        prev_actions = clamped_actions

        # ── 8. Apply commands (Isaac → SDK order) ───────────────────────────
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

        # ── 9. KP ramp ──────────────────────────────────────────────────────
        current_level = int(t // RAMP_INTERVAL_SECONDS)
        ramp_level    = min(RAMP_MAX_LEVEL, current_level)
        current_kp    = KP_START + ramp_level * KP_STEP

        # ── 10. Periodic status prints ───────────────────────────────────────
        if step_counter % 100 == 1:
            contact_str = "".join("●" if state.footForce[i] > 20 else "○" for i in range(4))
            print(f"\n{'='*90}", flush=True)
            print(f"t={t:5.1f}s | g=({gravity[0]:+.3f},{gravity[1]:+.3f},{gravity[2]:+.3f}) "
                  f"| tilt {tilt_angle:.1f}° | feet {contact_str} | kp {current_kp:.1f} "
                  f"| policy_ready={ready}", flush=True)
            print(f"{'='*90}", flush=True)

        if step_counter % 500 == 0:
            print(f"\n[OBS DEBUG] step {step_counter}", flush=True)
            print(f"  joint pos - stand_q: {base_obs[3:15].round(3)}", flush=True)
            print(f"  joint vel:           {base_obs[15:27].round(3)}", flush=True)
            print(f"  ang vel:             {base_obs[27:30].round(3)}", flush=True)
            print(f"  proj gravity:        {base_obs[30:33].round(3)}", flush=True)
            print(f"  prev actions:        {base_obs[33:45].round(3)}", flush=True)
            if ready:
                print(f"  clamped actions:     {clamped_actions.round(3)}", flush=True)
                print(f"  target_q:            {target_q.round(3)}", flush=True)
                print(f"  actual:              {joint_pos.round(3)}", flush=True)

        # ── 11. Maintain ~500Hz loop rate ─────────────────────────────────────
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