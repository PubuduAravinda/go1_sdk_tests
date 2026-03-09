#!/usr/bin/env python3
"""
Hip Sign Convention Test
========================
Commands FL_hip and FR_hip the SAME positive value (+0.3 rad).

Watch the robot and record what you see:

  RESULT A: Both legs splay OUTWARD


  RESULT B: FL leg goes OUT, FR leg goes IN (scissors)
            → Convention MISMATCH
            → Need to flip FR_hip and RR_hip signs in go1_deploy.py

Run from rack with robot hanging — safe to observe leg directions clearly.
"""

import time
import numpy as np
import robot_interface as sdk

# ─── CONFIG ──────────────────────────────────────────────────────────────────
KP       = 20.0   # moderate — enough to move clearly, not violent
KD       = 4.0
HIP_CMD  = 0.3    # rad — clearly visible movement without stressing hardware

# Start from standing default
DEFAULT_Q = np.array([
    0.1,  0.1,  0.1,  0.1,   # hips   FL FR RL RR  (Isaac grouped order)
    0.8,  0.8,  0.8,  0.8,   # thighs
   -1.5, -1.5, -1.5, -1.5,  # knees
], dtype=np.float32)

# SDK ↔ Isaac remapping (verified correct)
sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
isaac_to_sdk = [0] * 12
for i in range(12):
    isaac_to_sdk[sdk_to_isaac[i]] = i

# ─── TEST PHASES ─────────────────────────────────────────────────────────────
# Each phase: (name, delta_from_default in Isaac order, hold_seconds)

phases = [
    # ── Phase 0: Return to default standing ──────────────────────────────────
    ("STAND (baseline)",
     np.zeros(12),
     4.0),

    # ── Phase 1: FL_hip only ─────────────────────────────────────────────────
    # Expected: only FL leg moves outward
    ("FL_hip +0.3 only",
     np.array([+0.3, 0.0, 0.0, 0.0,  0,0,0,0,  0,0,0,0]),
     4.0),

    # ── Phase 2: Back to default ─────────────────────────────────────────────
    ("STAND (reset)",
     np.zeros(12),
     3.0),

    # ── Phase 3: FR_hip only ─────────────────────────────────────────────────
    # Expected: only FR leg moves — which direction?
    ("FR_hip +0.3 only",
     np.array([0.0, +0.3, 0.0, 0.0,  0,0,0,0,  0,0,0,0]),
     4.0),

    # ── Phase 4: Back to default ─────────────────────────────────────────────
    ("STAND (reset)",
     np.zeros(12),
     3.0),

    # ── Phase 5: THE KEY TEST — both SAME sign ────────────────────────────────
    # If conventions match:  both legs go OUTWARD
    # If mismatch:           FL goes out, FR goes IN
    ("KEY TEST: FL_hip+0.3 AND FR_hip+0.3 (same sign)",
     np.array([+0.3, +0.3, 0.0, 0.0,  0,0,0,0,  0,0,0,0]),
     5.0),

    # ── Phase 6: Back to default ─────────────────────────────────────────────
    ("STAND (reset)",
     np.zeros(12),
     3.0),

    # ── Phase 7: Both OPPOSITE sign (hardware-correct abduction) ─────────────
    # This is what your test code does: FL+, FR- for equal outward abduction
    ("COMPARE: FL_hip+0.3 AND FR_hip-0.3 (opposite sign)",
     np.array([+0.3, -0.3, 0.0, 0.0,  0,0,0,0,  0,0,0,0]),
     5.0),

    # ── Phase 8: Return to default ───────────────────────────────────────────
    ("STAND (final)",
     np.zeros(12),
     3.0),
]

# ─── INIT ────────────────────────────────────────────────────────────────────
udp   = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe  = sdk.Safety(sdk.LeggedType.Go1)
cmd   = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("\n" + "=" * 70)
print("HIP SIGN CONVENTION TEST")
print("=" * 70)
print(f"KP={KP}  KD={KD}  HIP_CMD=±{HIP_CMD} rad")
print()
print("WATCH CAREFULLY during Phase 5 (KEY TEST):")
print("  Both legs splay OUT → conventions match → no fix needed")
print("  FL out, FR IN       → mismatch → need sign flip in deploy.py")
print()
print("Hang robot on rack. Starting in 10 seconds...")
print("=" * 70 + "\n")
time.sleep(10)

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
phase_idx    = 0
phase_start  = time.time()

print(f"\n▶  Phase 0: {phases[0][0]}")

try:
    while phase_idx < len(phases):
        time.sleep(0.002)   # 500 Hz loop

        # Receive state
        try:
            udp.Recv()
            udp.GetRecv(state)
        except Exception as e:
            print(f"[UDP RECV] {e}")
            break

        name, delta_isaac, hold_s = phases[phase_idx]

        # Build target in Isaac order, convert to SDK
        target_isaac = DEFAULT_Q + delta_isaac
        target_sdk   = target_isaac[isaac_to_sdk]

        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q    = float(target_sdk[i])
            cmd.motorCmd[i].dq   = 0.0
            cmd.motorCmd[i].Kp   = KP
            cmd.motorCmd[i].Kd   = KD
            cmd.motorCmd[i].tau  = 0.0

        try:
            safe.PowerProtect(cmd, state, 9)
            udp.SetSend(cmd)
            udp.Send()
        except Exception as e:
            print(f"[UDP SEND] {e}")
            break

        # Read actual hip positions for logging
        real_sdk   = np.array([state.motorState[i].q for i in range(12)], dtype=np.float32)
        real_isaac = real_sdk[sdk_to_isaac]

        # Advance phase
        elapsed = time.time() - phase_start
        if elapsed >= hold_s:
            # Print summary for this phase
            fl_hip_real = real_isaac[0]
            fr_hip_real = real_isaac[1]
            fl_hip_tgt  = target_isaac[0]
            fr_hip_tgt  = target_isaac[1]

            print(f"\n  Phase {phase_idx}: {name}")
            print(f"  Target:  FL_hip={fl_hip_tgt:+.3f}  FR_hip={fr_hip_tgt:+.3f}")
            print(f"  Actual:  FL_hip={fl_hip_real:+.3f}  FR_hip={fr_hip_real:+.3f}")
            print(f"  Delta from default:  FL={fl_hip_real-0.1:+.3f}  FR={fr_hip_real-0.1:+.3f}")

            if phase_idx == 5:   # KEY TEST phase
                print()
                print("  ┌─────────────────────────────────────────────────┐")
                print("  │  RECORD YOUR OBSERVATION:                       │")
                print("  │  A) Both legs went OUTWARD  → conventions match │")
                print("  │  B) FL out, FR inward       → MISMATCH          │")
                print("  └─────────────────────────────────────────────────┘")

            if phase_idx == 7:   # OPPOSITE sign phase
                print()
                print("  ┌─────────────────────────────────────────────────┐")
                print("  │  Compare with Phase 5:                          │")
                print("  │  If this looks MORE symmetric than Phase 5      │")
                print("  │  → opposite signs are correct for real hardware │")
                print("  └─────────────────────────────────────────────────┘")

            phase_idx  += 1
            phase_start = time.time()

            if phase_idx < len(phases):
                print(f"\n▶  Phase {phase_idx}: {phases[phase_idx][0]}")

except KeyboardInterrupt:
    print("\nAborted by user.")

finally:
    # Return to default
    target_sdk = DEFAULT_Q[isaac_to_sdk]
    for i in range(12):
        cmd.motorCmd[i].q  = float(target_sdk[i])
        cmd.motorCmd[i].Kp = 15.0
        cmd.motorCmd[i].Kd = 4.0
        cmd.motorCmd[i].tau = 0.0
    udp.SetSend(cmd)
    udp.Send()
    print("\nReturned to default. Done.")
    print()
    print("=" * 70)
    print("RESULT GUIDE:")
    print("  Phase 5 both OUT  → deploy.py is correct as-is")
    print("  Phase 5 FL out FR IN → add to go1_deploy.py:")
    print("    obs[4]  = -obs[4]               # flip FR_hip in obs")
    print("    obs[6]  = -obs[6]               # flip RR_hip in obs")
    print("    scaled_actions[1] = -scaled_actions[1]  # flip FR_hip action")
    print("    scaled_actions[3] = -scaled_actions[3]  # flip RR_hip action")
    print("=" * 70)