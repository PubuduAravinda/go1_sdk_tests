import time
import robot_interface as sdk

udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
safe = sdk.Safety(sdk.LeggedType.Go1)

cmd = sdk.LowCmd()
state = sdk.LowState()
udp.InitCmdData(cmd)

print("Test Low-Level Mode — stand robot and enter low-level mode first")
time.sleep(5)

while True:
    time.sleep(0.002)

    try:
        udp.Recv()
        udp.GetRecv(state)
        print(f"State OK | hip_q {state.motorState[1].q:.3f} knee_q {state.motorState[2].q:.3f}", flush=True)
    except Exception as e:
        print(f"Recv failed: {e}", flush=True)
        break

    # Simple stand command
    for i in range(12):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q = [0.0, 0.85, -1.75, 0.0, 0.85, -1.75, 0.0, 0.95, -1.85, 0.0, 0.95, -1.85][i]
        cmd.motorCmd[i].dq = 0.0
        cmd.motorCmd[i].Kp = 60.0
        cmd.motorCmd[i].Kd = 3.0
        cmd.motorCmd[i].tau = 7.0 if i in [2,5,8,11] else 0.0

    safe.PowerProtect(cmd, state, 9)
    udp.SetSend(cmd)
    udp.Send()