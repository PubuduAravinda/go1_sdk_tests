#!/usr/bin/python

import sys
import time
import math

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# Motor mapping dictionary
MOTOR_MAP = {
    'FR_0': 0, 'FR_1': 1, 'FR_2': 2,  # Front Right
    'FL_0': 3, 'FL_1': 4, 'FL_2': 5,  # Front Left
    'RR_0': 6, 'RR_1': 7, 'RR_2': 8,  # Rear Right
    'RL_0': 9, 'RL_1': 10, 'RL_2': 11  # Rear Left
}


class LegController:
    def __init__(self, udp, cmd, state):
        self.udp = udp
        self.cmd = cmd
        self.state = state
        self.motiontime = 0
        self.movement_phase = 0
        self.leg_phase = 0  # 0:FL, 1:FR, 2:RL, 3:RR
        self.phase_start_time = 0
        self.phase_duration = 1000  # 2 seconds per movement phase

        # Leg sequence
        self.leg_sequence = ['FL', 'FR', 'RL', 'RR']

        # Initial positions for all legs
        self.initial_positions = {
            'FR_0': 0.0, 'FR_1': 1.7, 'FR_2': -3.0,
            'FL_0': 0.0, 'FL_1': 1.7, 'FL_2': -3.0,
            'RR_0': 0.0, 'RR_1': 1.7, 'RR_2': -3.0,
            'RL_0': 0.0, 'RL_1': 1.7, 'RL_2': -3.0
        }

    def enable_motors(self):
        """Enable all motors with moderate stiffness"""
        for i in range(12):
            self.cmd.motorCmd[i].mode = 0x0A  # Motor enable
            self.cmd.motorCmd[i].Kp = 20.0  # Moderate stiffness
            self.cmd.motorCmd[i].Kd = 0.8  # Moderate damping

    def set_initial_positions(self):
        """Set all legs to initial positions"""
        for motor_name, target_pos in self.initial_positions.items():
            motor_id = MOTOR_MAP[motor_name]
            self.cmd.motorCmd[motor_id].q = target_pos

    def move_single_joint(self, leg_prefix, joint_suffix, target_pos, current_time, duration):
        """Move a single joint of a specific leg"""
        progress = min(1.0, (current_time - self.phase_start_time) / duration)
        motor_name = f"{leg_prefix}_{joint_suffix}"
        motor_id = MOTOR_MAP[motor_name]

        # Get current position for smooth interpolation
        current_pos = self.state.motorState[motor_id].q
        self.cmd.motorCmd[motor_id].q = current_pos + (target_pos - current_pos) * progress

        return progress >= 1.0

    def execute_leg_sequence(self):
        """Execute movement sequence for one leg at a time"""
        current_leg = self.leg_sequence[self.leg_phase]
        progress = min(1.0, (self.motiontime - self.phase_start_time) / self.phase_duration)

        # Phase 0: Move hip from 1.7 to 0
        if self.movement_phase == 0:
            completed = self.move_single_joint(current_leg, '1', 0.0, self.motiontime, self.phase_duration)
            if completed:
                self.movement_phase = 1
                self.phase_start_time = self.motiontime
                print(f"{current_leg} hip moved to 0, now moving knee...")

        # Phase 1: Move knee from -3.0 to -1.0
        elif self.movement_phase == 1:
            completed = self.move_single_joint(current_leg, '2', -1.0, self.motiontime, self.phase_duration)
            if completed:
                self.movement_phase = 2
                self.phase_start_time = self.motiontime
                print(f"{current_leg} knee moved to -1.0, now moving abductor...")

        # Phase 2: Move abductor from 0 to -0.2
        elif self.movement_phase == 2:
            completed = self.move_single_joint(current_leg, '0', -0.2, self.motiontime, self.phase_duration)
            if completed:
                self.movement_phase = 3
                self.phase_start_time = self.motiontime
                print(f"{current_leg} abductor moved to -0.2, now moving to 0.5...")

        # Phase 3: Move abductor from -0.2 to 0.5
        elif self.movement_phase == 3:
            completed = self.move_single_joint(current_leg, '0', 0.5, self.motiontime, self.phase_duration)
            if completed:
                self.movement_phase = 4
                self.phase_start_time = self.motiontime
                print(f"{current_leg} abductor moved to 0.5, now moving to 0.0...")

        # Phase 4: Move abductor from 0.5 to 0.0
        elif self.movement_phase == 4:
            completed = self.move_single_joint(current_leg, '0', 0.0, self.motiontime, self.phase_duration)
            if completed:
                self.movement_phase = 5
                self.phase_start_time = self.motiontime
                print(f"{current_leg} abductor moved to 0.0, now resetting hip and knee...")

        # Phase 5: Reset hip to 1.7 and knee to -3.0
        elif self.movement_phase == 5:
            hip_completed = self.move_single_joint(current_leg, '1', 1.7, self.motiontime, self.phase_duration)
            knee_completed = self.move_single_joint(current_leg, '2', -3.0, self.motiontime, self.phase_duration)

            if hip_completed and knee_completed:
                self.movement_phase = 0
                self.leg_phase = (self.leg_phase + 1) % len(self.leg_sequence)
                self.phase_start_time = self.motiontime
                next_leg = self.leg_sequence[self.leg_phase]
                print(f"{current_leg} reset complete, now moving to {next_leg} leg...")


if __name__ == '__main__':
    HIGHLEVEL = 0xee
    LOWLEVEL = 0xff
    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)

    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    # Create leg controller
    leg_controller = LegController(udp, cmd, state)

    print("Starting sequential single leg movement...")
    print("Sequence: Front Left → Front Right → Rear Left → Rear Right")

    while True:
        time.sleep(0.002)
        leg_controller.motiontime += 1

        udp.Recv()
        udp.GetRecv(state)

        # Enable motors and set initial positions after 100 cycles
        if leg_controller.motiontime == 100:
            print("Enabling all motors and setting initial positions...")
            leg_controller.enable_motors()
            leg_controller.set_initial_positions()
            leg_controller.phase_start_time = leg_controller.motiontime

        # Start movement after motors are enabled
        if leg_controller.motiontime > 100:
            leg_controller.execute_leg_sequence()

        # Apply safety checks
        # safe.PowerProtect(cmd, state, 1)

        udp.SetSend(cmd)
        udp.Send()