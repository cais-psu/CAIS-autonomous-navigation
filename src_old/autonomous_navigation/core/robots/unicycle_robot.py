from __future__ import annotations

import numpy as np

from autonomous_navigation.core.robots.robot_state import RobotState


class UnicycleRobot:
    """
    Velocity integrator for a planar unicycle robot.

    State:
        x
        y
        theta

    Control inputs:
        v      Forward (body-frame) velocity
        omega  Angular velocity

    The robot state stores both body-frame and world-frame velocities.
    """

    def __init__(self, state: RobotState):
        self.state = state

    def step(
        self,
        v: float,
        omega: float,
        dt: float,
    ):
        """
        Advance the robot state one timestep using Euler integration.

        Parameters
        ----------
        v : float
            Forward velocity (m/s)
        omega : float
            Angular velocity (rad/s)
        dt : float
            Integration timestep (s)
        """

        theta = self.state.theta

        #
        # Convert body-frame velocity to world-frame velocity
        #
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        #
        # Integrate state
        #
        self.state.x += vx * dt
        self.state.y += vy * dt
        self.state.theta += omega * dt

        #
        # Wrap heading to [-pi, pi]
        #
        self.state.theta = np.arctan2(
            np.sin(self.state.theta),
            np.cos(self.state.theta),
        )

        #
        # Store body-frame velocities
        #
        self.state.v = v
        self.state.omega = omega

        #
        # Store world-frame velocities
        #
        self.state.vx = vx
        self.state.vy = vy