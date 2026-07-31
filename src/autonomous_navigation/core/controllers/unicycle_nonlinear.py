from __future__ import annotations

import numpy as np

from autonomous_navigation.core.controllers.mpc.base import MPCController
from pydrake.solvers import MathematicalProgram, Solve


class NonlinearUnicycleMPC(MPCController):
    """
    MPC for a planar unicycle robot.

    State:
        x = [px, py, theta]

    Control:
        u = [v, omega]
    """

    def __init__(
        self,
        dt: float,
        horizon: int,
        Q: np.ndarray,
        R: np.ndarray,
        v_max: float,
        omega_max: float,
    ):
        super().__init__(dt, horizon, Q, R)

        self.v_max = v_max
        self.omega_max = omega_max

    def solve(
        self,
        state: np.ndarray,
    ) -> np.ndarray:

        if self.reference is None:
            raise ValueError("Reference is not set")

        if len(self.reference) != self.N:
            raise ValueError(
                f"Expected {self.N} reference points, "
                f"received {len(self.reference)}."
            )

        x = np.asarray(state).copy()

        prog = MathematicalProgram()

        #
        # Decision variables
        #
        # u[:,k] = [v, omega]
        #
        u = prog.NewContinuousVariables(2, self.N, "u")

        cost = 0

        for k in range(self.N):

            #
            # Roll out dynamics
            #
            x = self.predict(
                x,
                u[:, k],
            )

            r = np.asarray(self.reference[k])

            #
            # Position tracking
            #
            cost += (
                (x[:2] - r[:2]).T
                @ self.Q
                @ (x[:2] - r[:2])
            )

            #
            # Control effort
            #
            cost += (
                u[:, k].T
                @ self.R
                @ u[:, k]
            )

        prog.AddCost(cost)

        #
        # Velocity constraints
        #
        for k in range(self.N):

            prog.AddBoundingBoxConstraint(
                -self.v_max,
                self.v_max,
                u[0, k],
            )

            prog.AddBoundingBoxConstraint(
                -self.omega_max,
                self.omega_max,
                u[1, k],
            )

        result = Solve(prog)

        if not result.is_success():
            print("MPC optimization failed.")
            return np.zeros(2)

        return result.GetSolution(u[:, 0])

    def predict(
        self,
        state,
        control,
    ):
        """
        Forward Euler discretization of the
        unicycle model.
        """

        x = state[0]
        y = state[1]
        theta = state[2]

        v = control[0]
        omega = control[1]

        x_next = x + self.dt * v * np.cos(theta)
        y_next = y + self.dt * v * np.sin(theta)
        theta_next = theta + self.dt * omega

        return np.array(
            [
                x_next,
                y_next,
                theta_next,
            ]
        )

    def reset(self):

        self.reference = None
        self.safe_region = None