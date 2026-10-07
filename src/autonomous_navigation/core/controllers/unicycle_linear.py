from __future__ import annotations

import numpy as np

from autonomous_navigation.core.controllers.mpc.base import MPCController
from pydrake.solvers import MathematicalProgram, Solve


class LinearUnicycleMPC(MPCController):
    """
    Linear MPC for a planar unicycle robot.

    State:
        x = [px, py, theta]

    Control:
        u = [v, omega]

    The dynamics are linearized about the current robot heading.
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

        #
        # Linearize about the current heading.
        #
        theta0 = float(state[2])

        A = np.eye(3)

        B = np.array(
            [
                [self.dt * np.cos(theta0), 0.0],
                [self.dt * np.sin(theta0), 0.0],
                [0.0,                      self.dt],
            ]
        )

        #
        # Initial predicted state
        #
        x = np.asarray(state).copy()

        prog = MathematicalProgram()

        #
        # Decision variables
        #
        # u[:, k] = [v, omega]
        #
        u = prog.NewContinuousVariables(2, self.N, "u")

        cost = 0

        for k in range(self.N):

            #
            # Linear state prediction
            #
            x = self.predict(
                x,
                u[:, k],
                A,
                B,
            )

            r = np.asarray(self.reference[k])

            #
            # State tracking
            #
            e = x - r

            cost += e.T @ self.Q @ e

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
        # Input constraints
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
        state: np.ndarray,
        control,
        theta0: float,
    ):
        """
        Linearized discrete-time unicycle model.

        The system is linearized about the heading theta0:

            x[k+1] = A x[k] + B u[k]
        """

        A = np.eye(3)

        B = np.array(
            [
                [self.dt * np.cos(theta0), 0.0],
                [self.dt * np.sin(theta0), 0.0],
                [0.0,                    self.dt],
            ]
        )

        return A @ state + B @ control

    def reset(self):

        self.reference = None
        self.safe_region = None