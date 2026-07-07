from __future__ import annotations

import numpy as np

from autonomous_navigation.core.controllers.mpc.base import MPCController
from pydrake.solvers import MathematicalProgram, Solve


class OmniMPC(MPCController):
    """
    MPC for a planar omnidirectional robot.

    State:
        x = [px, py]

    Control:
        u = [vx, vy]
    """

    def __init__(
        self,
        dt: float,
        horizon: int,
        Q: np.ndarray,
        R: np.ndarray,
        v_max: float,
    ):
        super().__init__(dt, horizon, Q, R)

        self.v_max = v_max

    def solve(self, state: np.ndarray) -> np.ndarray:
        x0 = state[:2]

        if self.reference is None:
            raise ValueError("Reference is not set")
        
        # take first reference point
        r = np.array(self.reference[0])

        prog = MathematicalProgram()

        # decision Variable: constant velocity over horizon
        u = prog.NewContinuousVariables(2, "u")

        #dynamics rollout (linear integrator)
        x = x0.copy()

        cost = 0

        for k in range(self.N):
            # x_{k+1} = x_k + dt*u
            x = x+self.dt*u

            cost += (x - r).T @ self.Q @ (x - r)

            cost += u.T @ self.R @ u
        
        prog.AddCost(cost)

        #constraints: velocity limits
        prog.AddBoundingBoxConstraint(
            -self.v_max,
            self.v_max,
            u
        )

        result = Solve(prog)

        if not result.is_success():
            return np.zeros(2)
        
        return result.GetSolution(u)

    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return state + self.dt * control

    def reset(self):
        self.reference = None
        self.safe_region = None