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
        
        if len(self.reference) != self.N:
            raise ValueError(
                f"Expected {self.N} reference points, "
                f"received {len(self.reference)}."
            )

        prog = MathematicalProgram()

        # Decision variables:
        # u[:, k] = control input at prediction step k
        u = prog.NewContinuousVariables(2, self.N, "u")

        #dynamics rollout (linear integrator)
        x = x0.copy()

        cost = 0

        for k in range(self.N):
            # x_{k+1} = x_k + dt*u
            x = self.predict(x, u[:, k])

            r = np.asarray(self.reference[k])

            cost += (x - r).T @ self.Q @ (x - r)

            cost += u[:, k].T @ self.R @ u[:, k]
        
        prog.AddCost(cost)

        #constraints: velocity limits
        for k in range(self.N):
            prog.AddBoundingBoxConstraint(
                -self.v_max,
                self.v_max,
                u[:, k]
            )

        result = Solve(prog)

        if not result.is_success():
            print("MPC optimization failed.")
            return np.zeros(2)
        
        #print("Solver success:", result.is_success())
        #print("Optimal cost:", result.get_optimal_cost())
        #print("Optimal u:")
        #print(result.GetSolution(u))

        return result.GetSolution(u[:, 0])

    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return state + self.dt * control

    def reset(self):
        self.reference = None
        self.safe_region = None