from __future__ import annotations

import numpy as np

from autonomous_navigation.core.controllers.mpc.base import MPCController


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
        raise NotImplementedError

    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return state + self.dt * control

    def reset(self):
        self.reference = None
        self.safe_region = None