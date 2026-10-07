from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class MPCController(ABC):
    """
    Abstract base class for Model Predictive Controllers.

    Derived classes implement the robot dynamics and optimization problem.
    """

    def __init__(
        self,
        dt: float,
        horizon: int,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        self.dt = dt
        self.N = horizon

        self.Q = Q
        self.R = R

        self.reference: Optional[np.ndarray] = None

    def set_reference(self, reference: np.ndarray) -> None:
        """
        Set the reference trajectory.

        Parameters
        ----------
        reference : np.ndarray
            Array of shape (N, state_dimension)
        """
        self.reference = reference

    def set_safe_region(self, region) -> None:
        """
        Set the current convex safe region.

        The default implementation simply stores the region.
        Derived classes may use it to construct optimization constraints.
        """
        self.safe_region = region

    @abstractmethod
    def solve(self, state: np.ndarray) -> np.ndarray:
        """
        Solve the MPC optimization.

        Parameters
        ----------
        state : np.ndarray
            Current robot state.

        Returns
        -------
        np.ndarray
            First optimal control input.
        """
        pass

    @abstractmethod
    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Propagate the robot dynamics one timestep.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any controller state.
        """
        pass