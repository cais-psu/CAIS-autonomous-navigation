import numpy as np


class HeatMap:
    """
    Heat-equation-based exploration map.

    Temperature represents exploration desirability:
        - Low temperature: desirable / unexplored regions
        - High temperature: recently visited regions

    Occupied cells are handled externally by OccupancyGrid.
    """

    def __init__(
        self,
        width: int,
        height: int,
        diffusion_coefficient: float = 0.5,
        cooling_rate: float = 1.0,
        dt: float = 0.1,
    ):
        """
        Initialize heat map.

        Parameters
        ----------
        width : int
            Grid width.

        height : int
            Grid height.

        diffusion_coefficient : float
            Heat diffusion coefficient (kappa).

        cooling_rate : float
            Heat source term coefficient (alpha).

        dt : float
            Discrete timestep.
        """

        self.width = width
        self.height = height

        self.kappa = diffusion_coefficient
        self.alpha = cooling_rate
        self.dt = dt

        # Temperature field
        self.temperature = np.zeros(
            (height, width),
            dtype=float,
        )


    def get_temperature(
        self,
        x: int,
        y: int,
    ) -> float:
        """
        Return temperature at a grid cell.
        """

        if not self._in_bounds(x, y):
            raise ValueError(
                f"Cell ({x},{y}) is outside the heat map."
            )

        return self.temperature[y, x]


    def set_temperature(
        self,
        x: int,
        y: int,
        value: float,
    ):
        """
        Set temperature at a grid cell.
        """

        if not self._in_bounds(x, y):
            raise ValueError(
                f"Cell ({x},{y}) is outside the heat map."
            )

        self.temperature[y, x] = value


    def mark_visited(
        self,
        x: int,
        y: int,
        amount: float = 1.0,
    ):
        """
        Increase temperature at a visited cell.

        This represents the robot leaving behind heat.
        """

        if not self._in_bounds(x, y):
            return

        self.temperature[y, x] += amount


    def diffuse(
        self,
    ):
        """
        Perform one discrete heat diffusion step.

        Implements:

        dT/dt = kappa * Laplacian(T)

        using a 5-point finite difference stencil.
        """

        T = self.temperature

        laplacian = (
            np.roll(T, 1, axis=0)
            + np.roll(T, -1, axis=0)
            + np.roll(T, 1, axis=1)
            + np.roll(T, -1, axis=1)
            - 4 * T
        )

        self.temperature = (
            T
            + self.dt * self.kappa * laplacian
        )


    def cool(
        self,
    ):
        """
        Apply cooling term.

        Implements:

        dT/dt = -alpha*T
        """

        self.temperature *= (
            1.0 - self.dt * self.alpha
        )


    def step(
        self,
    ):
        """
        Perform one complete heat update.
        """

        self.diffuse()
        self.cool()


    def reset(self):
        """
        Reset heat field.
        """

        self.temperature.fill(0.0)


    def _in_bounds(
        self,
        x: int,
        y: int,
    ) -> bool:

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )