import numpy as np

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid


class HeatMap:
    """
    Heat-equation-based exploration map.

    The occupancy grid defines the environment extent. The heat map uses
    an independently specified spatial resolution over the same extent.

    Temperature is increased around the robot through a spatially distributed
    heat source and subsequently diffuses according to the heat equation.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        heat_resolution: float = 0.5,
        diffusion_coefficient: float = 1.6,
        dt: float = 0.1,
        source_strength: float = 3.0,
        heat_cap: float = 300.0,
    ):
        self.occupancy_grid = occupancy_grid

        self.width = occupancy_grid.width
        self.height = occupancy_grid.height

        self.heat_resolution = heat_resolution
        self.kappa = diffusion_coefficient
        self.dt = dt
        self.source_strength = source_strength
        self.heat_cap = heat_cap

        # Heat-grid dimensions.
        self.heat_width = int(np.ceil(self.width / heat_resolution)) + 1
        self.heat_height = int(np.ceil(self.height / heat_resolution)) + 1

        # Physical coordinates of heat-grid cells.
        self.x = np.arange(self.heat_width) * heat_resolution
        self.y = np.arange(self.heat_height) * heat_resolution

        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Temperature field.
        self.temperature = np.zeros(
            (self.heat_height, self.heat_width),
            dtype=float,
        )

    def get_temperature(self, x: float, y: float) -> float:
        """
        Get temperature at the heat-grid cell corresponding to a
        physical-world coordinate.
        """
        ix, iy = self._world_to_heat_index(x, y)

        return self.temperature[iy, ix]

    def set_temperature(
        self,
        x: float,
        y: float,
        value: float,
    ):
        """
        Set temperature at the heat-grid cell corresponding to a
        physical-world coordinate.
        """
        ix, iy = self._world_to_heat_index(x, y)

        self.temperature[iy, ix] = value

    def apply_heat_source(
        self,
        robot_x: float,
        robot_y: float,
        source_radius: float,
    ):
        """
        Apply continuous heat around the robot.

        Heat follows a linear spatial falloff:

            weight = 1 - distance / source_radius

        and is accumulated according to:

            delta_T = source_strength * dt * weight

        Temperature is capped at heat_cap.
        """
        d2 = (
            (self.X - robot_x) ** 2
            + (self.Y - robot_y) ** 2
        )

        mask = d2 <= source_radius ** 2

        if not np.any(mask):
            return

        distance = np.sqrt(d2[mask])

        weight = 1.0 - distance / source_radius
        weight = np.maximum(weight, 0.0)

        delta = (
            self.source_strength
            * self.dt
            * weight
        )

        self.temperature[mask] += delta

        self.temperature[
            self.temperature > self.heat_cap
        ] = self.heat_cap

    def diffuse(self):
        """
        Perform one explicit heat-diffusion step using a five-point
        finite-difference stencil.
        """
        T = self.temperature

        laplacian = (
            np.roll(T, 1, axis=0)
            + np.roll(T, -1, axis=0)
            + np.roll(T, 1, axis=1)
            + np.roll(T, -1, axis=1)
            - 4.0 * T
        ) / (self.heat_resolution ** 2)

        T_new = (
            T
            + self.kappa
            * self.dt
            * laplacian
        )

        # Match MATLAB edge treatment.
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]

        # Prevent negative temperatures.
        T_new[T_new < 0.0] = 0.0

        self.temperature = T_new

    def step(
        self,
        robot_x: float,
        robot_y: float,
        source_radius: float,
    ):
        """
        Perform one complete heat update.

        1. Apply continuous heat source.
        2. Diffuse the heat field.
        """
        self.apply_heat_source(
            robot_x=robot_x,
            robot_y=robot_y,
            source_radius=source_radius,
        )

        self.diffuse()

    def reset(self):
        """Reset the heat field to zero."""
        self.temperature.fill(0.0)

    def _world_to_heat_index(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """Convert world coordinates to heat-grid indices."""
        ix = int(round(x / self.heat_resolution))
        iy = int(round(y / self.heat_resolution))

        if not (
            0 <= ix < self.heat_width
            and 0 <= iy < self.heat_height
        ):
            raise ValueError(
                f"Position ({x}, {y}) is outside the heat map."
            )

        return ix, iy