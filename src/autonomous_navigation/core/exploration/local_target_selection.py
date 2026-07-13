import math
from typing import Tuple

import numpy as np

from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_node import RRTNode


class LocalTargetSelector:
    """
    Selects a local exploration target from the current heat map.

    The initial implementation chooses the coldest free cell within the
    robot's sensing radius.

    Later versions can incorporate frontier detection, information gain,
    HEDAC weighting, semantic costs, etc., without changing the navigation
    pipeline.
    """

    def __init__(self, sensing_radius: float):
        self.sensing_radius = sensing_radius

    def select_target(
        self,
        robot_x: float,
        robot_y: float,
        occupancy_grid: OccupancyGrid,
        planner_step_size: float,
        heat_map: HeatMap,
    ) -> RRTNode:
        """
        Select the coldest free cell inside the sensing radius.

        Parameters
        ----------
        robot_x : float
            Current robot x-position.

        robot_y : float
            Current robot y-position.

        occupancy_grid : OccupancyGrid
            Current local occupancy map.

        heat_map : HeatMap
            Current heat field.

        Returns
        -------
        RRTNode
            Selected exploration target.
        """

        best_temperature = float("inf")
        best_cell = None
        best_distance = -1

        x_min = max(
            0,
            int(math.floor(robot_x - self.sensing_radius))
        )

        x_max = min(
            occupancy_grid.width - 1,
            int(math.ceil(robot_x + self.sensing_radius))
        )

        y_min = max(
            0,
            int(math.floor(robot_y - self.sensing_radius))
        )

        y_max = min(
            occupancy_grid.height - 1,
            int(math.ceil(robot_y + self.sensing_radius))
        )

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):

                dx = x - robot_x
                dy = y - robot_y

                if dx * dx + dy * dy > self.sensing_radius**2:
                    continue

                if occupancy_grid.is_occupied(x, y):
                    continue

                distance = math.hypot(x - robot_x, y - robot_y)

                if distance < 2 * planner_step_size:
                    continue

                temperature = heat_map.temperature[y, x]

                if (temperature < best_temperature
                    or (
                        temperature == best_temperature
                        and distance > best_distance
                    )
                ):
                    best_temperature = temperature
                    best_cell = (x, y)
        


        if best_cell is None:
            raise RuntimeError(
                "No reachable free cell found within sensing radius."
            )

        return RRTNode(best_cell[0], best_cell[1])