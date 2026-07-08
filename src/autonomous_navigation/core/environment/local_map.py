import math

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import (
    RectangleObstacle,
    CircleObstacle,
)


class LocalMap:
    """
    Generates a local occupancy grid based on a robot sensing radius.

    The global grid represents ground truth.
    The local grid represents currently perceived information.
    """

    def __init__(
        self,
        global_grid: OccupancyGrid,
        sensing_radius: float,
    ):
        self.global_grid = global_grid
        self.sensing_radius = sensing_radius


    def get_local_grid(
        self,
        robot_x: float,
        robot_y: float,
    ) -> OccupancyGrid:
        """
        Generate a local occupancy grid centered around the robot.

        Parameters
        ----------
        robot_x : float
            Robot x position.

        robot_y : float
            Robot y position.

        Returns
        -------
        OccupancyGrid
            Local perceived occupancy grid.
        """

        local_grid = OccupancyGrid(
            width=self.global_grid.width,
            height=self.global_grid.height,
        )

        # Copy only obstacles that are within sensing radius
        for obstacle in self.global_grid.obstacles:

            if self._obstacle_in_range(
                obstacle,
                robot_x,
                robot_y,
            ):
                local_grid.add_obstacle(obstacle)


        return local_grid


    def _obstacle_in_range(
        self,
        obstacle,
        robot_x: float,
        robot_y: float,
    ) -> bool:
        """
        Determine whether an obstacle is visible from the robot.

        Currently uses obstacle center distance.
        This is conservative enough for initial development.
        """

        if isinstance(obstacle, CircleObstacle):

            dx = obstacle.center_x - robot_x
            dy = obstacle.center_y - robot_y

            distance = math.sqrt(
                dx**2 + dy**2
            )

            return distance <= (
                self.sensing_radius + obstacle.radius
            )


        elif isinstance(obstacle, RectangleObstacle):

            closest_x = max(
                obstacle.x_min,
                min(robot_x, obstacle.x_max)
            )

            closest_y = max(
                obstacle.y_min,
                min(robot_y, obstacle.y_max)
            )

            dx = closest_x - robot_x
            dy = closest_y - robot_y

            distance = math.sqrt(
                dx**2 + dy**2
            )

            return distance <= self.sensing_radius


        else:
            raise TypeError(
                f"Unsupported obstacle type: {type(obstacle).__name__}"
            )