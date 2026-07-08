import numpy as np
import math
from autonomous_navigation.core.environment.obstacles import RectangleObstacle
from autonomous_navigation.core.environment.obstacles import CircleObstacle


class OccupancyGrid:
    def __init__(self, width: int, height: int):
        """
        Create an empty occupancy grid.
        
        0 = free
        1 = occupied
        """
        self.width = width
        self.height = height

        self.grid = np.zeros(
            (height, width),
            dtype = np.uint8
        )

        self.obstacles = []

    def in_bounds(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )
    def set_obstacle(self, x: int, y: int):

        if not self.in_bounds(x, y):
            raise ValueError(
                f"Cell ({x}, {y}) is outside the grid."
            )
        
        self.grid[y, x] = 1

    def clear_cell(self, x: int, y: int):

        if not self.in_bounds(x, y):
            raise ValueError(
                f"Cell ({x}, {y}) is outside the grid."
            )

        self.grid[y, x] = 0

    def is_occupied(self, x: int, y: int) -> bool:

        if not self.in_bounds(x, y):
            raise ValueError(
                f"Cell ({x}, {y}) is outside the grid."
            )

        return self.grid[y, x] == 1

    def is_free(self, x: int, y: int) -> bool:

        if not self.in_bounds(x, y):
            raise ValueError(
                f"Cell ({x}, {y}) is outside the grid."
            )

        return self.grid[y, x] == 0
    
    def add_rectangle_obstacle(
            self,
            obstacle: RectangleObstacle
    ):
        self.obstacles.append(obstacle)

        x_min = max(
            math.floor(obstacle.x_min),
            0
        )

        x_max = min(
            math.ceil(obstacle.x_max),
            self.width - 1
        )

        y_min = max(
            math.floor(obstacle.y_min),
            0
        )

        y_max = min(
            math.ceil(obstacle.y_max),
            self.height - 1
        )

        for x in range (
            x_min, 
            x_max + 1
            ):
            for y in range(
                y_min, 
                y_max + 1
                ):
                self.set_obstacle(x,y)

    def add_circle_obstacle(
            self,
            obstacle: CircleObstacle
    ):
        self.obstacles.append(obstacle)

        x_min = max(
            math.floor(obstacle.center_x - obstacle.radius),
            0
        )

        x_max = min(
            math.ceil(obstacle.center_x + obstacle.radius),
            self.width - 1
        )

        y_min = max(
            math.floor(obstacle.center_y - obstacle.radius),
            0
        )

        y_max = min(
            math.ceil(obstacle.center_y + obstacle.radius),
            self.height - 1
        )

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):

                dx = x - obstacle.center_x
                dy = y - obstacle.center_y

                if dx**2 + dy**2 <= obstacle.radius**2:
                    self.set_obstacle(x, y)

    def add_obstacle(
      self,
      obstacle      
    ):
        """Add a generic obstacle to the occupancy grid"""

        if isinstance(obstacle, RectangleObstacle):
            self.add_rectangle_obstacle(obstacle)
        
        elif isinstance(obstacle, CircleObstacle):
            self.add_circle_obstacle(obstacle)

        else:
            raise TypeError(
                f"Unsupported obstacle type: {type(obstacle).__name__}"
            )

    def line_is_free(
            self,
            x0: int,
            y0: int,
            x1: int,
            y1: int
    ):
        n_points = max(
            abs(x1 - x0),
            abs(y1 - y0)
        )

        for t in np.linspace(0, 1, n_points + 1):

            x = round(x0 + t * (x1 - x0))
            y = round(y0 + t * (y1 - y0))

            if self.is_occupied(x, y):
                return False

        return True
        
    def create_configuration_space(
        self,
        inflation_radius: float,
    ) -> "OccupancyGrid":
        """
        Create a new occupancy grid in configuration space by inflating
        every obstacle.
        """

        cspace = OccupancyGrid(
            width=self.width,
            height=self.height,
        )

        for obstacle in self.obstacles:
            cspace.add_obstacle(
                obstacle.inflate(inflation_radius)
            )

        return cspace