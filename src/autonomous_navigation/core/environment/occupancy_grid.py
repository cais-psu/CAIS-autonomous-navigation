import numpy as np
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
        for x in range (
            obstacle.x_min, 
            obstacle.x_max + 1
            ):
            for y in range(
                obstacle.y_min, 
                obstacle.y_max + 1
                ):
                self.set_obstacle(x,y)

    def add_circle_obstacle(
            self,
            obstacle: CircleObstacle
    ):
        for x in range(self.width):
            for y in range(self.height):

                dx = x - obstacle.center_x
                dy = y - obstacle.center_y

                if dx**2 + dy**2 <= obstacle.radius**2:
                    self.set_obstacle(x,y)

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
        