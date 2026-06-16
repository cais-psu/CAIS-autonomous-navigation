import numpy as np

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