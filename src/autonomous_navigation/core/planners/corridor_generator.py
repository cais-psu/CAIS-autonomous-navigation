import numpy as np

from autonomous_navigation.core.planners.corridor import ConvexCorridor

class CorridorGenerator:

    def __init__(
        self,
        local_grid,
        corridor_width: float = 2.0,
    ):

        self.local_grid = local_grid
        self.corridor_width = corridor_width


    def generate(
        self,
        path,
    ):

        corridors = []

        for waypoint in path:

            corridor = self.create_box_corridor(
                waypoint
            )

            if self.is_valid(corridor):
                corridors.append(corridor)

            else:
                print(
                    "Invalid corridor at:",
                    waypoint
                )

        return corridors
    
    def create_box_corridor(
        self,
        point,
    ):

        x, y = point

        w = self.corridor_width

        A = np.array(
            [
                [1,0],
                [-1,0],
                [0,1],
                [0,-1],
            ]
        )

        b = np.array(
            [
                x+w,
                -(x-w),
                y+w,
                -(y-w),
            ]
        )

        return ConvexCorridor(
            A,
            b,
        )
    
    def is_valid(
        self,
        corridor,
    ):
        """
        Check whether the corridor contains occupied cells.
        """

        for y in range(self.local_grid.height):
            for x in range(self.local_grid.width):

                if self.local_grid.is_occupied(x, y):

                    if corridor.contains(
                        np.array([x, y])
                    ):
                        return False

        return True
    
    def contains_path(
        self,
        corridor,
        waypoint,
    ):
        return corridor.contains(
            np.asarray(waypoint)
        )
    
    