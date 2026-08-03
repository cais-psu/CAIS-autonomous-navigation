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


    def generate(self, path):

        corridors = []

        for i in range(len(path)-1):

            corridor = self.create_segment_corridor(
                path[i],
                path[i+1],
            )
            if corridor is None:
                continue
            
            if not self.is_valid(corridor):

                print(
                    "Warning: corridor intersects obstacle between:",
                    path[i],
                    path[i+1],
                )

            corridors.append(corridor)

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

                if self.local_grid.is_occupied_continuous(x, y):

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
    
    def create_segment_corridor(
        self,
        p1,
        p2,
    ):
        """
        Create rectangular corridor around path segment.
        Equivalent to MATLAB path_to_corridor_simple().
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        direction = p2 - p1

        length = np.linalg.norm(direction)

        if length < 1e-6:
            return None


        direction = direction / length

        perpendicular = np.array(
            [
                -direction[1],
                direction[0],
            ]
        )


        offset = (
            perpendicular *
            self.corridor_width
        )


        vertices = np.array(
            [
                p1 + offset,
                p2 + offset,
                p2 - offset,
                p1 - offset,
            ]
        )


        return self.polygon_to_halfspace(vertices)
    
    def polygon_to_halfspace(
        self,
        vertices,
    ):

        A = []
        b = []


        center = np.mean(vertices, axis=0)


        for i in range(len(vertices)):

            p1 = vertices[i]

            p2 = vertices[
                (i+1)%len(vertices)
            ]

            edge = p2-p1


            normal = np.array(
                [
                    -edge[1],
                    edge[0],
                ]
            )

            normal = normal / np.linalg.norm(normal)

            # ensure normal points outward

            if normal @ center <= normal @ p1:
                A.append(normal)
                b.append(normal @ p1)

            else:
                A.append(-normal)
                b.append(-normal @ p1)


        return ConvexCorridor(
            np.array(A),
            np.array(b),
            vertices=vertices,
            center=center,
        )


    
    @staticmethod
    def corridor_to_polygon(corridor):

        A = corridor.A
        b = corridor.b

        points = []

        # intersection of each pair of constraints
        for i in range(len(A)):
            for j in range(i+1, len(A)):

                M = np.vstack(
                    [
                        A[i],
                        A[j],
                    ]
                )

                if abs(np.linalg.det(M)) < 1e-8:
                    continue

                x = np.linalg.solve(
                    M,
                    np.array(
                        [
                            b[i],
                            b[j],
                        ]
                    )
                )

                if np.all(
                    A @ x <= b + 1e-6
                ):
                    points.append(x)

        points = np.array(points)

        center = np.mean(points, axis=0)

        angles = np.arctan2(
            points[:,1]-center[1],
            points[:,0]-center[0],
        )

        return points[
            np.argsort(angles)
        ]
    