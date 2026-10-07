import numpy as np


class ConvexCorridor:
    """
    Convex polytope represented as:

        A x <= b

    where x = [x, y].
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        vertices: np.ndarray | None=None,
        center: np.ndarray | None = None,
    ):

        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.vertices = (
            np.asarray(vertices, dtype=float)
            if vertices is not None
            else None
        )
        self.center = (
            np.asarray(center, dtype = float)
            if center is not None
            else None
        )


    def contains(
        self,
        point: np.ndarray,
        tolerance: float = 1e-6,
    ) -> bool:

        point = np.asarray(point)

        if point.shape[0] > 2:
            point = point[:2]

        return np.all(
            self.A @ point <= self.b + tolerance
        )


    def num_constraints(self):

        return self.A.shape[0]


    def get_bounds(self):

        if self.vertices is None:
            raise ValueError("Vertices unavailable.")

        xmin = np.min(self.vertices[:,0])
        xmax = np.max(self.vertices[:,0])
        ymin = np.min(self.vertices[:,1])
        ymax = np.max(self.vertices[:,1])

        return xmin, xmax, ymin, ymax

    def get_vertices(self):
        return self.vertices

    @property
    def dimension(self):

        return self.A.shape[1]

    def halfspace(self):

        return self.A, self.b