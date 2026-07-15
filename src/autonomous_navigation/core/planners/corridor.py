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
    ):

        self.A = np.asarray(A)
        self.b = np.asarray(b)


    def contains(
        self,
        point: np.ndarray,
        tolerance: float = 1e-6,
    ) -> bool:

        point = np.asarray(point)

        return np.all(
            self.A @ point <= self.b + tolerance
        )


    def num_constraints(self):

        return self.A.shape[0]


    def get_vertices(self):
        """
        Return polygon vertices for visualization.

        For 2D rectangular corridors initially.
        """

        pass

    def get_bounds(self):

        xmin = -self.b[1]
        xmax = self.b[0]
        ymin = -self.b[3]
        ymax = self.b[2]

        return xmin,xmax,ymin,ymax