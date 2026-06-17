from dataclasses import dataclass


@dataclass
class RectangleObstacle:
    x_min: int
    y_min: int
    x_max: int
    y_max: int