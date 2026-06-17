from dataclasses import dataclass


@dataclass
class RectangleObstacle:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

@dataclass
class CircleObstacle:
    center_x: int
    center_y: int
    radius: int
    