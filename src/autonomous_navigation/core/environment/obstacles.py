from dataclasses import dataclass
import matplotlib.patches as patches


@dataclass
class RectangleObstacle:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def plot(self, ax):

        rect = patches.Rectangle(
            (self.x_min, self.y_min),
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            fill=False
        )

        ax.add_patch(rect)

@dataclass
class CircleObstacle:
    center_x: int
    center_y: int
    radius: int

    def plot(self, ax):

        circle = patches.Circle(
            (self.center_x, self.center_y),
            self.radius,
            fill=False
        )

        ax.add_patch(circle)
    