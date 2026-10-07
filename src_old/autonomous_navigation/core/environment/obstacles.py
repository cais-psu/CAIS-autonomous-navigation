from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.patches as patches


class Obstacle(ABC):
    """Abstract base class for all obstacle types."""

    @abstractmethod
    def plot(self, ax, **kwargs):
        """Plot the obstacle on a Matplotlib axis."""
        pass

    @abstractmethod
    def inflate(self, radius: float) -> "Obstacle":
        """Return a new obstacle inflated by the specified radius."""
        pass


@dataclass(frozen=True)
class RectangleObstacle(Obstacle):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def plot(self, ax, **kwargs):
        rect = patches.Rectangle(
            (self.x_min, self.y_min),
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            fill=False,
            **kwargs,
        )
        ax.add_patch(rect)

    def inflate(self, radius: float) -> "RectangleObstacle":
        return RectangleObstacle(
            x_min=self.x_min - radius,
            y_min=self.y_min - radius,
            x_max=self.x_max + radius,
            y_max=self.y_max + radius,
        )


@dataclass(frozen=True)
class CircleObstacle(Obstacle):
    center_x: float
    center_y: float
    radius: float

    def plot(self, ax, **kwargs):
        circle = patches.Circle(
            (self.center_x, self.center_y),
            self.radius,
            fill=False,
            **kwargs,
        )
        ax.add_patch(circle)

    def inflate(self, radius: float) -> "CircleObstacle":
        return CircleObstacle(
            center_x=self.center_x,
            center_y=self.center_y,
            radius=self.radius + radius,
        )