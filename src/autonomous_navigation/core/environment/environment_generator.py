import random

from autonomous_navigation.core.environment.obstacles import RectangleObstacle
from autonomous_navigation.core.environment.obstacles import CircleObstacle



class EnvironmentGenerator:

    def random_rectangle(
        self,
        width: int,
        height: int,
        min_size: int = 3,
        max_size: int = 10
    ):

        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)

        x = random.randint(0, width - w - 1)
        y = random.randint(0, height - h - 1)

        return RectangleObstacle(
            x,
            y,
            x + w,
            y + h
        )
    
    def random_circle(
        self,
        width: int,
        height: int,
        min_radius: int = 1,
        max_radius: int = 10
    ):

        radius = random.randint(
            min_radius,
            max_radius
        )

        center_x = random.randint(
            radius,
            width - radius - 1
        )

        center_y = random.randint(
            radius,
            height - radius - 1
        )

        return CircleObstacle(
            center_x=center_x,
            center_y=center_y,
            radius=radius
        )