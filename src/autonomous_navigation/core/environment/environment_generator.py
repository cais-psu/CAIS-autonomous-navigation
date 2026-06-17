import random

from autonomous_navigation.core.environment.obstacles import RectangleObstacle


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