from autonomous_navigation.core.environment.obstacles import RectangleObstacle


def test_rectangle_creation():

    obs = RectangleObstacle(
        x_min=1,
        y_min=2,
        x_max=5,
        y_max=6
    )

    assert obs.x_min == 1
    assert obs.y_min == 2
    assert obs.x_max == 5
    assert obs.y_max == 6