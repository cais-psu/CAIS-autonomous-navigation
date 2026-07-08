from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import (
    RectangleObstacle,
    CircleObstacle,
)


def test_configuration_space_preserves_obstacle_count():

    grid = OccupancyGrid(
        width=100,
        height=100
    )

    grid.add_rectangle_obstacle(
        RectangleObstacle(
            20,
            20,
            30,
            30
        )
    )

    grid.add_circle_obstacle(
        CircleObstacle(
            60,
            60,
            5
        )
    )

    cspace = grid.create_configuration_space(
        inflation_radius=2
    )

    assert len(cspace.obstacles) == len(grid.obstacles)

def test_configuration_space_does_not_modify_original():

    grid = OccupancyGrid(100,100)

    rectangle = RectangleObstacle(
        20,
        20,
        30,
        30
    )

    grid.add_rectangle_obstacle(rectangle)

    original = grid.obstacles[0]

    cspace = grid.create_configuration_space(
        inflation_radius=5
    )

    assert original.x_min == 20
    assert original.x_max == 30

    assert cspace.obstacles[0].x_min == 15
    assert cspace.obstacles[0].x_max == 35