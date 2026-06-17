from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import CircleObstacle

def test_line_hits_circle():

    grid = OccupancyGrid(50, 50)

    circle = CircleObstacle(
        center_x=25,
        center_y=25,
        radius=5
    )

    grid.add_circle_obstacle(circle)

    assert not grid.line_is_free(
        0, 0,
        49, 49
    )