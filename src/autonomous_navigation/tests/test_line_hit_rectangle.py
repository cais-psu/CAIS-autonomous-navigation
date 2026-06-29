from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import RectangleObstacle

def test_line_hits_rectangle():

    grid = OccupancyGrid(20, 20)

    obstacle = RectangleObstacle(
        8, 8,
        12, 12
    )

    grid.add_rectangle_obstacle(obstacle)

    assert not grid.line_is_free(
        0, 0,
        19, 19
    )