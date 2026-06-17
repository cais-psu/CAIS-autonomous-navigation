from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import RectangleObstacle

def test_add_rectangle():

    grid = OccupancyGrid(20, 20)
    
    obs = RectangleObstacle(
        x_min=1,
        y_min=2,
        x_max=5,
        y_max=6
    )

    grid.add_rectangle_obstacle(obs)

    assert grid.is_occupied(4, 4)
    assert grid.is_occupied(5, 6)
    assert grid.is_free(0, 0)