from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid 
from autonomous_navigation.core.environment.obstacles import CircleObstacle

def test_add_circle_obstacle():

    grid = OccupancyGrid(50, 50)

    circle = CircleObstacle(
        center_x=25,
        center_y=25,
        radius=5
    )

    grid.add_circle_obstacle(circle)

    assert grid.is_occupied(25, 25)
    assert grid.is_occupied(30, 25)
    assert grid.is_free(0, 0)