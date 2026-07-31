from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid


def test_obstacle_insertion():

    grid = OccupancyGrid(10,10)

    grid.set_obstacle(3,4)

    assert grid.is_occupied(3,4)
    assert not grid.is_free(3,4)