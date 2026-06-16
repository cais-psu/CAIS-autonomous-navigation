from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid

def test_add_rectangle():

    grid = OccupancyGrid(20, 20)

    grid.add_rectangle(
        5, 5,
        10, 10
    )

    assert grid.is_occupied(7, 7)
    assert grid.is_occupied(10, 10)
    assert grid.is_free(0, 0)