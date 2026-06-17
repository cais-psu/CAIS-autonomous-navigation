from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid

def test_line_free():

    grid = OccupancyGrid(20,20)

    assert grid.line_is_free(
        0,0,
        19,19
    )