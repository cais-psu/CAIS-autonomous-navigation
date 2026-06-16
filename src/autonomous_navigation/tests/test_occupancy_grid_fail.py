import pytest

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
import autonomous_navigation.core.environment.occupancy_grid as og

print(og.__file__)

def test_out_of_bounds():

    grid = OccupancyGrid(10,10)

    with pytest.raises(ValueError):
        grid.set_obstacle(20,20)