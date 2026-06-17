import pytest

from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid

def test_steer_diagonal_normalization():

    grid = OccupancyGrid(100, 100)
    planner = RRTPlanner(grid, step_size=5.0)

    nearest = RRTNode(0.0, 0.0)
    sample = RRTNode(3.0, 4.0)  # 3-4-5 triangle

    new_node = planner.steer(nearest, sample)

    assert new_node.x == pytest.approx(3.0)
    assert new_node.y == pytest.approx(4.0)