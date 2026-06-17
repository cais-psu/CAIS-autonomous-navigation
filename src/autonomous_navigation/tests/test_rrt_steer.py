import pytest

from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid


def test_steer_straight_line_x_axis():

    grid = OccupancyGrid(100, 100)
    planner = RRTPlanner(grid, step_size=2.0)

    nearest = RRTNode(0.0, 0.0)
    sample = RRTNode(10.0, 0.0)

    new_node = planner.steer(nearest, sample)

    assert new_node.x == pytest.approx(2.0)
    assert new_node.y == pytest.approx(0.0)


def test_steer_straight_line_y_axis():

    grid = OccupancyGrid(100, 100)
    planner = RRTPlanner(grid, step_size=3.0)

    nearest = RRTNode(0.0, 0.0)
    sample = RRTNode(0.0, 10.0)

    new_node = planner.steer(nearest, sample)

    assert new_node.x == pytest.approx(0.0)
    assert new_node.y == pytest.approx(3.0)