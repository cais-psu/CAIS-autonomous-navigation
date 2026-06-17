from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid


def test_sample_free():

    planner = RRTPlanner(
        OccupancyGrid(100, 100)
    )

    node = planner.sample_free()

    assert 0 <= node.x <= 100
    assert 0 <= node.y <= 100