from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid


def test_rrt_distance():

    planner = RRTPlanner(
        OccupancyGrid(10, 10)
    )

    n1 = RRTNode(0, 0)
    n2 = RRTNode(3, 4)

    assert planner.distance(
        n1,
        n2
    ) == 5.0