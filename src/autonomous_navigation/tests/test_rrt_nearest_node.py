from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_node import RRTNode

def test_nearest_node():

    planner = RRTPlanner(
        OccupancyGrid(100, 100)
    )

    nodes = [
        RRTNode(0, 0),
        RRTNode(10, 10),
        RRTNode(20, 20)
    ]

    sample = RRTNode(12, 11)

    idx = planner.nearest_node(
        sample,
        nodes
    )

    assert idx == 1