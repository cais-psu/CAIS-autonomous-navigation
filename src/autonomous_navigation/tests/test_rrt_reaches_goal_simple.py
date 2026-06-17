from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode


def test_rrt_reaches_goal_simple():

    grid = OccupancyGrid(50, 50)
    planner = RRTPlanner(grid, step_size=5.0)

    start = RRTNode(0, 0)
    goal = RRTNode(45, 45)

    path = planner.plan(start, goal, max_iters=500)

    assert path is not None
    assert len(path) > 1
    assert path[-1] == (goal.x, goal.y)