import numpy as np

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import RectangleObstacle
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.hedac_sampler import HEDACSampler
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode


def test_rrt_without_sampler_still_works():
    """
    Existing RRT behavior should remain unchanged when no sampler is provided.
    """

    grid = OccupancyGrid(
        width=50,
        height=50,
    )

    planner = RRTPlanner(
        grid,
        step_size=5,
    )

    start = RRTNode(5, 5)
    goal = RRTNode(45, 45)

    path, stats = planner.plan(
        start,
        goal,
        max_iters=2000,
    )

    assert stats["success"] is True
    assert path is not None


def test_rrt_accepts_hedac_sampler():
    """
    RRTPlanner should store and use a HEDAC sampler.
    """

    grid = OccupancyGrid(
        width=50,
        height=50,
    )

    heat_map = HeatMap(
        width=50,
        height=50,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    planner = RRTPlanner(
        grid,
        step_size=5,
        sampler=sampler,
    )

    assert planner.sampler is sampler


def test_rrt_sample_with_hedac_returns_rrt_node():
    """
    HEDAC sampling should return an RRTNode.
    """

    grid = OccupancyGrid(
        width=50,
        height=50,
    )

    heat_map = HeatMap(
        width=50,
        height=50,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    planner = RRTPlanner(
        grid,
        step_size=5,
        sampler=sampler,
    )

    sample = planner.sample()

    assert isinstance(
        sample,
        RRTNode,
    )

    assert 0 <= sample.x < grid.width
    assert 0 <= sample.y < grid.height


def test_rrt_uses_hedac_bias():
    """
    Verify that HEDAC sampling influences sample selection.

    A cold region should be selected more often than a hot region.
    """

    np.random.seed(42)

    grid = OccupancyGrid(
        width=50,
        height=50,
    )

    heat_map = HeatMap(
        width=50,
        height=50,
    )

    # Make one region undesirable
    heat_map.set_temperature(
        x=40,
        y=40,
        value=100.0,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    planner = RRTPlanner(
        grid,
        step_size=5,
        sampler=sampler,
    )

    candidates = np.array(
        [
            [10, 10],
            [40, 40],
        ]
    )

    selections = []

    for _ in range(500):

        selected = sampler.select_sample(
            candidates
        )

        selections.append(
            selected
        )

    cold_count = selections.count(
        (10, 10)
    )

    hot_count = selections.count(
        (40, 40)
    )

    assert cold_count > hot_count


def test_rrt_with_hedac_plans_around_obstacle():
    """
    RRT should still perform collision checking when HEDAC sampling is enabled.
    """

    grid = OccupancyGrid(
        width=50,
        height=50,
    )

    obstacle = RectangleObstacle(
        x_min=20,
        y_min=0,
        x_max=25,
        y_max=30,
    )

    grid.add_rectangle_obstacle(
        obstacle
    )

    heat_map = HeatMap(
        width=50,
        height=50,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    planner = RRTPlanner(
        grid,
        step_size=3,
        sampler=sampler,
    )

    start = RRTNode(5, 5)
    goal = RRTNode(45, 45)

    path, stats = planner.plan(
        start,
        goal,
        max_iters=5000,
    )

    assert stats["success"] is True
    assert path is not None