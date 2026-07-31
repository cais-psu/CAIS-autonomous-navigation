import math

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import RectangleObstacle
from autonomous_navigation.core.planners.local_rrt import LocalRRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode


def test_local_rrt_initialization():
    """Planner stores constructor arguments."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=20,
        step_size=5,
    )

    assert planner.global_map is grid
    assert planner.sensing_radius == 20
    assert planner.step_size == 5
    assert planner.sampler is None


def test_local_map_generation():
    """A local occupancy grid should be returned."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=20,
    )

    local_grid, bounds = planner.get_local_map(
        robot_x=50,
        robot_y=50,
    )

    assert local_grid.width == grid.width
    assert local_grid.height == grid.height

    assert bounds == (
        30,
        70,
        30,
        70,
    )


def test_sampling_bounds_near_boundary():
    """Bounds should be clipped to map edges."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=20,
    )

    _, bounds = planner.get_local_map(
        robot_x=5,
        robot_y=8,
    )

    assert bounds == (
        0,
        25,
        0,
        28,
    )


def test_local_rrt_finds_path_in_empty_environment():
    """Planner should succeed in free space."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=30,
        step_size=5,
    )

    start = RRTNode(
        10,
        10,
    )

    goal = RRTNode(
        25,
        25,
    )

    path, stats = planner.plan(
        start,
        goal,
        max_iters=2000,
    )

    assert stats["success"]
    assert path is not None
    assert len(path) >= 2


def test_local_rrt_respects_local_obstacles():
    """Planner should still avoid locally visible obstacles."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    grid.add_rectangle_obstacle(
        RectangleObstacle(
            x_min=18,
            y_min=5,
            x_max=22,
            y_max=30,
        )
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=30,
        step_size=3,
    )

    start = RRTNode(
        10,
        10,
    )

    goal = RRTNode(
        30,
        10,
    )

    path, stats = planner.plan(
        start,
        goal,
        max_iters=5000,
    )

    assert stats["success"]
    assert path is not None


def test_distant_obstacle_not_in_local_map():
    """Far-away obstacles should not appear in the local map."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    obstacle = RectangleObstacle(
        x_min=80,
        y_min=80,
        x_max=90,
        y_max=90,
    )

    grid.add_rectangle_obstacle(
        obstacle
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=20,
    )

    local_grid, _ = planner.get_local_map(
        robot_x=10,
        robot_y=10,
    )

    assert len(local_grid.obstacles) == 0


def test_nearby_obstacle_is_visible():
    """Nearby obstacles should appear in the local map."""

    grid = OccupancyGrid(
        width=100,
        height=100,
    )

    obstacle = RectangleObstacle(
        x_min=20,
        y_min=20,
        x_max=30,
        y_max=30,
    )

    grid.add_rectangle_obstacle(
        obstacle
    )

    planner = LocalRRTPlanner(
        global_map=grid,
        sensing_radius=25,
    )

    local_grid, _ = planner.get_local_map(
        robot_x=15,
        robot_y=15,
    )

    assert len(local_grid.obstacles) == 1