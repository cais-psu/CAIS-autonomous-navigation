import pytest

from autonomous_navigation.core.environment.local_map import LocalMap
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import (
    RectangleObstacle,
    CircleObstacle,
)


def test_local_map_returns_occupancy_grid():
    """LocalMap should return a valid OccupancyGrid."""

    global_grid = OccupancyGrid(
        width=100,
        height=100,
    )

    local_map = LocalMap(
        global_grid,
        sensing_radius=20,
    )

    local_grid = local_map.get_local_grid(
        robot_x=50,
        robot_y=50,
    )

    assert isinstance(local_grid, OccupancyGrid)


def test_obstacle_inside_sensing_radius_is_detected():
    """Obstacle inside sensing radius should appear in local map."""

    global_grid = OccupancyGrid(
        width=100,
        height=100,
    )

    obstacle = CircleObstacle(
        center_x=55,
        center_y=50,
        radius=3,
    )

    global_grid.add_circle_obstacle(obstacle)

    local_map = LocalMap(
        global_grid,
        sensing_radius=10,
    )

    local_grid = local_map.get_local_grid(
        robot_x=50,
        robot_y=50,
    )

    assert len(local_grid.obstacles) == 1
    assert isinstance(
        local_grid.obstacles[0],
        CircleObstacle
    )


def test_obstacle_outside_sensing_radius_is_not_detected():
    """Obstacle outside sensing radius should not appear."""

    global_grid = OccupancyGrid(
        width=100,
        height=100,
    )

    obstacle = CircleObstacle(
        center_x=80,
        center_y=80,
        radius=3,
    )

    global_grid.add_circle_obstacle(obstacle)

    local_map = LocalMap(
        global_grid,
        sensing_radius=10,
    )

    local_grid = local_map.get_local_grid(
        robot_x=50,
        robot_y=50,
    )

    assert len(local_grid.obstacles) == 0


def test_rectangle_inside_sensing_radius_is_detected():
    """Rectangle obstacle should be detected when within sensing range."""

    global_grid = OccupancyGrid(
        width=100,
        height=100,
    )

    obstacle = RectangleObstacle(
        x_min=45,
        y_min=45,
        x_max=50,
        y_max=50,
    )

    global_grid.add_rectangle_obstacle(obstacle)

    local_map = LocalMap(
        global_grid,
        sensing_radius=10,
    )

    local_grid = local_map.get_local_grid(
        robot_x=40,
        robot_y=40,
    )

    assert len(local_grid.obstacles) == 1
    assert isinstance(
        local_grid.obstacles[0],
        RectangleObstacle
    )


def test_local_map_does_not_modify_global_map():
    """Generating a local map should not modify the global environment."""

    global_grid = OccupancyGrid(
        width=100,
        height=100,
    )

    obstacle = CircleObstacle(
        center_x=50,
        center_y=50,
        radius=5,
    )

    global_grid.add_circle_obstacle(obstacle)

    original_obstacle_count = len(global_grid.obstacles)
    original_occupied_cells = global_grid.grid.sum()

    local_map = LocalMap(
        global_grid,
        sensing_radius=10,
    )

    _ = local_map.get_local_grid(
        robot_x=50,
        robot_y=50,
    )

    assert len(global_grid.obstacles) == original_obstacle_count
    assert global_grid.grid.sum() == original_occupied_cells


def test_multiple_obstacles_only_returns_visible_obstacles():
    """Only obstacles within sensing radius should be returned."""

    global_grid = OccupancyGrid(
        width=100,
        height=100,
    )

    near_obstacle = CircleObstacle(
        center_x=55,
        center_y=50,
        radius=2,
    )

    far_obstacle = RectangleObstacle(
        x_min=80,
        y_min=80,
        x_max=90,
        y_max=90,
    )

    global_grid.add_circle_obstacle(near_obstacle)
    global_grid.add_rectangle_obstacle(far_obstacle)

    local_map = LocalMap(
        global_grid,
        sensing_radius=10,
    )

    local_grid = local_map.get_local_grid(
        robot_x=50,
        robot_y=50,
    )

    assert len(local_grid.obstacles) == 1
    assert isinstance(
        local_grid.obstacles[0],
        CircleObstacle
    )