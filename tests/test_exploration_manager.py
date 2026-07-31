import numpy as np

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.exploration_manager import (
    ExplorationManager,
)
from autonomous_navigation.core.exploration.exploration_result import (
    ExplorationResult,
)
from autonomous_navigation.core.planners.rrt_node import RRTNode


def test_exploration_manager_returns_result():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
        planner_step_size=2,
    )

    robot = np.array([10.0, 10.0])

    result = manager.plan(
        robot_position=robot,
        max_rrt_iterations=2000,
    )

    assert isinstance(
        result,
        ExplorationResult,
    )


def test_path_exists():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
        planner_step_size=2,
    )

    result = manager.plan(
        np.array([10.0, 10.0]),
        max_rrt_iterations=2000,
    )

    assert result.path is not None
    assert len(result.path) >= 2


def test_target_inside_sensing_radius():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
    )

    robot = np.array([10.0, 10.0])

    result = manager.plan(robot)

    dx = result.target.x - robot[0]
    dy = result.target.y - robot[1]

    assert dx * dx + dy * dy <= manager.sensing_radius**2


def test_heat_is_updated():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
    )

    before = heat.temperature.copy()

    manager.plan(
        np.array([10.0, 10.0])
    )

    assert not np.array_equal(
        before,
        heat.temperature,
    )


def test_rrt_was_created():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
    )

    result = manager.plan(
        np.array([10.0, 10.0])
    )

    assert result.rrt is not None
    assert len(result.rrt.nodes) > 0


def test_stats_dictionary_contains_success():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
    )

    result = manager.plan(
        np.array([10.0, 10.0])
    )

    assert "success" in result.stats

def test_goal_mode_does_not_change_heat():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
    )

    before = heat.temperature.copy()

    manager.plan(
        robot_position=np.array([10.0,10.0]),
        mission_goal=RRTNode(30,30),
    )

    assert np.array_equal(
        before,
        heat.temperature,
    )

def test_goal_mode_selected():

    grid = OccupancyGrid(
        width=40,
        height=40,
    )

    heat = HeatMap(
        width=40,
        height=40,
    )

    manager = ExplorationManager(
        occupancy_grid=grid,
        heat_map=heat,
        sensing_radius=15,
    )

    result = manager.plan(
        robot_position=np.array([10.0,10.0]),
        mission_goal=RRTNode(30,30),
    )

    assert result.stats["mode"] == "goal"