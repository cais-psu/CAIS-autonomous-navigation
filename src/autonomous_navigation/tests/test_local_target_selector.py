import numpy as np
import pytest

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.local_target_selection import LocalTargetSelector


def test_selects_coldest_free_cell():
    """
    Verify that the selector chooses the lowest temperature
    free cell inside the sensing radius.
    """

    grid = OccupancyGrid(
        width=20,
        height=20
    )

    heat_map = HeatMap(
        width=20,
        height=20
    )

    # Set all temperatures high
    heat_map.temperature[:, :] = 100.0

    # Create a cold cell inside sensing radius
    cold_x = 12
    cold_y = 10

    heat_map.temperature[cold_y, cold_x] = 0.0

    selector = LocalTargetSelector(
        sensing_radius=10
    )

    target = selector.select_target(
        robot_x=10,
        robot_y=10,
        occupancy_grid=grid,
        planner_step_size=1,
        heat_map=heat_map,
    )

    assert target.x == cold_x
    assert target.y == cold_y


def test_does_not_select_occupied_cell():
    """
    Verify that occupied cells are ignored even if they have
    the lowest temperature.
    """

    grid = OccupancyGrid(
        width=20,
        height=20
    )

    heat_map = HeatMap(
        width=20,
        height=20
    )

    heat_map.temperature[:, :] = 100.0

    occupied_x = 12
    occupied_y = 10

    free_x = 13
    free_y = 10

    # Coldest cell is occupied
    heat_map.temperature[occupied_y, occupied_x] = 0.0
    heat_map.temperature[free_y, free_x] = 10.0

    grid.set_obstacle(
        occupied_x,
        occupied_y
    )

    selector = LocalTargetSelector(
        sensing_radius=10
    )

    target = selector.select_target(
        robot_x=10,
        robot_y=10,
        occupancy_grid=grid,
        planner_step_size=1,
        heat_map=heat_map,
    )

    assert target.x == free_x
    assert target.y == free_y


def test_does_not_select_outside_sensing_radius():

    grid = OccupancyGrid(
        width=50,
        height=50
    )

    heat_map = HeatMap(
        width=50,
        height=50
    )

    heat_map.temperature[:, :] = 100.0

    # Outside sensing radius
    heat_map.temperature[40, 40] = 0.0

    # Inside sensing radius
    heat_map.temperature[15, 10] = 5.0

    selector = LocalTargetSelector(
        sensing_radius=10
    )

    target = selector.select_target(
        robot_x=10,
        robot_y=10,
        occupancy_grid=grid,
        planner_step_size=1,
        heat_map=heat_map,
    )

    distance = np.linalg.norm(
        np.array([target.x, target.y]) -
        np.array([10,10])
    )

    assert distance <= 10
    assert (target.x, target.y) != (40,40)


def test_ignores_cells_too_close_to_robot():

    grid = OccupancyGrid(
        width=20,
        height=20
    )

    heat_map = HeatMap(
        width=20,
        height=20
    )

    heat_map.temperature[:, :] = 100.0

    # Very cold but invalid
    heat_map.temperature[11,10] = 0.0

    # Valid target
    heat_map.temperature[15,10] = 5.0

    selector = LocalTargetSelector(
        sensing_radius=10
    )

    target = selector.select_target(
        robot_x=10,
        robot_y=10,
        occupancy_grid=grid,
        planner_step_size=2,
        heat_map=heat_map,
    )

    distance = np.linalg.norm(
        np.array([target.x, target.y]) -
        np.array([10,10])
    )

    assert distance >= 4

def test_no_valid_target_raises_error():
    """
    Verify that an error is raised if every possible target
    is invalid.
    """

    grid = OccupancyGrid(
        width=10,
        height=10
    )

    heat_map = HeatMap(
        width=10,
        height=10
    )

    selector = LocalTargetSelector(
        sensing_radius=2
    )

    # Make every cell occupied
    for x in range(10):
        for y in range(10):
            grid.set_obstacle(x, y)

    with pytest.raises(RuntimeError):

        selector.select_target(
            robot_x=5,
            robot_y=5,
            occupancy_grid=grid,
            planner_step_size=1,
            heat_map=heat_map,
        )