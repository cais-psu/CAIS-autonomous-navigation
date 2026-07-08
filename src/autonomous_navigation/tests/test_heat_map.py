import numpy as np
import pytest

from autonomous_navigation.core.exploration.heat_map import HeatMap


def test_heat_map_initialization():
    """Heat map should initialize with zero temperature everywhere."""

    heat_map = HeatMap(
        width=50,
        height=50,
    )

    assert heat_map.temperature.shape == (50, 50)
    assert np.all(heat_map.temperature == 0.0)


def test_set_and_get_temperature():
    """Temperature values should be retrievable after setting."""

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    heat_map.set_temperature(
        x=5,
        y=10,
        value=3.5,
    )

    assert heat_map.get_temperature(
        x=5,
        y=10,
    ) == 3.5


def test_mark_visited_increases_temperature():
    """Visited cells should accumulate heat."""

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    heat_map.mark_visited(
        x=5,
        y=5,
        amount=2.0,
    )

    assert heat_map.get_temperature(
        x=5,
        y=5,
    ) == 2.0

    heat_map.mark_visited(
        x=5,
        y=5,
        amount=1.5,
    )

    assert heat_map.get_temperature(
        x=5,
        y=5,
    ) == 3.5


def test_diffusion_spreads_heat():
    """Heat should spread from a hot cell to neighboring cells."""

    heat_map = HeatMap(
        width=10,
        height=10,
        diffusion_coefficient=0.5,
        dt=0.1,
    )

    heat_map.set_temperature(
        x=5,
        y=5,
        value=10.0,
    )

    heat_map.diffuse()

    center_temperature = heat_map.get_temperature(
        x=5,
        y=5,
    )

    neighbor_temperature = heat_map.get_temperature(
        x=5,
        y=4,
    )

    assert center_temperature < 10.0
    assert neighbor_temperature > 0.0


def test_cooling_reduces_temperature():
    """Cooling should reduce temperature values."""

    heat_map = HeatMap(
        width=10,
        height=10,
        cooling_rate=1.0,
        dt=0.1,
    )

    heat_map.set_temperature(
        x=3,
        y=3,
        value=10.0,
    )

    heat_map.cool()

    assert heat_map.get_temperature(
        x=3,
        y=3,
    ) == pytest.approx(9.0)


def test_step_performs_diffusion_and_cooling():
    """Step should update the heat field."""

    heat_map = HeatMap(
        width=10,
        height=10,
    )

    heat_map.mark_visited(
        x=5,
        y=5,
        amount=10.0,
    )

    initial_temperature = heat_map.get_temperature(
        x=5,
        y=5,
    )

    heat_map.step()

    final_temperature = heat_map.get_temperature(
        x=5,
        y=5,
    )

    assert final_temperature != initial_temperature


def test_reset_clears_temperature():
    """Reset should return all temperatures to zero."""

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    heat_map.mark_visited(
        x=10,
        y=10,
        amount=5.0,
    )

    heat_map.reset()

    assert np.all(
        heat_map.temperature == 0.0
    )


def test_out_of_bounds_temperature_access():
    """Accessing invalid cells should raise an error."""

    heat_map = HeatMap(
        width=10,
        height=10,
    )

    with pytest.raises(ValueError):
        heat_map.get_temperature(
            x=20,
            y=20,
        )


def test_out_of_bounds_mark_visited_is_ignored():
    """
    Marking a cell outside the map should not modify the heat field.
    """

    heat_map = HeatMap(
        width=10,
        height=10,
    )

    heat_map.mark_visited(
        x=20,
        y=20,
        amount=5.0,
    )

    assert np.all(
        heat_map.temperature == 0.0
    )