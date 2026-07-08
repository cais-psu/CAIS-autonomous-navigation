import numpy as np
import pytest

from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.hedac_sampler import HEDACSampler


def test_sampler_initialization():
    """Sampler should store heat map and gamma."""

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    assert sampler.heat_map is heat_map
    assert sampler.gamma == 2.0


def test_empty_candidates_raise_error():
    """Selecting from an empty candidate set should fail."""

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    sampler = HEDACSampler(
        heat_map
    )

    candidates = np.empty(
        (0, 2),
        dtype=int,
    )

    with pytest.raises(ValueError):
        sampler.select_sample(
            candidates
        )


def test_selected_sample_is_valid_candidate():
    """Returned sample should always be one of the candidates."""

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    sampler = HEDACSampler(
        heat_map
    )

    candidates = np.array(
        [
            [1, 1],
            [5, 5],
            [10, 10],
        ]
    )

    sample = sampler.select_sample(
        candidates
    )

    assert sample in [
        (1, 1),
        (5, 5),
        (10, 10),
    ]


def test_low_temperature_is_preferred():
    """
    Low-temperature cells should be selected more frequently
    than high-temperature cells.
    """

    np.random.seed(42)

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    # Candidate A: cold
    heat_map.set_temperature(
        x=5,
        y=5,
        value=0.0,
    )

    # Candidate B: hot
    heat_map.set_temperature(
        x=10,
        y=10,
        value=10.0,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    candidates = np.array(
        [
            [5, 5],
            [10, 10],
        ]
    )

    selections = []

    for _ in range(1000):
        selections.append(
            sampler.select_sample(
                candidates
            )
        )

    cold_count = selections.count(
        (5, 5)
    )

    hot_count = selections.count(
        (10, 10)
    )

    assert cold_count > hot_count


def test_equal_temperature_gives_equal_probability():
    """
    Candidates with identical temperatures should have
    approximately equal selection probability.
    """

    np.random.seed(42)

    heat_map = HeatMap(
        width=20,
        height=20,
    )

    heat_map.set_temperature(
        x=5,
        y=5,
        value=2.0,
    )

    heat_map.set_temperature(
        x=10,
        y=10,
        value=2.0,
    )

    sampler = HEDACSampler(
        heat_map,
        gamma=2.0,
    )

    candidates = np.array(
        [
            [5, 5],
            [10, 10],
        ]
    )

    selections = []

    for _ in range(1000):
        selections.append(
            sampler.select_sample(
                candidates
            )
        )

    first_count = selections.count(
        (5, 5)
    )

    second_count = selections.count(
        (10, 10)
    )

    # Allow statistical variation
    ratio = first_count / second_count

    assert 0.75 < ratio < 1.33