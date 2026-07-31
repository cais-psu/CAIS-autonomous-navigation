import pytest
import matplotlib.pyplot as plt

from autonomous_navigation.core.environment.obstacles import (
    Obstacle,
    RectangleObstacle,
    CircleObstacle,
)


def test_rectangle_inflate():
    """Rectangle should inflate equally in all directions."""

    rect = RectangleObstacle(
        x_min=1.0,
        y_min=2.0,
        x_max=5.0,
        y_max=6.0,
    )

    inflated = rect.inflate(0.5)

    assert inflated.x_min == pytest.approx(0.5)
    assert inflated.y_min == pytest.approx(1.5)
    assert inflated.x_max == pytest.approx(5.5)
    assert inflated.y_max == pytest.approx(6.5)


def test_circle_inflate():
    """Circle radius should increase while center remains fixed."""

    circle = CircleObstacle(
        center_x=2.0,
        center_y=3.0,
        radius=1.5,
    )

    inflated = circle.inflate(0.75)

    assert inflated.center_x == pytest.approx(2.0)
    assert inflated.center_y == pytest.approx(3.0)
    assert inflated.radius == pytest.approx(2.25)


def test_inflate_returns_new_object():
    """inflate() should return a new object rather than modifying the original."""

    rect = RectangleObstacle(0, 0, 4, 4)

    inflated = rect.inflate(1.0)

    assert inflated is not rect


def test_original_rectangle_unchanged():
    """Original rectangle should remain unchanged after inflation."""

    rect = RectangleObstacle(0, 0, 4, 4)

    _ = rect.inflate(1.0)

    assert rect.x_min == 0
    assert rect.y_min == 0
    assert rect.x_max == 4
    assert rect.y_max == 4


def test_original_circle_unchanged():
    """Original circle should remain unchanged after inflation."""

    circle = CircleObstacle(3, 4, 2)

    _ = circle.inflate(0.5)

    assert circle.center_x == 3
    assert circle.center_y == 4
    assert circle.radius == 2


def test_rectangle_is_obstacle():
    """RectangleObstacle should inherit from Obstacle."""

    rect = RectangleObstacle(0, 0, 1, 1)

    assert isinstance(rect, Obstacle)


def test_circle_is_obstacle():
    """CircleObstacle should inherit from Obstacle."""

    circle = CircleObstacle(0, 0, 1)

    assert isinstance(circle, Obstacle)


def test_rectangle_plot():
    """Rectangle plot should execute without raising exceptions."""

    fig, ax = plt.subplots()

    rect = RectangleObstacle(0, 0, 2, 2)

    rect.plot(ax)

    plt.close(fig)


def test_circle_plot():
    """Circle plot should execute without raising exceptions."""

    fig, ax = plt.subplots()

    circle = CircleObstacle(0, 0, 1)

    circle.plot(ax)

    plt.close(fig)


def test_plot_kwargs():
    """plot() should accept arbitrary matplotlib keyword arguments."""

    fig, ax = plt.subplots()

    rect = RectangleObstacle(0, 0, 2, 2)
    rect.plot(
        ax,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
    )

    circle = CircleObstacle(4, 4, 1)
    circle.plot(
        ax,
        edgecolor="blue",
        linewidth=3,
    )

    plt.close(fig)