from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator


def test_random_rectangle_generation():

    generator = EnvironmentGenerator()

    obstacle = generator.random_rectangle(
        width=100,
        height=100
    )

    assert obstacle.x_min >= 0
    assert obstacle.y_min >= 0

    assert obstacle.x_max <= 100
    assert obstacle.y_max <= 100