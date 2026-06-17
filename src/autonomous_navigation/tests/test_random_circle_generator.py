from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator

def test_random_circle_generation():

    generator = EnvironmentGenerator()

    circle = generator.random_circle(
        width=100,
        height=100
    )

    assert circle.radius > 0

    assert circle.center_x - circle.radius >= 0
    assert circle.center_y - circle.radius >= 0

    assert circle.center_x + circle.radius < 100
    assert circle.center_y + circle.radius < 100