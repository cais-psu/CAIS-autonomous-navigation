from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator


def test_generated_obstacle_can_be_added():

    grid = OccupancyGrid(100, 100)

    generator = EnvironmentGenerator()

    obstacle = generator.random_rectangle(
        width=100,
        height=100
    )

    grid.add_rectangle_obstacle(obstacle)

    assert (
        grid.is_occupied(
            obstacle.x_min,
            obstacle.y_min
        )
    )