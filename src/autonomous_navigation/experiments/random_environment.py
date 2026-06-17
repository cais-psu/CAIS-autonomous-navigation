"""
Generate 10 random rectangles and 10 random circles.
"""
from matplotlib import pyplot as plt
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import CircleObstacle, RectangleObstacle
from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator

grid = OccupancyGrid(
    width=100,
    height=100
)

generator = EnvironmentGenerator()

for _ in range(0,10):
    circle = generator.random_circle(
        width=100,
        height=100,
        min_radius=4,
        max_radius=12
    )
    grid.add_circle_obstacle(circle)


for _ in range(0,10):
    rectangle = generator.random_rectangle(
        width=100,
        height=100,
        min_size=1,
        max_size=15
    )
    grid.add_rectangle_obstacle(rectangle)

# Display occupancy grid
plt.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r"
)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Environment')
plt.grid()

plt.show()