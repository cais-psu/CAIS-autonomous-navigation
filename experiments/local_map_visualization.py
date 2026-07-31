import matplotlib.pyplot as plt
import matplotlib.patches as patches

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator
from autonomous_navigation.core.environment.local_map import LocalMap


# ----------------------------
# Create ground truth environment
# ----------------------------

width = 100
height = 100

grid = OccupancyGrid(
    width=width,
    height=height,
)

generator = EnvironmentGenerator()


# Add random circles
for _ in range(10):
    circle = generator.random_circle(
        width=width,
        height=height,
        min_radius=4,
        max_radius=12,
    )

    grid.add_circle_obstacle(circle)


# Add random rectangles
for _ in range(10):
    rectangle = generator.random_rectangle(
        width=width,
        height=height,
        min_size=5,
        max_size=15,
    )

    grid.add_rectangle_obstacle(rectangle)


# ----------------------------
# Create local perception map
# ----------------------------

robot_x = 80
robot_y = 80

sensing_radius = 10

local_map = LocalMap(
    global_grid=grid,
    sensing_radius=sensing_radius,
)

local_grid = local_map.get_local_grid(
    robot_x,
    robot_y,
)


# ----------------------------
# Visualization
# ----------------------------

fig, ax = plt.subplots(
    figsize=(8, 8)
)


# Plot global obstacles
for obstacle in grid.obstacles:
    obstacle.plot(
        ax,
        edgecolor="black",
        linewidth=1,
    )


# Plot locally perceived obstacles
for obstacle in local_grid.obstacles:
    obstacle.plot(
        ax,
        edgecolor="red",
        linewidth=2,
    )


# Plot sensing radius
sensor_circle = patches.Circle(
    (robot_x, robot_y),
    sensing_radius,
    fill=False,
    edgecolor="blue",
    linestyle="--",
    linewidth=2,
)

ax.add_patch(sensor_circle)


# Plot robot
ax.plot(
    robot_x,
    robot_y,
    marker="o",
    markersize=8,
    color="blue",
)


# Formatting
ax.set_xlim(0, width)
ax.set_ylim(0, height)

ax.set_aspect("equal")

ax.set_title(
    "Local Map Perception Test"
)

ax.grid(True)

plt.show()