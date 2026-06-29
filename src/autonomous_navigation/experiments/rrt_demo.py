"""
Generate 10 random rectangles and 10 random circles.
"""
import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import CircleObstacle, RectangleObstacle
from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner

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

planner = RRTPlanner(grid, 5)
start = RRTNode(0, 0)
goal = RRTNode(90, 90)

path, stats = planner.plan(start, goal)

if path is None:
    print("No path found")
else:
    print(f"Path found with {len(path)} waypoints")

# Display occupancy grid
plt.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r"
)

# Draw tree
for node in planner.nodes:

    if node.parent is not None:

        parent = planner.nodes[node.parent]

        plt.plot(
            [parent.x, node.x],
            [parent.y, node.y],
            linewidth=0.5
        )

# Draw path
if path is not None:

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    plt.plot(
        xs,
        ys,
        linewidth=3
    )

# Start and goal
plt.plot(start.x, start.y, marker="o", markersize=8)
plt.plot(goal.x, goal.y, marker="x", markersize=8)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("RRT Demo")
plt.grid()

plt.show()