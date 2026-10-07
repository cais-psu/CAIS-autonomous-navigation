import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.local_map import LocalMap
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.hedac_sampler import HEDACSampler


WIDTH = 100
HEIGHT = 100

SENSING_RADIUS = 20.0
RRT_STEP_SIZE = 2.0
RRT_ITERATIONS = 1

BETA = 10.0

NUM_STEPS = 250

ROBOT_RADIUS = 1.0
SAFETY_MARGIN = 1.0

grid = OccupancyGrid.random_environment(
    width=WIDTH,
    height=HEIGHT,
    n_circles=10,
    n_rectangles=10,
    seed=6,
)

cspace = grid.create_configuration_space(
    ROBOT_RADIUS + SAFETY_MARGIN
)

heat = HeatMap(
    width=WIDTH,
    height=HEIGHT,
)

hedac_sampler = HEDACSampler(
    heat_map=heat,
    gamma=BETA,
)


robot = RRTNode(
    10.0,
    10.0,
)

trajectory = [
    (robot.x, robot.y)
]



for step in range(NUM_STEPS):

    print(f"\nStep {step}")

    # --------------------------------------------------------
    # 1. Deposit heat at robot position
    # --------------------------------------------------------

    heat.mark_visited(
        int(round(robot.x)),
        int(round(robot.y)),
        amount=1.0,
    )

    heat.step()

    local_map = LocalMap(
        global_grid=cspace,
        sensing_radius=SENSING_RADIUS,
    )

    local_grid = local_map.get_local_grid(
        robot.x,
        robot.y,
    )

    bounds = local_map.get_bounds(
        robot.x,
        robot.y,
    )

    rrt = RRTPlanner(
        occupancy_grid=local_grid,
        step_size=RRT_STEP_SIZE,
        sampler=hedac_sampler,
    )

    rrt.build_tree(
        start=RRTNode(robot.x, robot.y),
        max_iters=RRT_ITERATIONS,
        sampling_bounds=bounds,
    )

    if len(rrt.nodes) <= 1:
        print("No valid RRT extension.")
        continue

    new_node = rrt.nodes[-1]

    robot = RRTNode(
        new_node.x,
        new_node.y,
    )

    trajectory.append(
        (robot.x, robot.y)
    )

    print(
        f"Robot: ({robot.x:.2f}, {robot.y:.2f})"
    )

fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.imshow(
    heat.temperature.T,
    origin="lower",
    cmap="hot",
    alpha=0.75,
    extent=[0, WIDTH, 0, HEIGHT],
)

# Obstacles

for obstacle in cspace.obstacles:
    obstacle.plot(
        ax,
        edgecolor="black",
        linewidth=1,
    )

# Robot trajectory

trajectory = np.asarray(trajectory)

ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    linewidth=1.5,
    label="Robot trajectory",
)

# Start

ax.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    c="cyan",
    s=100,
    edgecolors="black",
    label="Start",
)

# Final position

ax.scatter(
    trajectory[-1, 0],
    trajectory[-1, 1],
    c="lime",
    s=100,
    edgecolors="black",
    label="Final position",
)

ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect("equal")
ax.grid(alpha=0.2)

ax.legend()

plt.tight_layout()
plt.show()