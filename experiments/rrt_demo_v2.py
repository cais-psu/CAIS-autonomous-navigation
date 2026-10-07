import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.exploration_manager import ExplorationManager
from autonomous_navigation.core.exploration.hedac_sampler import HEDACSampler


# ============================================================
# Environment
# ============================================================

grid = OccupancyGrid.random_environment(
    width=100,
    height=100,
    n_circles=10,
    n_rectangles=10,
    seed=5,
)

robot_radius = 1
safety_margin = 1

cspace = grid.create_configuration_space(
    robot_radius + safety_margin
)


# ============================================================
# Heat map
# ============================================================

heat = HeatMap(
    width=100,
    height=100,
)

hedac_sampler = HEDACSampler(
    heat_map=heat,
    gamma=10.0,
)


# ============================================================
# Exploration manager
# ============================================================

manager = ExplorationManager(
    occupancy_grid=cspace,
    heat_map=heat,
    sensing_radius=20,
    planner_step_size=2,
    sampler=hedac_sampler,
)


# ============================================================
# Robot
# ============================================================

start = RRTNode(
    10.0,
    10.0,
)


# ============================================================
# TEST 1:
# Build HEDAC-biased local RRT
# ============================================================

rrt, stats = manager.local_rrt.build_tree(
    start,
    max_iters=50,
)

print("\n==============================")
print("RRT BUILD")
print("==============================")

print("Success:", stats["success"])
print("Iterations:", stats["iterations"])
print("Nodes:", stats["nodes"])


# ============================================================
# Select exploration node
# ============================================================

index, target = manager.target_selector.select_exploration_node(
    robot_x=start.x,
    robot_y=start.y,
    heat_map=heat,
    rrt=rrt,
    planner_step_size=manager.local_rrt.step_size,
)

print("\n==============================")
print("TARGET SELECTION")
print("==============================")

print("Target index:", index)
print(
    "Target:",
    f"({target.x:.2f}, {target.y:.2f})"
)

print(
    "Target temperature:",
    heat.get_temperature(
        int(target.x),
        int(target.y),
    )
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Environment
ax.imshow(
    cspace.grid,
    origin="lower",
    cmap="gray_r",
    alpha=0.5,
)

# Configuration-space obstacles
for obstacle in cspace.obstacles:
    obstacle.plot(
        ax,
        edgecolor="red",
        linewidth=1,
        linestyle="--",
    )


# RRT edges
for i, node in enumerate(rrt.nodes):

    if node.parent is None:
        continue

    parent = rrt.nodes[node.parent]

    ax.plot(
        [parent.x, node.x],
        [parent.y, node.y],
        linewidth=0.5,
        alpha=0.5,
    )


# RRT nodes
nodes_x = [
    node.x
    for node in rrt.nodes
]

nodes_y = [
    node.y
    for node in rrt.nodes
]

ax.scatter(
    nodes_x,
    nodes_y,
    s=4,
)


# Robot
ax.scatter(
    start.x,
    start.y,
    s=100,
    label="Robot",
)


# Selected target
ax.scatter(
    target.x,
    target.y,
    marker="*",
    s=250,
    label="Selected target",
)


ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.grid()
ax.legend()

plt.show()