import numpy as np
import random

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.controllers.mpc.omnidirectional import OmniMPC
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.exploration_manager import ExplorationManager


grid = OccupancyGrid.random_environment(
    width=100,
    height=100,
    n_circles=10,
    n_rectangles=10,
    seed=1,
)

robot_radius = 2
safety_margin = 1

inflation_radius = robot_radius + safety_margin

cspace = grid.create_configuration_space(
    inflation_radius
)

heat = HeatMap(
    width=100,
    height=100,
)

manager = ExplorationManager(
    occupancy_grid=cspace,
    heat_map=heat,
    sensing_radius=20,
    planner_step_size=2,
)

mpc = OmniMPC(
    dt=0.1,
    horizon=20,
    Q=100*np.eye(2),
    R=0.1*np.eye(2),
    v_max=1.0,
)

start = np.array(
    [10.0,10.0]
)

goal = RRTNode(
    40,
    40,
)

x = start.copy()

trajectory = [
    x.copy()
]

planning_history = []
mode_history = []
target_history = []

max_steps = 5000

for step in range(max_steps):

    result = manager.plan(
        robot_position=x,
        mission_goal=goal,
        max_rrt_iterations=5000,
    )

    if result.path is None:
        print("Planner failed at step: ", step)
        print("Robot: ", x)
        print("Goal: ", goal.x, goal.y)
        break


    path = np.asarray(result.path)

    planning_history.append(
        result
    )


    reference = []

    for i in range(mpc.N):

        idx = min(
            i+1,
            len(path)-1
        )

        reference.append(
            path[idx]
        )


    mpc.set_reference(reference)

    u = mpc.solve(x)

    x = mpc.predict(
        x,
        u
    )


    trajectory.append(
        x.copy()
    )


    distance = np.linalg.norm(
        x - np.array([goal.x, goal.y])
    )

    mode_history.append(
        result.stats["mode"]
    )

    target_history.append(
        [
            result.target.x,
            result.target.y,
        ]
    )

    print(
        f"Step {step}: "
        f"State={x}, "
        f"Target={result.target.x,result.target.y}, "
        f"Mode={result.stats['mode']}"
    )


    if distance < 0.5:
        print("Goal reached")
        break

    if step == max_steps-1:
        print("Maximum iterations reached")


print(set(mode_history))

if len(planning_history) == 0:
    raise RuntimeError(
        "No successful planning iterations."
    )

last_result = planning_history[-1]

trajectory = np.asarray(
    trajectory
)


# -------------------------------------------------
# Visualization
# -------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8,8)
)


ax.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r",
)


# Original obstacles

for obstacle in grid.obstacles:

    obstacle.plot(
        ax,
        edgecolor="black",
        linewidth=2,
    )


# Configuration space obstacles

for obstacle in cspace.obstacles:

    obstacle.plot(
        ax,
        edgecolor="red",
        linewidth=1,
        linestyle="--",
    )


# -------------------------------------------------
# Final planning result
# -------------------------------------------------

path = np.asarray(
    last_result.path
)

rrt = last_result.rrt

bounds = last_result.bounds

corridors = last_result.corridors


# -------------------------------------------------
# RRT Tree
# -------------------------------------------------

for node in rrt.nodes:

    if node.parent is not None:

        parent = rrt.nodes[node.parent]

        ax.plot(
            [
                parent.x,
                node.x,
            ],
            [
                parent.y,
                node.y,
            ],
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )


# -------------------------------------------------
# RRT Path
# -------------------------------------------------

ax.plot(
    path[:,0],
    path[:,1],
    linewidth=3,
    label="RRT Path",
)


# -------------------------------------------------
# Corridors
# -------------------------------------------------

for corridor in corridors:

    xmin, xmax, ymin, ymax = corridor.get_bounds()

    rect = patches.Rectangle(
        (
            xmin,
            ymin,
        ),
        xmax-xmin,
        ymax-ymin,
        fill=False,
        linestyle=":",
        linewidth=1,
    )

    ax.add_patch(rect)


# -------------------------------------------------
# MPC trajectory
# -------------------------------------------------

ax.plot(
    trajectory[:,0],
    trajectory[:,1],
    linewidth=2,
    label="MPC trajectory",
)


# -------------------------------------------------
# Start / Goal
# -------------------------------------------------

ax.plot(
    start[0],
    start[1],
    marker="o",
    markersize=8,
    label="Start",
)


ax.plot(
    goal.x,
    goal.y,
    marker="x",
    markersize=10,
    label="Goal",
)


# -------------------------------------------------
# Robot
# -------------------------------------------------

robot = Circle(
    (
        trajectory[0,0],
        trajectory[0,1],
    ),
    robot_radius,
    fill=False,
)


ax.add_patch(robot)


# -------------------------------------------------
# Sensing radius
# -------------------------------------------------

sensor = Circle(
    (
        trajectory[0,0],
        trajectory[0,1],
    ),
    manager.sensing_radius,
    fill=False,
    linestyle="--",
)


ax.add_patch(sensor)


# -------------------------------------------------
# Local map window
# -------------------------------------------------

xmin, xmax, ymin, ymax = bounds


window = patches.Rectangle(
    (
        xmin,
        ymin,
    ),
    xmax-xmin,
    ymax-ymin,
    fill=False,
    linestyle=":",
)


ax.add_patch(window)


ax.set_xlim(
    0,
    grid.width,
)

ax.set_ylim(
    0,
    grid.height,
)


ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_title(
    "Closed Loop HEDAC-RRT-MPC Navigation"
)

ax.grid()

ax.legend()



# -------------------------------------------------
# Animation
# -------------------------------------------------

def update(frame):

    robot.center = (
        trajectory[frame,0],
        trajectory[frame,1],
    )


    sensor.center = (
        trajectory[frame,0],
        trajectory[frame,1],
    )


    xmin = max(
        0,
        trajectory[frame,0]-manager.sensing_radius,
    )

    xmax = min(
        grid.width,
        trajectory[frame,0]+manager.sensing_radius,
    )


    ymin = max(
        0,
        trajectory[frame,1]-manager.sensing_radius,
    )

    ymax = min(
        grid.height,
        trajectory[frame,1]+manager.sensing_radius,
    )


    window.set_xy(
        (
            xmin,
            ymin,
        )
    )

    window.set_width(
        xmax-xmin
    )

    window.set_height(
        ymax-ymin
    )


    return (
        robot,
        sensor,
        window,
    )


animation = FuncAnimation(
    fig,
    update,
    frames=len(trajectory),
    interval=100,
    blit=True,
)


plt.show()