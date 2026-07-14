import numpy as np
import random

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.environment_generator import (
    EnvironmentGenerator,
)

from autonomous_navigation.core.planners.rrt_node import RRTNode

from autonomous_navigation.core.controllers.mpc.omnidirectional import OmniMPC

from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.exploration_manager import (
    ExplorationManager,
)



# -------------------------------------------------
# Environment
# -------------------------------------------------
random.seed(0)
np.random.seed(0)

grid = OccupancyGrid(
    width=100,
    height=100,
)

generator = EnvironmentGenerator()


for _ in range(10):

    circle = generator.random_circle(
        width=100,
        height=100,
        min_radius=4,
        max_radius=12,
    )

    grid.add_circle_obstacle(circle)



for _ in range(10):

    rectangle = generator.random_rectangle(
        width=100,
        height=100,
        min_size=1,
        max_size=15,
    )

    grid.add_rectangle_obstacle(rectangle)



# -------------------------------------------------
# Configuration space
# -------------------------------------------------

robot_radius = 2
safety_margin = 1

inflation_radius = (
    robot_radius + safety_margin
)


cspace = grid.create_configuration_space(
    inflation_radius
)



# -------------------------------------------------
# Exploration manager
# -------------------------------------------------

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



# -------------------------------------------------
# MPC
# -------------------------------------------------

mpc = OmniMPC(
    dt=0.1,
    horizon=20,
    Q=100*np.eye(2),
    R=0.1*np.eye(2),
    v_max=1.0,
)



# -------------------------------------------------
# Mission
# -------------------------------------------------

goal = RRTNode(
    30,
    25,
)

print(f"Goal occupied: {cspace.is_occupied(goal.x, goal.y)}")
if cspace.is_occupied(goal.x, goal.y):
    exit()


print(
    f"Goal is free: {cspace.is_free(
        goal.x,
        goal.y
    )}"
)
if not cspace.is_free(goal.x, goal.y):
    exit()

initial_start = RRTNode(
    20,
    20,
)


x = np.array(
    [
        initial_start.x,
        initial_start.y,
    ],
    dtype=float,
)


trajectory = [
    x.copy()
]


last_result = None


mission_complete = False


max_steps = 5000


# -------------------------------------------------
# Main loop
# -------------------------------------------------

for step in range(max_steps):


    result = manager.plan(
        robot_position=x,
        mission_goal=goal,
        max_rrt_iterations=5000,
    )

    if result.path is None:

        print("No path found.")
        break
    
    last_result = result


    path = np.asarray(result.path)


    reference = []

    for i in range(mpc.N):

        idx = min(
            i + 1,
            len(path)-1,
        )

        reference.append(
            np.asarray(path[idx])
        )



    mpc.set_reference(reference)

    u = mpc.solve(x)


    x = mpc.predict(
        x,
        u,
    )


    trajectory.append(
        x.copy()
    )


    print(
        f"Step {step}: "
        f"State={x}, "
        f"Target={result.target.x,result.target.y}"
    )


    distance = np.linalg.norm(
        x - np.array(
            [
                goal.x,
                goal.y,
            ]
        )
    )


    if distance < 0.5:

        mission_complete = True

        print("Goal reached!")

        break



if not mission_complete:

    print(
        "Maximum iterations reached."
    )



trajectory = np.asarray(
    trajectory
)



# -------------------------------------------------
# Visualization
# -------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8,8)
)


fig.canvas.manager.window.wm_geometry(
    "+300+100"
)



ax.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r",
)



for obstacle in grid.obstacles:

    obstacle.plot(
        ax,
        edgecolor="black",
        linewidth=2,
    )



# Last planning result

path = np.asarray(last_result.path)
rrt = last_result.rrt
local_grid = last_result.local_grid
bounds = last_result.bounds



for obstacle in local_grid.obstacles:

    obstacle.plot(
        ax,
        edgecolor="green",
        linewidth=2,
    )



# RRT tree

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
        )



# Planned path

ax.plot(
    path[:,0],
    path[:,1],
    linewidth=3,
    label="RRT Path",
)



# MPC trajectory

ax.plot(
    trajectory[:,0],
    trajectory[:,1],
    linewidth=2,
    label="MPC trajectory",
)



# Start

ax.plot(
    initial_start.x,
    initial_start.y,
    marker="o",
    markersize=8,
    label="Start",
)



# Goal

ax.plot(
    goal.x,
    goal.y,
    marker="x",
    markersize=8,
    label="Goal",
)



# Robot

robot = Circle(
    (
        trajectory[0,0],
        trajectory[0,1],
    ),
    robot_radius,
    fill=False,
)


ax.add_patch(robot)



# Sensing radius

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



# Local map window

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



ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_title(
    "Exploration Manager + MPC"
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
        cspace.width-1,
        trajectory[frame,0]+manager.sensing_radius,
    )


    ymin = max(
        0,
        trajectory[frame,1]-manager.sensing_radius,
    )

    ymax = min(
        cspace.height-1,
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