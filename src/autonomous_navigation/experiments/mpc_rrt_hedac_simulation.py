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
from autonomous_navigation.core.planners.corridor_generator import CorridorGenerator


grid = OccupancyGrid.random_environment(
    width=100,
    height=100,
    n_circles=10,
    n_rectangles=10,
    seed=6,
)

robot_radius = 1
safety_margin = 1

inflation_radius = robot_radius + safety_margin

cspace = grid.create_configuration_space(
    inflation_radius
)

for x in range(80,85):
    for y in range(35,40):

        grid_value = cspace.is_occupied(x,y)

        print(
            (x,y),
            "grid:",
            grid_value
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

global_goal = RRTNode(
    80,
    80,
)

robot = start.copy()

trajectory = [
    robot.copy()
]

max_steps = 5000

planning_history = []
target_history = []
mode_history = []
heat_history = []

for step in range(max_steps):

    #
    # Create a new mission if needed
    #
    if not manager.has_local_mission():

        result = manager.begin_navigation_mission(
            robot_position=robot,
            global_goal=global_goal,
        )

        if result is not None:

            planning_history.append(
                result
            )

            target_history.append(
                [
                    result.target.x,
                    result.target.y
                ]
            )

            mode_history.append(
                manager.mode
            )

            heat_history.append(
                heat.temperature.copy()
            )

        if result is None:

            print("===================")
            print("Planning failed")
            print("Robot:", robot)
            print("Mode:", manager.mode)
            print(
                "Target:",
                manager.current_target
            )
            print("===================")

            break

    target = manager.current_target


    #
    # Current MPC target
    #
    waypoint = manager.get_current_waypoint()


    reference = np.tile(
        waypoint,
        (mpc.N,1)
    )

    mpc.set_reference(
        reference
    )


    #
    # Solve MPC
    #
    u = mpc.solve(
        robot
    )


    robot = mpc.predict(
        robot,
        u
    )


    trajectory.append(
        robot.copy()
    )


    #
    # Mission progress
    #
    if manager.waypoint_reached(robot):

        if manager.mission_complete():

            print(
                f"{manager.mode} mission complete"
            )


            if manager.mode == "goal":

                print("Reached global goal")
                break


            #
            # Exploration target reached:
            # update HEDAC heat and restart
            #
            manager.complete_current_mission(
                robot
            )


        else:

            manager.advance_waypoint()


print("Finished")

fig, ax = plt.subplots(
    figsize=(8,8)
)


#
# Occupancy grid
#
ax.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r",
    alpha=0.5,
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
#
# MPC trajectory
#
traj = np.array(trajectory)

ax.plot(
    traj[:,0],
    traj[:,1],
    linewidth=2,
    label="MPC trajectory"
)


#
# Start and goal
#
ax.scatter(
    start[0],
    start[1],
    c="green",
    s=100,
    label="Start"
)

ax.scatter(
    global_goal.x,
    global_goal.y,
    c="red",
    s=100,
    label="Goal"
)


#
# Local targets
#
for target in target_history:

    ax.scatter(
        target[0],
        target[1],
        marker="x",
        s=100,
        label="HEDAC target"
    )


#
# Final robot
#
ax.scatter(
    robot[0],
    robot[1],
    c="blue",
    s=100,
    label="Final robot"
)

last = planning_history[-1]

for node in last.rrt.nodes:

    ax.scatter(
        node.x,
        node.y,
        s=2,
        c="blue"
    )

last = planning_history[-1]

for corridor in last.corridors:

    vertices = CorridorGenerator.corridor_to_polygon(corridor)

    polygon = patches.Polygon(
        vertices,
        fill=False,
        linewidth=1,
    )

    ax.add_patch(
        polygon
    )

ax.set_xlim(0,100)
ax.set_ylim(0,100)

ax.set_aspect("equal")
ax.grid()
ax.legend()

plt.show()

plt.figure(figsize=(8,8))

plt.imshow(
    heat.temperature.T,
    origin="lower",
    cmap="hot"
)

plt.colorbar(
    label="Temperature"
)

plt.scatter(
    traj[:,0],
    traj[:,1],
    c="cyan",
    s=5
)

plt.scatter(
    start[0],
    start[1],
    c="green",
    label="start"
)

plt.scatter(
    global_goal.x,
    global_goal.y,
    c="red",
    label="goal"
)


plt.legend()
plt.axis("equal")
plt.show()