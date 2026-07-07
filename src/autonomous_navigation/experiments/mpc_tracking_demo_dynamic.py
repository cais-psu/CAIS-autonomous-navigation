import numpy as np

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import CircleObstacle, RectangleObstacle
from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.controllers.mpc.omnidirectional import OmniMPC
from autonomous_navigation.core.planners.path_utils import interpolate_path


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

planner = RRTPlanner(grid, 2)
start = RRTNode(0, 0)
goal = RRTNode(90, 90)

path, stats = planner.plan(start, goal, 5000)

path = interpolate_path(path, spacing=0.5)


if path is None:
    print("No path found")
    exit()
else:
    print(f"Path found with {len(path)} waypoints")


mpc = OmniMPC(
    dt = 0.1,
    horizon = 20,
    Q = 100 * np.eye(2),
    R = 0.1 * np.eye(2),
    v_max = 1.0
)

x = np.array([start.x, start.y])
trajectory = [x.copy()]

max_steps = 5000
steps = 0
i = 0

while i < len(path) - 1 and steps < max_steps:

    target = np.asarray(path[i + 1])

    distances = [
        np.linalg.norm(x - np.asarray(p))
        for p in path
    ]

    closest = np.argmin(distances)

    reference = []

    for j in range(mpc.N):
        idx = min(closest + j, len(path)-1)
        reference.append(np.asarray(path[idx]))

    # u = mpc.solve(current_state, reference_horizon)
    print(np.array(reference))
    
    mpc.set_reference(reference)
    u = mpc.solve(x)
    
    x = mpc.predict(x, u)
    print(f"State:  {x}, Control: {u}")
    trajectory.append(x.copy())

    if np.linalg.norm(target - x) < 2:
        i += 1

    goal_position = np.array([goal.x, goal.y])

    if np.linalg.norm(x - goal_position) < 0.5:
        print("Goal reached!")
        break


    steps += 1

if steps == max_steps:
    print("Simulation terminated: maximum number of steps reached.")
    

trajectory = np.array(trajectory)

fig, ax = plt.subplots(figsize=(8,8))

# Move window position (TkAgg)
fig.canvas.manager.window.wm_geometry("+300+100")

# Occupancy grid
ax.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r"
)

for obstacle in grid.obstacles:
    obstacle.plot(ax)

# Draw RRT tree
for node in planner.nodes:

    if node.parent is not None:

        parent = planner.nodes[node.parent]

        ax.plot(
            [parent.x, node.x],
            [parent.y, node.y],
            linewidth=0.5
        )


# Draw RRT path
xs = [p[0] for p in path]
ys = [p[1] for p in path]

ax.plot(
    xs,
    ys,
    linewidth=3,
    label="RRT Path"
)


# Draw MPC trajectory
ax.plot(
    trajectory[:,0],
    trajectory[:,1],
    linewidth=2,
    label="MPC Trajectory"
)


# Start and goal
ax.plot(
    start.x,
    start.y,
    marker="o",
    markersize=8,
    label="Start"
)

ax.plot(
    goal.x,
    goal.y,
    marker="x",
    markersize=8,
    label="Goal"
)



# Robot marker
robot_radius = 2.0  # grid units

robot = Circle(
    (trajectory[0,0], trajectory[0,1]),
    robot_radius,
    fill=False
)

ax.add_patch(robot)


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("RRT + MPC Tracking")

ax.grid()
ax.legend()


def update(frame):

    robot.center = (
        trajectory[frame,0],
        trajectory[frame,1]
    )

    return robot,


animation = FuncAnimation(
    fig,
    update,
    frames=len(trajectory),
    interval=100,
    blit=True
)


plt.show()