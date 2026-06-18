from autonomous_navigation.core.controllers.path_follower import PathFollower
from autonomous_navigation.core.robots.omni_robot import OmniRobot
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner, RRTNode
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.environment_generator import EnvironmentGenerator
from autonomous_navigation.core.robots.robot_state import RobotState

import matplotlib.pyplot as plt

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

start_node = RRTNode(0, 0)
goal_node = RRTNode(90, 90)

planner = RRTPlanner(grid, step_size=4)

path, stats = planner.plan(start_node, goal_node)

if path is None:
    print("No path found")
    raise SystemExit
else:
    print(f"Path found with {len(path)} waypoints")

start_state = RobotState(
    x=0.0,
    y=0.0,
    theta=0.0
)

robot = OmniRobot(start_state)
follower = PathFollower(kp=0.8)

x_hist, y_hist = [], []

for _ in range(2000):

    vx, vy, omega = follower.compute(robot.state, path)

    robot.step(vx, vy, omega, dt=0.1)

    x_hist.append(robot.state.x)
    y_hist.append(robot.state.y)

print(
    f"Final robot position: "
    f"({robot.state.x:.2f}, {robot.state.y:.2f})"
)

plt.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r"
)

for node in planner.nodes:

    if node.parent is not None:

        parent = planner.nodes[node.parent]

        plt.plot(
            [parent.x, node.x],
            [parent.y, node.y],
            linewidth=0.5
        )

path_x = [p[0] for p in path]
path_y = [p[1] for p in path]

plt.plot(
    path_x,
    path_y,
    linewidth=3,
    label="RRT Path"
)

plt.plot(
    x_hist,
    y_hist,
    linewidth=2,
    linestyle="--",
    label="Robot Trajectory"
)

plt.plot(
    start_node.x,
    start_node.y,
    marker="o",
    markersize=10,
    label="Start"
)

plt.plot(
    goal_node.x,
    goal_node.y,
    marker="x",
    markersize=10,
    label="Goal"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("RRT Path Following")

plt.legend()
plt.axis("equal")

plt.show()

print(stats)