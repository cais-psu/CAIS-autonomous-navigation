import matplotlib.pyplot as plt

from autonomous_navigation.core.robots.robot_state import RobotState
from autonomous_navigation.core.robots.omni_robot import OmniRobot
from autonomous_navigation.core.controllers.p_controller import PController


def run_simulation():

    state = RobotState(
        x=2.0,
        y=1.0,
        theta=-.30
    )

    robot = OmniRobot(state)

    controller = PController(
        kp_xy=1.2,
        kp_theta=0.8
    )

    goal = RobotState(
        x=5.0,
        y=5.0,
        theta=0.0
    )

    x_hist = []
    y_hist = []

    dt = 0.1

    for _ in range(100):

        vx, vy, omega = controller.compute(
            state,
            goal
        )

        robot.step(
            vx=vx,
            vy=vy,
            omega=omega,
            dt=dt
        )

        x_hist.append(state.x)
        y_hist.append(state.y)

    plt.plot(x_hist, y_hist)
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    run_simulation()