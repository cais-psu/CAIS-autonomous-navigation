from autonomous_navigation.core.robots.robot_state import RobotState
from autonomous_navigation.core.robots.omni_robot import OmniRobot


def test_forward_motion():

    state = RobotState(
        x=0.0,
        y=0.0,
        theta=0.0
    )

    robot = OmniRobot(state)

    robot.step(
        vx=1.0,
        vy=0.0,
        omega=0.0,
        dt=0.1
    )

    assert state.x == 0.1
    assert state.y == 0.0