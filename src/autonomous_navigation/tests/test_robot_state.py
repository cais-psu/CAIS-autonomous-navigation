from autonomous_navigation.core.robots.robot_state import RobotState


def test_robot_state_creation():

    state = RobotState(
        x=1.0,
        y=2.0,
        theta=0.5
    )

    assert state.x == 1.0
    assert state.y == 2.0
    assert state.theta == 0.5