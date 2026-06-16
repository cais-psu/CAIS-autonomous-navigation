from autonomous_navigation.core.robots.robot_state import RobotState

class OmniRobot:

    """
    Velocity integrator for an omnidirectional robot model.
    """
    
    def __init__(self, state: RobotState):
        self.state = state

    def step(
        self,
        vx: float,
        vy: float,
        omega: float,
        dt: float
    ):

        self.state.x += vx* dt
        self.state.y += vy* dt
        self.state.theta += omega* dt

        self.state.vx = vx
        self.state.vy = vy
        self.state.omega = omega
