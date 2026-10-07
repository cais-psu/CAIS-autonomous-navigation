from dataclasses import dataclass


@dataclass
class RobotState:
    """
    Define the state variables a robot's position and velocity.
    """
    x: float
    y: float
    theta: float

    v: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0