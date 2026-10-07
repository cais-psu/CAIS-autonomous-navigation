from autonomous_navigation.core.robots.robot_state import RobotState
import numpy as np

class PController:

    def __init__(self, kp_xy: float = 1.0, kp_theta: float = 1.0):
        self.kp_xy = kp_xy
        self.kp_theta = kp_theta

    def compute(self, state: RobotState, goal: RobotState):

        ex = goal.x - state.x
        ey = goal.y - state.y
        et = goal.theta - state.theta

        vx = self.kp_xy * ex
        vy = self.kp_xy * ey
        omega = self.kp_theta * et

        return vx, vy, omega