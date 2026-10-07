import math

class PathFollower: 

    def __init__(self, kp=1.0, goal_tol=0.5):
        self.kp = kp
        self.goal_tol = goal_tol
        self.current_idx = 0

    def compute(self, state, path):
        
        if self.current_idx >= len(path):
            return 0.0, 0.0, 0.0
        
        target_x, target_y = path[self.current_idx]

        dx = target_x - state.x
        dy = target_y - state.y

        dist = math.sqrt(dx**2 + dy**2)

        if dist < self.goal_tol:
            self.current_idx += 1
            return 0.0, 0.0, 0.0
        
        vx = self.kp * dx
        vy = self.kp * dy
        omega = 0.0

        return vx, vy, omega
    