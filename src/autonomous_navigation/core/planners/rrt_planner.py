import math
import random
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid



class RRTPlanner:

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        step_size=5
    ):
        self.grid = occupancy_grid
        self.step_size = step_size
        
        self.nodes = []
    
    def distance(
            self,
            node1: RRTNode,
            node2: RRTNode
    ):
        return math.sqrt(
            (node1.x - node2.x)**2
            +
            (node1.y - node2.y)**2
        )
    
    def sample_free(self):

        x = random.uniform(
            0,
            self.grid.width
        )

        y = random.uniform(
            0,
            self.grid.height
        )

        return RRTNode(x, y)
        
    def nearest_node(
            self,
            sample: RRTNode,
            nodes
    ):
        nearest_index = 0
        nearest_distance = float("inf")

        for i, node in enumerate(nodes):
            d = self.distance(
                node,
                sample
            )

            if d < nearest_distance:

                nearest_distance = d
                nearest_index = i

        return nearest_index
    
    def steer(
            self,
            nearest: RRTNode,
            sample: RRTNode
    ):
        dx = sample.x - nearest.x
        dy = sample.y - nearest.y

        distance = self.distance(nearest, sample)

        if distance == 0:
            return nearest  # or copy

        ux = dx / distance 
        uy = dy / distance

        new_x = nearest.x + self.step_size * ux
        new_y = nearest.y + self.step_size * uy

        return RRTNode(new_x, new_y)
    
    def reconstruct_path(self, goal_index):
        path = []
        idx = goal_index

        while idx is not None:
            node = self.nodes[idx]
            path.append((node.x, node.y))
            idx = node.parent
        return path[::-1]
    
    def plan(self, start: RRTNode, goal: RRTNode, max_iters: int = 1000):

        self.nodes = [start]

        for _ in range(max_iters):

            sample = self.sample_free()

            nearest_idx = self.nearest_node(sample, self.nodes)
            nearest = self.nodes[nearest_idx]

            new_node = self.steer(nearest, sample)

            if self.grid.line_is_free(
                int(nearest.x), int(nearest.y),
                int(new_node.x), int(new_node.y)
            ):
                new_node.parent = nearest_idx
                self.nodes.append(new_node)

                #goal check (simple radius)

                if (
                    self.distance(new_node, goal) < self.step_size
                    and
                    self.grid.line_is_free(
                        int(new_node.x),
                        int(new_node.y),
                        int(goal.x),
                        int(goal.y)
                    )
                ):
                    goal.parent = len(self.nodes) - 1
                    self.nodes.append(goal)
                    return self.reconstruct_path(len(self.nodes) - 1)
                
        return None