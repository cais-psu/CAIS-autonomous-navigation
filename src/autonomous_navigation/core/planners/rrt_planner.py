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
        while True:
            x = random.uniform(
                0,
                self.grid.width - 1
            )

            y = random.uniform(
                0,
                self.grid.height - 1 
            )
            if self.grid.is_free(
                int(x),
                int(y)
            ):
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
        """
        new_x = max(
            0,
            min(new_x, self.grid.width - 1)
        )

        new_y = max(
            0,
            min(new_y, self.grid.height - 1)
        )
        """
        return RRTNode(new_x, new_y)
    
    def reconstruct_path(self, goal_index):
        path = []
        idx = goal_index

        while idx is not None:
            node = self.nodes[idx]
            path.append((node.x, node.y))
            idx = node.parent
        return path[::-1]
    
    def compute_path_length(self, path):
        total = 0.0

        for i in range(len(path) - 1):

            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]

            total += math.sqrt(dx**2 + dy**2)

        return total
    
    def plan(self, start: RRTNode, goal: RRTNode, max_iters: int = 1000):

        self.nodes = [start]

        for iteration in range(max_iters):

            sample = self.sample_free()

            nearest_idx = self.nearest_node(sample, self.nodes)
            nearest = self.nodes[nearest_idx]

            new_node = self.steer(nearest, sample)
            """
            if (
                new_node.x < 0
                or new_node.x >= self.grid.width
                or
                new_node.y < 0
                or new_node.y >= self.grid.height
            ):
                print(
                    f"OUT OF BOUNDS NODE: "
                    f"({new_node.x:.2f}, {new_node.y:.2f})"
                )
            """
            if not self.grid.in_bounds(
                int(new_node.x),
                int(new_node.y)
            ):
                continue

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

                    path = self.reconstruct_path(len(self.nodes) - 1)

                    straight_line = self.distance(start, goal)

                    stats = {
                        "success": True,
                        "iterations": iteration + 1,
                        "nodes": len(self.nodes),
                        "path_length": self.compute_path_length(path),
                        "path_efficiency": straight_line/self.compute_path_length(path)
                    }

                    return path, stats

        stats = {
            "success": False,
            "iterations": max_iters,
            "nodes": len(self.nodes),
            "path_length": None

        }

        return None, stats
    
