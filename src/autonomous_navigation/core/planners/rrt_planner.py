import math
import random
from autonomous_navigation.core.planners.rrt_node import RRTNode


class RRTPlanner:

    def __init__(
        self,
        occupancy_grid,
        step_size=5
    ):
        self.grid = occupancy_grid
        self.step_size = step_size
    
    def distance(
            self,
            node1,
            node2
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
            sample,
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