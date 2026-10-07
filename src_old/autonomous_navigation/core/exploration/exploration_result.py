from dataclasses import dataclass

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode


@dataclass
class ExplorationResult:
    path: list | None
    corridors: list | None
    target: RRTNode
    stats: dict
    local_grid: OccupancyGrid
    rrt: RRTPlanner
    bounds: tuple