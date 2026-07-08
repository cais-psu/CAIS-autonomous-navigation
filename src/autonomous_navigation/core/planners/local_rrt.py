from autonomous_navigation.core.environment.local_map import LocalMap
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_planner import RRTPlanner
from autonomous_navigation.core.planners.rrt_node import RRTNode
from typing import Tuple


class LocalRRTPlanner:
    """
    Local RRT planner for limited-perception exploration.

    Creates a local occupancy map based on robot sensing radius
    and performs RRT planning only within the perceived region.
    """

    def __init__(
        self,
        global_map: OccupancyGrid,
        sensing_radius: float,
        step_size: float = 5,
        sampler=None,
    ):
        """
        Parameters
        ----------
        global_map : OccupancyGrid
            Ground truth environment.

        sensing_radius : float
            Robot perception radius.

        step_size : float
            RRT extension distance.

        sampler :
            Optional sampling strategy (HEDACSampler, etc.)
        """

        self.global_map = global_map

        self.sensing_radius = sensing_radius

        self.step_size = step_size

        self.sampler = sampler


    def get_local_map(
        self,
        robot_x: float,
        robot_y: float,
    ) -> tuple[OccupancyGrid, tuple]:
        """
        Generate the locally perceived occupancy map
        and local sampling bounds.
        """

        local_map = LocalMap(
            self.global_map,
            self.sensing_radius,
        )

        local_grid = local_map.get_local_grid(
            robot_x,
            robot_y,
        )

        bounds = local_map.get_bounds(
            robot_x,
            robot_y,
        )

        return local_grid, bounds


    def plan(
        self,
        start: RRTNode,
        goal: RRTNode,
        max_iters: int = 5000,
    ):
        """
        Plan a path using only locally sensed obstacles.

        Parameters
        ----------
        start : RRTNode
            Current robot position.

        goal : RRTNode
            Local target location.

        max_iters : int
            Maximum RRT iterations.

        Returns
        -------
        path, stats
        """

        local_grid, bounds = self.get_local_map(
            start.x,
            start.y,
        )


        planner = RRTPlanner(
            local_grid,
            step_size=self.step_size,
            sampler=self.sampler,
        )


        return planner.plan(
            start,
            goal,
            max_iters,
            sampling_bounds=bounds
        )