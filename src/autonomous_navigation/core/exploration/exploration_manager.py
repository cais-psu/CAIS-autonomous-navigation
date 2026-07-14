from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.exploration.local_target_selection import (
    LocalTargetSelector,
)
from autonomous_navigation.core.planners.local_rrt import LocalRRTPlanner
from autonomous_navigation.core.planners.path_utils import interpolate_path
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.exploration.exploration_result import ExplorationResult


class ExplorationManager:
    """
    High-level exploration pipeline.

    This class coordinates

        Robot pose
             ↓
        Heat map update
             ↓
        Local map generation
             ↓
        Local target selection
             ↓
        Local RRT planning
             ↓
        Path interpolation
             ↓
        MPC

    It does not perform control itself—it simply supplies a local
    reference trajectory every planning cycle.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        heat_map: HeatMap,
        sensing_radius: float,
        planner_step_size: float = 2.0,
        interpolation_spacing: float = 0.5,
        sampler=None,
    ):
        self.grid = occupancy_grid
        self.heat_map = heat_map

        self.sensing_radius = sensing_radius
        self.interpolation_spacing = interpolation_spacing

        self.target_selector = LocalTargetSelector(
            sensing_radius=sensing_radius
        )

        self.local_rrt = LocalRRTPlanner(
            global_map=occupancy_grid,
            sensing_radius=sensing_radius,
            step_size=planner_step_size,
            sampler=sampler,
        )

    def plan(
        self,
        robot_position,
        max_rrt_iterations: int = 5000,
    ):
        """
        Perform one exploration planning cycle.

        Parameters
        ----------
        robot_position : ndarray-like
            Current robot position [x, y].

        max_rrt_iterations : int
            Maximum RRT iterations.

        Returns
        -------
        path : list[(x,y)]
            Interpolated path.

        stats : dict
            Planner statistics.

        target : RRTNode
            Selected exploration target.
        """

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])

        #
        # Update heat field
        #
        self.heat_map.mark_visited(
            robot_x,
            robot_y,
        )

        self.heat_map.step()

        #
        # Local occupancy map
        #
        local_grid, _ = self.local_rrt.get_local_map(
            robot_x,
            robot_y,
        )

        #
        # Select exploration target
        #
        target = self.target_selector.select_target(
            robot_x=robot_x,
            robot_y=robot_y,
            occupancy_grid=local_grid,
            planner_step_size=self.local_rrt.step_size,
            heat_map=self.heat_map,
        )

        #
        # Local RRT
        #
        start = RRTNode(robot_x, robot_y)

        path, stats = self.local_rrt.plan(
            start,
            target,
            max_rrt_iterations,
        )

        if path is None:
            return ExplorationResult(
                path=None,
                target=target,
                stats=stats,
                local_grid=local_grid,
                rrt=self.local_rrt.rrt
            )

        #
        # Interpolate path for MPC
        #
        path = interpolate_path(
            path,
            spacing=self.interpolation_spacing,
        )

        return ExplorationResult(
            path=path,
            target=target,
            stats=stats,
            local_grid=local_grid,
            rrt=self.local_rrt.rrt
        )