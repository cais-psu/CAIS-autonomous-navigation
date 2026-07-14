import numpy as np

from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.exploration.local_target_selection import (
    LocalTargetSelector,
)
from autonomous_navigation.core.planners.local_rrt import LocalRRTPlanner
from autonomous_navigation.core.planners.path_utils import interpolate_path
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.exploration.exploration_result import (
    ExplorationResult,
)


class ExplorationManager:
    """
    High-level exploration/navigation pipeline.

    Pipeline:

        Robot Pose
            ↓
        Heat Update
            ↓
        Local Map Extraction
            ↓
        Target Selection
            ↓
        Local RRT
            ↓
        Path Interpolation
            ↓
        MPC Reference

    Exploration mode:
        Selects a low-temperature region.

    Goal mode:
        Plans toward a provided mission goal.

    This class does not perform control.
    It provides local trajectories for MPC.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        heat_map: HeatMap,
        sensing_radius: float,
        planner_step_size: float = 2.0,
        interpolation_spacing: float = 0.5,
        sampler=None
    ):

        self.last_result = None

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

        self.iteration = 0

    def plan(
        self,
        robot_position,
        mission_goal: RRTNode | None = None,
        max_rrt_iterations: int = 5000,
    ) -> ExplorationResult:

        self.iteration += 1

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])
        
        #
        # Update exploration heat only during exploration
        #
        if mission_goal is None:

            self.heat_map.mark_visited(
                int(round(robot_x)),
                int(round(robot_y)),
            )

            self.heat_map.step()


        #
        # Generate local map
        #
        local_grid, bounds = self.local_rrt.get_local_map(
            robot_x,
            robot_y,
        )


        #
        # Select target
        #
        if mission_goal is None:

            target = self.target_selector.select_target(
                robot_x=robot_x,
                robot_y=robot_y,
                occupancy_grid=local_grid,
                planner_step_size=self.local_rrt.step_size,
                heat_map=self.heat_map,
            )

            mode = "exploration"

        else:

            target = mission_goal
            mode = "goal"

            dx = mission_goal.x - robot_x
            dy = mission_goal.y - robot_y

            distance = np.sqrt(dx**2 + dy**2)

            print(
                f"Goal Distance: {distance:.2f}, "
                f"Sensing Radius: {self.sensing_radius}"
            )

            if distance <= self.sensing_radius:

                target = mission_goal
                mode = "goal"

            else:

                target = self.target_selector.select_target(
                    robot_x=robot_x,
                    robot_y=robot_y,
                    occupancy_grid=local_grid,
                    planner_step_size=self.local_rrt.step_size,
                    heat_map=self.heat_map,
                )

                mode = "exploration"

        #
        # Local RRT
        #
        start = RRTNode(
            robot_x,
            robot_y,
        )
        print(f"Start : ({start.x:.1f}, {start.y:.1f})")
        print(f"Goal  : ({target.x:.1f}, {target.y:.1f})")
        print(f"Bounds: {bounds}")
        
        path, stats = self.local_rrt.plan(
            start,
            target,
            max_rrt_iterations,
        )

        if stats is None:
            stats = {}

        stats["mode"] = mode

        stats["iteration"] = self.iteration
        stats["success"] = False

        if path is None:

            result = ExplorationResult(
                path=None,
                target=target,
                stats=stats,
                local_grid=local_grid,
                rrt=self.local_rrt.rrt,
                bounds=bounds,
            )

            self.last_result = result

            return result


        #
        # Interpolate for MPC
        #
        path = interpolate_path(
            path,
            spacing=self.interpolation_spacing,
        )


        stats["success"] = True

        result = ExplorationResult(
            path=path,
            target=target,
            stats=stats,
            local_grid=local_grid,
            rrt=self.local_rrt.rrt,
            bounds=bounds,
        )

        self.last_result = result

        return result