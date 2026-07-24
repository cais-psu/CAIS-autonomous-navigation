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
from autonomous_navigation.core.planners.corridor_generator import CorridorGenerator


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
        Convex Corridor Generation
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

        self.current_result = None

        self.current_target = None
        self.current_path = None
        self.current_corridors = None

        self.current_waypoint_index = 0
        self.current_corridor_index = 0

        self.mode = "exploration"

        self.grid = occupancy_grid
        self.heat_map = heat_map

        self.sensing_radius = sensing_radius
        self.interpolation_spacing = interpolation_spacing

        self.target_selector = LocalTargetSelector(
            sensing_radius=sensing_radius
        )

        self.corridor_generator = None

        self.local_rrt = LocalRRTPlanner(
            global_map=occupancy_grid,
            sensing_radius=sensing_radius,
            step_size=planner_step_size,
            sampler=sampler,
        )

        self.iteration = 0
    """
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

        corridor_generator = CorridorGenerator(
            local_grid,
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
                corridors=None,
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

        corridors = corridor_generator.generate(
            path
        )

        stats["num_corridors"] = len(corridors)

        result = ExplorationResult(
            path=path,
            corridors=corridors,
            target=target,
            stats=stats,
            local_grid=local_grid,
            rrt=self.local_rrt.rrt,
            bounds=bounds,
        )

        self.last_result = result

        stats["success"] = True

        return result
    """
    def plan_local_mission(
        self,
        robot_position,
        local_target: RRTNode,
        max_rrt_iterations: int = 5000,
    ) -> ExplorationResult:

        self.iteration += 1

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])

        #
        # Generate current local map
        #
        local_grid, bounds = self.local_rrt.get_local_map(
            robot_x,
            robot_y,
        )

        self.current_local_grid = local_grid
        self.current_bounds = bounds

        corridor_generator = CorridorGenerator(
            local_grid,
        )

        #
        # Plan from current robot position to the
        # previously-selected local target.
        #
        start = RRTNode(
            robot_x,
            robot_y,
        )

        print(f"Start : ({start.x:.1f}, {start.y:.1f})")
        print(f"Target: ({local_target.x:.1f}, {local_target.y:.1f})")

        path, stats = self.local_rrt.plan(
            start,
            local_target,
            max_rrt_iterations,
        )

        if stats is None:
            stats = {}

        stats["mode"] = "local_mission"
        stats["iteration"] = self.iteration
        stats["success"] = path is not None

        if path is None:

            result = ExplorationResult(
                path=None,
                corridors=None,
                target=local_target,
                stats=stats,
                local_grid=local_grid,
                rrt=self.local_rrt.rrt,
                bounds=bounds,
            )

            self.last_result = result
            return result

        #
        # Interpolate path
        #
        path = interpolate_path(
            path,
            spacing=self.interpolation_spacing,
        )

        #
        # Generate corridors
        #
        corridors = corridor_generator.generate(
            path,
        )

        if corridors is None:

            print(
                "Corridor generation failed"
            )

            result = ExplorationResult(
                path=path,
                corridors=None,
                target=local_target,
                stats=stats,
                local_grid=local_grid,
                rrt=self.local_rrt.rrt,
                bounds=bounds,
            )

            self.last_result = result

            return result

        stats["num_corridors"] = len(corridors)

        result = ExplorationResult(
            path=path,
            corridors=corridors,
            target=local_target,
            stats=stats,
            local_grid=local_grid,
            rrt=self.local_rrt.rrt,
            bounds=bounds,
        )

        self.last_result = result

        return result
    
    def select_local_target(
        self,
        robot_position,
    ) -> RRTNode:

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])

        local_grid, _ = self.local_rrt.get_local_map(
            robot_x,
            robot_y,
        )

        return self.target_selector.select_target(
            robot_x=robot_x,
            robot_y=robot_y,
            occupancy_grid=local_grid,
            planner_step_size=self.local_rrt.step_size,
            heat_map=self.heat_map,
        )
    
    def begin_exploration_mission(
        self,
        robot_position,
    ):

        self.current_target = self.select_local_target(robot_position)

        result = self.plan_local_mission(
            robot_position,
            self.current_target,
        )

        if result.path is None:
            return None

        self.current_rrt = result.rrt
        self.current_bounds = result.bounds

        self.current_result = result
        self.current_path = result.path
        self.current_corridors = result.corridors

        self.current_waypoint_index = 1
        self.current_corridor_index = 0

        return result
    
    def get_current_waypoint(self):
        
        return self.current_path[self.current_waypoint_index]
    
    def get_current_corridor(self):

        return self.current_corridors[self.current_corridor_index]
    
    def get_current_target(self):

        return self.current_target
    
    def advance_waypoint(self):

        if self.current_waypoint_index < len(self.current_path) - 1:
            self.current_waypoint_index += 1

        if self.current_corridors is not None:
            if self.current_corridor_index < len(self.current_corridors) - 1:
                self.current_corridor_index += 1

    def mission_complete(self):

        if self.current_path is None:
            return True
        
        return self.current_waypoint_index >= len(self.current_path) - 1
        
    def clear_local_mission(self):

        self.current_result = None

        self.current_target = None
        self.current_path = None
        self.current_corridors = None

        self.current_waypoint_index = 0
        self.current_corridor_index = 0

        self.current_rrt = None
        self.current_bounds = None

    def waypoint_reached(
        self,
        robot_position,
        tolerance=0.5,
    ):

        waypoint = self.get_current_waypoint()

        return (
            np.linalg.norm(
                np.asarray(robot_position) - np.asarray(waypoint)
            )
            < tolerance
        )
    
    def has_local_mission(self):

        return self.current_target is not None
    
    def begin_navigation_mission(
        self,
        robot_position,
        global_goal: RRTNode,
        max_rrt_iterations=5000,
    ):

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])

        distance = np.linalg.norm(
            np.array([
                global_goal.x - robot_x,
                global_goal.y - robot_y
            ])
        )


        #
        # Goal is visible
        #
        if distance <= self.sensing_radius:

            self.current_target = global_goal

            result = self.plan_local_mission(
                robot_position,
                global_goal,
                max_rrt_iterations,
            )

            self.mode = "goal"


        #
        # Continue exploration
        #
        else:

            self.current_target = self.select_local_target(
                robot_position
            )

            result = self.plan_local_mission(
                robot_position,
                self.current_target,
                max_rrt_iterations,
            )

            self.mode = "exploration"


        if result.path is None:
            self.clear_local_mission()
            print("RRT could not find path")
            return None


        self.current_result = result
        self.current_path = result.path
        self.current_corridors = result.corridors

        self.current_rrt = result.rrt
        self.current_bounds = result.bounds

        self.current_waypoint_index = 1
        self.current_corridor_index = 0

        return result
    
    def update_heat(
        self,
        robot_position,
    ):

        self.heat_map.mark_visited(
            int(round(robot_position[0])),
            int(round(robot_position[1])),
        )

        self.heat_map.step()

    def complete_current_mission(
        self,
        robot_position,
    ):

        if self.mode == "exploration":

            self.update_heat(
                robot_position
            )

        self.clear_local_mission()