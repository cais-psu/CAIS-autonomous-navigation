import numpy as np
import random

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.planners.rrt_node import RRTNode
from autonomous_navigation.core.controllers.mpc.omnidirectional import OmniMPC
from autonomous_navigation.core.exploration.heat_map import HeatMap
from autonomous_navigation.core.exploration.exploration_manager import ExplorationManager
from autonomous_navigation.core.planners.corridor_generator import CorridorGenerator
from autonomous_navigation.core.exploration.hedac_sampler import HEDACSampler

grid = OccupancyGrid.random_environment(
    width=100,
    height=100,
    n_circles=10,
    n_rectangles=10,
    seed=6,
)

robot_radius = 1
safety_margin = 1

inflation_radius = robot_radius + safety_margin

cspace = grid.create_configuration_space(
    inflation_radius
)

heat = HeatMap(
    width=100,
    height=100,
)

hedac_sampler = HEDACSampler(
    heat_map=heat,
    gamma=10.0,
)

manager = ExplorationManager(
    occupancy_grid=cspace,
    heat_map=heat,
    sensing_radius=20,
    planner_step_size=2,
    sampler=hedac_sampler,
)

mpc = OmniMPC(
    dt=0.1,
    horizon=20,
    Q=100*np.eye(2),
    R=0.1*np.eye(2),
    v_max=1.0,
)

start = np.array(
    [10.0,10.0]
)

global_goal = RRTNode(
    80,
    80,
)

robot = start.copy()

trajectory = [
    robot.copy()
]

planning_history = []
target_history = []
mode_history = []
heat_history = []