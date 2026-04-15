# Environment Parameters
gridSizeX = 20
gridSizeY = 10
numShapes = 20

cellsPerRow = math.ceil(math.sqrt(numShapes))
cellSizeX = gridSizeX / cellsPerRow
cellSizeY = gridSizeY / cellsPerRow
shapeVertices = 6
minScale = 0.3; maxScale = 0.9
bufferDist = 0.5

senseRadius = 5

max_global_iters = 600

start = [0,0]
goal = [np.random.rand()*gridSizeX, np.random.rand()*gridSizeY]

# Control Parameters
dt_ctrl = 0.5
N_mpc = 10
rho = 0.01
Q = np.diag([20,20,5])
R = rho * np.eye(2)
u_min = np.array([0, -100])
u_max = np.array([1, 100])
tol_waypoint = 0.5
lambda_s = 1000
nu = 2
nx = 3

# Tolerancing Parameters
tol_pos = 0.25
tol_theta = 0.2
move_threshold = 1e-3
max_mpc_steps = 80
tol_goal = 0.25

# Heat Map Parameters
heatRes = 0.5               # spatial resolution of heat grid (m)
xHeat = np.arange(0, gridSizeX + heatRes, heatRes)
yHeat = np.arange(0, gridSizeY + heatRes, heatRes)
HX, HY = np.meshgrid(xHeat, yHeat);  # HX: columns = x positions, HY: rows = y positions
heatT = np.zeros_like(HX)     # temperature field
diffusion_alpha = 1.6       # diffusion coefficient (tunable; larger -> faster spreading)
dt_heat = 0.02              # time step for heat integration (s)
heat_source_strength = 6.0  # temperature increase per second while heating (continuous)
heat_cap = 300              # cap maximum temperature
bias_beta = 1.8             # sampling bias factor
max_rrt_bias_trials = 200   # not used in new approach, kept for compatibility
heat_avoid_threshold = 4.0  # temperatures above this are treated as 'too hot' for MPC to step into

