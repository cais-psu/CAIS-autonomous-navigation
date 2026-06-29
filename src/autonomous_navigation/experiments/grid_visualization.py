import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid
from autonomous_navigation.core.environment.obstacles import RectangleObstacle


grid = OccupancyGrid(50, 50)

rect_1 = RectangleObstacle(10, 10, 15, 40)
rect_2 = RectangleObstacle(30, 5, 35, 25)
rect_3 = RectangleObstacle(40, 30, 45, 45)

grid.add_rectangle_obstacle(rect_1)
grid.add_rectangle_obstacle(rect_2)
grid.add_rectangle_obstacle(rect_3)

plt.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r"
)

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()