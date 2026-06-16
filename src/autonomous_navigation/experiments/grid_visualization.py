import matplotlib.pyplot as plt

from autonomous_navigation.core.environment.occupancy_grid import OccupancyGrid


grid = OccupancyGrid(50, 50)

grid.add_rectangle(10, 10, 15, 40)
grid.add_rectangle(30, 5, 35, 25)
grid.add_rectangle(40, 30, 45, 45)

plt.imshow(
    grid.grid,
    origin="lower",
    cmap="gray_r"
)

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()