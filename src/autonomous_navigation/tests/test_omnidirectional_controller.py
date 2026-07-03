from autonomous_navigation.core.controllers.mpc.base import MPCController
from autonomous_navigation.core.controllers.mpc.omnidirectional import OmniMPC
import numpy as np

controller = OmniMPC(
    dt=0.1,
    horizon=10,
    Q=np.eye(2),
    R=np.eye(2),
    v_max=1.0,
)

x = np.array([0.0, 0.0])
u = np.array([1.0, 0.5])

print(controller.predict(x, u))