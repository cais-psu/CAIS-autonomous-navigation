import numpy as np


def interpolate_path(path, spacing=0.5):

    dense_path = []

    for i in range(len(path)-1):

        p1 = np.asarray(path[i])
        p2 = np.asarray(path[i+1])

        distance = np.linalg.norm(p2-p1)

        steps = max(int(distance/spacing),1)

        for j in range(steps):
            alpha = j/steps
            point = (1-alpha)*p1 + alpha*p2
            dense_path.append(point)

    dense_path.append(np.asarray(path[-1]))

    return dense_path