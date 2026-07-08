import numpy as np

from autonomous_navigation.core.exploration.heat_map import HeatMap


class HEDACSampler:
    """
    Heat Equation Driven Area Coverage (HEDAC) sampler.

    Selects exploration samples according to a heat distribution.

    Low temperature regions are favored:
    
        w(x,y) = exp(-gamma * T(x,y))

    where:
        T(x,y) = heat value
        gamma = sampling bias strength
    """

    def __init__(
        self,
        heat_map: HeatMap,
        gamma: float = 2.0,
    ):
        """
        Parameters
        ----------
        heat_map : HeatMap
            Current exploration heat field.

        gamma : float
            Strength of HEDAC sampling bias.
            Larger values favor low-temperature regions more strongly.
        """

        self.heat_map = heat_map
        self.gamma = gamma


    def select_sample(
        self,
        candidates: np.ndarray,
    ) -> tuple[int, int]:
        """
        Select a candidate point using HEDAC weighting.

        Parameters
        ----------
        candidates : np.ndarray
            Candidate points with shape (N,2):

            [
                [x1,y1],
                [x2,y2],
                ...
            ]

        Returns
        -------
        tuple[int,int]
            Selected (x,y) sample.
        """

        if len(candidates) == 0:
            raise ValueError(
                "Candidate list cannot be empty."
            )


        temperatures = np.array(
            [
                self.heat_map.get_temperature(
                    int(x),
                    int(y),
                )
                for x, y in candidates
            ]
        )


        # HEDAC probability weighting
        weights = np.exp(
            -self.gamma * temperatures
        )


        # Normalize
        total_weight = np.sum(weights)

        if total_weight == 0:
            probabilities = np.ones(
                len(candidates)
            ) / len(candidates)

        else:
            probabilities = (
                weights / total_weight
            )


        index = np.random.choice(
            len(candidates),
            p=probabilities,
        )


        selected = candidates[index]


        return (
            int(selected[0]),
            int(selected[1]),
        )