from typing import Tuple

import numpy as np


class Surface:
    def __init__(self, material_index: int):
        self.material_index = material_index

    def intersect(
        self, source: np.ndarray, ray_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        raise NotImplementedError()

    def reflection(
        self, ray_vec: np.ndarray, distance: float, intersection: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()
