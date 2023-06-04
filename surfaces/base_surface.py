from typing import Tuple

import numpy as np

from .material import Material


class Surface:
    def __init__(self, material: Material):
        self.material = material

    def intersect(
        self, source: np.ndarray, ray_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        raise NotImplementedError()

    def reflection(self, ray_vec: np.ndarray, intersection: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
