from typing import List, Tuple

import numpy as np

from . import Surface
from .material import Material


class Surface:
    def __init__(self, material: Material):
        self.material = material

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        raise NotImplementedError()

    def reflection(self, ray_vec: np.ndarray, intersection: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def light_hit(
        self,
        light_source: np.ndarray,
        intersection_point: np.ndarray,
        surfaces: List[Surface],
    ) -> bool:
        raise NotImplementedError()

    def get_normal(self, intersection: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
