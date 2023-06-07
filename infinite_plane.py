from typing import List

import numpy as np

from base_surface import Material, Surface
from consts import EPSILON


class InfinitePlane(Surface):
    def __init__(self, normal: List[float], offset: float, material: Material):
        super().__init__(material)
        self.normal = np.array(normal) / np.linalg.norm(normal)
        self.offset = offset

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        denom = self.normal @ ray_vec

        # Check if ray and plane are parallel
        if abs(denom) < EPSILON:
            return None

        # Ray and plane are not parallel, so an intersection exists
        t = (self.offset - np.dot(source, self.normal)) / denom
        return source + t * ray_vec

    def normal_at_point(self, point: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        dot = self.normal @ ray_vec
        return self.normal if dot > 0 else -self.normal
