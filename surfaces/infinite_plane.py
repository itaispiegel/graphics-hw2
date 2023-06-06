from typing import List

import numpy as np

from ..utils import get_closest_surface
from .base_surface import Material, Surface


class InfinitePlane(Surface):
    def __init__(self, normal: List[float], offset: float, material: Material):
        super().__init__(material)
        self.normal = np.array(normal)
        self.offset = offset

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        denom = np.dot(self.normal, ray_vec)
        if abs(denom) < 1e-6:
            return None

        # Ray and plane are not parallel, so an intersection exists
        t = (self.offset - np.dot(source, self.normal)) / denom
        return source + t * ray_vec

    def reflection(self, ray_vec: np.ndarray, intersection: np.ndarray) -> np.ndarray:
        normal = self.get_normal(intersection)

        # Calculate the reflection vector
        reflection_vec = ray_vec - 2 * (ray_vec @ normal) * normal
        return reflection_vec / np.linalg.norm(reflection_vec)

    def light_hit(
        self,
        light_source: np.ndarray,
        intersection: np.ndarray,
        surfaces: List[Surface],
    ) -> bool:
        light_vec = intersection - light_source
        closest_surface, _ = get_closest_surface(
            light_source, light_vec, surfaces, None
        )
        if closest_surface == self:
            return True
        return False

    def get_normal(self, intersection: np.ndarray) -> np.ndarray:
        return self.normal / np.linalg.norm(self.normal)
