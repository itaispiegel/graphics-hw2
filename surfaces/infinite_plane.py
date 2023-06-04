from typing import List, Tuple

import numpy as np

from .base_surface import Material, Surface


class InfinitePlane(Surface):
    def __init__(self, normal: List[float], offset: float, material: Material):
        super().__init__(material)
        self.normal = np.array(normal)
        self.offset = offset

    def intersect(
        self, source: np.ndarray, ray_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        t = -(source @ self.normal + self.offset) / (ray_vec @ self.normal)

        # Check if intersection is behind the source (according to the ray's direction)
        # or if the ray is parallel to the plane
        if t < 0:
            return None, None

        # calculate the intersection point and the distance from the source to the intersection point
        intersection = source + t * ray_vec
        distance = np.linalg.norm(intersection - source)
        return intersection, distance

    def reflection(self, ray_vec: np.ndarray, intersection: np.ndarray) -> np.ndarray:
        # Calculate the reflection vector
        reflection_ray_vec = ray_vec - 2 * ray_vec @ self.normal * self.normal
        return reflection_ray_vec
