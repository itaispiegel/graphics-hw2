from typing import List

import numpy as np

from .base_surface import Surface


class InfinitePlane(Surface):
    def __init__(self, normal: List[float], offset: float, material_index: int):
        super().__init__(material_index)
        self.normal = np.array(normal)
        self.offset = offset

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray):
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        t = np.dot(self.normal, source - (self.normal * self.offset)) / np.dot(
            self.normal, ray_vec
        )

        # Check if intersection is behind the source (according to the ray's direction)
        # or if the ray is parallel to the plane
        if t < 0:
            return None, None

        # calculate the intersection point and the distance from the source to the intersection point
        intersection = source + t * ray_vec
        distance = np.linalg.norm(intersection - source)

        # Calculate the reflection vector
        reflection_ray_vec = ray_vec - 2 * np.dot(ray_vec, self.normal) * self.normal
        return intersection, distance, reflection_ray_vec
