from typing import List, Tuple

import numpy as np

from .base_surface import Material, Surface


class Cube(Surface):
    def __init__(self, position: List[float], scale: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.scale = scale

    def intersect(
        self, source: np.ndarray, ray_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return the intersection point and distance from the source to the
        intersection point, using the slab method.
        """
        t_near = float("-inf")
        t_far = float("inf")

        for i in range(3):
            if ray_vec[i] == 0:
                if (
                    source[i] < self.position[i]
                    or source[i] > self.position[i] + self.scale
                ):
                    return None, None
            else:
                t1 = (self.position[i] - source[i]) / ray_vec[i]
                t2 = (self.position[i] + self.scale - source[i]) / ray_vec[i]
                if t1 > t2:
                    t1, t2 = t2, t1

                t_near = max(t_near, t1)
                t_far = min(t_far, t2)

                if t_near > t_far or t_far < 0:
                    return None, None

        t = t_near if t_near >= 0 else t_far

        # calculate the intersection point and the distance from the source to the intersection point
        intersection = source + t * ray_vec
        distance = np.linalg.norm(intersection - source)
        return intersection, distance

    def reflection(
        self, ray_vec: np.ndarray, distance: float, intersection: np.ndarray
    ) -> np.ndarray:
        # Calculate the reflection vector
        reflection_ray_vec = np.zeros(3)
        for i in range(3):
            if abs(intersection[i] - self.position[i]) < 1e-10:
                reflection_ray_vec[i] = -ray_vec[i]
            elif abs(intersection[i] - (self.position[i] + self.scale)) < 1e-10:
                reflection_ray_vec[i] = ray_vec[i]
            else:
                reflection_ray_vec[i] = 0

        return reflection_ray_vec