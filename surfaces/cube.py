from typing import List

import numpy as np

from ..utils import get_closest_surface
from .base_surface import Material, Surface


class Cube(Surface):
    def __init__(self, position: List[float], scale: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.scale = scale

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
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
                    return None
            else:
                t1 = (self.position[i] - source[i]) / ray_vec[i]
                t2 = (self.position[i] + self.scale - source[i]) / ray_vec[i]
                if t1 > t2:
                    t1, t2 = t2, t1

                t_near = max(t_near, t1)
                t_far = min(t_far, t2)

                if t_near > t_far or t_far < 0:
                    return None

        t = t_near if t_near >= 0 else t_far

        # calculate the intersection point
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
        _, light_intersection = get_closest_surface(
            light_source, light_vec, surfaces, None
        )
        if np.allclose(intersection, light_intersection, atol=1e-5):
            return True
        return False

    def get_normal(self, intersection: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
