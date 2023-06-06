from typing import List

import numpy as np

from ..utils import get_closest_surface
from .base_surface import Material, Surface


class Sphere(Surface):
    def __init__(self, position: List[int], radius: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.radius = radius

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Return the intersection point and distance from the source to the intersection point
        using the algebraic method shown in class.
        """
        oc = source - self.position
        a = ray_vec @ ray_vec
        b = 2.0 * oc @ ray_vec
        c = oc @ oc - self.radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        if t1 < 0 and t2 < 0:
            return None
        t = t1 if t1 >= 0 else t2

        # calculate the intersection point and the distance from the source to the intersection point
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
        normal = intersection - self.position
        return normal / np.linalg.norm(normal)
