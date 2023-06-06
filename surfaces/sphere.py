from math import sqrt
from typing import List, Tuple

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
        # Calculate the surface normal at the intersection point
        normal = intersection - self.position
        normal /= np.linalg.norm(normal)

        # Calculate the reflection vector
        return ray_vec - 2 * (ray_vec @ normal) * normal

    def light_hit(
        self,
        light_source: np.ndarray,
        intersection_point: np.ndarray,
        surfaces: List[Surface],
    ) -> bool:
        light_vec = intersection_point - light_source
        _, light_intersection = get_closest_surface(
            light_source, light_vec, surfaces, None
        )
        if np.allclose(intersection_point, light_intersection, atol=1e-5):
            return True
        return False
