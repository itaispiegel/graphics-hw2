from typing import List, Tuple

import numpy as np

from .base_surface import Material, Surface


class Sphere(Surface):
    def __init__(self, position: List[int], radius: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.radius = radius

    def intersect(
        self, source: np.ndarray, ray_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
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
            return None, None

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        if t1 < 0 and t2 < 0:
            return None, None
        t = t1 if t1 >= 0 else t2

        # calculate the intersection point and the distance from the source to the intersection point
        intersection = source + t * ray_vec
        distance = np.linalg.norm(intersection - source)
        return intersection, distance

    def reflection(
        self, ray_vec: np.ndarray, distance: float, intersection: np.ndarray
    ) -> np.ndarray:
        # Calculate the surface normal at the intersection point
        normal = (intersection - self.position) / self.radius

        # Calculate the reflection vector
        reflection_ray_vec = ray_vec - 2 * ray_vec @ normal * normal
        return intersection, distance, reflection_ray_vec
