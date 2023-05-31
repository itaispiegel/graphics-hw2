from typing import List

import numpy as np

from .base_surface import Surface


class Sphere(Surface):
    def __init__(self, position: List[int], radius: float, material_index: int):
        super().__init__(material_index)
        self.position = np.array(position)
        self.radius = radius

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray):
        """
        Return the intersection point and distance from the source to the intersection point
        using the algebraic method shown in class.
        """
        oc = source - self.position
        a = np.dot(ray_vec, ray_vec)
        b = 2.0 * np.dot(oc, ray_vec)
        c = np.dot(oc, oc) - self.radius**2

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

        # Calculate the surface normal at the intersection point
        normal = (intersection - self.position) / self.radius

        # Calculate the reflection vector
        reflection_ray_vec = ray_vec - 2 * np.dot(ray_vec, normal) * normal
        return intersection, distance, reflection_ray_vec
