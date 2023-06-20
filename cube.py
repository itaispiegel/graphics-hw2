from typing import List, Optional

import numpy as np

from base_surface import Material, Surface
from ray import Ray

NORMAL_TUPLE = (
    np.array([1, 0, 0], dtype=np.float64),
    np.array([-1, 0, 0], dtype=np.float64),
    np.array([0, 1, 0], dtype=np.float64),
    np.array([0, -1, 0], dtype=np.float64),
    np.array([0, 0, 1], dtype=np.float64),
    np.array([0, 0, -1], dtype=np.float64)
)


class Cube(Surface):
    def __init__(self, position: List[float], scale: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.scale = scale

    def intersect(self, ray: Ray) -> Optional[np.ndarray]:
        """
        Return the intersection point using the slab method.
        """
        t_near = float("-inf")
        t_far = float("inf")

        for i in range(3):
            if ray.direction[i] == 0:
                if (
                    ray.source[i] < self.position[i]
                    or ray.source[i] > self.position[i] + self.scale
                ):
                    return None
            else:
                t1 = (self.position[i] - ray.source[i]) / ray.ray_vec[i]
                t2 = (self.position[i] + self.scale - ray.source[i]) / ray.ray_vec[i]
                if t1 > t2:
                    t1, t2 = t2, t1

                t_near = max(t_near, t1)
                t_far = min(t_far, t2)

                if t_near > t_far or t_far < 0:
                    return None

        t = t_near if t_near >= 0 else t_far
        return ray.at(t)

    def normal_at_point(self, point: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        # we are assuming that the point is on the surface of the cube
        closest_normal = None
        closest_dist = float("inf")

        for normal in NORMAL_TUPLE:
            dist = np.linalg.norm(point - (self.position + (normal * self.scale / 2.0)))
            if dist < closest_dist:
                closest_dist = dist
                closest_normal = normal

        return closest_normal
