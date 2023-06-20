from typing import List, Optional, Tuple

import numpy as np

from consts import EPSILON
from material import Material
from ray import Ray


class Surface:
    def __init__(self, material: Material):
        self.material = material

    def intersect(self, ray: Ray) -> Optional[np.ndarray]:
        """
        Return the intersection point between the ray and surface, or None if they don't intersect.
        """
        raise NotImplementedError()

    def reflection(self, ray_vec: np.ndarray, intersection: np.ndarray) -> np.ndarray:
        """
        Receive the ray vector and intersection point on the surface, and return the reflection vector.
        """
        normal = self.normal_at_point(intersection, ray_vec)
        reflection_vec = ray_vec - 2 * (ray_vec @ normal) * normal
        return reflection_vec / np.linalg.norm(reflection_vec)

    def is_path_clear(
        self,
        source: np.ndarray,
        dest: np.ndarray,
        surfaces: List["Surface"],
    ) -> bool:
        """
        Returns true iff the light source hits the surface at the intersection point
        without hitting any other surface on the way.
        """
        light_ray = Ray.ray_between_points(source, dest)
        _, light_intersection = get_closest_surface(light_ray, surfaces)

        if light_intersection is None:
            return True
        return np.allclose(dest, light_intersection, atol=EPSILON)

    def normal_at_point(self, point: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Return the normal vector of the surface at the given point.
        """
        raise NotImplementedError()


def get_closest_surface(
    ray: Ray,
    surfaces: List[Surface],
    source_surface: Surface = None,
) -> Tuple[Surface, np.ndarray]:
    """
    Receive a ray and the list of surfaces, and return a pair of the closest surface and its
    intersection point with the ray.
    :param ray: The shot ray.
    :param surfaces: A list of the surfaces in the scene.
    :param source_surface: An optional parameter indicating the surface, the ray is shot from.
    This surface will be ignored when searching for the closest surface.
    :returns: A pair of the closest surface and the intersection point.
    """
    closest_surface = None
    closest_intersection = None
    min_dist = float("inf")

    for surface in surfaces:
        if surface == source_surface:
            continue

        intersection = surface.intersect(ray)
        if intersection is None:
            continue

        dist = np.linalg.norm(intersection - ray.source)
        if dist < min_dist:
            closest_surface = surface
            closest_intersection = intersection
            min_dist = dist

    return closest_surface, closest_intersection
