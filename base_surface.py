from typing import List, Tuple

import numpy as np

from material import Material

EPSILON = 1e-5


class Surface:
    def __init__(self, material: Material):
        self.material = material

    def intersect(self, source: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Return the intersection point and distance from the source to the intersection point.
        """
        raise NotImplementedError()

    def reflection(self, ray_vec: np.ndarray, intersection: np.ndarray) -> np.ndarray:
        normal = self.normal_at_point(intersection)

        # Calculate the reflection vector
        reflection_vec = ray_vec - 2 * (ray_vec @ normal) * normal
        return reflection_vec / np.linalg.norm(reflection_vec)

    def light_hit(
        self,
        light_source: np.ndarray,
        intersection: np.ndarray,
        surfaces: List["Surface"],
    ) -> bool:
        """
        Returns true iff the light source hits the surface at the intersection point
        without hitting any other surface on the way.
        """
        light_vec = intersection - light_source
        _, light_intersection = get_closest_surface(light_source, light_vec, surfaces)
        
        return np.allclose(intersection, light_intersection, atol=EPSILON) 

    def normal_at_point(self, point: np.ndarray) -> np.ndarray:
        """
        Return the normal vector of the surface at the given point.
        """
        raise NotImplementedError()


# returns the closet surface to the source and the intersection point of the ray on object
def get_closest_surface(
    source: np.ndarray,
    ray_vec: np.ndarray,
    surfaces: List[Surface],
    curr_surface: Surface = None,
) -> Tuple[Surface, np.ndarray]:
    closest_surface = None
    closest_intersection = None
    min_dist = float("inf")

    for surface in surfaces:
        if surface == curr_surface:
            continue

        intersection = surface.intersect(source, ray_vec)
        if intersection is None:
            continue

        dist = np.linalg.norm(intersection - source)
        if dist < min_dist:
            closest_surface = surface
            closest_intersection = intersection
            min_dist = dist

    return closest_surface, closest_intersection
