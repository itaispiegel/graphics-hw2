from typing import List

import numpy as np

import vector
from base_surface import Surface, get_closest_surface
from consts import EPSILON
from ray import Ray


class Light:
    def __init__(
        self,
        position: List[float],
        color: List[float],
        specular_intensity: float,
        shadow_intensity: float,
        radius: float,
    ):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    @staticmethod
    def is_path_clear(
        source: np.ndarray,
        dest: np.ndarray,
        surfaces: List[Surface],
    ) -> bool:
        """
        Returns true iff the light source hits the surface at the intersection point
        without hitting any other surface on the way.
        This method expects point dest to be on a surface and the source to be a light source.
        """
        light_ray = Ray.ray_between_points(source, dest)
        _, light_intersection = get_closest_surface(light_ray, surfaces)
        return light_intersection is None or np.allclose(
            dest, light_intersection, atol=EPSILON
        )

    def calculate_intensity(
        self,
        surfaces: List[Surface],
        root_number_shadow_rays: int,
        point: np.ndarray,
    ) -> float:
        """
        Receive the list of surfaces, the number of shadow rays to case and a point, and
        return the light intensity value on the point measured as coefficient in the range [0, 1].
        The calculation is done according to the "soft shadows" algorithm.
        If the number of shadow rays is 1, we calculate "hard shadows" instead.
        """
        if root_number_shadow_rays == 1:
            return (
                1.0
                - Light.is_path_clear(self.position, point, surfaces)
                * self.shadow_intensity
            )

        normal = Ray.ray_between_points(self.position, point)
        vec1, vec2 = vector.orthonormal_vector_pair(normal.direction)

        row_indices = np.repeat(
            np.arange(root_number_shadow_rays),
            root_number_shadow_rays,
        )[:, np.newaxis]
        col_indices = np.tile(
            np.arange(root_number_shadow_rays),
            root_number_shadow_rays,
        )[:, np.newaxis]

        number_shadow_rays = root_number_shadow_rays * root_number_shadow_rays
        vec1_repeated = np.repeat(vec1[np.newaxis, :], number_shadow_rays, axis=0)
        vec2_repeated = np.repeat(vec2[np.newaxis, :], number_shadow_rays, axis=0)

        half_r = self.radius / 2.0
        top_left = self.position - (half_r * vec1) - (half_r * vec2)
        grid_square_length = self.radius / root_number_shadow_rays
        corners1 = (
            top_left
            + row_indices * grid_square_length * vec1_repeated
            + col_indices * grid_square_length * vec2_repeated
        )
        corners2 = (
            corners1
            + grid_square_length * vec1_repeated
            + grid_square_length * vec2_repeated
        )

        min_coords = np.minimum(corners1, corners2)
        max_coords = np.maximum(corners1, corners2)
        light_sources = np.random.uniform(min_coords, max_coords)
        light_hit_cnt = sum(
            Light.is_path_clear(light_source, point, surfaces)
            for light_source in light_sources
        )
        return (1 - self.shadow_intensity) + self.shadow_intensity * (
            light_hit_cnt / (root_number_shadow_rays**2)
        )
