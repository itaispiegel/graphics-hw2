from typing import List, Optional

import numpy as np

from light import Light
from scene import SceneSettings
from surfaces import Surface

COLOR_CHANNELS = 3


def get_color(
    source: np.ndarray,
    ray_vec: np.ndarray,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
    source_surface: Optional[Surface] = None,
    iteration: int = 0,
):
    if iteration == scene_settings.max_recursions:
        return scene_settings.background_color

    surface, intersection_point = get_closest_surface(
        source, ray_vec, surfaces, source_surface
    )
    if not surface:
        if not lights:
            return np.zeros(COLOR_CHANNELS)
        return scene_settings.background_color

    return surface.material.diffuse_color * 255


# returns the closet surface to the source and the intersection point of the ray on object
def get_closest_surface(
    source: np.ndarray,
    ray_vec: np.ndarray,
    surfaces: List[Surface],
    curr_surface: Surface,
):
    closest_surface = None
    closest_intersection_point = None
    min_dist = float("inf")

    for surface in surfaces:
        if surface == curr_surface:
            continue

        intersection_point = surface.intersect(source, ray_vec)
        if intersection_point is None:
            continue

        dist = np.linalg.norm(intersection_point - source)
        if dist < min_dist:
            closest_surface = surface
            closest_intersection_point = intersection_point
            min_dist = dist

    return closest_surface, closest_intersection_point


def get_light_intensity(
    curr_surface: Surface,
    surfaces: List[Surface],
    light: Light,
    scene_settings: SceneSettings,
    intersection_point: np.ndarray,
):
    light_hit_cnt = 0

    # get 2 vectors that are orthogonal to the normal vector
    normal = intersection_point - light.position
    fixed_vector = np.array([1, 1, 1])
    vec1 = fixed_vector - np.cross(normal, fixed_vector) * normal
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.cross(normal, vec1)
    vec2 /= np.linalg.norm(vec2)

    # get the top left corner of the grid
    n = light.radius / 2.0
    top_left = light.position - (n * vec1) - (n * vec2)

    # check if the light hits the surface from each square in the grid
    r = light.radius / scene_settings.num_shadow_samples
    for i in range(scene_settings.num_shadow_samples):
        for j in range(scene_settings.num_shadow_samples):
            # get the corners of the square
            corner1 = top_left + (i * r * vec1) + (j * r * vec2)
            corner2 = (
                top_left
                + ((i + 1) * light.radius * vec1)
                + ((j + 1) * light.radius * vec2)
            )

            # Calculate the minimum and maximum coordinates for each dimension
            min_coords = np.minimum(corner1, corner2)
            max_coords = np.maximum(corner1, corner2)

            # Generate random coordinates within the square
            light_source = np.random.uniform(min_coords, max_coords)
            if curr_surface.light_hit(light_source, intersection_point, surfaces):
                light_hit_cnt += 1

    # return the light intensity
    return (1 - light.shadow_intensity) + light.shadow_intensity * (
        light_hit_cnt / (scene_settings.num_shadow_samples**2)
    )
