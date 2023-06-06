from typing import List, Optional, Tuple

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
) -> np.ndarray:
    if iteration == scene_settings.max_recursions:
        if not lights:
            return get_black_color()
        return scene_settings.background_color

    surface, intersection = get_closest_surface(
        source, ray_vec, surfaces, source_surface
    )
    if not surface:
        if not lights:
            return get_black_color()
        return scene_settings.background_color

    # return surface.material.diffuse_color * 255

    color = phong(
        source, intersection, surface, surfaces, lights, scene_settings, iteration
    ) * (1 - surface.material.transparency)
    if surface.material.transparency > 0:
        color += (
            get_color(
                intersection,
                ray_vec,
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.transparency
        )
    is_reflective = True  # TODO: LEARN HOW TO CHECK IF A SURFACE IS REFLECTIVE
    if is_reflective:
        color += (
            get_color(
                intersection,
                surface.reflection(ray_vec, intersection),
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.reflection_color
        )

    return np.clip(color, 0, 1)


def get_black_color() -> np.ndarray:
    return np.zeros(COLOR_CHANNELS, dtype=np.float64)


# returns the closet surface to the source and the intersection point of the ray on object
def get_closest_surface(
    source: np.ndarray,
    ray_vec: np.ndarray,
    surfaces: List[Surface],
    curr_surface: Surface,
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


def get_light_intensity(
    curr_surface: Surface,
    surfaces: List[Surface],
    light: Light,
    scene_settings: SceneSettings,
    intersection: np.ndarray,
) -> float:
    light_hit_cnt = 0

    # get 2 vectors that are orthogonal to the normal vector
    normal = intersection - light.position
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
            if curr_surface.light_hit(light_source, intersection, surfaces):
                light_hit_cnt += 1

    # return the light intensity
    return (1 - light.shadow_intensity) + light.shadow_intensity * (
        light_hit_cnt / (scene_settings.num_shadow_samples**2)
    )


def phong(
    source: np.ndarray,
    intersection: np.ndarray,
    surface: Surface,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
) -> np.ndarray:
    color = get_black_color()

    for light in lights:
        if not surface.light_hit(light.position, intersection, surfaces):
            continue

        l_vec = light.position - intersection
        l_vec /= np.linalg.norm(l_vec)
        normal = surface.get_normal(intersection)
        v_vec = source - intersection
        v_vec /= np.linalg.norm(v_vec)
        r_vec = surface.reflection(
            -l_vec, intersection
        )  # the reflection methods need a vector FROM the source TO the intersection

        light_intensity = get_light_intensity(
            surface, surfaces, light, scene_settings, intersection
        )
        diffuse = surface.material.diffuse_color * light.color * (normal @ l_vec)
        specular = (
            surface.material.specular_color
            * light.color
            * light.specular_intensity
            * (r_vec @ v_vec) ** surface.material.shininess
        )

        color += (diffuse + specular) * light_intensity

    return color
