from typing import List, Optional

import numpy as np

import vector
from base_surface import Surface, get_closest_surface
from consts import COLOR_CHANNELS, EPSILON
from light import Light
from ray import Ray
from scene import SceneSettings


def get_color(
    ray: Ray,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
    source_surface: Optional[Surface] = None,
    iteration: int = 0,
) -> np.ndarray:
    if iteration == scene_settings.max_recursions:
        return scene_settings.background_color

    surface, intersection = get_closest_surface(
        ray, surfaces, source_surface=source_surface
    )
    if not surface:
        return scene_settings.background_color

    color = phong(ray, intersection, surface, surfaces, lights, scene_settings) * (
        1 - surface.material.transparency
    )
    if surface.material.transparency > 0:
        color += (
            get_color(
                Ray(intersection, ray.direction),
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.transparency
        )
    if surface.material.is_reflective():
        color += (
            get_color(
                surface.reflection_ray(ray, intersection),
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.reflection_color
        )

    return color


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


def calculate_light_intensity(
    surfaces: List[Surface],
    light: Light,
    scene_settings: SceneSettings,
    point: np.ndarray,
) -> float:
    # if we need to produce hard shadows
    if scene_settings.root_number_shadow_rays == 1:
        return (
            1.0
            - is_path_clear(light.position, point, surfaces) * light.shadow_intensity
        )

    total_lights_cnt = (
        scene_settings.root_number_shadow_rays * scene_settings.root_number_shadow_rays
    )
    light_hit_cnt = 0
    normal = Ray.ray_between_points(light.position, point)
    vec1, vec2 = vector.orthonormal_vector_pair(normal.direction)

    half_r = light.radius / 2.0
    top_left = light.position - (half_r * vec1) - (half_r * vec2)

    grid_square_length = light.radius / scene_settings.root_number_shadow_rays
    row_indices = np.repeat(
        np.arange(scene_settings.root_number_shadow_rays),
        scene_settings.root_number_shadow_rays,
    )[:, np.newaxis]
    col_indices = np.tile(
        np.arange(scene_settings.root_number_shadow_rays),
        scene_settings.root_number_shadow_rays,
    )[:, np.newaxis]

    vec1_repeated = np.repeat(vec1[np.newaxis, :], total_lights_cnt, axis=0)
    vec2_repeated = np.repeat(vec2[np.newaxis, :], total_lights_cnt, axis=0)

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
        is_path_clear(light_source, point, surfaces)
        for light_source in light_sources.reshape(25, 3)
    )

    # return the light intensity
    return (1 - light.shadow_intensity) + light.shadow_intensity * (
        light_hit_cnt / (scene_settings.root_number_shadow_rays**2)
    )


def phong(
    ray: Ray,
    intersection: np.ndarray,
    surface: Surface,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
) -> np.ndarray:
    color = np.zeros(COLOR_CHANNELS, dtype=np.float64)
    v = -ray

    for light in lights:
        l = Ray.ray_between_points(intersection, light.position)
        normal = surface.normal_at_point(intersection, -l.direction)
        reflected_ray = surface.reflection_ray(-l, intersection)

        light_intensity = calculate_light_intensity(
            surfaces, light, scene_settings, intersection
        )
        diffuse = surface.material.diffuse_color * (normal @ l.direction)
        specular = (
            surface.material.specular_color
            * light.specular_intensity
            * (v.direction @ reflected_ray.direction) ** surface.material.shininess
        )
        color += (diffuse + specular) * light.color * light_intensity

    return color
