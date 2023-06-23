from typing import List, Optional

import numpy as np

from base_surface import Surface, get_closest_surface
from consts import COLOR_CHANNELS
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

        light_intensity = light.calculate_intensity(
            surfaces, scene_settings.root_number_shadow_rays, intersection
        )
        diffuse = surface.material.diffuse_color * (normal @ l.direction)
        specular = (
            surface.material.specular_color
            * light.specular_intensity
            * (v.direction @ reflected_ray.direction) ** surface.material.shininess
        )
        color += (diffuse + specular) * light.color * light_intensity

    return color
