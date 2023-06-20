from typing import List, Optional

import numpy as np

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
      
    color = phong(
        ray.source, intersection, surface, surfaces, lights, scene_settings
    ) * (1 - surface.material.transparency)
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
                Ray(intersection, surface.reflection(ray.direction, intersection)),
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.reflection_color
        )

    return color


def get_light_intensity(
    curr_surface: Surface,
    surfaces: List[Surface],
    light: Light,
    scene_settings: SceneSettings,
    intersection: np.ndarray,
) -> float:
    light_hit_cnt = 0

    # get 2 vectors that are orthogonal to the normal vector
    normal = Ray.ray_between_points(light.position, intersection)
    fixed_vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if np.allclose(normal.direction, fixed_vector, atol=EPSILON):
        fixed_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vec1 = np.cross(normal.direction, fixed_vector)
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.cross(normal.direction, vec1)
    vec2 /= np.linalg.norm(vec2)

    # get the top left corner of the grid
    top_left = light.position - (light.radius * vec1) - (light.radius * vec2)

    # check if the light hits the surface from each square in the grid
    grid_square_length = (2.0 * light.radius) / scene_settings.root_number_shadow_rays
    for i in range(scene_settings.root_number_shadow_rays):
        for j in range(scene_settings.root_number_shadow_rays):
            # get the corners of the square
            corner1 = (
                top_left
                + (i * grid_square_length * vec1)
                + (j * grid_square_length * vec2)
            )
            corner2 = (
                top_left
                + ((i + 1) * grid_square_length * vec1)
                + ((j + 1) * grid_square_length * vec2)
            )

            # Calculate the minimum and maximum coordinates for each dimension
            min_coords = np.minimum(corner1, corner2)
            max_coords = np.maximum(corner1, corner2)

            # Generate random coordinates within the square
            light_source = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            for i in range(COLOR_CHANNELS):
                light_source[i] = np.random.uniform(min_coords[i], max_coords[i])
            
            if curr_surface.is_path_clear(light_source, intersection, surfaces):
                light_hit_cnt += 1

    # return the light intensity
    return (1 - light.shadow_intensity) + light.shadow_intensity * (
        light_hit_cnt / (scene_settings.root_number_shadow_rays**2)
    )


def phong(
    source: np.ndarray,
    intersection: np.ndarray,
    surface: Surface,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
) -> np.ndarray:
    color = np.zeros(COLOR_CHANNELS, dtype=np.float64)

    for light in lights:
        if not surface.is_path_clear(light.position, intersection, surfaces):           
            continue

        l = Ray.ray_between_points(intersection, light.position)
        v = Ray.ray_between_points(intersection, source)
        normal = surface.normal_at_point(intersection, -l.direction)
        r_vec = surface.reflection(
            -l.direction, intersection
        )  # the reflection methods need a vector FROM the source TO the intersection

        light_intensity = get_light_intensity(
            surface, surfaces, light, scene_settings, intersection
        )
        diffuse = surface.material.diffuse_color * light.color * (normal @ l.direction)
        specular = (
            surface.material.specular_color
            * light.color
            * light.specular_intensity
            * (r_vec @ v.direction) ** surface.material.shininess
        )
        
        color += (diffuse + specular) * light_intensity

    return color
