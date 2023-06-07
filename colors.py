from typing import List, Optional, Tuple

import numpy as np

from base_surface import Surface, get_closest_surface
from consts import COLOR_SCALE, EPSILON
from light import Light
from scene import SceneSettings


def get_color(
    source: np.ndarray,
    ray_vec: np.ndarray,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
    source_surface: Optional[Surface] = None,
    iteration: int = 0,
    is_reflection: bool = False,
) -> np.ndarray:
    if iteration == scene_settings.max_recursions:
        if not lights:
            return vector()
        return scene_settings.background_color

    surface, intersection = get_closest_surface(
        source, ray_vec, surfaces, source_surface
    )
    if not surface:
        if not lights:
            return vector()
        return scene_settings.background_color

    color = phong(source, intersection, surface, surfaces, lights, scene_settings) * (
        1 - surface.material.transparency
    )
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
        reflection_color = get_color(
            intersection,
            surface.reflection(ray_vec, intersection),
            surfaces,
            lights,
            scene_settings,
            surface,
            iteration + 1,
            True,
        )
        if not np.allclose(
            reflection_color, scene_settings.background_color, atol=EPSILON
        ):
            color += reflection_color

    if is_reflection:
        color *= surface.material.reflection_color
    return np.clip(color, 0, 1) * COLOR_SCALE


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def get_light_intensity(
    curr_surface: Surface,
    surfaces: List[Surface],
    light: Light,
    scene_settings: SceneSettings,
    intersection: np.ndarray,
) -> float:
    return 0.5
    light_hit_cnt = 0

    # get 2 vectors that are orthogonal to the normal vector
    normal = intersection - light.position
    normal /= np.linalg.norm(normal)
    fixed_vector = vector(x=1.0)
    if np.allclose(normal, fixed_vector, atol=EPSILON):
        fixed_vector = vector(y=1.0)
    vec1 = np.cross(normal, fixed_vector)
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.cross(normal, vec1)
    vec2 /= np.linalg.norm(vec2)

    # get the top left corner of the grid
    half_w = light.radius / 2.0
    top_left = light.position - (half_w * vec1) - (half_w * vec2)

    # check if the light hits the surface from each square in the grid
    grid_square_length = light.radius / scene_settings.root_number_shadow_rays
    for i in range(int(scene_settings.root_number_shadow_rays)):
        for j in range(int(scene_settings.root_number_shadow_rays)):
            # get the corners of the square
            corner1 = (
                top_left
                + (i * grid_square_length * vec1)
                + (j * grid_square_length * vec2)
            )
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
    color = vector()

    for light in lights:
        if not surface.light_hit(light.position, intersection, surfaces):
            continue

        l_vec = light.position - intersection
        l_vec /= np.linalg.norm(l_vec)
        normal = surface.normal_at_point(intersection, -l_vec)
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
