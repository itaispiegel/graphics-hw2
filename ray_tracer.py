import argparse
from typing import List, Optional

import numpy as np
from PIL import Image

from light import Light
from scene import SceneSettings, parse_scene_file
from surfaces import Surface


def get_color(
    source: np.ndarray,
    ray_vec: np.ndarray,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
    curr_surface: Optional[Surface] = None,
    iteration: int = 0,
):
    if iteration == scene_settings.max_recursions:
        return scene_settings.background_color

    obj, intersection_point = get_closest_surface(
        source, ray_vec, surfaces, curr_surface
    )
    if not obj:
        return scene_settings.background_color

    return (0, 0, 0)


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

        intersection_point, dist = surface.intersect(source, ray_vec)
        if dist and dist < min_dist:
            closest_surface = surface
            closest_intersection_point = intersection_point
            min_dist = dist

    return closest_surface, closest_intersection_point


def save_image(image_array: np.ndarray):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description="Python Ray Tracer")
    parser.add_argument("scene_file", type=str, help="Path to the scene file")
    parser.add_argument("output_image", type=str, help="Name of the output image file")
    parser.add_argument("--width", type=int, default=500, help="Image width")
    parser.add_argument("--height", type=int, default=500, help="Image height")
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, lights = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    image_array = np.zeros((args.height, args.width, 3))

    # calculate image's center, towards vector, right vector and up vector and the ratio
    v_to = camera.look_at - camera.position
    v_to /= np.linalg.norm(v_to)
    p_c = camera.position + (camera.screen_distance * v_to)
    v_right = np.cross(v_to, camera.up_vector)
    v_right /= np.linalg.norm(v_right)
    v_up = np.cross(v_right, camera.look_at)
    v_up /= np.linalg.norm(v_up)
    ratio = camera.screen_width / args.width

    # calculate the color of each pixel
    for i in range(args.height):
        for j in range(args.width):
            # calculate the ray's vector and the point (p) on the screen
            p = (
                p_c
                + ((j - int(args.width / 2)) * ratio * v_right)
                - ((i - int(args.height / 2)) * ratio * v_up)
            )
            ray_vec = p - camera.position

            # calculate the color of the pixel using ray tracing
            image_array[i][j] = get_color(
                camera.position, ray_vec, surfaces, lights, scene_settings
            )

    # Save the output image
    save_image(image_array)


if __name__ == "__main__":
    main()
