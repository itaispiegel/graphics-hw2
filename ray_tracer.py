import argparse

import numpy as np
from PIL import Image

from scene import parse_scene_file


def get_color(source, ray_vec, objects, scene_settings, curr_object=None, iteration=0):
    if iteration == scene_settings.max_recursions:
        return scene_settings.background_color

    obj, intersection_point, reflection_ray = get_closest_object(
        source, ray_vec, objects, curr_object
    )
    if not obj:
        return scene_settings.background_color

    return (0, 0, 0)


# returns the closet object to the source and the intersection point of the ray on object
def get_closest_object(source, ray_vec, objects, curr_object):
    closest_obj = None
    closest_intersection_point = None
    closest_reflection_ray_vec = None
    min_dist = float("inf")

    for obj in objects:
        if obj == curr_object:
            continue

        intersection_point, dist, reflection_ray = obj.intersect(source, ray_vec)
        if dist and dist < min_dist:
            closest_obj = obj
            closest_intersection_point = intersection_point
            closest_reflection_ray_vec = reflection_ray
            min_dist = dist

    return closest_obj, closest_intersection_point, closest_reflection_ray_vec


def save_image(image_array):
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
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    image_array = np.zeros((args.height, args.width, 3))

    # calculate image's center, towards vector, right vector and up vector and the ratio
    v_to = camera.look_at - camera.position
    v_to = v_to / np.linalg.norm(v_to)
    p_c = camera.position + (camera.screen_distance * v_to)
    v_r = np.cross(v_to, camera.up_vector)
    v_r = v_r / np.linalg.norm(v_r)
    v_up = np.cross(v_r, camera.look_at)
    v_up = v_up / np.linalg.norm(v_up)
    ratio = camera.screen_width / args.width

    # calculate the color of each pixel
    for i in range(args.height):
        for j in range(args.width):
            # calculate the ray's vector and the point (p) on the screen
            p = (
                p_c
                + ((j - int(args.width / 2)) * ratio * v_r)
                - ((i - int(args.height / 2)) * ratio * v_up)
            )
            ray_vec = p - camera.position

            # calculate the color of the pixel using ray tracing
            image_array[i][j] = get_color(
                camera.position, ray_vec, objects, scene_settings
            )

    # Save the output image
    save_image(image_array)


if __name__ == "__main__":
    main()
