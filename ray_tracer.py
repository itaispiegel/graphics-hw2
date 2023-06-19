import argparse
import itertools

import numpy as np
from PIL import Image

from colors import get_color
from consts import COLOR_CHANNELS, COLOR_SCALE
from progressbar import progressbar
from scene import parse_scene_file


def save_image(image_array: np.ndarray, save_path: str) -> None:
    image = Image.fromarray(np.uint8(image_array))
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Python Ray Tracer")
    parser.add_argument("scene_file", type=str, help="Path to the scene file")
    parser.add_argument("output_image", type=str, help="Name of the output image file")
    parser.add_argument("--width", type=int, default=500, help="Image width")
    parser.add_argument("--height", type=int, default=500, help="Image height")
    args = parser.parse_args()

    camera, scene_settings, surfaces, lights = parse_scene_file(args.scene_file)
    image_array = np.zeros((args.height, args.width, COLOR_CHANNELS))

    v_to = camera.look_at - camera.position
    v_to /= np.linalg.norm(v_to)
    p_c = camera.position + (camera.screen_distance * v_to)
    v_right = np.cross(v_to, camera.up_vector)
    v_right /= np.linalg.norm(v_right)
    v_up = np.cross(v_right, v_to)
    v_up /= np.linalg.norm(v_up)
    ratio = camera.screen_width / args.width

    for i, j in progressbar(
        itertools.product(range(args.height), range(args.width)),
        count=args.height * args.width,
        prefix="Computing: ",
    ):
        p = (
            p_c
            + ((j - args.width // 2) * ratio * v_right)
            - ((i - args.height // 2) * ratio * v_up)
        )
        ray_vec = p - camera.position
        ray_vec /= np.linalg.norm(ray_vec)

        color = get_color(p, ray_vec, surfaces, lights, scene_settings)
        image_array[i][j] = np.clip(color, 0, 1) * COLOR_SCALE
        print(
            f"Pixel: {i},{j} - color: {image_array[i][j][0]} {image_array[i][j][1]} {image_array[i][j][2]}"
        )

    save_image(image_array, args.output_image)


if __name__ == "__main__":
    main()
