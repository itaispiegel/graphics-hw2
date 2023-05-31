from typing import List

import numpy as np

from camera import Camera
from light import Light
from material import Material
from surfaces import Cube, InfinitePlane, Sphere


class SceneSettings:
    def __init__(
        self,
        background_color: List[float],
        root_number_shadow_rays: float,
        max_recursions: float,
    ):
        self.background_color = np.array(background_color)
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_recursions = max_recursions


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(
                    params[:3], params[3:6], params[6:9], params[9], params[10]
                )
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(
                    params[:3], params[3:6], params[6:9], params[9], params[10]
                )
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects
