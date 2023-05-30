import argparse

import numpy as np
from PIL import Image

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


DIMENSIONS = 3


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


# aaaaaaaaaaaaaaaa
def get_color(source, ray_vec, objects, scene_settings, curr_object=None, iteration=0):
    if iteration == scene_settings.max_recursions:
        return scene_settings.background_color
    
    obj, intersection_point, reflection_ray = get_closest_object(source, ray_vec, objects, curr_object)
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
        
        intersection_point, dist, reflection_ray = intersect(source, ray_vec, obj)
        if dist and dist < min_dist:
            closest_obj = obj
            closest_intersection_point = intersection_point
            closest_reflection_ray_vec = reflection_ray
            min_dist = dist
            
    return closest_obj, closest_intersection_point, closest_reflection_ray_vec


# returns the intersection point and distance from the source to the intersection point
def intersect(source, ray_vec, obj):
    if isinstance(obj, Sphere):
        return sphere_intersect(source, ray_vec, obj)
    if isinstance(obj, InfinitePlane):
        return plane_intersect(source, ray_vec, obj)
    if isinstance(obj, Cube):
        return cube_intersect(source, ray_vec, obj)
   
    raise ValueError(f"Unknown object (obj) type: {type(obj)}")
    

# returns the intersection point and distance from the source to the intersection point
# this method detects if the ray intersects the sphere using the algebraic method shown in class
def sphere_intersect(source, ray_vec, sphere):
    oc = source - sphere.position
    a = np.dot(ray_vec, ray_vec)
    b = 2.0 * np.dot(oc, ray_vec)
    c = np.dot(oc, oc) - sphere.radius**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None

    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)
    if t1 < 0 and t2 < 0:
        return None, None
    t = t1 if t1 >= 0 else t2
    
    # calculate the intersection point and the distance from the source to the intersection point
    intersection = source + t * ray_vec
    distance = np.linalg.norm(intersection - source)
    
    # Calculate the surface normal at the intersection point
    normal = (intersection - sphere.position) / sphere.radius
    
    # Calculate the reflection vector
    reflection_ray_vec = ray_vec - 2 * np.dot(ray_vec, normal) * normal
    return intersection, distance, reflection_ray_vec


# returns the intersection point and distance from the source to the intersection point
def plane_intersect(source, ray_vec, plane):
    t = np.dot(plane.normal, source - (plane.normal * plane.offset)) / np.dot(plane.normal, ray_vec)
    
    # Check if intersection is behind the source (according to the ray's direction)
    # or if the ray is parallel to the plane
    if t < 0:
        return None, None
    
    # calculate the intersection point and the distance from the source to the intersection point
    intersection = source + t * ray_vec
    distance = np.linalg.norm(intersection - source)
    
    # Calculate the reflection vector
    reflection_ray_vec = ray_vec - 2 * np.dot(ray_vec, plane.normal) * plane.normal
    return intersection, distance, reflection_ray_vec


# returns the intersection point and distance from the source to the intersection point
# this method detects if the ray intersects the cube using the slab method
def cube_intersect(source, ray_vec, cube):
    t_near = float("-inf")
    t_far = float("inf")

    for i in range(DIMENSIONS):
        if ray_vec[i] == 0:
            if source[i] < cube.position[i] or source[i] > cube.position[i] + cube.scale:
                return None, None
        else:
            t1 = (cube.position[i] - source[i]) / ray_vec[i]
            t2 = (cube.position[i] + cube.scale - source[i]) / ray_vec[i]
            if t1 > t2:
                t1, t2 = t2, t1

            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            if t_near > t_far or t_far < 0:
                return None, None

    t = t_near if t_near >= 0 else t_far
    
    # calculate the intersection point and the distance from the source to the intersection point
    intersection = source + t * ray_vec
    distance = np.linalg.norm(intersection - source)
    
    # Calculate the reflection vector
    reflection_ray_vec = np.zeros(3)
    for i in range(3):
        if abs(intersection[i] - cube.position[i]) < 1e-10:
            reflection_ray_vec[i] = -ray_vec[i]
        elif abs(intersection[i] - (cube.position[i] + cube.scale)) < 1e-10:
            reflection_ray_vec[i] = ray_vec[i]
        else:
            reflection_ray_vec[i] = 0

    return intersection, distance, reflection_ray_vec
    

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
    image_array = np.zeros((args.height, args.width, DIMENSIONS))
    
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
            p = p_c + ((j - int(args.width / 2)) * ratio * v_r) - ((i - int(args.height / 2)) * ratio * v_up)
            ray_vec = p - camera.position

            # calculate the color of the pixel using ray tracing
            image_array[i][j] = get_color(camera.position, ray_vec, objects, scene_settings)
            
    # Save the output image
    save_image(image_array)


if __name__ == "__main__":
    main()
