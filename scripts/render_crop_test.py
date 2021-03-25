import os
import gc
import trimesh
import numpy as np
from PIL import Image

from mesh import Mesh
from scripts.helper import load_from_json, visualize_scene
from scripts.box import Box
from renderer import Render

# define scene_name nad paths.
scene_name = 'Z6MFQCViBuw_room0.json'
center_obj = '32'
scene_graph_dir = '../data/matterport3d/scene_graphs/all'
models_dir = '../data/matterport3d/models'

# load the accepted cats
accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')


def prepare_obj_centered_scene(graph, objects, models_dir, center_obj, ceiling_cats=['ceiling', 'void']):
    object_scene = []
    context_scene = []
    for obj in objects:
        cat = graph[obj]['category'][0]

        # if the mesh is a ceiling mesh skip
        if cat in ceiling_cats:
            continue

        # load the mesh and save it to the scene.
        model_path = os.path.join(models_dir, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path, graph[obj]['transform'])
        mesh = mesh_obj.load(with_transform=True)
        context_scene.append(mesh)

        # if the object is the center object add it to the object centered scene
        if obj == center_obj:
            object_scene.append(mesh)

    del mesh
    gc.collect()

    # extract the room dimention and the camera pose
    context_scene = trimesh.Scene(context_scene)
    object_scene = trimesh.Scene(object_scene)
    room_dimension = context_scene.extents
    camera_pose, _ = context_scene.graph[context_scene.camera.name]

    # translate the camera pose to be above the
    obj_centroid = np.asarray(graph[center_obj]['obbox'][0])
    camera_pose[:2, 3] = obj_centroid[:2]

    # find the axis along which the rotation of the camera happens.
    obj_to_scene = context_scene.centroid - obj_centroid
    obj_to_scene_xy = obj_to_scene[:2]
    obj_to_scene_xy = obj_to_scene_xy / np.linalg.norm(obj_to_scene_xy)
    direction = [-obj_to_scene_xy[1], obj_to_scene_xy[0], 0]

    # the sign of the angle is positive if the room's centroid is on the right of the direction otherwise negative.
    mid_to_scene_centroid = context_scene.centroid - (obj_centroid + obj_to_scene / 2.0)
    mid_to_scene_centroid_xy = mid_to_scene_centroid[:2]
    mid_to_scene_centroid_xy = mid_to_scene_centroid_xy / np.linalg.norm(mid_to_scene_centroid_xy)

    perpendicular_direction = [mid_to_scene_centroid_xy[0], mid_to_scene_centroid_xy[1], 0]
    z_axis = np.cross(direction, perpendicular_direction)
    angle = np.radians(20)
    if z_axis[2] > 0:
        angle = -angle

    # rotate the camera pose to point at the object of interest
    rotation = trimesh.transformations.rotation_matrix(angle=angle, direction=direction, point=obj_centroid)
    camera_pose = np.dot(rotation, camera_pose)

    return object_scene, context_scene, camera_pose, room_dimension


def center_crop(img, crop_size):
    new_width, new_height = crop_size
    width, height = img.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))

    return img


# prepare and object centered view of the scene
graph = load_from_json(os.path.join(scene_graph_dir, scene_name))
object_scene, context_scene, camera_pose, room_dimension = prepare_obj_centered_scene(graph, graph.keys(), models_dir,
                                                                                      center_obj=center_obj)

# render the object centered view
resolution = (512, 512)
rendering_kwargs = {'fov': np.pi / 6, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                    'wall_thickness': 5}
r = Render(rendering_kwargs)
img_context = r.center_view_render(context_scene, resolution, camera_pose, room_dimension)
img_object = r.center_view_render(object_scene, resolution, camera_pose, room_dimension)
imgs = [img_object, img_context]

# crop the rendered image based on the max scale of the centred object. we include the object and its context
# crop_sizes = [(256, 256)]
crop_sizes = []
center_obj_vertices = np.asarray(graph[center_obj]['obbox'])
center_obj_obbox = Box(center_obj_vertices)
max_scale = np.max(center_obj_obbox.scale)
scales = [max_scale, np.minimum(max_scale * 5, np.min(room_dimension)/2)]

prev_crop_dim = 0
for scale in scales:
    # if the zoomed version reached max resolution we, stay at max resolution
    if prev_crop_dim == resolution[0]:
        crop_dim = resolution[0]
    else:
        crop_dim = int(scale * resolution[0] / np.min(room_dimension))
        crop_dim = np.minimum(crop_dim, resolution[0])
        prev_crop_dim = crop_dim

    crop_sizes.append((crop_dim, crop_dim))
    print(crop_sizes)

# center crop the object and its context.
for i in range(len(imgs)):
    img_cropped = center_crop(Image.fromarray(imgs[i]), crop_sizes[i])
    img_resized = img_cropped.resize(resolution)
    img_resized.show()

# visualize the scene
# visualize_scene(scene_graph_dir, models_dir, scene_name, accepted_cats,
#                 highlighted_objects=[center_obj], with_backbone=False, as_obbox=False)

