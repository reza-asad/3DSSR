import os
import numpy as np
import trimesh

from mesh import Mesh
from scripts.helper import load_from_json


def visualize_scene(graph, objects, ceiling_cats=['ceiling', 'void']):
    scene = []
    for obj in objects:
        cat = graph[obj]['category'][0]

        # if the mesh is a ceiling mesh skip
        if cat in ceiling_cats:
            continue

        # load the mesh
        model_path = os.path.join(models_dir, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path, graph[obj]['transform'])
        mesh = mesh_obj.load(with_transform=True)

        # add elements to the scene and cropped scene (if necessary)
        scene.append(mesh)

    # extract the room dimension and the camera pose
    scene = trimesh.Scene(scene)

    return scene


def transform_scene(s):
    transformation_z = trimesh.transformations.rotation_matrix(angle=alpha, direction=[0, 0, 1],
                                                               point=s.centroid)
    transformation_y = trimesh.transformations.rotation_matrix(angle=beta, direction=[0, 1, 0],
                                                               point=s.centroid)
    transformation_x = trimesh.transformations.rotation_matrix(angle=gamma, direction=[1, 0, 0],
                                                               point=s.centroid)

    transformation = np.matmul(np.matmul(transformation_z, transformation_y), transformation_x)
    s.apply_transform(transformation)


# scene names for query, top1, top2 and top3
models_dir = '../data/matterport3d/models'
scene_dir = '../data/matterport3d/scenes/all'
room_names = ['pa4otMbVnkk_room18', 'q9vSo1VnCiC_room12', 'jtcxE69GiFV_room1']
pc_dir = '../data/matterport3d/pc_regions/test'
mesh_region_dir = '../data/matterport3d/mesh_regions/test'
regions = ['rqfALeAoiTq_room11-8', 'yqstnuAEVhm_room14-26', 'yqstnuAEVhm_room24-24']
room_ = False
pc_ = True
mesh_ = False

if room_:
    scene_names = [os.path.join(scene_dir, '{}.json'.format(scene_name)) for scene_name in room_names]
elif mesh_:
    scene_names = [os.path.join(mesh_region_dir, '{}.ply'.format(mesh_region)) for mesh_region in regions]
else:
    scene_names = [os.path.join(pc_dir, '{}.npy'.format(pc_region)) for pc_region in regions]

for i, scene_name in enumerate(scene_names):
    # load the scene.
    if room_:
        scene_graph = load_from_json(scene_name)
        scene = visualize_scene(scene_graph, scene_graph.keys())
    elif mesh_:
        scene = trimesh.load(scene_name)
        radii = np.linalg.norm(scene.vertices, axis=1)
        scene.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')
    else:
        pc = np.load(scene_name)
        radii = np.linalg.norm(pc, axis=1)
        colors = trimesh.visual.interpolate(radii, color_map='viridis')
        scene = trimesh.points.PointCloud(pc, colors=colors)

    # apply transformation
    if i == 0:
        alpha = 0 * np.pi / 180
        beta = 10 * np.pi / 180
        gamma = -90 * np.pi / 180
    elif i == 1:
        alpha = 0 * np.pi / 180
        beta = -15 * np.pi / 180
        gamma = -70 * np.pi / 180
    elif i == 2:
        alpha = 0 * np.pi / 180
        beta = -70 * np.pi / 180
        gamma = -90 * np.pi / 180
    transform_scene(scene)
    scene.show()



