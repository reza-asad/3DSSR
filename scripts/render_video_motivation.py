import os
import numpy as np
import trimesh

from mesh import Mesh
from scripts.helper import load_from_json


def visualize_scene(graph, objects, models_dir, query_objects, ceiling_cats=['ceiling', 'void'], sub=False,
                    edit_obj_info={}):
    scene, subscene = [], []
    for obj in objects:
        cat = graph[obj]['category'][0]

        # if the mesh is a ceiling mesh skip
        if cat in ceiling_cats:
            continue

        # load the mesh
        if obj in edit_obj_info:
            model_path = os.path.join(models_dir, edit_obj_info[obj])
        else:
            model_path = os.path.join(models_dir, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path, graph[obj]['transform'])
        mesh = mesh_obj.load(with_transform=True)

        # highlight node if its important
        if obj in query_objects and sub:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#1E90FF")

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


# load accepted cats
accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')

# scene names for query, top1, top2 and top3
scene_names = ['wc2JMjhGNzB_room23', 'pa4otMbVnkk_room19']
q_and_context_per_scene = [['1', '4', '48', '52'], ['26', '32', '43']]

# path to the mesh models and scene graph.
models_dir = '../data/matterport3d/models'
scene_dir = '../data/matterport3d/scenes/all'

for i, scene_name in enumerate(scene_names):
    # if i == 0:
    #     continue
    q_and_context = q_and_context_per_scene[i]
    scene_graph = load_from_json(os.path.join(scene_dir, scene_name + '.json'))
    # prepare the scene
    scene = visualize_scene(scene_graph, scene_graph.keys(), models_dir, q_and_context,
                            ceiling_cats=['ceiling', 'void'])
    # prepare the subscene
    subscene = visualize_scene(scene_graph, scene_graph.keys(), models_dir, q_and_context,
                               ceiling_cats=['ceiling', 'void'], sub=True)

    # edit the
    subscene_edited = None
    if i == 1:
        subscene_edited = visualize_scene(scene_graph, scene_graph.keys(), models_dir, q_and_context,
                                          ceiling_cats=['ceiling', 'void'], sub=True,
                                          edit_obj_info={'43': 'RPmz2sHmrrY_room5-30.ply',
                                                         '32': 'pa4otMbVnkk_room7-19.ply'})
        scene_edited = visualize_scene(scene_graph, scene_graph.keys(), models_dir, q_and_context,
                                       ceiling_cats=['ceiling', 'void'], sub=False,
                                       edit_obj_info={'43': 'RPmz2sHmrrY_room5-30.ply',
                                                      '32': 'pa4otMbVnkk_room7-19.ply'})

    # apply transformation
    alpha = 0 * np.pi / 180
    beta = 0 * np.pi / 180
    if i == 0:
        gamma = -30 * np.pi / 180
    else:
        gamma = -45 * np.pi / 180
    transform_scene(scene)
    transform_scene(subscene)
    scene.show()
    subscene.show()
    if subscene_edited is not None:
        transform_scene(subscene_edited)
        transform_scene(scene_edited)
        subscene_edited.show()
        scene_edited.show()




