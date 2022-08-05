import os
import numpy as np
import trimesh
from PIL import Image

from mesh import Mesh
from scripts.helper import load_from_json
from renderer import Render


def prepare_scene_for_rendering(graph, objects, models_dir, query_object, context_objects=[],
                                ceiling_cats=['ceiling', 'void'], with_color=False, with_boxes=False, with_crops=False):
    default_color = '#aec7e8'
    scene = []
    cropped_scene = []
    for obj in objects:
        cat = graph[obj]['category'][0]

        # if the mesh is a ceiling mesh skip
        if cat in ceiling_cats:
            continue

        # load the mesh
        model_path = os.path.join(models_dir, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path, graph[obj]['transform'])
        mesh = mesh_obj.load(with_transform=True)
        if cat in accepted_cats:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba('#FAA0A0')
        else:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)

        # highlight node if its important
        if with_color and obj == query_object:
            if with_boxes:
                box = mesh.bounding_box
                box.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#8A2BE2")
                scene.append(box)
            else:
                mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#8A2BE2")
                if with_crops:
                    cropped_scene.append(mesh)

        # faded color if object is not important
        if with_color and obj in context_objects:
            if with_boxes:
                box = mesh.bounding_box
                box.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#1E90FF")
                scene.append(box)
            else:
                mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#1E90FF")
                if with_crops:
                    cropped_scene.append(mesh)

        # add elements to the scene and cropped scene (if necessary)
        scene.append(mesh)

    # extract the room dimension and the camera pose
    scene = trimesh.Scene(scene)
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]

    if with_crops:
        cropped_scene = trimesh.Scene(cropped_scene)
        room_dimension = cropped_scene.extents
        camera_pose, _ = cropped_scene.graph[cropped_scene.camera.name]


    return scene, camera_pose, room_dimension


def render_scene(camera_pose, room_dimension):
    r = Render(rendering_kwargs)
    img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose,
                               room_dimension=room_dimension, with_height_offset=True)

    return img


# load accepted cats
accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')

# scene names for query, top1, top2 and top3
query_scene_name = 'wc2JMjhGNzB_room23'
ranked_scene_names = ['pa4otMbVnkk_room19', 'jtcxE69GiFV_room11', 'ARNzJeq3xxb_room4']
q_and_context_per_rank = [['26', '32', '43'], ['3', '16', '34'], ['6', '5']]
target_scene_names = ['jtcxE69GiFV_room11', 'RPmz2sHmrrY_room4', 'RPmz2sHmrrY_room4', 'pa4otMbVnkk_room18']

# path to the mesh models and scene graph.
file_name = 'wc2JMjhGNzB_room23-4'
models_dir = '../data/matterport3d/models'
scene_dir = '../data/matterport3d/scenes/test'

# rendering arguments.
rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                    'wall_thickness': 10.0}
resolution = (512, 512)

####################### render the query img without colors ##########################.
query_and_context = ['1', '4', '48']
query_scene = load_from_json(os.path.join(scene_dir, query_scene_name + '.json'))
scene, camera_pose, room_dimension = prepare_scene_for_rendering(query_scene, query_scene.keys(), models_dir,
                                                                 query_and_context[0],
                                                                 context_objects=query_and_context[1:],
                                                                 ceiling_cats=['ceiling', 'void'], with_color=False,
                                                                 with_boxes=False)
img_query = render_scene(camera_pose, room_dimension)
img_query = Image.fromarray(img_query)
img_query.save(os.path.join('../figures/3DSSR Overview', 'T0.png'))
# img_query.show()

####################### render the query subscene img with colors ##########################.
scene, camera_pose, room_dimension = prepare_scene_for_rendering(query_scene, query_scene.keys(), models_dir,
                                                                 query_and_context[0],
                                                                 context_objects=query_and_context[1:],
                                                                 ceiling_cats=['ceiling', 'void'], with_color=True,
                                                                 with_crops=True)
img_query = render_scene(camera_pose, room_dimension)
img_query = Image.fromarray(img_query)
img_query.save(os.path.join('../figures/3DSSR Overview', 'Q.png'))
# img_query.show()

####################### render T1, T2, T3 and T4 without colors ##########################.
for i, target_scene_name in enumerate(target_scene_names):
    target_scene = load_from_json(os.path.join(scene_dir, target_scene_name + '.json'))
    scene, camera_pose, room_dimension = prepare_scene_for_rendering(target_scene, target_scene.keys(), models_dir,
                                                                     query_object='',
                                                                     context_objects=[],
                                                                     ceiling_cats=['ceiling', 'void'], with_color=False)
    img_target = render_scene(camera_pose, room_dimension)
    img_target = Image.fromarray(img_target)
    img_target.save(os.path.join('../figures/3DSSR Overview', 'T_{}.png'.format(i+1)))
    # img_target.show()

####################### render R1, R2 and R3 with colors ##########################.
for i, ranked_scene_name in enumerate(ranked_scene_names):
    target_scene = load_from_json(os.path.join(scene_dir, ranked_scene_name + '.json'))
    q = q_and_context_per_rank[i][0]
    context = q_and_context_per_rank[i][1:]
    scene, camera_pose, room_dimension = prepare_scene_for_rendering(target_scene, target_scene.keys(), models_dir,
                                                                     query_object=q,
                                                                     context_objects=context,
                                                                     ceiling_cats=['ceiling', 'void'], with_color=True)
    img_target = render_scene(camera_pose, room_dimension)
    img_target = Image.fromarray(img_target)
    img_target.save(os.path.join('../figures/3DSSR Overview', 'R_{}.png'.format(i+1)))
    # img_target.show()



