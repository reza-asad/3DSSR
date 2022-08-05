import os
import gc
import shutil
import random
import json
import numpy as np
import trimesh
from PIL import Image
from matplotlib import pyplot as plt

from .renderer import Render
from .mesh import Mesh


def load_from_json(path, mode='r'):
    with open(path, mode) as f:
        return json.load(f)


def write_to_json(dictionary, path, mode='w', indent=4):
    with open(path, mode) as f:
        json.dump(dictionary, f, indent=indent)


def vanilla_plot(values, cp_dir, plot_label='Train', xlabel='Epoch', ylabel='Loss', plot_name='loss.png',
                 with_legend=False, scatter=False):
    # plot
    if scatter:
        plt.scatter(range(1, len(values)+1), values, label=plot_label)
    else:
        plt.plot(range(1, len(values)+1), values, label=plot_label)

    # add label
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if with_legend:
        plt.legend()

    # save the plot
    plt.savefig(os.path.join(cp_dir, plot_name))


def render_single_scene(graph, objects, highlighted_object, path, model_dir, colormap, resolution=(512, 512),
                        faded_nodes=[], rendering_kwargs=None, alpha=0, beta=0, gamma=0, with_height_offset=True):
    # setup default rendering conditions such as lighting
    if rendering_kwargs is None:
        rendering_kwargs = {'fov': np.pi/4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                            'wall_thickness': 5}

    # prepare scene, camera pose and the room dimensions and render the entire scene.
    r = Render(rendering_kwargs)
    scene, camera_pose, room_dimension = prepare_scene_for_rendering(graph, objects, models_dir=model_dir,
                                                                     query_objects=highlighted_object,
                                                                     faded_nodes=faded_nodes, colormap=colormap,
                                                                     crop=False, alpha=alpha, beta=beta, gamma=gamma)
    if scene is not None:
        img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose, room_dimension=room_dimension,
                                   with_height_offset=with_height_offset)
        # save the side-by-side image
        Image.fromarray(img).save(path)


def render_scene_subscene(graph, objects, highlighted_object, path, model_dir, colormap, resolution=(512, 512),
                          faded_nodes=[], rendering_kwargs=None, alpha=0, beta=0, gamma=0, with_height_offset=True):
    # setup default rendering conditions such as lighting
    if rendering_kwargs is None:
        rendering_kwargs = {'fov': np.pi/4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                            'wall_thickness': 5}

    # prepare scene, camera pose and the room dimensions and render the entire scene as well as the cropped and aligned
    # subscene
    imgs = []
    for crop in [False, True]:
        r = Render(rendering_kwargs)
        scene, camera_pose, room_dimension = prepare_scene_for_rendering(graph, objects, models_dir=model_dir,
                                                                         query_objects=highlighted_object,
                                                                         faded_nodes=faded_nodes, colormap=colormap,
                                                                         crop=crop, alpha=alpha, beta=beta, gamma=gamma)
        # do no render if either the scene or subscene is empty.
        if scene is None:
            return
        else:
            img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose,
                                       room_dimension=room_dimension, with_height_offset=with_height_offset)
            imgs.append(img)

    # put the scene image in the top left corner and the subscene in the bottom right corner.
    small_img_shape = (128, 128)
    width, height = img.shape[:-1]
    new_img = Image.new('RGB', (width, height + small_img_shape[1]), color=(255, 255, 255))
    img_scene, img_subscene = Image.fromarray(imgs[0]), Image.fromarray(imgs[1])
    img_scene = img_scene.resize(small_img_shape)
    new_img.paste(img_scene, (0, 0))
    new_img.paste(img_subscene, (0, small_img_shape[1]))
    new_img.save(path)


def render_subscene(graph, objects, highlighted_object, path, model_dir, colormap, resolution=(512, 512),
                    faded_nodes=[], rendering_kwargs=None, alpha=0, beta=0, gamma=0, with_height_offset=True):
    # setup default rendering conditions such as lighting
    if rendering_kwargs is None:
        rendering_kwargs = {'fov': np.pi/4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                            'wall_thickness': 5}

    # prepare scene, camera pose and the room dimensions and render the cropped subscene
    r = Render(rendering_kwargs)
    scene, camera_pose, room_dimension = prepare_scene_for_rendering(graph, objects, models_dir=model_dir,
                                                                     query_objects=highlighted_object,
                                                                     faded_nodes=faded_nodes, colormap=colormap,
                                                                     crop=True, alpha=alpha, beta=beta, gamma=gamma)
    if scene is not None:
        img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose,
                                   room_dimension=room_dimension, with_height_offset=with_height_offset)

        # save the side-by-side image
        Image.fromarray(img).save(path)


def prepare_scene_for_rendering(graph, objects, models_dir, query_objects=[], faded_nodes=[], colormap={},
                                ceiling_cats=['ceiling', 'void'], crop=False, alpha=0, beta=0, gamma=0):
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
        mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)

        # find color based on category
        if cat in colormap:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(colormap[cat])

        # highlight node if its important
        if obj in query_objects:
            # "#8A2BE2"
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#8A2BE2")

        # faded color if object is not important
        if obj in faded_nodes:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)

        # add elements to the scene and cropped scene (if necessary)
        scene.append(mesh)
        if crop and (obj not in faded_nodes):
            cropped_scene.append(mesh)

        del mesh
        gc.collect()

    if len(scene) == 0:
        return None, None, None

    # extract the room dimension and the camera pose
    scene = trimesh.Scene(scene)
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]

    # if the scene is cropped, camera pose is rotated by theta and room dimension is extracted from the subscene.
    if crop:
        if len(cropped_scene) == 0:
            return None, None, None
        cropped_scene = trimesh.Scene(cropped_scene)
        room_dimension = cropped_scene.extents
        camera_pose, _ = cropped_scene.graph[cropped_scene.camera.name]
        transformation_z = trimesh.transformations.rotation_matrix(angle=alpha, direction=[0, 0, 1],
                                                                   point=cropped_scene.centroid)
        transformation_y = trimesh.transformations.rotation_matrix(angle=beta, direction=[0, 1, 0],
                                                                   point=cropped_scene.centroid)
        transformation_x = trimesh.transformations.rotation_matrix(angle=gamma, direction=[1, 0, 0],
                                                                   point=cropped_scene.centroid)

        transformation = np.matmul(np.matmul(transformation_z, transformation_y), transformation_x)
        scene.apply_transform(transformation)

    return scene, camera_pose, room_dimension


def create_img_table(img_dir, img_folder, imgs, html_file_name, topk=25, ncols=5, captions=[],
                     with_query_scene=False, evaluation_plot=None, query_img=None, query_caption=None):
    def insert_into_table(file, img_name, caption=None):
        img_path = os.path.join(img_folder, img_name)
        file.write('\n')
        file.write('<td align="center" valign="center">\n')
        file.write('<img src="{}" />\n'.format(img_path))
        file.write('<br />\n')
        # add caption
        file.write('<br />\n')
        if caption is not None:
            file.write(img_name)
            file.write('<br />\n')
            file.write(caption)
            file.write('<br />\n')
        file.write('</td>\n')
        file.write('\n')

    img_dir_parent = '/'.join(img_dir.split('/')[:-1])
    html_file_dir = os.path.join(img_dir_parent, html_file_name)
    with open(html_file_dir, 'w+') as f:
        # add the table
        f.write('<table width="500" border="0" cellpadding="5">\n')

        # insert the query into the table
        if with_query_scene:
            f.write('<tr>\n')
            insert_into_table(f, query_img, query_caption)
            # insert the evaluation plot for the query, if necessary.
            if evaluation_plot is not None:
                insert_into_table(f, evaluation_plot, caption=None)
            f.write('</tr>\n')

        # add the rows of the table
        nrows = int(np.ceil(topk/ncols))
        for i in range(nrows):
            f.write('<tr>\n')
            # add the rendered scenes
            for j in range(ncols):
                if i*ncols+j >= topk or i*ncols+j >= len(imgs):
                    break
                if len(captions) > 0:
                    insert_into_table(f, imgs[i*ncols+j], captions[i*ncols+j])
                else:
                    insert_into_table(f, imgs[i * ncols + j])
            f.write('</tr>\n')

        # end the table
        f.write('</table>\n')


def create_img_table_scrollable(img_dir, img_folder, imgs, html_file_name, query_img, topk=25, ncols=7, captions=[],
                                query_caption=None):
    def insert_into_table(file, img_name, caption=None):
        img_path = os.path.join(img_folder, img_name)
        file.write('\n')
        file.write('<td align="center" valign="center">\n')
        file.write('<img src="{}" />\n'.format(img_path))
        file.write('<br />\n')
        # add caption
        if caption is None:
            file.write(img_name)
        else:
            file.write(caption)
        file.write('</td>\n')
        file.write('\n')

    img_dir_parent = '/'.join(img_dir.split('/')[:-1])
    html_file_dir = os.path.join(img_dir_parent, html_file_name)
    with open(html_file_dir, 'w+') as f:
        # initialize the html file and add a div class
        f.write('<!DOCTYPE html><html><head><link rel="stylesheet" href="mystyle.css"></head><body>\n')
        f.write('<div class="tableFixHead">\n')
        # add the table
        f.write('<table width="500" border="0" cellpadding="5">\n')

        # insert the query into the table with table head
        f.write('<thead>\n')
        f.write('<tr>\n')
        f.write('<th align="center" valign="center">\n')
        img_path = os.path.join(img_folder, query_img)
        f.write('<img src="{}" />\n'.format(img_path))
        if query_caption is not None:
            f.write('<br />\n')
            f.write(query_caption)
        f.write('</th>\n')
        f.write('</tr>\n')
        f.write('</thead>\n')

        # add the rows of the table
        nrows = int(np.ceil(topk/ncols))
        for i in range(nrows):
            f.write('<tr>\n')
            f.write('<td></td>\n')
            # add the rendered scenes
            for j in range(ncols):
                if i*ncols+j >= topk or i*ncols+j >= len(imgs):
                    break
                if len(captions) > 0:
                    insert_into_table(f, imgs[i*ncols+j], captions[i*ncols+j])
                else:
                    insert_into_table(f, imgs[i * ncols + j])
            f.write('</tr>\n')

        # end the table
        f.write('</table>\n')
        f.write('<tr>\n')
        f.write('<td></td>\n')
        for i in range(15):
            f.write('<br />\n')
        f.write('</tr>\n')

    # add CSS
    css_file_dir = os.path.join(img_dir_parent, 'mystyle.css')
    with open(css_file_dir, 'w+') as f:
        f.write('.tableFixHead          { overflow-y: auto; height: 950px;}\n')
        f.write('.tableFixHead thead th { position: sticky; top: 0; }')


def create_train_val_test(data_dir, train_path, val_path, test_path, split_char='_'):
    # make sure the 3 folders exist
    folder_to_path = {'train': train_path, 'val': val_path, 'test': test_path}
    for folder in folder_to_path.keys():
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)

    # for each house find out which folder (train, val and test) it belongs to
    house_to_folder = {}
    for folder, path in folder_to_path.items():
        with open(path, 'r') as f:
            house_names = f.readlines()
        for house_name in house_names:
            house_name = house_name.strip()
            house_to_folder[house_name] = folder

    # for each scene find out which folder it belongs to and copy it there
    scene_names = os.listdir(os.path.join(data_dir, 'all'))
    for scene_name in scene_names:
        house_name = scene_name.split('.')[0]
        if split_char is not None:
            house_name = house_name.split(split_char)[0]
        folder = house_to_folder[house_name]
        d1 = os.path.join(data_dir, 'all', scene_name)
        d2 = os.path.join(data_dir, folder, scene_name)
        shutil.copy(d1, d2)


def prepare_mesh_for_scene(models_dir, graph, obj):
    model_path = os.path.join(models_dir, graph[obj]['file_name'])
    mesh_obj = Mesh(model_path, graph[obj]['transform'])
    return mesh_obj.load(with_transform=True)


def visualize_scene(scene_graph_dir, models_dir, scene_name, accepted_cats=set(), objects=[], highlighted_objects=[],
                    with_backbone=True, as_obbox=False):
    # load the graph and accepted cats
    graph = load_from_json(os.path.join(scene_graph_dir, scene_name))

    scene = []
    visited = set()
    # if true the backbone of the scene is included
    if with_backbone and len(accepted_cats) > 0:
        for obj in graph.keys():
            cat = graph[obj]['category'][0]
            if cat not in accepted_cats:
                mesh = prepare_mesh_for_scene(models_dir, graph, obj)
                scene.append(mesh)
                visited.add(obj)

    # if no objects specified the entire scene is visualized
    if len(objects) == 0:
        for obj in graph.keys():
            if obj not in visited:
                objects.append(obj)

    # include the requested objects in the visualization
    for obj in objects:
        mesh = prepare_mesh_for_scene(models_dir, graph, obj)
        if obj in highlighted_objects:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
        if as_obbox:
            scene.append(mesh.bounding_box)
        else:
            scene.append(mesh)
    scene = trimesh.Trimesh.scene(scene)
    scene.show()


def sample_mesh(mesh, num_points=1000):
    """
    Sample points from the mesh.
    :param mesh: Mesh representing the 3d object.
    :param count: Number of query points/
    :return: Sample points on the mesh and the face index corresponding to them.
    """
    faces_idx = np.zeros(num_points, dtype=int)
    points = np.zeros((num_points, 3), dtype=float)
    # pick a triangle randomly proportional to its area
    cum_area = np.cumsum(mesh.area_faces)
    random_areas = np.random.uniform(0, cum_area[-1]+0.001, num_points)
    for i in range(num_points):
        face_idx = np.argmin(np.abs(cum_area - random_areas[i]))
        faces_idx[i] = face_idx
        r1, r2, = np.random.uniform(0, 1), np.random.uniform(0, 1)
        triangle = mesh.triangles[face_idx, ...]
        point = (1 - np.sqrt(r1)) * triangle[0, ...] + \
            np.sqrt(r1) * (1 - r2) * triangle[1, ...] + \
            np.sqrt(r1) * r2 * triangle[2, ...]
        points[i, :] = point
    return points, faces_idx


def visualize_labled_pc(pc, labels, center_segment=None):
    # define a color map
    pc_colors = np.zeros((len(pc), 4))
    number_of_colors = len(np.unique(labels))
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    color_map = {e: colors[i] for i, e in enumerate(np.unique(labels))}

    # make the center segment black.
    if center_segment is not None:
        for k in color_map.keys():
            if k != center_segment:
                color_map[k] = "#000000"

    # color the points
    for i, seg_idx in enumerate(labels):
        pc_colors[i, :] = trimesh.visual.color.hex_to_rgba(color_map[seg_idx])

    trimesh.points.PointCloud(pc, colors=pc_colors).show()


def visualize_pc(pc):
    radii = np.linalg.norm(pc, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    trimesh.points.PointCloud(pc, colors=colors).show()
