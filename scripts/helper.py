import os
import gc
import shutil
import json
import numpy as np
import trimesh
from PIL import Image
import cv2

from .renderer import Render
from .mesh import Mesh


def load_from_json(path, mode='r'):
    with open(path, mode) as f:
        return json.load(f)


def write_to_json(dictionary, path, mode='w', indent=4):
    with open(path, mode) as f:
        json.dump(dictionary, f, indent=indent)


def draw_bbox_img(img, bbox_info, resolution, fov, camera_pose, theta, expansion_factor=1.0, arrow_pixel=30):
    # unload the bbox dimensions and the vector connecting the centroid of the scene to the centroid of the box.
    bbox_dims, vec = bbox_info

    # convert distances to pixels
    room_size = np.tan(fov/2) * camera_pose[2, 3] * 2
    vec_pixel = vec * resolution[0] / room_size #* np.sqrt(2)
    vec_pixel = np.round(vec_pixel).astype(int)

    # find the pixel for the center of the subscene
    subscene_center_pixel = [resolution[0] // 2 + vec_pixel[0], resolution[1] // 2 - vec_pixel[1]]

    # convert bbox dims to pixels
    bbox_x_pixel = bbox_dims[0] / room_size * resolution[0] * expansion_factor * np.sqrt(2)
    bbox_y_pixel = bbox_dims[1] / room_size * resolution[1] * expansion_factor * np.sqrt(2)

    # find all the corners of the bbox in pixel values
    pts = np.zeros((4, 3), dtype=np.float)
    # top left
    pts[0, :] = [-bbox_x_pixel, -bbox_y_pixel, 1]
    # top right
    pts[1, :] = [bbox_x_pixel, -bbox_y_pixel, 1]
    # bottom right
    pts[2, :] = [bbox_x_pixel, bbox_y_pixel, 1]
    # bottom left
    pts[3, :] = [-bbox_x_pixel, bbox_y_pixel, 1]

    # rotate the box around the origin
    rotation = np.asarray([[np.cos(-theta), -np.sin(-theta), 0],
                           [np.sin(-theta), np.cos(-theta), 0],
                           [0, 0, 1]])
    rot_pts = np.dot(pts, rotation)

    # translate the box to where it should be
    rot_trans_pts = np.zeros((4, 2), dtype=np.int)
    for i in range(4):
        rot_trans_pts[i, :] = rot_pts[i, :2] + np.asarray(subscene_center_pixel)

    # draw the bbox
    img = cv2.polylines(img, [rot_trans_pts], True, (0, 255, 0), 2)

    # find the x-axis of the local frame and add it
    x_axis = [int(np.round(arrow_pixel * np.cos(theta))), int(np.round(arrow_pixel * np.sin(theta)))]
    end_point = (subscene_center_pixel[0] + x_axis[0], subscene_center_pixel[1] + x_axis[1])
    img = cv2.arrowedLine(img, (subscene_center_pixel[0], subscene_center_pixel[1]), end_point, color=(0, 0, 255),
                          thickness=3)

    # find the y-axis of the local frame and add it
    y_axis = [int(np.round(arrow_pixel * np.cos(theta - np.pi/2))), int(np.round(arrow_pixel * np.sin(theta - np.pi/2)))]
    end_point = (subscene_center_pixel[0] + y_axis[0], subscene_center_pixel[1] + y_axis[1])
    img = cv2.arrowedLine(img, (subscene_center_pixel[0], subscene_center_pixel[1]), end_point, color=(205, 0, 0),
                          thickness=3)

    return img


def render_single_scene(graph, objects, highlighted_object, path, model_dir, colormap, resolution=(512, 512),
                        faded_nodes=[], rendering_kwargs=None, alpha=0, beta=0, gamma=0):
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
        img = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose, room_dimension=room_dimension)
        imgs.append(img)

    # put images of the full and subscene side-by-side
    width, height = img.shape[:-1]
    new_img = Image.new('RGB', (2*width, height))
    x_offset = 0
    for img in imgs:
        img = Image.fromarray(img)
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    new_width, new_height = new_img.size
    new_img = new_img.resize((new_width//2, new_height//2))

    # save the side-by-side image
    new_img.save(path)


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
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#1E90FF")

        # faded color if object is not important
        if obj in faded_nodes:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)

        # add elements to the scene and cropped scene (if necessary)
        scene.append(mesh)
        if crop and (obj not in faded_nodes):
            cropped_scene.append(mesh)

        del mesh
        gc.collect()

    # extract the room dimension and the camera pose
    scene = trimesh.Scene(scene)
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]

    # if the scene is cropped, camera pose is rotated by theta and room dimension is extracted from the subscene.
    if crop:
        cropped_scene = trimesh.Scene(cropped_scene)
        bbox_dims = cropped_scene.extents[:2] / 2.0
        vec = cropped_scene.bounding_box.centroid - scene.bounding_box.centroid
        vec = vec[:2]
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


def create_train_val_test(data_dir, train_path, val_path, test_path):
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
        house_name = scene_name.split('_')[0]
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


def sample_mesh(mesh, count=1000):
    """
    Sample points from the mesh.
    :param mesh: Mesh representing the 3d object.
    :param count: Number of query points/
    :return: Sample points on the mesh and the face index corresponding to them.
    """
    faces_idx = np.zeros(count, dtype=int)
    points = np.zeros((count, 3), dtype=float)
    # pick a triangle randomly promotional to its area
    cum_area = np.cumsum(mesh.area_faces)
    random_areas = np.random.uniform(0, cum_area[-1]+0.001, count)
    for i in range(count):
        face_idx = np.argmin(np.abs(cum_area - random_areas[i]))
        faces_idx[i] = face_idx
        r1, r2, = np.random.uniform(0, 1), np.random.uniform(0, 1)
        triangle = mesh.triangles[face_idx, ...]
        point = (1 - np.sqrt(r1)) * triangle[0, ...] + \
            np.sqrt(r1) * (1 - r2) * triangle[1, ...] + \
            np.sqrt(r1) * r2 * triangle[2, ...]
        points[i, :] = point
    return points, faces_idx

