import os
import sys
import shutil
from collections import Counter
import numpy as np
import trimesh

from scripts.mesh import Mesh
from scripts.helper import load_from_json, sample_mesh, visualize_labled_pc, create_img_table
from render_scene_functions import render_pc, render_mesh


def find_majority_label(vertices, point, face, segIndices):
    # find the closest vertex to the point
    min_dist = 1000
    closest_vertex = None
    for v in face:
        if np.linalg.norm(vertices[v, :] - point) < min_dist:
            closest_vertex = v

    return segIndices[closest_vertex]


def save_pc(pc, results_dir, model_name):
    path = os.path.join(results_dir, model_name)
    np.save(path, pc)


def visualize(pc):
    trimesh.points.PointCloud(pc).show()


def build_cube_mesh(s_x, e_x, s_y, e_y, s_z, e_z):
    # compute scale and the centroid of the cube
    beg_points = np.array([s_x, s_y, s_z])
    end_points = np.array([e_x, e_y, e_z])
    scale = end_points - beg_points
    centroid = (beg_points + end_points) / 2.0

    # build the cube
    transformation = np.eye(4)
    transformation[:3, 3] = centroid
    cube = trimesh.creation.box(scale, transform=transformation)

    return cube


def find_submeshes(mesh, cube):
    # mesh.show()
    # cube.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
    # trimesh.Scene([mesh, cube]).show()
    # read the original vertices
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # find vertices that are inside the cube
    inside_vertices = np.abs(vertices - cube.centroid) < (cube.extents / 2.0)
    inside_vertices = np.sum(inside_vertices, axis=1) == 3
    filtered_vertices = vertices[inside_vertices, :]

    # find the corresponding face for the inside vertices
    inside_faces = np.arange(len(vertices))[inside_vertices]

    # find a map that sends the inside faces in the old mesh to the face ids in the submesh.
    face_id_map = dict(zip(inside_faces, np.arange(len(filtered_vertices))))

    # filter the faces to only contain the submesh vertices.
    filtered_faces = []
    for i, face in enumerate(faces):
        if np.sum([True if f in face_id_map else False for f in face]) == 3:
            new_face = [face_id_map[f] for f in face]
            filtered_faces.append(new_face)

    # build the submesh from the filtered vertices and faces
    submesh = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces)

    return submesh


class PointCloud:
    def __init__(self, model_name):
        # load the mesh
        self.model_name = model_name

    def sample(self, sampling_factor=5000, with_label=False):
        # sample points on the mesh
        mesh_path = os.path.join(models_dir, self.model_name)
        mesh = trimesh.load(mesh_path, process=False)

        # find the number of points to be sampled.
        num_points = int(sampling_factor * np.sqrt(mesh.area))
        pc, sampled_faces_idx = sample_mesh(mesh, num_points=num_points)

        # use the face ids to extract the vertices of the face that the point was sampled from. choose the label with
        # highest vote among the thre vertices.
        labels = np.zeros(len(pc))
        mesh_faces = mesh.faces
        mesh_vertices = mesh.vertices
        if with_label:
            # load the over-segmentations for the room
            file_name = self.model_name.split('/')[0] + '.annotated.{}.segs.json'.format(kThreshold)
            seg_dict = load_from_json(os.path.join(results_dir_over_segments, file_name))
            for i, face_idx in enumerate(sampled_faces_idx):
                face = mesh_faces[face_idx]
                labels[i] = find_majority_label(mesh_vertices, pc[i, :], face, seg_dict['segIndices'])

        return pc, labels


class PointCloudOverlapping:
    def __init__(self, models_dir, model_name, num_points=4096, block_size=2.0, stride=1.0, padding=0.001):
        self.models_dir = models_dir
        self.model_name = model_name

        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride
        self.padding = padding

    def sample(self):
        # load the mesh
        mesh_path = os.path.join(self.models_dir, self.model_name)
        mesh = Mesh(mesh_path).load()

        # find the mind and max vertex coordinates of the mesh along each axis x, y and z.
        coord_min, coord_max = np.amin(mesh.vertices, axis=0)[:3], np.amax(mesh.vertices, axis=0)[:3]

        # setup a grid in the x-y plane for sampling point clouds.
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)

        # make sure the grids are never empty
        grid_x = np.maximum(grid_x, 1)
        grid_y = np.maximum(grid_y, 1)

        # find the extents of the mesh in the z direction
        mesh_bounds = mesh.bounds
        s_z = mesh_bounds[0, 2]
        e_z = mesh_bounds[1, 2]

        # find the coordinates of the grid in the x-y plane and store all the cube blocks
        pc_room = np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                # find the start and end x-y coordinates for the block.
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size

                # create a cube mesh representing the block and store it.
                cube = build_cube_mesh(s_x - self.padding, e_x + self.padding, s_y - self.padding, e_y + self.padding,
                                       s_z, e_z)

                # find the submesh enclosed by each cube block in the x-y plan along the entire z-axis
                submesh = find_submeshes(mesh, cube)

                # skip if the submesh is empty
                if len(submesh.area_faces) == 0:
                    continue

                # sample point clouds within the submesh extracted in the block
                pc_block, _ = sample_mesh(submesh, num_points=self.num_points)

                # update the point cloud for the entire 3D scene
                pc_room = np.vstack([pc_room, pc_block]) if pc_room.size else pc_block

        return pc_room


def derive_pc(room_names_chunk, with_label):
    for model_name in room_names_chunk:
        visited = set(os.listdir(results_dir_pc))
        output_model_name = model_name.split('/')[-1].split('.')[0] + '.npy'
        if output_model_name not in visited:
            if overlapping:
                pc_object = PointCloudOverlapping(models_dir, model_name)
            else:
                pc_object = PointCloud(model_name)

            # sample point clouds
            pc, labels = pc_object.sample(with_label=with_label)
            # visualize(pc)
            # visualize_labled_pc(pc, labels)
            # t=y
            # save point clouds and labels
            save_pc(pc, results_dir_pc, output_model_name)
            if with_label:
                save_pc(labels, results_dir_pc_labels, output_model_name)

            visited.add(output_model_name)


def main(num_chunks, chunk_idx, action='extract_pc'):
    if action == 'extract_pc':
        # extract point clouds either from a region or entire scene
        with_label = True
        chunk_size = int(np.ceil(len(room_names) / num_chunks))
        derive_pc(room_names_chunk=room_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                  with_label=with_label)
    elif action == 'render':
        # render the pc
        pc_file_names = os.listdir(results_dir_pc)[:num_imgs]
        render_pc(results_dir_pc, pc_file_names, results_dir_rendered, resolution, rendering_kwargs)

    elif action == 'create_img_table':
        imgs = os.listdir(results_dir_rendered)
        imgs = sorted(imgs)
        create_img_table(results_dir_rendered, 'imgs', imgs, 'img_table.html', ncols=4, captions=imgs, topk=40)
    elif action == 'save_room_subset':
        pc_file_names = os.listdir(results_dir_pc)[:num_imgs]
        out_dir = 'pc_rooms_subset'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for pc_file_name in pc_file_names:
            shutil.copy(os.path.join(results_dir_pc, pc_file_name), os.path.join(out_dir, pc_file_name))


if __name__ == '__main__':
    # set up rendering parameters
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    num_imgs = 20
    mesh_dir = '/media/reza/Large/Matterport3D_rooms/rooms'

    # set up paths and params
    overlapping = False
    kThreshold = '0.010000'
    results_dir_pc = '/media/reza/Large/Matterport3D_rooms/rooms_pc'
    results_dir_pc_labels = '/media/reza/Large/Matterport3D_rooms/rooms_pc_labels'
    results_dir_rendered = os.path.join('../data/matterport3d/pc_rooms_rendered/imgs')
    results_dir_over_segments = os.path.join('/media/reza/Large/Matterport3D_rooms/rooms_over_segments')
    models_dir = '/media/reza/Large/Matterport3D_rooms/rooms'
    room_names = os.listdir(models_dir)
    # np.random.seed(21)
    # np.random.shuffle(room_names)

    # skip rooms with no mesh
    room_names = [os.path.join(room_name, '{}.annotated.ply'.format(room_name)) for room_name in room_names if
                  len(os.listdir(os.path.join(models_dir, room_name))) > 0]

    for folder in [results_dir_pc, results_dir_pc_labels, results_dir_rendered]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except FileExistsError:
                pass

    if len(sys.argv) == 1:
        main(1, 0, 'extract_pc')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u extract_point_clouds.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_pc
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
