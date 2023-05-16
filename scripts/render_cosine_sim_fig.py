import os
import numpy as np
import trimesh
from PIL import Image

from render_scene_functions import render_single_pc, render_single_mesh


def load_pc(pc_name):
    pc = np.load(os.path.join(pc_dir, pc_name))
    radii = np.linalg.norm(pc, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    pc_trimesh = trimesh.points.PointCloud(pc, colors=colors)

    return pc_trimesh


def apply_transformation(pc):
    transformation_z = trimesh.transformations.rotation_matrix(angle=alpha, direction=[0, 0, 1],
                                                               point=[0, 0, 0])
    transformation_y = trimesh.transformations.rotation_matrix(angle=beta, direction=[0, 1, 0],
                                                               point=[0, 0, 0])
    transformation_x = trimesh.transformations.rotation_matrix(angle=gamma, direction=[1, 0, 0],
                                                               point=[0, 0, 0])

    transformation = np.matmul(np.matmul(transformation_z, transformation_y), transformation_x)
    pc.apply_transform(transformation)

    return pc


def render_pc(pc, file_name):
    img_pc = render_single_pc(pc.vertices, resolution, rendering_kwargs)
    img_pc = Image.fromarray(img_pc)
    img_pc.save(os.path.join('../figures/ObjectObjectSim', file_name))
    img_pc.show()


# path to the mesh region and pc
q_2_file_name = '5ZKStnWn8Zo_room11-50.npy'
t_2_file_name = '5ZKStnWn8Zo_room25-5.npy'
t_3_file_name = 'WYY7iVyf5p8_room14-14.npy'
q_3_file_name = '5ZKStnWn8Zo_room25-22.npy'
t_4_file_name = '5ZKStnWn8Zo_room25-2.npy'
pc_dir = '../data/matterport3d/pc_regions/test'

# load pc for q_2, t_2 and t_3.
q_2 = load_pc(q_2_file_name)
t_2 = load_pc(t_2_file_name)
t_3 = load_pc(t_3_file_name)
q_3 = load_pc(q_3_file_name)
t_4 = load_pc(t_4_file_name)

# build a transformation for rendering the point clouds.
alpha = 10 * np.pi / 180
beta = 10 * np.pi / 180
gamma = -120

q_2 = apply_transformation(q_2)
q_2.show()
t_2 = apply_transformation(t_2)
t_2.show()
t_3 = apply_transformation(t_3)
t_3.show()

alpha = 0 * np.pi / 180
beta = -35 * np.pi / 180
gamma = -90 * np.pi / 180

q_3 = apply_transformation(q_3)
q_3.show()
t_4 = apply_transformation(t_4)
t_4.show()

# render the pc.
rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                    'wall_thickness': 0}
resolution = (512, 512)
file_name_to_pc = {q_2_file_name: q_2, t_2_file_name: t_2, t_3_file_name: t_3, q_3_file_name: q_3, t_4_file_name: t_4}
for file_name, pc in file_name_to_pc.items():
    render_pc(pc, file_name.split('.')[0] + '.png')

