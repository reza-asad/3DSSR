import os
import numpy as np
import trimesh
from PIL import Image

from render_scene_functions import render_single_pc, render_single_mesh

# path to the mesh region and pc
file_name = 'wc2JMjhGNzB_room23-4'
mesh_path = '../data/matterport3d/mesh_regions/test/{}.ply'.format(file_name)
pc_dir = '../data/matterport3d/pc_regions/test/{}.npy'.format(file_name)

# load the mesh
mesh = trimesh.load(mesh_path)
# mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#83c2bc")
radii = np.linalg.norm(mesh.vertices - mesh.center_mass, axis=1)
mesh.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')

# build a transformation for rendering the mesh and point clouds.
alpha = 85 * np.pi / 180
beta = 75 * np.pi / 180
gamma = 0
transformation_z = trimesh.transformations.rotation_matrix(angle=alpha, direction=[0, 0, 1],
                                                           point=mesh.centroid)
transformation_y = trimesh.transformations.rotation_matrix(angle=beta, direction=[0, 1, 0],
                                                           point=mesh.centroid)
transformation_x = trimesh.transformations.rotation_matrix(angle=gamma, direction=[1, 0, 0],
                                                           point=mesh.centroid)

transformation = np.matmul(np.matmul(transformation_z, transformation_y), transformation_x)
mesh.apply_transform(transformation)
# mesh.show()

# show the pc.
pc = np.load(pc_dir)
radii = np.linalg.norm(pc, axis=1)
colors = trimesh.visual.interpolate(radii, color_map='viridis')
pc_trimesh = trimesh.points.PointCloud(pc, colors=colors)
pc_trimesh.apply_transform(transformation)
# pc_trimesh.show()

# render the original mesh and the pc.
rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                    'wall_thickness': 0}
resolution = (512, 512)
img_mesh = render_single_mesh(mesh, resolution, rendering_kwargs)
img_mesh = Image.fromarray(img_mesh)
img_mesh.save(os.path.join('../figures/PointCrop Overview', 'img_mesh.png'))
# img_mesh.show()
img_pc = render_single_pc(pc_trimesh.vertices, resolution, rendering_kwargs)
img_pc = Image.fromarray(img_pc)
img_pc.save(os.path.join('../figures/PointCrop Overview', 'img_pc.png'))
# img_pc.show()


def find_center_crop(point_cloud, threshold_min, threshold_max, seed):
    norm = np.linalg.norm(point_cloud, axis=1)
    indices = (threshold_min < norm) * (norm < threshold_max)

    if len(indices) > 0:
        indices = list(zip(range(len(indices)), indices))
        np.random.seed(seed)
        np.random.shuffle(indices)
        for idx, bounded in indices:
            if bounded:
                return idx

    return None


# extract a global crop.
# (pc, 0.30, 0.35, 2) and (pc, 0.15, 0.20, 4)
idx1 = find_center_crop(pc, 0.30, 0.35, 2)
if idx1 is not None:
    global_crop_bound = 0.7
    cube_center = pc[idx1, ...]
    cube_extents = pc_trimesh.bounding_box.extents * global_crop_bound / 2.0
    is_inside = np.abs(pc - cube_center) <= cube_extents
    is_inside = np.sum(is_inside, axis=1) == 3
    pc_global = pc[is_inside, ...]

    # visualize
    radii = np.linalg.norm(pc_global, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    pc_global_trimesh = trimesh.points.PointCloud(pc_global, colors=colors)
    pc_global_trimesh.apply_transform(transformation)
    # pc_global_trimesh.show()

    # render
    img_pc = render_single_pc(pc_global_trimesh.vertices, resolution, rendering_kwargs)
    img_pc = Image.fromarray(img_pc)
    img_pc.save(os.path.join('../figures/PointCrop Overview', 'img_pc_global2.png'))
    # img_pc.show()

# extract a local crop.
# (pc, 0.3, 0.35, 0) (pc, 0.3, 0.35, 2) (pc, 0.25, 0.30, 11)
idx1 = find_center_crop(pc, 0.25, 0.30, 11)
if idx1 is not None:
    global_crop_bound = 0.4
    cube_center = pc[idx1, ...]
    cube_extents = pc_trimesh.bounding_box.extents * global_crop_bound / 2.0
    is_inside = np.abs(pc - cube_center) <= cube_extents
    is_inside = np.sum(is_inside, axis=1) == 3
    pc_local = pc[is_inside, ...]

    # visualize
    radii = np.linalg.norm(pc_local, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    pc_local_trimesh = trimesh.points.PointCloud(pc_local, colors=colors)
    pc_local_trimesh.apply_transform(transformation)
    # pc_local_trimesh.show()

    # render
    img_pc = render_single_pc(pc_local_trimesh.vertices, resolution, rendering_kwargs)
    img_pc = Image.fromarray(img_pc)
    img_pc.save(os.path.join('../figures/PointCrop Overview', 'img_pc_local3.png'))
    img_pc.show()
