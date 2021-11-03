import os
import numpy as np
import trimesh
from PIL import Image

from scripts.renderer import Render


def render_single_pc(pc, resolution, rendering_kwargs, region=False, with_obbox=False, obbox=None):
    # initialize the renderer.
    r = Render(rendering_kwargs)

    radii = np.linalg.norm(pc, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')

    # build the scene object for rendering
    pc_scene = trimesh.Trimesh.scene(trimesh.points.PointCloud(pc, colors=colors))

    # setup the camera pose and extract the dimensions of the room
    room_dimension = pc_scene.extents
    camera_pose, _ = pc_scene.graph[pc_scene.camera.name]
    if region:
        camera_pose[0:2, 3] = 0

    # render the pc
    img, _ = r.pyrender_render(pc, resolution=resolution, camera_pose=camera_pose, room_dimension=room_dimension,
                               points=True, colors=colors, with_obbox=with_obbox, obbox=obbox)

    return img


def render_pc(pc_dir, pc_file_names, results_dir, resolution, rendering_kwargs, region=False, with_obbox=False,
              obbox=None):
    # render and save all the images
    for pc_file_name in pc_file_names:
        # load the pc
        pc = np.load(os.path.join(pc_dir, pc_file_name))

        # add positional color to the point clouds for better visualization
        if len(pc) == 0:
            continue

        img = render_single_pc(pc, resolution, rendering_kwargs, region, with_obbox, obbox)

        # save the rendered img
        if results_dir is not None:
            img_path = os.path.join(results_dir, pc_file_name.split('.')[0] + '_pc.png')
            Image.fromarray(img).save(img_path)


def render_mesh(mesh_dir, mesh_file_names, results_dir, resolution, rendering_kwargs, region=False):
    # initialize the renderer.
    r = Render(rendering_kwargs)

    # render and save all the images
    for mesh_file_name in mesh_file_names:
        # load the mesh
        mesh = trimesh.Trimesh.scene(trimesh.load(os.path.join(mesh_dir, mesh_file_name)))

        # setup the camera pose and extract the dimensions of the room
        room_dimension = mesh.extents
        camera_pose, _ = mesh.graph[mesh.camera.name]
        if region:
            camera_pose[0:2, 3] = 0

        # render the pc
        img, _ = r.pyrender_render(mesh, resolution=resolution, camera_pose=camera_pose, room_dimension=room_dimension)

        # save the rendered img
        if '/' in mesh_file_name:
            mesh_file_name = mesh_file_name.split('/')[-1]
        img_path = os.path.join(results_dir, mesh_file_name.split('.')[0] + '_mesh.png')
        Image.fromarray(img).save(img_path)

