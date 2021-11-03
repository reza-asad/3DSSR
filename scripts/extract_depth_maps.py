import os
import sys
import numpy as np
import trimesh
from PIL import Image

from scripts.renderer import Render
from scripts.helper import create_img_table


def derive_depth_map(mesh_regions_names):
    # initialize the renderer.
    r = Render(rendering_kwargs)

    # extract and save depth images.
    for region_name in mesh_regions_names:
        # load the room mesh.
        region_name = region_name.split('.')[0]
        mesh_region = trimesh.load(os.path.join(mesh_regions_dir, region_name+'.ply'))
        mesh_region = trimesh.Trimesh.scene(mesh_region)
        # mesh_region.show()

        # render an object centric depth map from the region.
        room_dimension = mesh_region.extents
        camera_pose, _ = mesh_region.graph[mesh_region.camera.name]
        camera_pose[0:2, 3] = 0
        _, depth_map = r.pyrender_render(mesh_region, resolution=resolution, camera_pose=camera_pose.copy(),
                                         room_dimension=room_dimension)

        # save the depth map
        output_file_path = os.path.join(results_dir, region_name)
        np.save(output_file_path, depth_map)


def main(num_chunks, chunk_idx, action='extract_depth_maps'):
    mesh_regions_names = os.listdir(mesh_regions_dir)
    chunk_size = int(np.ceil(len(mesh_regions_names) / num_chunks))
    if action == 'extract_depth_maps':
        derive_depth_map(mesh_regions_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])
    elif action == 'create_img_table':
        results_dir_rendered = os.path.join(data_dir, 'depth_regions_rendered', 'imgs')
        if not os.path.exists(results_dir_rendered):
            os.makedirs(results_dir_rendered)

        imgs = []
        # read images and scale them for visual clarity.
        file_names = os.listdir(results_dir)[:num_imgs]
        for file_name in file_names:
            depth_map = np.load(os.path.join(results_dir, file_name))
            max_depth = np.max(depth_map)
            depth_map = depth_map / max_depth * 255
            img = Image.fromarray(depth_map).convert('L')
            img_name = file_name.split('.')[0] + '.png'
            img.save(os.path.join(results_dir_rendered, img_name))
            imgs.append(img_name)

        # create the image table
        create_img_table(results_dir_rendered, 'imgs', imgs, 'img_table.html', captions=imgs)


if __name__ == '__main__':
    dataset_name = 'matterport3d'
    mode = 'train'
    data_dir = '../data/{}'.format(dataset_name)
    mesh_regions_dir = os.path.join(data_dir, 'mesh_regions', mode)
    results_dir = '/media/reza/Large/mesh_regions_depth/{}'.format(mode)#os.path.join(data_dir, 'mesh_regions_depth', mode)
    num_imgs = 20
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except FileExistsError:
            pass

    # set up rendering parameters
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}

    # ensure no mesh region is rendered twice.
    visited = set()

    if len(sys.argv) == 1:
        main(1, 0, 'extract_depth_maps')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u extract_depth_maps.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_depth_maps
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

