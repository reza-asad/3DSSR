import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import Counter
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


class Region(Dataset):
    def __init__(self, pc_dir, scene_dir, metadata_path, mode, max_coord, transforms=None, cat_to_idx=None,
                 file_name_to_idx=None, num_global_crops=2, num_local_crops=8, num_points=4096, global_crop_bounds=(0.4, 1.),
                 local_crop_bounds=(0.05, 0.4), save_crops=False, global_to_local_points_ratio=4, theta=0):
        self.pc_dir = pc_dir
        self.scene_dir = scene_dir
        self.metadata_path = metadata_path
        self.mode = mode
        self.transforms = transforms
        self.cat_to_idx = cat_to_idx
        self.file_name_to_idx = file_name_to_idx
        self.file_names = self.extract_files()

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_points = num_points

        self.global_crop_bounds = global_crop_bounds
        self.local_crop_bounds = local_crop_bounds
        self.global_to_local_points_ratio = global_to_local_points_ratio
        self.max_coord = max_coord

        self.save_crops = save_crops
        self.results_to_render = []

        # if pc is rotated
        self.theta = theta
        self.transformation = np.eye(4)
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        self.transformation[:3, :3] = rotation

    def extract_files(self):
        # read the metadata
        df = pd.read_csv(self.metadata_path)

        # filter the metadata by the accepted cats
        if 'mpcat40' in df.keys():
            is_accepted = df['mpcat40'].apply(lambda x: x in self.cat_to_idx)
            df = df.loc[is_accepted]

        # filter the metadata by the mode
        df = df.loc[df['split'] == self.mode]

        # create the file_names
        df['key'] = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1)

        return df['key'].values.tolist()

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def build_colored_pc(pc):
        radii = np.linalg.norm(pc, axis=1)
        colors = trimesh.visual.interpolate(radii, color_map='viridis')
        pc_vis = trimesh.points.PointCloud(pc, colors=colors)

        return pc_vis

    @staticmethod
    def sample_cube_centers(pc, pc_extents):
        indices = np.arange(len(pc))
        is_center = np.abs(pc) <= (1/3 * pc_extents)
        is_center = np.sum(is_center, axis=1) == 3

        return indices[is_center]

    @staticmethod
    def build_cube_crop(center, pc_extents, coverage_percent):
        # find the scale of the cube
        scale = pc_extents * coverage_percent
        # build the cube
        cube = {'extents': scale, 'center': center}

        return cube

    def find_subpc(self, pc, cube, crop_id, idx, num_sampled_points):
        # if self.file_names[idx] == 'zsNo4HB9uLZ_room15-14.npy':
        # pc_vis = self.build_colored_pc(pc)
        # pc_vis.show()
        # transformation = np.eye(4)
        # transformation[:3, 3] = cube['center']
        # cube_vis = trimesh.creation.box(cube['extents'], transform=transformation)
        # cube_vis.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
        # trimesh.Scene([pc_vis, cube_vis]).show()

        # take points inside the cube.
        # TODO: change 1
        is_inside = np.abs(pc - cube['center']) <= (cube['extents'] / 2.0)
        is_inside = np.sum(is_inside, axis=1) == 3
        subpc = pc[is_inside, :]

        # find the number of points that you will sample form the subpc

        # np.random.seed(crop_id)
        sampled_subpc = None
        if len(subpc) >= num_sampled_points:
            sampled_indices = np.random.choice(range(len(subpc)), num_sampled_points, replace=False)
            sampled_subpc = subpc[sampled_indices, :]

        return sampled_subpc

    def extract_crops(self, pc, pc_extents, num_crops, crop_bounds, idx, crop_type, num_tries=200):
        # find the number of point to smaple from the crop
        if crop_type == 'global':
            num_sampled_points = self.num_points
        else:
            num_sampled_points = self.num_points // self.global_to_local_points_ratio

        # extract local or global crops
        crops_pc = np.zeros((num_crops, num_sampled_points, 3), dtype=np.float)
        for i in range(num_crops):
            # sample what percentage of the region you are going to cover.
            crop_id = idx + i
            # np.random.seed(crop_id)
            coverage_percent = np.random.uniform(crop_bounds[0], crop_bounds[1], 3)

            # sample a set of seed points as the center for the crop.
            # np.random.seed(crop_id)
            # TODO: change 2
            sampled_indices = self.sample_cube_centers(pc, pc_extents)
            replace = num_tries > len(sampled_indices)
            sampled_indices = np.random.choice(sampled_indices, num_tries, replace=replace)
            # sampled_indices = np.random.choice(range(len(pc)), num_tries, replace=False)

            # iterate through the points and exit once you find enough points in the crop.
            for sampled_index in sampled_indices:
                center = pc[sampled_index, :]
                cube = self.build_cube_crop(center, pc_extents, coverage_percent)

                # find the points inside the cube crop
                subpc = self.find_subpc(pc, cube, crop_id, idx, num_sampled_points)

                # if you find enough points, you found the crop so exit.
                if subpc is not None:
                    break

            # skip the scene if you still did not find non-empty local crops
            if subpc is None:
                return None

            # record the crop
            crops_pc[i, :] = subpc

            # store the results for rendering if necessary.
            if self.save_crops:
                results_template = dict(cube=..., subpc=...)
                # store the scene and the cube
                transformation = np.eye(4)
                transformation[:3, 3] = cube['center']
                cube_vis = trimesh.creation.box(cube['extents'], transform=transformation)
                cube_vis.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")

                results_template['cube'] = cube_vis

                # store the submesh
                results_template['subpc'] = subpc

                # store the crop
                self.results_to_render[-1]['crops'][crop_type].append(results_template)

            # if self.file_names[idx] == 'zsNo4HB9uLZ_room15-14.npy':
            # sub_pc_vis = self.build_colored_pc(subpc)
            # sub_pc_vis.show()

        return crops_pc

    def apply_transformation(self, crops):
        for i in range(len(crops)):
            crops[i, ...] = self.transforms(crops[i, ...])

        return crops

    def __getitem__(self, idx):
        # load the point cloud region.
        pc_region = np.load(os.path.join(self.pc_dir, self.mode, self.file_names[idx]))

        # normalize the mesh region.
        pc_region /= self.max_coord
        pc_region_extents = trimesh.points.PointCloud(pc_region).extents

        if self.save_crops:
            # find the scene and obj name
            if '-' in self.file_names[idx]:
                scene_name, obj = self.file_names[idx].split('-')
                scene_name = scene_name + '.json'
                obj = obj.split('.')[0]
            else:
                scene_name = self.file_names[idx].split('.')[0] + '.json'
                obj = '0'

            # load the scene.
            scene = load_from_json(os.path.join(self.scene_dir, self.mode, scene_name))

            # find the OBB for the center obj and translate it to the origin
            vertices = np.asarray(scene[obj]['obbox'], dtype=np.float64)
            obbox = Box(vertices)
            transformation = np.eye(4)
            transformation[:3, 3] = -obbox.translation
            scale = obbox.scale
            obbox = obbox.apply_transformation(transformation)
            obbox = trimesh.creation.box(scale, obbox.transformation)
            obbox.vertices /= self.max_coord
            obbox.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("##FF0000")

            rendered_scene = {'scene_name': self.file_names[idx], 'pc_region': pc_region,
                              'crops': {'local': [], 'global': []}}
            self.results_to_render.append(rendered_scene)

        # load global crops of the region and augment.
        if self.num_global_crops == 0:
            if len(pc_region) >= self.num_points:
                np.random.seed(0)
                sampled_indices = np.random.choice(range(len(pc_region)), self.num_points, replace=False)
                sampled_pc_region = pc_region[sampled_indices, :]

                if self.theta == 0:
                    global_crops = np.expand_dims(sampled_pc_region, axis=0)
                else:
                    # rotate if necessary
                    sampled_pc_region_rot = np.ones((4, len(sampled_pc_region)), dtype=np.float64)
                    sampled_pc_region_rot[0, :] = sampled_pc_region[:, 0]
                    sampled_pc_region_rot[1, :] = sampled_pc_region[:, 1]
                    sampled_pc_region_rot[2, :] = sampled_pc_region[:, 2]
                    sampled_pc_region_rot = np.dot(self.transformation, sampled_pc_region_rot)
                    sampled_pc_region_rot = sampled_pc_region_rot[:3, :].T
                    global_crops = np.expand_dims(sampled_pc_region_rot, axis=0)
            else:
                return None
        else:
            global_crops = self.extract_crops(pc_region, pc_region_extents, self.num_global_crops,
                                              self.global_crop_bounds, idx, crop_type='global')
            if global_crops is None:
                return None

        # apply augmentation.
        global_crops = torch.from_numpy(global_crops).to(dtype=torch.float32)
        if self.transforms:
            global_crops = self.apply_transformation(global_crops)

        # load local views of the region and normalize them.
        local_crops = []
        if self.num_local_crops > 0:
            local_crops = self.extract_crops(pc_region, pc_region_extents, self.num_local_crops,
                                             self.local_crop_bounds, idx, crop_type='local')
            if local_crops is None:
                return None

            # apply augmentation if asked.
            local_crops = torch.from_numpy(local_crops).to(dtype=torch.float32)
            if self.transforms:
                local_crops = self.apply_transformation(local_crops)

        # add labels if necessary
        labels = torch.zeros(1)
        if self.cat_to_idx is not None:
            if '-' in self.file_names[idx]:
                scene_name, obj = self.file_names[idx].split('-')
                scene_name = scene_name + '.json'
                obj = obj.split('.')[0]
            else:
                scene_name = self.file_names[idx].split('.')[0] + '.json'
                obj = '0'

            # load the scene name and prepare the labels
            scene = load_from_json(os.path.join(self.scene_dir, self.mode, scene_name))
            cat = scene[obj]['category'][0]
            labels[0] = self.cat_to_idx[cat]

        # prepare the data
        crops = [crop for crop in global_crops]
        for crop in local_crops:
            crops.append(crop)

        # prepare file_name
        file_names = torch.zeros(1)
        file_names[0] = self.file_name_to_idx[self.file_names[idx]]

        data = {'file_name': file_names, 'crops': crops, 'labels': labels}

        return data
