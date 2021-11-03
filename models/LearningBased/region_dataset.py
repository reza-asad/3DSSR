import os
import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


class Region(Dataset):
    def __init__(self, pc_dir, scene_dir, mode, accepted_regions=None, transforms=None, cat_to_idx=None,
                 num_global_crops=2, num_local_crops=8, num_points=4096, global_crop_bounds=(0.4, 1.),
                 local_crop_bounds=(0.05, 0.4), save_crops=False, models_dir=None):
        self.pc_dir = pc_dir
        self.scene_dir = scene_dir
        self.mode = mode
        self.transforms = transforms
        self.cat_to_idx = cat_to_idx
        self.file_names = os.listdir(os.path.join(pc_dir, mode))
        if accepted_regions is not None:
            filtered_file_names = [f for f in self.file_names if f in accepted_regions]
            self.file_names = filtered_file_names

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_points = num_points

        self.global_crop_bounds = global_crop_bounds
        self.local_crop_bounds = local_crop_bounds

        self.save_crops = save_crops
        self.models_dir = models_dir
        self.results_to_render = []

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def build_colored_pc(pc):
        radii = np.linalg.norm(pc, axis=1)
        colors = trimesh.visual.interpolate(radii, color_map='viridis')
        pc_vis = trimesh.points.PointCloud(pc, colors=colors)

        return pc_vis

    def build_cube_crop(self, pc_bounds, coverage_percent, crop_idx, interval_id, grid_res, crop_type=None):
        # find the lower and upper bounds for the starting points of the cube.
        pc_extents = pc_bounds[1, :] - pc_bounds[0, :]
        pc_extents = np.max(pc_extents[:2])
        end_points = pc_bounds[1, :]

        beg_points_upper_bound = end_points - pc_extents * coverage_percent
        beg_points_lower_bound = pc_bounds[0, :]

        # find the subinterval to sample the cube's starting point from.
        interval_id_xy = interval_id % (grid_res**2)
        interval_id_x = interval_id_xy // grid_res
        interval_id_y = interval_id_xy % grid_res
        interval_ids = [interval_id_x, interval_id_y]

        # sample the starting point of the cube in x, y and z direction, knowing the extent that you cover.
        beg_points = np.zeros(3, dtype=np.float)
        beg_points[2] = pc_bounds[0, 2]
        for i in range(2):
            all_subintervals = np.linspace(beg_points_lower_bound[i], beg_points_upper_bound[i], grid_res+1)
            # np.random.seed(crop_idx)
            np.random.shuffle(all_subintervals)
            subinterval = all_subintervals[interval_ids[i]: interval_ids[i] + 2]
            # np.random.seed(crop_idx)
            beg_points[i] = np.random.uniform(subinterval[0], subinterval[1])

        # compute the end point using the sampled starting points
        end_points = beg_points + pc_extents * coverage_percent
        end_points[2] = pc_bounds[1, 2]

        # compute scale and the centroid of the cube
        scale = end_points - beg_points
        centroid = (beg_points + end_points) / 2.0

        # build the cube
        cube = {'extents': scale, 'centroid': centroid}

        return cube

    def find_subpc(self, pc, cube, crop_id, idx):
        # if self.file_names[idx] == 'zsNo4HB9uLZ_room15-14.npy':
        # pc_vis = self.build_colored_pc(pc)
        # pc_vis.show()
        # transformation = np.eye(4)
        # transformation[:3, 3] = cube['centroid']
        # cube_vis = trimesh.creation.box(cube['extents'], transform=transformation)
        # cube_vis.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
        # trimesh.Scene([pc_vis, cube_vis]).show()

        # take points inside the cube.
        is_inside = np.abs(pc[:, :2] - cube['centroid'][:2]) < (cube['extents'][:2] / 2.0)
        is_inside = np.sum(is_inside, axis=1) == 2
        subpc = pc[is_inside, :]

        # sample N points among the points in subpc
        points_indices = np.arange(len(subpc))
        # np.random.seed(crop_id)
        np.random.shuffle(points_indices)
        sub_indices = points_indices[:self.num_points]
        subpc = subpc[sub_indices, :]

        return subpc

    def extract_crops(self, pc, num_crops, crop_bounds, idx, pc_bounds=None, grid_res=10, crop_type='local'):
        # find the number of intervals to sample local crops from
        num_intervals = grid_res**2

        # extract local or global crops
        crops_pc = np.zeros((num_crops, self.num_points, 3), dtype=np.float)
        for i in range(num_crops):
            # sample what percentage of the mesh region you are going to cover.
            interval_id = 0
            crop_id = idx + i
            # np.random.seed(crop_id)
            coverage_percent = np.random.uniform(crop_bounds[0], crop_bounds[1], 1)

            # find the original bounds of the region (i.e., including potential empty space)
            original_pc_bounds = np.zeros((2, 3), dtype=np.float)
            max_dist = np.max(np.abs(pc), axis=0)
            original_pc_bounds[0, :] = -max_dist
            original_pc_bounds[1, :] = max_dist
            original_pc_extents = np.max(2 * max_dist[:2])

            # if the crop is local use the current mesh bounds but adjust the percent coverage (i.e., make sure coverage
            # percentage is wrt to the bounds of the original mesh region)
            if crop_type == 'local':
                pc_extents = np.max(pc_bounds[1, :2] - pc_bounds[0, :2])
                coverage_percent = coverage_percent * original_pc_extents / pc_extents
                # create a cube to crop the mesh region.
                cube = self.build_cube_crop(pc_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                            grid_res=grid_res, crop_type=crop_type)
            else:
                cube = self.build_cube_crop(original_pc_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                            grid_res=grid_res, crop_type=crop_type)

            # find the intersection of the cube and the region
            subpc = self.find_subpc(pc, cube, crop_id, idx)
            interval_id = 1
            while (len(subpc) < self.num_points) and (interval_id < num_intervals):
                # if no vertices found try the next interval for sampling
                if crop_type == 'local':
                    cube = self.build_cube_crop(pc_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                                grid_res=grid_res, crop_type=crop_type)
                else:
                    cube = self.build_cube_crop(original_pc_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                                grid_res=grid_res, crop_type=crop_type)

                # compute the submesh with the new cube
                subpc = self.find_subpc(pc, cube, crop_id, idx)

                # update the interval_id
                interval_id += 1

            # skip the scene if you still did not find non-empty local crops
            if len(subpc) < self.num_points:
                return None

            # sample points on the view
            crops_pc[i, :] = subpc

            # store the results for rendering if necessary.
            if self.save_crops:
                results_template = dict(cube=..., subpc=...)
                # store the scene and the cube
                transformation = np.eye(4)
                transformation[:3, 3] = cube['centroid']
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
        max_coord = np.max(pc_region)
        pc_region /= max_coord
        pc_region_bounds = trimesh.points.PointCloud(pc_region).bounds

        if self.save_crops:
            # find the scene and obj name
            scene_name, obj = self.file_names[idx].split('-')
            scene_name = scene_name + '.json'
            obj = obj.split('.')[0]

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
            obbox.vertices /= max_coord
            obbox.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("##FF0000")

            # take the object and object-centric scene and normalize them
            obj_mesh = trimesh.load(os.path.join(self.models_dir, self.file_names[idx].split('.')[0] + '.ply'))
            obj_mesh.vertices /= max_coord
            obj_mesh = trimesh.Scene([obj_mesh])

            rendered_scene = {'scene_name': self.file_names[idx], 'pc_region': pc_region, 'obj': obj_mesh,
                              'obbox': obbox, 'crops': {'local': [], 'global': []}}
            self.results_to_render.append(rendered_scene)

            # if self.file_names[idx] == 'SN83YJsR3w2_room40-2.ply':
            #     print(mesh_region.extents)
            #     print(scene_obj.extents)
            #     print(obj_mesh.extents)
            #     mesh_region.show()
            #     scene_obj.show()
            #     obj_mesh.show()
            #     t=y

        # load global crops of the region and augment.
        indices = np.arange(len(pc_region))
        # np.random.seed(idx)
        np.random.shuffle(indices)
        sampled_indices = indices[:self.num_points]
        if len(sampled_indices) < self.num_points:
            return None
        global_crops = np.expand_dims(pc_region[sampled_indices, :], axis=0)
        if self.num_global_crops > 0:
            global_crops = self.extract_crops(pc_region, self.num_global_crops, self.global_crop_bounds, idx,
                                              crop_type='global')
            if global_crops is None:
                return None

        # apply augmentation.
        global_crops = torch.from_numpy(global_crops).to(dtype=torch.float32)
        if self.transforms:
            global_crops = self.apply_transformation(global_crops)

        # load local views of the region and normalize them.
        local_crops = []
        if self.num_local_crops > 0:
            local_crops = self.extract_crops(pc_region, self.num_local_crops, self.local_crop_bounds,
                                             idx, pc_bounds=pc_region_bounds, crop_type='local')
            if local_crops is None:
                return None

            # apply augmentation if asked.
            local_crops = torch.from_numpy(local_crops).to(dtype=torch.float32)
            if self.transforms:
                local_crops = self.apply_transformation(local_crops)

        # add labels if necessary
        labels = torch.zeros(1)
        if self.cat_to_idx is not None:
            # find the scene and obj name
            scene_name, obj = self.file_names[idx].split('-')
            scene_name = scene_name + '.json'
            obj = obj.split('.')[0]

            # load the scene name and prepare the labels
            scene = load_from_json(os.path.join(self.scene_dir, self.mode, scene_name))
            cat = scene[obj]['category'][0]
            labels[0] = self.cat_to_idx[cat]

        # prepare the data
        crops = [crop for crop in global_crops]
        for crop in local_crops:
            crops.append(crop)
        data = {'file_name': self.file_names[idx], 'crops': crops, 'labels': labels}

        return data
