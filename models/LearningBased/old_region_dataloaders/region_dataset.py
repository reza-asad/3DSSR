import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import trimesh

from scripts.helper import load_from_json, sample_mesh
from scripts.box import Box


class Region(Dataset):
    def __init__(self, mesh_dir, scene_dir, metadata_path, accepted_cats_path, mode, transforms=None,
                 cat_to_idx=None, num_global_crops=2, num_local_crops=8, num_points=4096, global_crop_bounds=(0.4, 1),
                 local_crop_bounds=(0.05, 0.4), num_files=None, fixed_crop_height=True, models_dir=None, render=False):
        self.mesh_dir = mesh_dir
        self.scene_dir = scene_dir
        self.metadata_path = metadata_path
        self.accepted_cats_path = accepted_cats_path
        self.mode = mode
        self.transforms = transforms
        self.cat_to_idx = cat_to_idx
        self.file_names = self.filter_file_names(num_files)

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_points = num_points

        self.global_crop_bounds = global_crop_bounds
        self.local_crop_bounds = local_crop_bounds
        self.fixed_crop_height = fixed_crop_height

        self.models_dir = models_dir
        self.render = render
        self.results_to_render = []

    def filter_file_names(self, num_files):
        cat_to_file_names = {}

        def build_map(cat, key):
            if cat in cat_to_file_names:
                cat_to_file_names[cat].append(key)
            else:
                cat_to_file_names[cat] = [key]

        if num_files is None:
            file_names = os.listdir(os.path.join(self.mesh_dir, self.mode))
        else:
            file_names = []
            accepted_cats = load_from_json(self.accepted_cats_path)
            # find a mapping form each category to the file_names.
            df_metadata = pd.read_csv(self.metadata_path)
            is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
            df_metadata = df_metadata[is_accepted]
            df_metadata = df_metadata[df_metadata['split'] == self.mode]
            df_metadata['key'] = df_metadata[['room_name', 'objectId']]. \
                apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]) + '.ply', axis=1)
            df_metadata[['mpcat40', 'key']].apply(lambda x: build_map(x[0], x[1]), axis=1)

            # filter to take objects from each category.
            cat_to_file_names = sorted(cat_to_file_names.items(), key=lambda x: len(x[1]))
            num_cats = len(self.cat_to_idx)
            chunk_size = int(np.floor(num_files / num_cats))
            for cat, f_names in cat_to_file_names:
                file_names += f_names[:chunk_size]
            if len(file_names) < num_files:
                difference = num_files - len(file_names)
                file_names += f_names[chunk_size: chunk_size + difference]
            # from collections import Counter
            # I = df_metadata['key'].apply(lambda x: x in file_names)
            # c = Counter(df_metadata.loc[I, 'mpcat40'])
            # print(c)
            # print(len(c))
            # t=y
        return file_names

    def __len__(self):
        return len(self.file_names)

    def build_cube_crop(self, mesh_bounds, coverage_percent, crop_idx, interval_id, grid_res):
        # find the lower and upper bounds for the starting points of the cube.
        mesh_extents = mesh_bounds[1, :] - mesh_bounds[0, :]
        end_points = mesh_bounds[1, :]
        beg_points_upper_bound = end_points - mesh_extents * coverage_percent
        beg_points_lower_bound = mesh_bounds[0, :]

        # find the subinterval to sample the cube's starting point from.
        interval_id_z = interval_id // (grid_res**2)
        interval_id_xy = interval_id % (grid_res**2)
        interval_id_x = interval_id_xy // grid_res
        interval_id_y = interval_id_xy % grid_res
        interval_ids = [interval_id_x, interval_id_y, interval_id_z]

        # sample the starting point of the cube in x, y and z direction, knowing the extent that you cover.
        beg_points = np.zeros(3, dtype=np.float)
        for i in range(3):
            all_subintervals = np.linspace(beg_points_lower_bound[i], beg_points_upper_bound[i], grid_res+1)
            np.random.seed(crop_idx)
            np.random.shuffle(all_subintervals)
            subinterval = all_subintervals[interval_ids[i]: interval_ids[i] + 2]
            np.random.seed(crop_idx)
            beg_points[i] = np.random.uniform(subinterval[0], subinterval[1])
        if self.fixed_crop_height:
            beg_points[2] = mesh_bounds[0, 2]

        # compute the end point using the sampled starting points
        end_points = beg_points + mesh_extents * coverage_percent
        if self.fixed_crop_height:
            end_points[2] = mesh_bounds[1, 2]

        # compute scale and the centroid of the cube
        scale = end_points - beg_points
        centroid = (beg_points + end_points) / 2.0

        # build the cube
        transformation = np.eye(4)
        transformation[:3, 3] = centroid
        cube = trimesh.creation.box(scale, transform=transformation)

        return cube

    @staticmethod
    def find_submesh(mesh, cube):
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
        # trimesh.points.PointCloud(filtered_vertices).show()

        # find the corresponding face for the inside vertices
        inside_faces = np.arange(len(vertices))[inside_vertices]

        # find a map that sends the inside faces in the old mesh to the face ids in the submesh.
        face_id_map = dict(zip(inside_faces, np.arange(len(filtered_vertices))))

        # filter the faces to only contain the submesh vertices.
        filtered_faces = []
        if len(inside_faces) > 0:
            for i, face in enumerate(faces):
                if np.sum([True if f in face_id_map else False for f in face]) == 3:
                    new_face = [face_id_map[f] for f in face]
                    filtered_faces.append(new_face)

        # build the submesh from the filtered vertices and faces
        submesh = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces)

        return submesh

    def extract_crops_mesh(self, mesh, num_crops, crop_bounds, idx, crop_type, grid_res=3):
        # find the number of intervals to sample local crops from
        num_intervals = grid_res**2
        if not self.fixed_crop_height:
            num_intervals *= grid_res

        # extract local or global crops
        crops_pc = np.zeros((num_crops, self.num_points, 3), dtype=np.float)
        for i in range(num_crops):
            # sample what percentage of the mesh region you are going to cover.
            interval_id = 0
            crop_id = idx + i
            np.random.seed(crop_id)
            coverage_percent = np.random.uniform(crop_bounds[0], crop_bounds[1], 3)
            # find the original bounds of the mesh (i.e., including potential empty space)
            original_mesh_bounds = np.zeros((2, 3), dtype=np.float)
            max_dist = np.max(np.abs(mesh.vertices), axis=0)
            original_mesh_bounds[0, :] = -max_dist
            original_mesh_bounds[1, :] = max_dist
            original_mesh_extents = 2 * max_dist

            # if the crop is local use the current mesh bounds but adjust the percent coverage (i.e., make sure coverage
            # percentage is wrt to the bounds of the original mesh region)
            if crop_type == 'local':
                mesh_bounds = mesh.bounds
                mesh_extents = mesh_bounds[1, :] - mesh_bounds[0, ...]
                coverage_percent = coverage_percent * mesh_extents / original_mesh_extents

                # create a cube to crop the mesh region.
                cube = self.build_cube_crop(mesh_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                            grid_res=grid_res)
            else:
                cube = self.build_cube_crop(original_mesh_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                            grid_res=grid_res)

            # find the intersection of the cube and the region
            submesh = self.find_submesh(mesh, cube)
            interval_id = 1
            while (len(submesh.area_faces) == 0) and (interval_id < num_intervals):
                # if no vertices found try the next interval for sampling
                if crop_type == 'local':
                    cube = self.build_cube_crop(mesh_bounds, coverage_percent, crop_id, interval_id=interval_id,
                                                grid_res=grid_res)
                else:
                    cube = self.build_cube_crop(original_mesh_bounds, coverage_percent, crop_id,
                                                interval_id=interval_id, grid_res=grid_res)

                # compute the submesh with the new cube
                submesh = self.find_submesh(mesh, cube)

                # update the interval_id
                interval_id += 1

            # skip the scene if you still did not find non-empty local crops
            if len(submesh.area_faces) == 0:
                return None

            # sample points on the view
            pc, _ = sample_mesh(submesh, num_points=self.num_points)
            crops_pc[i, :] = pc

            # store the results for rendering if necessary.
            if self.render:
                results_template = dict(scene_crop=..., submesh=..., pc=...)
                # store the scene and the cube
                cube.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
                results_template['scene_crop'] = trimesh.Scene([mesh, cube])

                # store the submesh
                results_template['submesh'] = trimesh.Scene([submesh])

                # store the pc
                results_template['pc'] = pc

                # store the crop
                self.results_to_render[-1]['crops'][crop_type].append(results_template)

            # submesh.show()
            # trimesh.points.PointCloud(pc).show()
            # t=y

        return crops_pc

    def apply_transformation(self, crops):
        for i in range(len(crops)):
            crops[i, ...] = self.transforms(crops[i, ...])

        return crops

    def __getitem__(self, idx):
        # load the mesh region.
        mesh_region = trimesh.load(os.path.join(self.mesh_dir, self.mode, self.file_names[idx]))

        # normalize the mesh region.
        max_coord = np.max(mesh_region.vertices)
        mesh_region.vertices /= max_coord

        if self.render:
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
            obbox.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("##FF0000")

            # take the object and object-centric scene and normalize them
            obj_mesh = trimesh.load(os.path.join(self.models_dir, self.file_names[idx]))
            obj_mesh.vertices /= max_coord
            obj_mesh = trimesh.Scene([obj_mesh])
            obbox.vertices /= max_coord
            scene_obj = trimesh.Scene([mesh_region, obbox])

            rendered_scene = {'scene_name': self.file_names[idx], 'obj': obj_mesh, 'scene': mesh_region,
                              'scene_obj': scene_obj, 'crops': {'local': [], 'global': []}}
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
        if self.num_global_crops > 0:
            global_crops = self.extract_crops_mesh(mesh_region, self.num_global_crops, self.global_crop_bounds, idx,
                                                   crop_type='global')
            if global_crops is None:
                return None
        else:
            pc, _ = sample_mesh(mesh_region, num_points=self.num_points)
            global_crops = np.expand_dims(pc, axis=0)

        # apply normalization and augmentation.
        global_crops = torch.from_numpy(global_crops).to(dtype=torch.float32)
        if self.transforms:
            global_crops = self.apply_transformation(global_crops)

        # load local views of the region and normalize them.
        if self.num_local_crops > 0:
            local_crops = self.extract_crops_mesh(mesh_region, self.num_local_crops, self.local_crop_bounds, idx,
                                                  crop_type='local')
            if local_crops is None:
                return None
            # apply augmentation if asked.
            local_crops = torch.from_numpy(local_crops).to(dtype=torch.float32)
            if self.transforms:
                local_crops = self.apply_transformation(local_crops)
        else:
            local_crops = global_crops

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
        crops = [crop for crop in torch.cat([global_crops, local_crops], dim=0)]
        data = {'file_name': self.file_names[idx], 'crops': crops, 'labels': labels}

        return data
