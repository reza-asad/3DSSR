import os
from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


class Scene(Dataset):
    def __init__(self, scene_dir, pc_dir, metadata_path, accepted_cats_path, max_coord_box, max_coord_scene,
                 mode='train', num_points=4096, crop_bounds=(0.7, 0.7), max_query_size=5, max_sample_size=30,
                 random_subscene=False):
        self.scene_dir = os.path.join(scene_dir, mode)
        self.pc_dir = os.path.join(pc_dir, mode)
        self.max_coord_box = max_coord_box
        self.max_coord_scene = max_coord_scene
        self.num_points = num_points
        self.crop_bounds = crop_bounds

        # synthetic query parameters.
        self.max_query_size = max_query_size
        self.max_sample_size = max_sample_size
        self.min_radial_threshold = 0.3
        self.max_radial_threshold = 0.5
        self.min_angle_diff = 20
        self.random_subscene = random_subscene

        # files to process.
        self.accepted_cats = load_from_json(accepted_cats_path)
        self.file_names = self.find_file_names(metadata_path, mode)

    def find_file_names(self, metadata_path, mode):
        # load the metadata and accepted cats.
        df = pd.read_csv(metadata_path)

        # filter the metadata by the accepted cats
        is_accepted = df['mpcat40'].apply(lambda x: x in self.accepted_cats)
        df = df.loc[is_accepted]

        # filter the metadata by the mode
        df = df.loc[df['split'] == mode]
        print(Counter(df['mpcat40']))

        # filter the remaining scenes, so they have at least 1 object.
        scene_name_to_obj_count = df.groupby(['room_name'])['objectId'].count()
        scene_filter_criteria = scene_name_to_obj_count >= self.max_query_size
        scene_name_to_obj_count = scene_name_to_obj_count[scene_filter_criteria]

        # take objects in the filtered scenes.
        scene_names = set(scene_name_to_obj_count.keys())
        df = df.loc[df['room_name'].apply(lambda x: x in scene_names)]
        df['key'] = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1])]) + '.npy', axis=1)

        return df['key'].values.tolist()

    def __len__(self):
        return len(self.file_names)

    def find_random_subscene(self, scene, anchor_obj):
        # shuffle the context objects.
        context_objects = [obj for obj in scene.keys() if obj != anchor_obj]
        np.random.shuffle(context_objects)

        # sample.
        subscene_objects = [anchor_obj] + context_objects[: self.max_query_size - 1]

        # populate the properties of the subscene and return it.
        subscene = {}
        for obj in subscene_objects:
            subscene[obj] = scene[obj].copy()

        return subscene

    def find_radial_subscene(self, scene, anchor_obj):
        # find the closest N objects to the anchor object to build the subscene.
        anchor_centroid = np.asarray(scene[anchor_obj]['aabb'][0], dtype=np.float32)
        distances = []
        for obj in scene.keys():
            obj_centroid = np.asarray(scene[obj]['aabb'][0], dtype=np.float32)
            distance = np.linalg.norm(obj_centroid - anchor_centroid)
            distances.append((obj, distance))

        # sort the distances from smallest to largest and take the subscene objects
        distances = sorted(distances, key=lambda x: x[1])
        subscene_objects = [e[0] for e in distances[:self.max_query_size]]

        # populate the properties of the subscene and return it.
        subscene = {}
        for obj in subscene_objects:
            subscene[obj] = scene[obj].copy()

        return subscene

    def filter_scene_by_cat(self, scene, q_subscene):
        # find all objects in the scene with same category as the query objects.
        t_subscene_objects = []
        q_subscene_cats = set([obj_info['category'][0] for _, obj_info in q_subscene.items()])
        for obj, obj_info in scene.items():
            if obj_info['category'][0] in q_subscene_cats:
                t_subscene_objects.append(obj)

        # sample from the target objects.
        sample_size = np.minimum(self.max_sample_size - self.max_query_size, len(t_subscene_objects))
        np.random.shuffle(t_subscene_objects)
        t_subscene_objects_sample = t_subscene_objects[:sample_size] + list(q_subscene.keys())

        # build the target subscene
        t_subscene = {}
        for obj in t_subscene_objects_sample:
            t_subscene[obj] = scene[obj].copy()

        return t_subscene

    def populate_scene_info(self, scene, scene_name):
        # for each object in the scene load the object's dense point cloud along with aabb.
        scene_to_pc = {}
        for obj, obj_info in scene.items():
            # load the dense point cloud for the obj.
            pc_filename = '-'.join([scene_name, obj]) + '.npy'
            obj_pc = np.load(os.path.join(self.pc_dir, pc_filename))

            # find the axis-aligned bounding box for the object.
            aabb = Box(np.asarray(obj_info['aabb'], dtype=np.float32))

            # record normalized results.
            scene_to_pc[obj] = {'pc': obj_pc, 'aabb': aabb}

        return scene_to_pc

    @staticmethod
    def visualize_scene_boxes(anchor, curr_scene, prev_scene):
        color = "#0000FF"
        if len(prev_scene) != 0:
            color = "#ff0000"
        for obj, obj_info in curr_scene.items():
            box = trimesh.creation.box(obj_info['aabb'].scale, obj_info['aabb'].transformation)
            if obj == anchor:
                box.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#00FF00")
            else:
                box.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(color)
            prev_scene.append(box)

        return prev_scene

    @staticmethod
    def build_colored_pc(pc):
        radii = np.linalg.norm(pc, axis=1)
        colors = trimesh.visual.interpolate(radii, color_map='viridis')
        pc_vis = trimesh.points.PointCloud(pc, colors=colors)

        return pc_vis

    @staticmethod
    def sample_cube_centers(pc, pc_extents):
        denominator = 3
        is_center = []
        while np.sum(is_center) == 0:
            indices = np.arange(len(pc))
            fraction = 1/denominator
            is_center = np.abs(pc) <= (fraction * pc_extents / 2)
            is_center = np.sum(is_center, axis=1) == 3
            denominator -= 1

        return indices[is_center]

    @staticmethod
    def build_cube(center, pc_extents, coverage_percent):
        # find the scale of the cube
        scale = pc_extents * coverage_percent
        # build the cube
        cube = {'extents': scale, 'center': center}

        return cube

    @staticmethod
    def new_box(old_box, translation, rotation):
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        transformation[:3, :3] = rotation
        new_box = old_box.apply_transformation(transformation)
        # boxes = trimesh.Scene([trimesh.creation.box(old_box.scale, old_box.transformation),
        #                        trimesh.creation.box(new_box.scale, new_box.transformation)])

        return new_box

    @staticmethod
    def transform_pc(pc, translation, rotation):
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        transformation[:3, :3] = rotation

        new_pc = np.ones((len(pc), 4), dtype=np.float)
        new_pc[:, 0] = pc[:, 0]
        new_pc[:, 1] = pc[:, 1]
        new_pc[:, 2] = pc[:, 2]

        new_pc = np.dot(transformation, new_pc.T).T
        new_pc = new_pc[:, :3]

        return new_pc

    def find_sampled_crop(self, pc, cube):
        # if self.file_names[idx] == 'zsNo4HB9uLZ_room15-14.npy':
        # pc_vis = self.build_colored_pc(pc)
        # pc_vis.show()
        # transformation = np.eye(4)
        # transformation[:3, 3] = cube['center']
        # cube_vis = trimesh.creation.box(cube['extents'], transform=transformation)
        # cube_vis.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
        # trimesh.Scene([pc_vis, cube_vis]).show()

        # take points inside the cube.
        is_inside = np.abs(pc - cube['center']) <= (cube['extents'] / 2.0)
        is_inside = np.sum(is_inside, axis=1) == 3
        crop = pc[is_inside, :]

        sampled_crop = None
        if len(crop) >= self.num_points:
            sampled_indices = np.random.choice(range(len(crop)), self.num_points, replace=False)
            sampled_crop = crop[sampled_indices, :]

        return sampled_crop

    def sample_crop_pc(self, scene_pc, num_tries=200):
        # for each object sample points and extract a crop.
        for obj, obj_info in scene_pc.items():
            coverage_percent = np.random.uniform(self.crop_bounds[0], self.crop_bounds[1], 3)

            # load the pc and sample the center of the cube near the center of the object.
            pc = obj_info['pc']
            pc_extents = trimesh.points.PointCloud(pc).extents
            sampled_indices = self.sample_cube_centers(pc, pc_extents)

            # try at least x times to get enough points.
            replace = num_tries > len(sampled_indices)
            sampled_indices = np.random.choice(sampled_indices, num_tries, replace=replace)

            # iterate through the points and exit once you find enough points in the crop.
            sampled_crop = None
            for sampled_index in sampled_indices:
                center = pc[sampled_index, :]
                cube = self.build_cube(center, pc_extents, coverage_percent)

                # find the points inside the cube crop
                sampled_crop = self.find_sampled_crop(pc, cube)

                # if you find enough points, you found the crop so exit.
                if sampled_crop is not None:
                    break

            # skip the scene if you still did not find non-empty local crops
            if sampled_crop is None:
                return None

            # record the crop
            scene_pc[obj]['pc'] = sampled_crop

            # sampled_crop_vis = self.build_colored_pc(sampled_crop)
            # sampled_crop_vis.show()

        return scene_pc

    def sample_points(self, scene_pc):
        for obj, obj_info in scene_pc.items():
            # for validation use the same sample.
            np.random.seed(0)
            sampled_indices = np.random.choice(range(len(obj_info['pc'])), self.num_points, replace=False)
            obj_info['pc'] = obj_info['pc'][sampled_indices, :]

        return scene_pc

    def perturb_radial(self, aabb_obj, aabb_anchor):
        # find the centroids of the boxes.
        obj_c = aabb_obj.translation
        anchor_c = aabb_anchor.translation

        # find the vector connecting anchor to obj.
        anchor_to_obj = obj_c - anchor_c

        # compute the norm of the distance between obj and anchor
        anchor_to_obj_norm = np.linalg.norm(anchor_to_obj)
        if anchor_to_obj_norm == 0:
            return aabb_obj

        # pick a positive or negative direction along the vec.
        alpha = 1
        if np.random.uniform(0, 1) > 0.5:
            alpha = -1

        # randomly pick the amount of perturbation.
        perturbation_magnitude = np.random.uniform(self.min_radial_threshold * anchor_to_obj_norm,
                                                   self.max_radial_threshold * anchor_to_obj_norm)

        # move the centroid of the context object according to the perturbation magnitude and direction.
        translation = alpha * perturbation_magnitude * (anchor_to_obj / anchor_to_obj_norm)

        # build the new aabb
        return self.new_box(aabb_obj, translation, np.eye(3))

    def perturb_angular(self, subscene_pc, anchor, context_obj, theta):
        # create the rotation matrix.
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])

        # translate the object's aabb so the centroid of the anchor object is at the origin.
        anchor_translation = -subscene_pc[anchor]['aabb'].translation
        subscene_pc[context_obj]['aabb'] = self.new_box(subscene_pc[context_obj]['aabb'], anchor_translation, np.eye(3))

        # rotate the object's aabb by theta.
        subscene_pc[context_obj]['aabb'] = self.new_box(subscene_pc[context_obj]['aabb'], np.zeros(3), rotation)

        # translate the object's aabb back to the world coordinate.
        subscene_pc[context_obj]['aabb'] = self.new_box(subscene_pc[context_obj]['aabb'], -anchor_translation, np.eye(3))

        # vis only
        # pc1 = self.build_colored_pc(subscene_pc[context_obj]['pc'])
        # pc1.show()

        # rotate the pc
        subscene_pc[context_obj]['pc'] = self.transform_pc(subscene_pc[context_obj]['pc'], np.zeros(3), rotation)

        # pc2 = self.build_colored_pc(subscene_pc[context_obj]['pc'])
        # pc2.show()

        return subscene_pc[context_obj]

    def choose_random_angle(self, random_angles):
        bad_angle = True
        while bad_angle:
            # take a random angle.
            random_angle = np.random.uniform(0, 2*np.pi)

            # examine the new angle against the previous ones.
            bad_angle = False
            for angle in random_angles:
                if (np.abs(random_angle - angle) * 180 / np.pi) < self.min_angle_diff:
                    bad_angle = True
                    break

        return random_angle

    def perturb(self, query_pc, anchor):
        # visualize the query boxes.
        # prev_scene = self.visualize_scene_boxes(anchor, query_pc, [])

        # select the matching query objects.
        query_size = len(query_pc)

        # edge case if there are two or three objects.
        if query_size in [2, 3]:
            # majority is 2.
            min_num_matching_objects = 2
        else:
            # round down.
            min_num_matching_objects = int(query_size//2)
        num_matching_objects = np.random.choice(np.arange(min_num_matching_objects, query_size+1), 1)[0]
        query_objects = list(query_pc.keys())
        np.random.shuffle(query_objects)
        matching_objects = set(query_objects[:num_matching_objects])

        # choose a random rotation for the matching objects.
        theta = np.random.uniform(0, 2*np.pi)
        random_angles = [theta]

        # for each object in the query, perturb its radial and angle relative to the anchor object
        for obj in query_objects:
            # perturb radially relative to the anchor obj.
            query_pc[obj]['aabb'] = self.perturb_radial(query_pc[obj]['aabb'], query_pc[anchor]['aabb'])

            # perturb angular relative to the anchor.
            if obj in matching_objects:
                query_pc[obj] = self.perturb_angular(query_pc, anchor, obj, theta)
            else:
                random_angle = self.choose_random_angle(random_angles)
                random_angles.append(random_angle)
                query_pc[obj] = self.perturb_angular(query_pc, anchor, obj, random_angle)

        # visualize the query boxes after perturbation compared to no perturbation.
        # matching_query_pc = {obj: obj_info for obj, obj_info in query_pc.items() if obj in matching_objects}
        # combined_scene = self.visualize_scene_boxes(anchor, matching_query_pc, prev_scene)
        # print('{}/{} matches'.format(len(matching_objects), query_size))
        # print(anchor in matching_objects, [c * 180/np.pi for c in random_angles])
        # trimesh.Scene(combined_scene).show()

        return query_pc, theta, matching_objects

    def normalize_scene(self, scene):
        for obj, obj_info in scene.items():
            # normalize the pc.
            obj_info['pc'] /= self.max_coord_box

            # normalize the centroid and scale of the aabb
            aabb_vertices = obj_info['aabb'].vertices
            aabb_vertices /= self.max_coord_scene
            obj_info['aabb'] = Box(aabb_vertices)

        return scene

    def prepare_tensor_data(self, scene):
        pc_data = np.zeros((len(scene), self.num_points, 3), dtype=np.float32)
        centroid_data = np.zeros((len(scene), 3), dtype=np.float32)
        objects = sorted(scene.keys(), key=int)
        for i, obj in enumerate(objects):
            # add pc data.
            pc_data[i, ...] = scene[obj]['pc']

            # add centroid data
            centroid_data[i, :] = scene[obj]['aabb'].translation

        # convert data to tensor.
        pc_data = torch.from_numpy(pc_data)
        centroid_data = torch.from_numpy(centroid_data)

        return pc_data, centroid_data

    @staticmethod
    def prepare_label(scene, matching_objects):
        match_label = torch.zeros(len(scene), dtype=torch.bool)
        objects = sorted(scene.keys(), key=int)
        for i, obj in enumerate(objects):
            if obj in matching_objects:
                match_label[i] = 1

        return match_label

    def __getitem__(self, idx):
        # load the anchor object and the scene it belongs to.
        file_name = self.file_names[idx]
        scene_name = file_name.split('-')[0]
        scene = load_from_json(os.path.join(self.scene_dir, scene_name + '.json'))

        # take a subscene.
        anchor_obj = file_name.split('.')[0].split('-')[-1]
        if self.random_subscene:
            q_subscene = self.find_random_subscene(scene, anchor_obj)
        else:
            q_subscene = self.find_radial_subscene(scene, anchor_obj)

        # find the filtered target scene based on the categories present in the query subscene
        t_subscene = self.filter_scene_by_cat(scene, q_subscene)

        # populate dense pc and aabb for each object in the target subscene and query subscene.
        t_scene_pc = self.populate_scene_info(t_subscene, scene_name)
        q_scene_pc = {}
        for obj, obj_info in t_scene_pc.items():
            if obj in q_subscene:
                q_scene_pc[obj] = t_scene_pc[obj].copy()

        # transformations:
        # sample and crop point clouds from the target and query objects.
        t_scene_pc = self.sample_crop_pc(t_scene_pc)
        q_scene_pc = self.sample_crop_pc(q_scene_pc)

        # perturb the query objects along radial and angular directions.
        q_scene_pc, theta, matching_objects = self.perturb(q_scene_pc, anchor_obj)

        # normalize the pc and centroid
        t_scene_pc = self.normalize_scene(t_scene_pc)
        q_scene_pc = self.normalize_scene(q_scene_pc)

        # prepare the data for training.
        pc_t, centroid_t = self.prepare_tensor_data(t_scene_pc)
        pc_q, centroid_q = self.prepare_tensor_data(q_scene_pc)

        # prepare the label.
        match_label_t = self.prepare_label(t_scene_pc, matching_objects)
        match_label_q = self.prepare_label(q_scene_pc, matching_objects)

        data = {'pc_t': pc_t,
                'centroid_t': centroid_t,
                'match_label_t': match_label_t,
                'pc_q': pc_q,
                'centroid_q': centroid_q,
                'match_label_q': match_label_q,
                'theta': theta}

        return data
