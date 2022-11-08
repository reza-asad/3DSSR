# Copyright (c) Facebook, Inc. and its affiliates.

"""
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys

import json
import pandas as pd
import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid
from scripts.helper import load_from_json
from scripts.box import Box


IGNORE_LABEL = -1
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = "/home/reza/Documents/research/3DSSR/data/matterport3d/matterport_train_detection_data"  ## Replace with path to dataset
DATASET_METADATA_DIR = "/home/reza/Documents/research/3DSSR/data/matterport3d/meta_data" ## Replace with path to dataset
#DATASET_ROOT_DIR = "/home/rasad/scratch/3dssr/data/matterport3d/matterport_train_detection_data"  ## Replace with path to dataset
#DATASET_METADATA_DIR = "/home/rasad/scratch/3dssr/data/matterport3d/meta_data" ## Replace with path to dataset


class MatterportRealDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 28
        self.num_angle_bin = 1

        self.type2class = {
            'appliances': 0,
            'bathtub': 1,
            'bed': 2,
            'blinds': 3,
            'cabinet': 4,
            'chair': 5,
            'chest_of_drawers': 6,
            'clothes': 7,
            'curtain': 8,
            'cushion': 9,
            'fireplace': 10,
            'furniture': 11,
            'gym_equipment': 12,
            'lighting': 13,
            'mirror': 14,
            'objects': 15,
            'picture': 16,
            'plant': 17,
            'seating': 18,
            'shelving': 19,
            'shower': 20,
            'sink': 21,
            'sofa': 22,
            'stool': 23,
            'table': 24,
            'toilet': 25,
            'towel': 26,
            'tv_monitor': 27
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            range(self.num_semcls)
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        # # Semantic Segmentation Classes. Not used in 3DETR
        # self.num_class_semseg = 20
        # self.type2class_semseg = {
        #     "wall": 0,
        #     "floor": 1,
        #     "cabinet": 2,
        #     "bed": 3,
        #     "chair": 4,
        #     "sofa": 5,
        #     "table": 6,
        #     "door": 7,
        #     "window": 8,
        #     "bookshelf": 9,
        #     "picture": 10,
        #     "counter": 11,
        #     "desk": 12,
        #     "curtain": 13,
        #     "refrigerator": 14,
        #     "showercurtrain": 15,
        #     "toilet": 16,
        #     "sink": 17,
        #     "bathtub": 18,
        #     "garbagebin": 19,
        # }
        # self.class2type_semseg = {
        #     self.type2class_semseg[t]: t for t in self.type2class_semseg
        # }
        self.nyu40ids_semseg = np.array(
            range(self.num_semcls)
        )
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class MatterportRealDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        meta_data_dir=None,
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=False,
        query_info=None,
        scene_dir=None,
        random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        assert split_set in ["train", "val", "test"]
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir

        # load all scan names.
        with open(os.path.join(DATASET_METADATA_DIR, 'accepted_cats.json'), 'r') as f:
            accepted_cats = json.load(f)
        df_metadata = pd.read_csv(os.path.join(DATASET_METADATA_DIR, 'metadata.csv'))
        is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
        df_metadata = df_metadata.loc[is_accepted]
        all_scan_names = np.unique(df_metadata['room_name'].values)
        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            self.scan_names = np.unique(df_metadata.loc[df_metadata['split'] == split_set, 'room_name'].values)
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [
                sname for sname in self.scan_names if sname in all_scan_names
            ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        # TODO: load the query dict for real queries and the scene dir for taking bboxes for the query subscene.
        self.query_info = query_info
        self.scene_dir = scene_dir
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        # TODO: add split set for adding randomness for train subscene queries only.
        self.split_set = split_set

    def __len__(self):
        return len(self.scan_names)

    #  TODO: add binary mask to the point cloud where 1 represents a point that belongs to the subscene.
    @staticmethod
    def add_subscene_mask(point_cloud, bboxes_q):
        N, D = point_cloud.shape
        masked_point_cloud = np.zeros((N, D+1), dtype=np.float32)
        masked_point_cloud[:, :D] = point_cloud[:, :D]

        for box in bboxes_q:
            # find the points corresponding to the box
            centroid, scale = box[:3], box[3:6]
            is_in_box = np.abs(point_cloud[:, :3] - centroid) < (scale / 2.0)
            is_in_box = np.sum(is_in_box, axis=1) == 3

            # add the mask for the object.
            masked_point_cloud[is_in_box, D] = 1.0

        return masked_point_cloud

    @staticmethod
    def format_bbox(vertices, label_id):
        vertices = np.asarray(vertices, dtype=np.float32)

        # centroid.
        centroid = vertices[0, :].tolist()

        # scale
        d1 = np.max(vertices[1:, 0]) - np.min(vertices[1:, 0])
        d2 = np.max(vertices[1:, 1]) - np.min(vertices[1:, 1])
        d3 = np.max(vertices[1:, 2]) - np.min(vertices[1:, 2])
        scale = [d1, d2, d3]

        return centroid + scale + [label_id]

    # TODO: load the query subscene as a set of bboxes and compute the subscene radius.
    def build_subscene(self, scan_name_q):
        context_objects = self.query_info['example']['context_objects']
        bboxes_q = np.zeros((len(context_objects) + 1, 7), dtype=np.float32)

        # load the query scene.
        scene_q = load_from_json(os.path.join(self.scene_dir, '{}.json'.format(scan_name_q)))

        # load the bbox for the anchor object.
        q_node = self.query_info['example']['query']
        anchor_bbox = scene_q[q_node]['aabb']
        anchor_cat = scene_q[q_node]['category'][0]
        label_id = self.dataset_config.type2class[anchor_cat]
        anchor_bbox = self.format_bbox(anchor_bbox, label_id)
        bboxes_q[0, :] = anchor_bbox

        # load the bboxes for each context object.
        for i, context_object in enumerate(context_objects):
            context_bbox = scene_q[context_object]['aabb']
            context_cat = scene_q[context_object]['category'][0]
            label_id = self.dataset_config.type2class[context_cat]
            context_bbox = self.format_bbox(context_bbox, label_id)
            bboxes_q[i+1, :] = context_bbox

        # compute the distance from each context bbox to the anchor and pick the largest as the subscene radius.
        anchor_centroid = bboxes_q[0, :3]
        bboxes_q_centroid = bboxes_q[:, :3]
        dist = np.linalg.norm(bboxes_q_centroid - anchor_centroid, axis=1)
        subscene_radius = np.max(dist)

        return bboxes_q, subscene_radius

    def __getitem__(self, idx):
        # load the mesh for target scene.
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + "_vert.npy")

        # load the instance labels for the target.
        instance_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        )

        # TODO: load the mesh vertics and instancel labels for query scene.
        scan_name_q = self.query_info['example']['scene_name'].split('.')[0]
        mesh_vertices_q = np.load(os.path.join(self.data_path, scan_name_q) + "_vert.npy")
        instance_labels_q = np.load(
            os.path.join(self.data_path, scan_name_q) + "_ins_label.npy"
        )
        # instance_bboxes_q = np.load(os.path.join(self.data_path, scan_name_q) + "_bbox.npy")
        # print(instance_bboxes_q)
        # print('*'*50)
        # tt
        point_cloud = mesh_vertices[:, 0:3]  # do not use color for now

        # TODO: add zero value as the last dimension of the original point cloud.
        zero_feat = np.zeros((len(point_cloud), 1), dtype=np.float32)
        point_cloud = np.concatenate([point_cloud, zero_feat], 1)

        # ------------------------------- LABELS ------------------------------

        # TODO: use the query info to create the subscene as a set of bboxes.
        bboxes_q, subscene_radius = self.build_subscene(scan_name_q)

        # TODO: add a binary mask to the point cloud with 1 representing point belonging to the subscene.
        point_cloud_with_mask = self.add_subscene_mask(mesh_vertices_q, bboxes_q)
        # print(point_cloud.shape)
        # print(point_cloud_with_mask.shape)
        # import trimesh
        # trimesh.points.PointCloud(point_cloud[:, :3]).show()
        # subscene = point_cloud_with_mask[point_cloud_with_mask[:, 3] == 1, :3]
        # trimesh.points.PointCloud(subscene).show()
        # t=y

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        instance_labels = instance_labels[choices]
        point_cloud_with_mask, choices = pc_util.random_sampling(
            point_cloud_with_mask, self.num_points, return_choices=True
        )
        instance_labels_q = instance_labels_q[choices]

        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["point_clouds_with_mask"] = point_cloud_with_mask.astype(np.float32)
        ret_dict["instance_labels"] = instance_labels.astype(np.long)
        ret_dict["instance_labels_q"] = instance_labels_q.astype(np.long)
        ret_dict["subscene_radius"] = subscene_radius.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["scan_name"] = scan_name
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        return ret_dict