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

IGNORE_LABEL = -1
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = "/home/reza/Documents/research/3DSSR/data/matterport3d/matterport_train_detection_data"  ## Replace with path to dataset
DATASET_METADATA_DIR = "/home/reza/Documents/research/3DSSR/data/matterport3d/meta_data" ## Replace with path to dataset


class MatterportDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 42
        self.num_angle_bin = 1
        # TODO: changed from 64 to 5.
        self.max_num_obj = 5

        self.type2class = {
            'appliances': 0,
            'bathtub': 1,
            'beam': 2,
            'bed': 3,
            'blinds': 4,
            'board_panel': 5,
            'cabinet': 6,
            'ceiling': 7,
            'chair': 8,
            'chest_of_drawers': 9,
            'clothes': 10,
            'column': 11,
            'counter': 12,
            'curtain': 13,
            'cushion': 14,
            'door': 15,
            'fireplace': 16,
            'floor': 17,
            'furniture': 18,
            'gym_equipment': 19,
            'lighting': 20,
            'mirror': 21,
            'misc': 22,
            'objects': 23,
            'picture': 24,
            'plant': 25,
            'railing': 26,
            'seating': 27,
            'shelving': 28,
            'shower': 29,
            'sink': 30,
            'sofa': 31,
            'stairs': 32,
            'stool': 33,
            'table': 34,
            'toilet': 35,
            'towel': 36,
            'tv_monitor': 37,
            'unlabeled': 38,
            'void': 39,
            'wall': 40,
            'window': 41
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


class MatterportDetectionDataset(Dataset):
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
        use_random_cuboid=True,
        aggressive_rot=False,
        augment_eval=False,
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
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.aggressive_rot = aggressive_rot
        self.augment_eval = augment_eval
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        # TODO: add split set for adding randomness for train subscene queries only.
        self.split_set = split_set

    def __len__(self):
        return len(self.scan_names)

    #  TODO: add binary mask to the point cloud where 1 represents a point that belongs to the subscene.
    def add_subscene_mask(self, point_cloud, instance_bboxes):
        N, D = point_cloud.shape
        masked_point_cloud = np.zeros((N, D), dtype=np.float32)
        masked_point_cloud[:, :D-1] = point_cloud[:, :D-1]

        for box in instance_bboxes:
            # find the points corresponding to the box
            centroid, scale = box[:3], box[3:6]
            is_in_box = np.abs(point_cloud[:, :3] - centroid) < (scale / 2.0)
            is_in_box = np.sum(is_in_box, axis=1) == 3

            # add the mask for the object.
            masked_point_cloud[is_in_box, D-1] = 1.0

        return masked_point_cloud

    # TODO: picking some closest objects to an anchor object to form a subscene.
    def sample_subscene(self, instance_bboxes, MAX_NUM_OBJ):
        # randomly take an anchor object.
        if self.split_set != 'train':
            np.random.seed(0)
        num_boxes = len(instance_bboxes)
        anchor_idx = np.random.choice(num_boxes, 1)[0]
        anchor_box = instance_bboxes[anchor_idx, ...]

        # pick MAX_NUM_OBJ - 1 closest boxes to anchor box
        anchor_centroid = anchor_box[:3]
        instance_bboxes_centroids = instance_bboxes[:, :3]
        dist = np.linalg.norm(instance_bboxes_centroids - anchor_centroid, axis=1)
        idx_dist = zip(range(num_boxes), dist)

        # sort boxes from closest to furthest from the anchor box.
        sorted_idx_dist = sorted(idx_dist, key=lambda x: x[1])

        # taking the anchor box too.
        closest_indices = list(list(zip(*sorted_idx_dist[:MAX_NUM_OBJ]))[0])

        instance_bboxes = instance_bboxes[closest_indices, ...]

        return instance_bboxes, sorted_idx_dist[-1][-1]

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + "_vert.npy")
        instance_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        )
        semantic_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_sem_label.npy"
        )
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_bbox.npy")

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # TODO: add zero value as the last dimension of the original point cloud.
        zero_feat = np.zeros((len(point_cloud), 1), dtype=np.float32)
        point_cloud = np.concatenate([point_cloud, zero_feat], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes, [instance_labels, semantic_labels]
            )
            instance_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]

        # TODO: randomly take MAX_NUM_OBJ - 1 many closest boxes (to an anchor) among the instance_bboxes.
        if len(instance_bboxes) == 0:
            return None
        instance_bboxes, subscene_radius = self.sample_subscene(instance_bboxes, MAX_NUM_OBJ)

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

        for _c in self.dataset_config.nyu40ids_semseg:
            sem_seg_labels[
                semantic_labels == _c
            ] = self.dataset_config.nyu40id2class_semseg[_c]

        pcl_color = pcl_color[choices]

        target_bboxes_mask[0 : instance_bboxes.shape[0]] = 1
        target_bboxes[0 : instance_bboxes.shape[0], :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]

        # TODO: add a binary mask to the point cloud with 1 representing point belonging to the subscene.
        point_cloud_with_mask = self.add_subscene_mask(point_cloud, instance_bboxes)
        # print(point_cloud.shape)
        # print(point_cloud_with_mask.shape)
        # import trimesh
        # trimesh.points.PointCloud(point_cloud[:, :3]).show()
        # subscene = point_cloud_with_mask[point_cloud_with_mask[:, 3] == 1, :3]
        # trimesh.points.PointCloud(subscene).show()
        # t=y
        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            # TODO: skip fliping as it does not make sense for 3DSSR.
            # if np.random.random() > 0.5:
            #     # Flipping along the YZ plane
            #     point_cloud[:, 0] = -1 * point_cloud[:, 0]
            #     target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            # if np.random.random() > 0.5:
            #     # Flipping along the XZ plane
            #     point_cloud[:, 1] = -1 * point_cloud[:, 1]
            #     target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # TODO: allow for more aggressive rotation
            # Rotation along up-axis/Z-axis
            if self.aggressive_rot:
                rot_angle = np.random.uniform(0, 2 * np.pi)
            else:
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

            # TODO: allow for rotation during evaluation.
        if self.augment_eval and self.aggressive_rot:
            np.random.seed(idx)
            rot_angle = np.random.uniform(0, 2 * np.pi)
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        # TODO: add point cloud with mask representing the subscene.
        ret_dict["point_clouds_with_mask"] = point_cloud_with_mask.astype(np.float32)
        # TODO: add the instance labels per point.
        ret_dict["instance_labels"] = instance_labels.astype(np.long)
        # TODO: add the radius of the subscene.
        ret_dict["subscene_radius"] = subscene_radius.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0 : instance_bboxes.shape[0]] = [
            self.dataset_config.nyu40id2class[int(x)]
            for x in instance_bboxes[:MAX_NUM_OBJ, -1][0 : instance_bboxes.shape[0]]
        ]
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        # TODO: take the scan name and rotation angle between query and target.
        ret_dict["scan_name"] = scan_name
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        return ret_dict