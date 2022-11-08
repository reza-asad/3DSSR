# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .matterport3d import MatterportDetectionDataset, MatterportDatasetConfig
from .matterport3d_real_queries import MatterportRealDetectionDataset, MatterportRealDatasetConfig

DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "matterport3d": [MatterportDetectionDataset, MatterportDatasetConfig],
    "matterport3d_real": [MatterportRealDetectionDataset, MatterportRealDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()

    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="train",
            root_dir=args.dataset_root_dir,
            meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=True
        ),
        args.test_split: dataset_builder(
            dataset_config,
            split_set=args.test_split,
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False,
            query_info=args.query_info,
            scene_dir=args.scene_dir
        ),
    }
    return dataset_dict, dataset_config
