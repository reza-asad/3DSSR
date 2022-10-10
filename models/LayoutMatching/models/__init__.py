# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr, build_seed_corr

# TODO: adding alignment module as an option.
MODEL_FUNCS = {
    "3detr": build_3detr,
    "seed_corr": build_seed_corr
}


def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor