# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr, build_alignment_module

# TODO: adding alignment module as an option.
MODEL_FUNCS = {
    "3detr": build_3detr,
    "alignment": build_alignment_module
}


def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor