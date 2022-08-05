import os
import argparse
import numpy as np
import pandas as pd
import trimesh

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


def add_gt_aabbs(args, mode):
    # load the accepted cats
    accepted_cats = load_from_json(args.accepted_cats_path)

    scene_names = os.listdir(os.path.join(args.scene_dir_gt, mode))
    for scene_name in scene_names:
        # load the scene.
        scene = load_from_json(os.path.join(args.scene_dir_gt, mode, scene_name))

        # add the axis aligned bounding box (aabb) for each accepted object
        scene_out = {}
        for obj in scene.keys():
            if scene[obj]['category'][0] in accepted_cats:
                # copy the object metadata if it has an accepted category.
                scene_out[obj] = scene[obj]

                # load the region corresponding to the object
                region_file_name = scene_out[obj]['file_name'].split('.')[0] + '.npy'
                pc = np.load(os.path.join(args.pc_dir, mode, region_file_name))

                # load the translation that brings the centroid of the region from origin to its location in the
                # scene.
                transform = np.asarray(scene_out[obj]['transform'], dtype=np.float64).reshape(4, 4).transpose()
                centroid = transform[0:3, 3]

                # find the aabb for the pc
                pc = trimesh.points.PointCloud(pc)
                pc.apply_transform(transform)
                vertices = pc.bounding_box.vertices.tolist()
                scene_out[obj]['aabb'] = [centroid.tolist()] + vertices

        # save the scene
        write_to_json(scene_out, os.path.join(args.scene_dir_output, mode, scene_name))


def visualize_boxes(args, scene_name, boxes, num_boxes=10):
    mesh = trimesh.load(os.path.join(args.room_dir, '{}/{}.annotated.ply'.format(scene_name, scene_name)))
    mesh.show()
    boxes_vis = []
    for i, box in enumerate(boxes):
        if i > num_boxes:
            continue
        transformation = np.eye(4)
        transformation[:3, 3] = box[:3]
        scale = box[3:]
        box_vis = trimesh.creation.box(scale, transform=transformation)
        boxes_vis.append(box_vis)
    scene = trimesh.Scene([mesh] + boxes_vis)
    scene.show()


def add_pred_aabbs(args, mode, predicted_boxes):
    # find a mapping from int labels to actual categories.
    cats = load_from_json(args.accepted_cats_path)
    labels = np.arange(len(cats))
    label_to_cat = dict(zip(labels, sorted(cats)))

    # for each scene record the boxes, their ids.
    for scene_name, scene_info in predicted_boxes.items():
        scene = {}
        for i, box in enumerate(scene_info['boxes']):
            # find the vertices of the box.
            transformation = np.eye(4)
            transformation[:3, 3] = box[:3]
            scale = box[3:]
            box_trimesh = trimesh.creation.box(scale, transform=transformation)
            centroid = box_trimesh.centroid.tolist()
            vertices = box_trimesh.vertices.tolist()
            scene[i+1] = {'aabb': [centroid] + vertices}

            # add predicted category.
            label = scene_info['labels'][i]
            cat = label_to_cat[label]
            scene[i+1]['predicted_category'] = [cat]

            # add the file name for the box
            scene[i + 1]['file_name'] = '{}-{}.ply'.format(scene_name, i+1)

        # save the scene.
        write_to_json(scene, os.path.join(args.scene_dir_output, mode, '{}.json'.format(scene_name)))


def clean_up_scenes(args, mode):
    # find all boxes with objects in them.
    file_names = set(os.listdir(os.path.join(args.pc_dir, mode)))

    scene_names = os.listdir(os.path.join(args.scene_dir_in, mode))
    for scene_name in scene_names:
        # load the gt scene.
        scene_gt = load_from_json(os.path.join(args.scene_dir_gt, mode, scene_name))

        # load the raw predicted scene.
        scene_in = load_from_json(os.path.join(args.scene_dir_in, mode, scene_name))

        # build the box for each predicted object.
        scene_out = {}
        for obj, obj_info in scene_in.items():
            # find the pc file name for the objects within the box
            file_name = '{}-{}.npy'.format(scene_name.split('.')[0], obj)

            # proceed with adding the predicted object if the box is non-empty
            if file_name in file_names:
                # initialize the gt category for the predicted box.
                for threshold in args.thresholds:
                    cat_key = 'category_{}'.format(threshold)
                    obj_info[cat_key] = []

                # build the box for the predicted object.
                vertices = np.asarray(obj_info['aabb'])
                aabb = Box(vertices)

                # find the IoU between the gt and each predicted box. if IoU above threshold take the gt category.
                for obj_gt, obj_info_gt in scene_gt.items():
                    # build the box for gt object.
                    vertices_gt = np.asarray(obj_info_gt['aabb'])
                    aabb_gt = Box(vertices_gt)

                    # proceed with adding the gt category if IoU above threshold.
                    iou = IoU(aabb_gt, aabb).iou()
                    for threshold in args.thresholds:
                        if iou >= threshold:
                            cat_key = 'category_{}'.format(threshold)
                            gt_cat = obj_info_gt['category'][0]
                            obj_info[cat_key].append(gt_cat)

                # record the output scene.
                scene_out[obj] = obj_info

        # save the scene.
        write_to_json(scene_out, os.path.join(args.scene_dir_output, mode, scene_name))


def add_metadata_df(args, mode, prev_frame_data):
    # gather data for metadata.
    pc_dir_curr = os.path.join(args.pc_dir, mode)
    file_names = os.listdir(pc_dir_curr)
    frame_data = {'room_name': [], 'objectId': [], 'split': []}
    for file_name in file_names:
        scene_name, obj = file_name.split('.')[0].split('-')
        frame_data['room_name'].append(scene_name)
        frame_data['objectId'].append(obj)
        frame_data['split'].append('test')

    # combine the data to form a frame.
    for k, v in frame_data.items():
        prev_frame_data[k] += v

    return prev_frame_data


def get_args():
    parser = argparse.ArgumentParser('Adding metadata', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--data_root', default='../../data/{}/')
    parser.add_argument('--results_dir', default='../../results/{}')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../../data/{}/metadata_predicted_nms.csv')
    parser.add_argument('--scene_dir_gt', default='scenes')
    parser.add_argument('--scene_dir_in', default='scenes_predicted_nms_raw')
    parser.add_argument('--scene_dir_output', default='scenes')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_objects')
    parser.add_argument('--room_dir', default='/media/reza/Large/{}/rooms')
    parser.add_argument('--predicted_boxes_path', default='predicted_boxes_large.npy')
    parser.add_argument('--add_gt_aabb', action='store_true', default=True)
    parser.add_argument('--add_pred_aabb', action='store_true', default=False)
    parser.add_argument('--clean_up_predicted_scenes', action='store_true', default=False)
    parser.add_argument('--add_metadata', action='store_true', default=False)
    parser.add_argument('--thresholds', default=[0.25, 0.5], type=float, nargs='+')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Adding Metadata', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.scene_dir_gt = os.path.join(args.data_root, args.scene_dir_gt)
    args.scene_dir_in = os.path.join(args.results_dir, args.scene_dir_in)
    args.scene_dir_output = os.path.join(args.results_dir, args.scene_dir_output)
    args.predicted_boxes_path = os.path.join(args.results_dir, args.predicted_boxes_path)

    # make sure the output directory exists.
    modes = ['train', 'val']
    for mode in modes:
        if not os.path.exists(os.path.join(args.scene_dir_output, mode)):
            os.makedirs(os.path.join(args.scene_dir_output, mode))

    # add AABB's if necessary.
    frame_data = {'room_name': [], 'objectId': [], 'split': []}
    for mode in modes:
        if args.add_gt_aabb:
            add_gt_aabbs(args, mode)
        if args.add_pred_aabb:
            predicted_boxes = np.load(args.predicted_boxes_path, allow_pickle=True).item()
            add_pred_aabbs(args, mode, predicted_boxes)
        if args.clean_up_predicted_scenes:
            clean_up_scenes(args, mode)
        if args.add_metadata:
            frame_data = add_metadata_df(args, mode, frame_data)

    if args.add_metadata:
        df = pd.DataFrame(frame_data)
        df.sort_values(by='room_name', inplace=True)
        df.to_csv(args.metadata_path, index=False)


if __name__ == '__main__':
    main()
