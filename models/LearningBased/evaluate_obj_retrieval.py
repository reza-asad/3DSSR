import os
import argparse
import numpy as np
import trimesh
from PIL import Image
import torch

from scripts.helper import load_from_json, write_to_json, create_img_table
from scripts.renderer import Render
from extract_point_transformer_features_old import extract_features_pipeline, create_data_loader


def retrieve_topk(query_idx, test_features, test_labels, file_names, topk, cat_to_idx):
    # take the query feature
    query_features = test_features[query_idx: query_idx + 1, :]

    # find the similarity between the query and target features.
    similarity = torch.mm(test_features, query_features.t())
    # take the topk most similar features.
    topk_sim, topk_idx = torch.topk(similarity, topk, dim=0)

    # record the label and the file names for topk features.
    results = []
    topk_labels = test_labels[topk_idx, 0]
    topk_files = file_names[topk_idx.cpu()]
    for i, idx1 in enumerate(topk_labels):
        for cat, idx2 in cat_to_idx.items():
            if idx1 == idx2:
                results.append((cat, topk_files[i][0]))

    return results


def retrieve_results(args, data_loader, checkpoints, query_dict):
    for checkpoint, checkpoint_number in checkpoints:
        # load or extract the features
        features_dir = '{}_{}'.format(args.features_dir_name, checkpoint_number)
        features_dir = os.path.join(args.cp_dir, args.results_folder_name, features_dir)
        if args.load_features:
            test_features = torch.load(os.path.join(features_dir, "testfeat.pth"))
            test_labels = torch.load(os.path.join(features_dir, "testlabels.pth"))
        else:
            # need to extract features !
            args.pretrained_weights_name = checkpoint
            args.dump_features = features_dir
            # if dumping feature, make sure the directory for dumping exists.
            if not os.path.exists(args.dump_features):
                try:
                    os.mkdir(args.dump_features)
                except FileExistsError:
                    pass
            test_features, test_labels = extract_features_pipeline(args, data_loader, checkpoint)

        # load the filenames
        file_names = load_from_json(os.path.join(args.cp_dir, args.results_folder_name, 'file_names.json'))
        file_names = np.asarray(file_names)

        # find the topk results for each query
        query_results = {cat: {} for cat in query_dict.keys()}
        for cat, query_file_names in query_dict.items():
            for query_file_name in query_file_names:
                # find the query idx
                for i, file_name in enumerate(file_names):
                    if file_name.split('.')[0] == query_file_name:
                        query_idx = i
                query_results[cat][query_file_name] = retrieve_topk(query_idx, test_features, test_labels, file_names,
                                                                    args.topk, args.cat_to_idx)

        # save the query results for the checkpoint
        write_to_json(query_results, os.path.join(args.cp_dir, args.results_folder_name, 'query_results_{}.json'
                                                  .format(checkpoint_number)))


def render_obj(scene):
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]
    # camera_pose[0:2, 3] = 0
    r = Render(rendering_kwargs)
    img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose,
                               room_dimension=room_dimension)
    return Image.fromarray(img)


def get_args():
    parser = argparse.ArgumentParser('OBJ retrieval', add_help=False)
    # paths
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats_top4.json')
    parser.add_argument('--pc_dir', default='../../data/{}/objects_pc')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--models_dir', default='../../data/{}/models')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata_equal_full_top4.csv')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', dest='results_folder_name',
                        default='3D_DINO_objects_equal_full_top4')
    parser.add_argument('--features_dir_name', default='features', type=str)
    parser.add_argument('--queries_path', default='../../queries/{}/queries_obj_val.json', type=str)

    # parameters
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--topk', dest='topk', default=10)
    parser.add_argument('--classifier_type', dest='classifier_type', default='dino',
                        help='supervised|dino')
    parser.add_argument('--batch_size_per_gpu', default=10, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--load_features', default=False)
    parser.add_argument('--render', default=False)
    parser.add_argument('--img_table', default=True)
    parser.add_argument('--retrieve', default=False)

    # transformer parameters
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=4, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=64, type=int)
    parser.add_argument('--crop_normalized', action='store_true', default=True)
    parser.add_argument('--max_coord', default=15.24, type=float, help='15.24 for MP3D| 5.02 for shapenetsem')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('DINO', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # map each category to an index
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.cat_to_idx = cat_to_idx
    args.num_class = len(cat_to_idx)

    # create data loaders.
    data_loader = create_data_loader(args)

    # compute object retrieval accuracy
    if args.classifier_type == 'supervised':
        checkpoints = [c for c in os.listdir(os.path.join(args.cp_dir, args.results_folder_name)) if 'best' in c]
        checkpoints = [(checkpoints[0], 0)]
    else:
        checkpoints = [c for c in os.listdir(os.path.join(args.cp_dir, args.results_folder_name)) if
                       (c.endswith('.pth')) and (len(c.split('.')[0]) > 10)]
        checkpoints = [(checkpoint, int(checkpoint.split('.')[0][10:])) for checkpoint in checkpoints]
        checkpoints = sorted(checkpoints, key=lambda x: x[1])

    # render results if necessary
    if args.render:
        # load results for each checkpoint
        for _, checkpoint_number in checkpoints:
            query_results = load_from_json(os.path.join(args.cp_dir, args.results_folder_name, 'query_results_{}.json'
                                                        .format(checkpoint_number)))
            render_dir = os.path.join(args.cp_dir, args.results_folder_name, 'rendered_results')
            if not os.path.exists(render_dir):
                os.mkdir(render_dir)
            for cat, results_info in query_results.items():
                render_dir_cat = os.path.join(render_dir, cat + '_{}'.format(checkpoint_number))
                if not os.path.exists(render_dir_cat):
                    os.mkdir(render_dir_cat)
                for query_file, results in results_info.items():
                    # save the query img
                    render_dir_cat_curr = os.path.join(render_dir_cat, query_file, 'imgs')
                    if not os.path.exists(render_dir_cat_curr):
                        os.makedirs(render_dir_cat_curr)
                    obj_path = os.path.join(args.models_dir, query_file+'.ply')
                    obj_mesh = trimesh.load(obj_path)
                    obj_scene = trimesh.Trimesh.scene(obj_mesh)
                    img = render_obj(obj_scene)
                    img_path = os.path.join(render_dir_cat_curr, 'query_'+query_file+'.png')
                    img.save(img_path)

                    # save the retrieved imgs
                    for i, (cat, file_name) in enumerate(results):
                        obj_path = os.path.join(args.models_dir, file_name.split('.')[0] + '.ply')
                        obj_mesh = trimesh.load(obj_path)
                        obj_scene = trimesh.Trimesh.scene(obj_mesh)
                        img = render_obj(obj_scene)
                        img_path = os.path.join(render_dir_cat_curr, 'top{}_'.format(i+1) + file_name.split('.')[0] + '.png')
                        img.save(img_path)
    if args.img_table:
        # load results for each checkpoint
        for _, checkpoint_number in checkpoints:
            query_results = load_from_json(os.path.join(args.cp_dir, args.results_folder_name, 'query_results_{}.json'
                                                        .format(checkpoint_number)))
            render_dir = os.path.join(args.cp_dir, args.results_folder_name, 'rendered_results')
            for cat, results_info in query_results.items():
                render_dir_cat = os.path.join(render_dir, cat + '_{}'.format(checkpoint_number))
                for query_file, results in results_info.items():
                    # take the images to create a table
                    render_dir_cat_curr = os.path.join(render_dir_cat, query_file, 'imgs')
                    imgs = []
                    captions = []
                    for i, (c, file_name) in enumerate(results):
                        captions.append(c)
                        img_name = 'top{}_'.format(i + 1) + file_name.split('.')[0] + '.png'
                        imgs.append(img_name)
                    query_img = 'query_' + query_file + '.png'
                    query_caption = cat
                    create_img_table(render_dir_cat_curr, 'imgs', imgs, 'img_table.html', ncols=4, captions=captions,
                                     with_query_scene=True, query_img=query_img, query_caption=query_caption,
                                     topk=len(imgs)-1)
    if args.retrieve:
        # load queries
        query_dict = load_from_json(os.path.join(args.queries_path))
        retrieve_results(args, data_loader, checkpoints, query_dict)


if __name__ == '__main__':
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    main()



