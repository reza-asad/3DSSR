import json

from eval_knn_transformer import *


def main(args):
    # find all the checkpoints
    checkpoints = [e for e in os.listdir(args.cp_dir) if (e.endswith('.pth'))
                   and (len(e.split('.')[0]) > 10)]
    checkpoints_number = [(checkpoint, int(checkpoint.split('.')[0][10:])) for checkpoint in checkpoints]
    checkpoints_number = sorted(checkpoints_number, key=lambda x: x[1])
    checkpoints = list(zip(*checkpoints_number))[0]

    # load features if necessary.
    knn_accuracies = {}
    for checkpoint in checkpoints:
        # if loading features find the directory to load from
        checkpoint_number = int(checkpoint.split('.')[0][10:])
        if checkpoint_number <= args.cp_offset:
            continue
        features_dir = '{}_{}'.format(args.features_dir_name, checkpoint_number)
        features_dir = os.path.join(args.cp_dir, features_dir)
        if args.load_features:
            train_features = torch.load(os.path.join(features_dir, "trainfeat.pth"))
            test_features = torch.load(os.path.join(features_dir, "testfeat.pth"))
            train_labels = torch.load(os.path.join(features_dir, "trainlabels.pth"))
            test_labels = torch.load(os.path.join(features_dir, "testlabels.pth"))
        else:
            args.pretrained_weights_name = checkpoint
            args.dump_features = features_dir
            # if dumping feature, make sure the directory for dumping exists.
            if not os.path.exists(args.dump_features):
                try:
                    os.mkdir(args.dump_features)
                except FileExistsError:
                    pass
            train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

        if utils.get_rank() == 0:
            if args.use_cuda:
                train_features = train_features.cuda()
                test_features = test_features.cuda()
                train_labels = train_labels.cuda()
                test_labels = test_labels.cuda()

            # apply knn.
            print('Checkpoint: {}'.format(checkpoint))
            knn_accuracies[checkpoint] = {}
            for k in args.nb_knn:
                top1, top5 = knn_classifier(train_features, train_labels, test_features, test_labels, k,
                                            args.temperature)
                knn_accuracies[checkpoint][k] = top1
                print('KNN with K: {}'.format(k))
                print('TOP1 Accuracy is {}'.format(top1))

    # save the knn accuracies
    if args.cp_offset <= 0:
        with open(os.path.join(args.cp_dir, 'knn_accuracies.json'), 'w') as f:
            json.dump(knn_accuracies, f, indent=4)


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on Matterport3D')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--accepted_cats_path', dest='accepted_cats_path',
                        default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', dest='pc_dir', default='../../data/{}/objects_pc')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', dest='results_folder_name', default='3D_DINO_objects_default')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--features_dir_name', default='', type=str, help="name of the directory to dump and load "
                                                                          "features from.")

    # point transformer arguments
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--num_classes', dest='num_classes', default=28, type=int)
    parser.add_argument('--crop_normalized', default=False, type=utils.bool_flag)
    parser.add_argument('--max_coord', default=14.30, type=float, help='14.30 for MP3D| 5.02 for shapenetsem')

    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--cp_offset', default=0, type=int, help='only add epochs after this checkpoint')

    args = parser.parse_args()
    adjust_paths(args, exceptions=['dist_url'])

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # find a mapping from the accepted categories into indices
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.cat_to_idx = cat_to_idx
    args.num_class = len(cat_to_idx)

    # prepare the checkpoint dir
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name)

    main(args)
