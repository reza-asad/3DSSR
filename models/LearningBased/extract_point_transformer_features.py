import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from scripts.helper import load_from_json, write_to_json
import utils
from eval_knn_transformer import extract_features
from transformer_models import PointTransformerSeg
from projection_models import DINOHead
from eval_knn_transformer import ReturnIndexDataset, ReturnIndexDatasetNorm


def create_data_loader(args):
    # ============ preparing data ... ============
    if args.crop_normalized:
        dataset_train = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, args.metadata_path,
                                               max_coord=args.max_coord, num_local_crops=0, num_global_crops=0,
                                               mode='train', cat_to_idx=args.cat_to_idx, num_points=args.num_point)
        dataset_val = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                                             num_local_crops=0, num_global_crops=0, mode=args.mode,
                                             cat_to_idx=args.cat_to_idx, num_points=args.num_point)
    else:
        dataset_train = ReturnIndexDataset(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0,
                                           num_global_crops=0, mode='train', cat_to_idx=args.cat_to_idx,
                                           num_points=args.num_point)
        dataset_val = ReturnIndexDataset(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0,
                                         num_global_crops=0, mode=args.mode, cat_to_idx=args.cat_to_idx,
                                         num_points=args.num_point)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val regions.")

    return data_loader_train, data_loader_val


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    all_labels = []
    all_file_names = []
    for data, index in metric_logger.log_every(data_loader, 10):
        all_file_names += data['file_name'].copy()
        samples = data['crops'][0]
        samples = samples.cuda(non_blocking=True)

        labels = data['labels'].long().cuda()
        all_labels.append(labels)

        index = index.cuda(non_blocking=True)

        output = model(samples)
        feats = output.mean(1)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    all_labels = torch.cat(all_labels, dim=0)

    return features, all_labels, all_file_names


def extract_features_pipeline(args, data_loader_train, data_loader_val):
    # initialize a point transformer model
    if args.classifier_type == 'supervised':
        backbone = PointTransformerSeg(args)
        head = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                        norm_last_layer=args.norm_last_layer)
        classifier = utils.DINO(backbone, head, num_local_crops=0, num_global_crops=1, network_type='teacher')
        classifier = torch.nn.DataParallel(classifier).cuda()
        checkpoint = torch.load(os.path.join(args.cp_dir, args.results_folder_name, args.checkpoint))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        model = classifier.module.backbone
    else:
        checkpoint_path = os.path.join(args.cp_dir, args.results_folder_name, args.checkpoint)
        backbone = PointTransformerSeg(args)
        head = DINOHead(in_dim=args.transformer_dim, out_dim=args.out_dim, use_bn=args.use_bn_in_head)
        DINO = utils.DINO(backbone, head, num_local_crops=0, num_global_crops=0, network_type='teacher')
        DINO.cuda()

        utils.load_pretrained_weights(DINO, checkpoint_path, 'teacher')
        model = DINO.backbone

    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_labels, train_file_names = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features, test_labels, test_file_names = extract_features(model, data_loader_val)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    features_dir = os.path.join(args.cp_dir, args.results_folder_name, args.features_dir_name)
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)
    if dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(features_dir, "trainfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(features_dir, "trainlabels.pth"))
        write_to_json(train_file_names, os.path.join(features_dir, "train_file_names.json"))

        torch.save(test_features.cpu(), os.path.join(features_dir, "testfeat.pth"))
        torch.save(test_labels.cpu(), os.path.join(features_dir, "testlabels.pth"))
        write_to_json(test_file_names, os.path.join(features_dir, "test_file_names.json"))

    print("Features are ready!.")

    return train_features, train_labels, train_file_names, test_features, test_labels, test_file_names


def sample_training_data(train_features, train_labels, sample_ratio):
    # map each label to the data indices of that label.
    label_to_indices = {}
    for i, label in enumerate(train_labels):
        label = label[0].item()
        if label not in label_to_indices:
            label_to_indices[label] = [i]
        else:
            label_to_indices[label].append(i)

    sampled_indices = []
    # sample from each label.
    for label, indices in label_to_indices.items():
        sample_size = int(np.ceil(sample_ratio * len(indices)))
        sample = np.random.choice(indices, sample_size, replace=False)
        sampled_indices.append(sample)
    sampled_indices = np.concatenate(sampled_indices)

    return train_features[sampled_indices, ...], train_labels[sampled_indices, ...]


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    top1, total = 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    all_predictions = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        all_predictions.append(predictions[:, 0:1])

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    all_predictions = torch.cat(all_predictions, dim=0)

    return top1, all_predictions


def get_args():
    parser = argparse.ArgumentParser('Extracting Features Labels and Predicted Labels', add_help=False)
    # paths
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats_top10.json')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', dest='results_folder_name',
                        default='region_classification_transformer_2_32_4096_non_equal_full_region_top10')
    parser.add_argument('--features_dir_name', default='features', type=str)
    parser.add_argument('--output_file_name', default='predicted_labels_knn.json', type=str)
    parser.add_argument('--checkpoint', default='CP_best.pth', type=str)

    # parameters
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--classifier_type', dest='classifier_type', default='supervised', help='supervised|dino')
    parser.add_argument('--batch_size_per_gpu', default=8, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--sample_train', action='store_true', default=False)
    parser.add_argument('--sample_ratio', default=1.0, type=float)
    parser.add_argument('--load_features', action='store_true', default=False)

    # transformer parameters
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--crop_normalized', action='store_true', default=True)
    parser.add_argument('--max_coord', default=14.30, type=float, help='14.30 MP3D| 5.02 shapenetsem')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--out_dim', dest='out_dim', default=2000, type=int)

    # knn params
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in the voting coefficient')
    parser.add_argument('--nb_knn', default=20, type=int, help='Number of NN to use. 20 is usually working the best.')

    # the rest
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Extracting Features Labels and Predicted Labels', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=['dist_url'])

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # map each category to an index
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.cat_to_idx = cat_to_idx
    args.num_class = len(cat_to_idx)

    # create data loader
    data_loader_train, data_loader_val = create_data_loader(args)

    if args.load_features:
        features_dir = os.path.join(args.cp_dir, args.results_folder_name, args.features_dir_name)
        train_features = torch.load(os.path.join(features_dir, "trainfeat.pth"))
        test_features = torch.load(os.path.join(features_dir, "testfeat.pth"))
        train_labels = torch.load(os.path.join(features_dir, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(features_dir, "testlabels.pth"))
        test_file_names = load_from_json(os.path.join(features_dir, "test_file_names.json"))
    else:
        # need to extract features !
        train_features, train_labels, train_file_names, test_features, test_labels, test_file_names = \
            extract_features_pipeline(args, data_loader_train, data_loader_val)

    if utils.get_rank() == 0:
        # determine if you sample from the training data as opposed to taking it all.
        if args.sample_train:
            train_features, train_labels = sample_training_data(train_features, train_labels, args.sample_ratio)

        # apply knn and save the predicted results.
        top1, predictions = knn_classifier(train_features, train_labels, test_features, test_labels, args.nb_knn,
                                           args.temperature, args.num_class)
        print(f"{args.nb_knn}-NN classifier result: Top1: {top1}")

        # save predictions.
        file_name_to_predictions = {}
        for i, prediction in enumerate(predictions):
            file_name_to_predictions[test_file_names[i]] = prediction[0].item()
        write_to_json(file_name_to_predictions, os.path.join(args.cp_dir, args.results_folder_name,
                                                             args.features_dir_name, args.output_file_name))

        dist.barrier()


if __name__ == '__main__':
    main()

