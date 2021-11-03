import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

import utils
from region_dataset import Region
from transformer_models import Backbone, PointTransformerCls
from train_linear_classifier import compute_accuracy
from scripts.helper import load_from_json


def extract_feature_pipeline(args):
    # create train and test datasets
    dataset_train = Region(args.pc_dir, args.scene_dir, num_local_crops=0, num_global_crops=0, mode='train',
                           cat_to_idx=args.cat_to_idx, num_points=args.num_point)
    dataset_val = Region(args.pc_dir, args.scene_dir, num_local_crops=0, num_global_crops=0, mode=args.mode,
                         cat_to_idx=args.cat_to_idx, num_points=args.num_point)
    tr_sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    val_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)

    # create train and test dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=tr_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=val_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # load the encoder with pre-trained weights
    # TODO: load the model from DINO instead of the point transformer classifier
    model = Backbone(args)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key)
    model.eval()
    # classifier = PointTransformerCls(args)
    # classifier = torch.nn.DataParallel(classifier).cuda()
    # checkpoint = torch.load(args.pretrained_weights)
    # classifier.load_state_dict(checkpoint['model_state_dict'])
    # model = classifier.module.backbone
    # model.eval()

    # extract features
    print("Extracting features for train set...")
    train_features, train_labels = extract_features(model, data_loader_train)
    print("Extracting features for {} set...".format(args.mode))
    test_features, test_labels = extract_features(model, data_loader_val)

    # save features and labels
    if not os.path.exists(args.feature_dir):
        os.mkdir(args.feature_dir)
    if dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.feature_dir, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.feature_dir, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.feature_dir, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.feature_dir, "testlabels.pth"))

    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    all_labels = []
    for data in metric_logger.log_every(data_loader, 100):
        # load data
        crop = data['crops'][0]
        labels = data['labels'].squeeze(dim=1)

        # move data to the right device
        crop = crop.to(dtype=torch.float32).cuda()
        labels = labels.to(dtype=torch.long).cuda()

        output = model(crop)
        blocks = output[0].clone()
        features.append(blocks.mean(1))
        all_labels.append(labels)

    features = torch.cat(features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return features, all_labels


def compute_mean_accuracy_stats(df_metadata, mode, per_class_accuracy, cat_to_freq, k, topk=10):
    # take the topk categories by frequency
    topk_cats = [e[0] for e in sorted(cat_to_freq.items(), reverse=True, key=lambda x: x[1])[:topk]]

    # compute accuracy for topk and macro average accuracy for all cats.
    topk_accuracy = [per_class_accuracy[cat] for cat in topk_cats]
    topk_mean_accuracy = np.mean(topk_accuracy)
    mean_accuracy = np.mean(list(per_class_accuracy.values()))

    # compute micro average accuracy for all cats.
    # filter the metadata to only include test objects and accpeted cats
    df_metadata = df_metadata.loc[df_metadata['split'] == mode]
    df_metadata = df_metadata.loc[df_metadata['mpcat40'].apply(lambda x: x in per_class_accuracy)]
    # find the number of objects per accepted category.
    cat_to_freq_test = Counter(df_metadata['mpcat40'])
    num_total = df_metadata.shape[0]
    num_correct = 0
    for cat, freq in cat_to_freq_test.items():
        num_correct += per_class_accuracy[cat] * freq
    micro_accuracy = num_correct / num_total

    print('Displaying KNN accuracy for {} neighbours'.format(k))
    print('topk accuracy: {}'.format(topk_accuracy))
    print('topk mean accuracy: {}'.format(topk_mean_accuracy))
    print('Macro average accuracy: {}'.format(mean_accuracy))
    print('Micro average accuracy: {}'.format(micro_accuracy))

    return topk_mean_accuracy, mean_accuracy, micro_accuracy


@torch.no_grad()
def vanilla_knn_classifier(train_features, train_labels, test_features, k):
    # fit knn to the training data
    knn_cls = KNeighborsClassifier(n_neighbors=k)
    knn_cls.fit(train_features, train_labels)

    # predict knn on test features
    predictions = knn_cls.predict(test_features)

    return predictions


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    train_features = train_features.t()
    num_test_shapes, num_chunks = test_labels.shape[0], 100
    shapes_per_chunk = num_test_shapes // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    all_predictions = torch.zeros(num_test_shapes).cuda()
    for idx in range(0, num_test_shapes, shapes_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + shapes_per_chunk), num_test_shapes), :
        ]
        targets = test_labels[idx : min((idx + shapes_per_chunk), num_test_shapes)]
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
        all_predictions[idx : min((idx + shapes_per_chunk), num_test_shapes)] = predictions[:, 0]

    return all_predictions


def get_args():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on PointClouds')
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--accepted_cats_path', dest='accepted_cats_path',
                        default='../../data/matterport3d/accepted_cats.json')
    parser.add_argument('--cat_to_freq_path', dest='cat_to_freq_path',
                        default='../../data/matterport3d/accepted_cats_to_frequency.json')
    parser.add_argument('--pc_dir', dest='mesh_dir', default='../../data/matterport3d/pc_regions')
    parser.add_argument('--scene_dir', default='../../data/matterport3d/scenes')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../../data/matterport3d/metadata.csv')

    # point transformer arguments
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=4, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=512, type=int)
    parser.add_argument('--num_classes', dest='num_classes', default=28, type=int)

    # dino arguments
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='../../results/matterport3d/LearningBased/'
                                                        '3D_DINO_exact_regions_transformer_none_pretrain/'
                                                        'checkpoint0020.pth', type=str)
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--feature_dir', default='../../results/matterport3d/LearningBased/transformer_feats',
                        help="""If the features have 
    already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    args = parser.parse_args()
    return args


def main():
    # get the arguments
    args = get_args()
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # find a mapping from the accepted categories into indices
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(cat_to_idx)

    if os.path.exists(args.feature_dir):
        train_features = torch.load(os.path.join(args.feature_dir, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.feature_dir, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.feature_dir, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.feature_dir, "testlabels.pth"))
    else:
        # need to extract features !
        args.cat_to_idx = cat_to_idx
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    # convert to numpy
    train_features = train_features.cpu().detach().numpy()
    test_features = test_features.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()

    # read the metadata
    df_metadata = pd.read_csv(args.metadata_path)

    if utils.get_rank() == 0:
        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            predictions = vanilla_knn_classifier(train_features, train_labels, test_features, k)

            # find per class accuracy.
            per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
            compute_accuracy(per_class_accuracy, predictions, test_labels, cat_to_idx)
            per_class_accuracy_final = {cat: 0 for cat in cat_to_idx.keys()}
            for c, (num_correct, num_total) in per_class_accuracy.items():
                if num_total != 0:
                    per_class_accuracy_final[c] = float(num_correct) / num_total

            # compute stats on the mean accuracy and the mean accuracy on topk most frequent categories.
            cat_to_freq = load_from_json(args.cat_to_freq_path)
            compute_mean_accuracy_stats(df_metadata, args.mode, per_class_accuracy_final, cat_to_freq, k)
            print('*' * 50)

    dist.barrier()


if __name__ == '__main__':
    main()

