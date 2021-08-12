import os
import argparse
import trimesh
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils
from train_linear_classifier import compute_accuracy
from scripts.helper import load_from_json, sample_mesh
from transformations import PointcloudToTensor
from models import PointCapsNet


def compute_mean_accuracy_stats(per_class_accuracy, cat_to_freq, k, topk=10):
    # take the topk categories by frequency
    topk_cats = [e[0] for e in sorted(cat_to_freq.items(), reverse=True, key=lambda x: x[1])[:topk]]

    # print accuracy for topk, mean accuracy for topk and mean for all cats
    topk_accuracy = [per_class_accuracy[cat] for cat in topk_cats]
    topk_mean_accuracy = np.mean(topk_accuracy)
    mean_accuracy = np.mean(list(per_class_accuracy.values()))

    print('Displaying KNN accuracy for {} neighbours'.format(k))
    print('topk accuracy: {}'.format(topk_accuracy))
    print('topk mean accuracy: {}'.format(topk_mean_accuracy))
    print('mean accuracy: {}'.format(mean_accuracy))


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
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


def extract_latent_caps(accepted_cats, device, args):
    # create a dataloader to sample points on the mesh regions.
    transform = transforms.Compose([
        PointcloudToTensor(),
    ])
    dataset = Region(args.models_dir, accepted_cats, args.metadata_path, is_raw=True, num_points=args.num_points,
                     transform=transform)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    # load the capsule_net model that was trained using the 3D DINO pipeline.
    capsule_net = PointCapsNet(1024, 16, 64, 64, args.num_points)
    capsule_net.load_state_dict(torch.load(args.best_encoder))
    capsule_net = capsule_net.to(device=device)
    capsule_net.eval()

    # iterate thorough the data and apply 3D DINO to extract latent caps
    for i, data in enumerate(data_loader):
        # load the points
        pc = data['pc']
        file_names = data['file_names']
        pc = pc.to(device=device, dtype=torch.float32)

        # apply the trained model
        pc = pc.transpose(2, 1)
        latent_caps = capsule_net(pc).reshape(-1, 64 * 64)

        # save the latent caps
        if not os.path.exists(args.latent_caps_dir):
            os.makedirs(args.latent_caps_dir)

        for j in range(len(pc)):
            room_name = file_names[j].split('-')[0]
            obj_id = file_names[j].split('-')[1].split('.')[0]
            file_name = room_name + '-' + obj_id + '.npy'
            np.save(os.path.join(args.latent_caps_dir, file_name), latent_caps[j, ...].cpu().detach().numpy())


def load_latent_caps(accepted_cats, cat_to_idx, device, args):
    # create a dataloader to take the extracted latent caps.
    dataset = Region(args.latent_caps_dir, accepted_cats, args.metadata_path, is_raw=False, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    # iterate through the data and prepare the latent caps.
    all_caps = torch.zeros(len(dataset), 64*64, dtype=torch.float32, device=device)
    file_names = []
    for i, data in enumerate(data_loader):
        all_caps[i * args.batch_size: (i + 1) * args.batch_size, ...] = data['latent_caps']
        file_names += data['file_names']

    # create a dataframe of the extracted filename, their category and split.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata['file_names'] = df_metadata[['room_name', 'objectId']]. \
        apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]) + '.npy', axis=1)
    df_file_names = pd.DataFrame({'file_names': file_names})
    df = pd.merge(df_file_names, df_metadata, on='file_names')[['file_names', 'mpcat40', 'split']]
    df['labels'] = df['mpcat40'].apply(lambda cat: cat_to_idx[cat])

    # split the features and categories into train and test
    is_train = (df['split'] == 'train').values
    is_test = (df['split'] == args.mode).values
    train_features = all_caps[is_train, ...]
    test_features = all_caps[is_test, ...]
    train_labels = torch.from_numpy(df.loc[is_train, 'labels'].values)
    test_labels = torch.from_numpy(df.loc[is_test, 'labels'].values)

    train_labels = train_labels.to(device=device, dtype=torch.long)
    test_labels = test_labels.to(device=device, dtype=torch.long)

    return train_features, test_features, train_labels, test_labels


class Region(Dataset):
    def __init__(self, data_dir, accepted_cats, metadata_path, is_raw=False, num_points=None, transform=None):
        self.data_dir = data_dir
        self.accepted_cats = accepted_cats
        self.df_metadata = pd.read_csv(metadata_path)
        self.is_raw = is_raw
        self.num_points = num_points
        self.transform = transform
        if self.is_raw:
            self.file_names = self.filter_by_accepted_cats()
        else:
            self.file_names = os.listdir(self.data_dir)

    def filter_by_accepted_cats(self):
        self.df_metadata['file_names'] = self.df_metadata[['room_name', 'objectId']]. \
            apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]) + '.ply', axis=1)
        is_cat_accpeted = self.df_metadata['mpcat40'].apply(lambda x: x in self.accepted_cats)

        return self.df_metadata.loc[is_cat_accpeted, 'file_names'].values

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # read the raw mesh and sample points from it. otherwise, read the latent caps.
        data = {'file_names': self.file_names[idx]}
        if self.is_raw:
            # load the mesh region.
            mesh_region = trimesh.load(os.path.join(self.data_dir, self.file_names[idx]))

            # sample points on the mesh
            pc, _ = sample_mesh(mesh_region, self.num_points)

            # normalize the points
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            std = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc /= std
            data['pc'] = self.transform(pc)

        else:
            data['latent_caps'] = np.load(os.path.join(self.data_dir, self.file_names[idx])).reshape(-1)
            data['latent_caps'] = torch.from_numpy(data['latent_caps'])

        return data


def prepare_latent_caps(accepted_cats, cat_to_idx, device, args):
    if args.extract_features:
        extract_latent_caps(accepted_cats, device, args)

    # load the latent caps from their directory
    train_features, test_features, train_labels, test_labels = load_latent_caps(accepted_cats, cat_to_idx, device, args)

    return train_features, test_features, train_labels, test_labels


def get_args():
    # parser = OptionParser()
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on PointClouds')
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--extract_features', dest='extract_features', default=False, type=utils.bool_flag)
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')

    parser.add_argument('--accepted_cats_path', dest='accepted_cats_path',
                        default='../../data/matterport3d/accepted_cats.json')
    parser.add_argument('--cat_to_freq_path', dest='cat_to_freq_path',
                        default='../../data/matterport3d/accepted_cats_to_frequency.json')
    parser.add_argument('--models_dir', dest='models_dir', default='../../data/matterport3d/mesh_regions/all')
    parser.add_argument('--latent_caps_dir', dest='latent_caps_dir',
                        default='../../results/matterport3d/LearningBased/latent_caps_exact_regions_80DINO')
    parser.add_argument('--best_encoder', dest='best_encoder',
                        default='../../results/matterport3d/LearningBased/3D_DINO_exact_regions/'
                                'checkpoint0080_caps_net.pth')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../../data/matterport3d/metadata.csv')

    parser.add_argument('--num_points', dest='num_points', default=4096, type=int)
    parser.add_argument('--num_classes', dest='num_classes', default=28, type=int)
    parser.add_argument('--nb_knn', default=[10, 20, 100, 1000], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--freeze_last_layer', dest='freeze_last_layer', default=1, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=1000, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=4, type=int)

    args = parser.parse_args()
    return args


def main():
    # get the arguments
    args = get_args()

    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')

    # find a mapping from the accepted categories into indices
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}

    # prepare latent caps
    train_features, test_features, train_labels, test_labels = prepare_latent_caps(accepted_cats, cat_to_idx, device,
                                                                                   args)

    # apply knn for different value of k
    test_labels = test_labels.cpu().detach().numpy()
    for k in args.nb_knn:
        predictions = knn_classifier(train_features, train_labels, test_features, test_labels, k, args.temperature,
                                     args.num_classes)

        # find per class accuracy.
        predictions = predictions.cpu().detach().numpy()
        per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
        compute_accuracy(per_class_accuracy, predictions, test_labels, cat_to_idx)
        per_class_accuracy_final = {cat: 0 for cat in cat_to_idx.keys()}
        for c, (num_correct, num_total) in per_class_accuracy.items():
            if num_total != 0:
                per_class_accuracy_final[c] = float(num_correct) / num_total

        # compute stats on the mean accuracy and the mean accuracy on topk most frequent categories.
        cat_to_freq = load_from_json(args.cat_to_freq_path)
        compute_mean_accuracy_stats(per_class_accuracy_final, cat_to_freq, k)
        print('*' * 50)


if __name__ == '__main__':
    main()

