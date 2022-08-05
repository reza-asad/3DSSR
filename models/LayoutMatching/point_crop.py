import os
import argparse
from time import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from query_target_scene_dataset import Scene
from models.LearningBased.transformer_models import PointTransformerSeg
from models.LearningBased.projection_models import DINOHead
from scripts.helper import load_from_json
import models.LearningBased.utils as utils
from classifier_subscene import Backbone, Classifier


def train_net(args):
    # create the training dataset
    dataset = Scene(scene_dir=args.scene_dir, pc_dir=args.pc_dir, metadata_path=args.metadata_path,
                    accepted_cats_path=args.accepted_cats_path, max_coord_box=args.max_coord_box,
                    max_coord_scene=args.max_coord_scene, num_points=args.num_point, crop_bounds=args.crop_bounds,
                    max_query_size=args.max_query_size, max_sample_size=args.max_sample_size,
                    random_subscene=args.random_subscene)

    dataset_len = len(dataset)
    print('{} train point clouds'.format(dataset_len))

    # create the dataloaders
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)

    # build the shape backbone.
    shape_backbone = PointTransformerSeg(args)

    # arguments for the layout matching model.
    layout_args = argparse.Namespace()
    exceptions = {'input_dim': 35, 'nblocks': 0}
    for k, v in vars(args).items():
        if k in exceptions:
            vars(layout_args)[k] = exceptions[k]
        else:
            vars(layout_args)[k] = v

    # build layout matching backbone.
    layout_backbone = PointTransformerSeg(layout_args)

    # combine the backbones and attach a projection module to that.
    backbone = Backbone(shape_backbone=shape_backbone, layout_backbone=layout_backbone, num_point=args.num_point,
                        num_objects=args.max_sample_size + args.max_query_size)
    projection = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                          norm_last_layer=args.norm_last_layer)
    model = Classifier(backbone=backbone, head=projection)
    model = torch.nn.DataParallel(model).cuda()

    # load the pre-trained classification weights
    checkpoint_path = os.path.join(args.cp_dir, args.results_folder_name_pret, args.pre_training_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # replace the projection head from classifier with one that will learn layout matching.
    model.module.head = DINOHead(in_dim=args.transformer_dim, out_dim=args.out_dim, use_bn=args.use_bn_in_head,
                                 norm_last_layer=args.norm_last_layer)

    # freeze the backbone weights if necessary but let the projection module learn new weights.
    if args.freeze_backbone:
        model.module.backbone.requires_grad_(False)
    else:
        model.module.backbone.requires_grad_(True)
    model.module.head.requires_grad_(True)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # define the optimizer and the scheduler.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    # check if training is form scratch or from the best model sofar.
    try:
        checkpoint = torch.load(os.path.join(args.cp_dir, args.results_folder_name, args.checkpoint))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print('Use pretrain model')
    except FileNotFoundError:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    # track losses and use that along with patience to stop early.
    global_epoch = 0
    global_step = 0

    print('Start training...')
    for epoch in range(start_epoch, args.epochs):
        t1 = time()
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epochs))
        epoch_loss = 0

        for i, data in enumerate(loader):
            # load data
            pc_t = data['pc_t'].squeeze(dim=0)
            centroid_t = data['centroid_t'].squeeze(dim=0)
            match_label_t = data['match_label_t'].squeeze(dim=0)

            pc_q = data['pc_q'].squeeze(dim=0)
            centroid_q = data['centroid_q'].squeeze(dim=0)
            match_label_q = data['match_label_q'].squeeze(dim=0)

            # print(pc_t.shape, centroid_t.shape, match_label_t.shape)
            # print(pc_q.shape, centroid_q.shape, match_label_q.shape)

            # move data to the right device
            pc_t = pc_t.to(dtype=torch.float32).cuda()
            centroid_t = centroid_t.to(dtype=torch.float32).cuda()
            match_label_t = match_label_t.to(dtype=torch.bool).cuda()

            pc_q = pc_q.to(dtype=torch.float32).cuda()
            centroid_q = centroid_q.to(dtype=torch.float32).cuda()
            match_label_q = match_label_q.to(dtype=torch.bool).cuda()

            # apply the model on query subscene and target subscenes.
            pc_q_t = torch.cat([pc_q, pc_t], dim=0)
            centroid_q_t = torch.cat([centroid_q, centroid_t], dim=0)
            q_t = model(pc_q_t, centroid_q_t)
            q = q_t[:args.max_query_size, :]
            t = q_t[args.max_query_size:, :]

            # filter the output to matching objects.
            q = q[match_label_q, :]
            t = t[match_label_t, :]

            # representation loss.
            repr_loss = F.mse_loss(q, t)

            # std loss
            q = q - q.mean(dim=0)
            t = t - t.mean(dim=0)
            std_q = torch.sqrt(q.var(dim=0) + 0.0001)
            std_t = torch.sqrt(t.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_q)) / 2 + torch.mean(F.relu(1 - std_t)) / 2

            # cov loss.
            batch_size = len(q)
            cov_q = (q.T @ q) / (batch_size - 1)
            cov_t = (t.T @ t) / (batch_size - 1)
            cov_loss = off_diagonal(cov_q).pow_(2).sum().div(args.out_dim) + \
                       off_diagonal(cov_t).pow_(2).sum().div(args.out_dim)

            # combined loss.
            loss = (args.sim_coeff * repr_loss + args.std_coeff * std_loss + args.cov_coeff * cov_loss)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # keep track of the losses
            epoch_loss += loss.item()

        scheduler.step()
        print('Epoch %d finished! - Loss: %.10f' % (epoch + 1, epoch_loss / (i + 1)))
        t2 = time()
        print("Training epoch took %.3f minutes" % ((t2 - t1) / 60))

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(state, os.path.join(args.cp_dir, args.results_folder_name, 'checkpoint.pth'))
        global_epoch += 1
        print('-' * 50)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_args():
    parser = argparse.ArgumentParser('Point Transformer Classification', add_help=False)
    # Data
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path',  default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--cp_dir', default='../../results/{}/LayoutMatching/')
    parser.add_argument('--results_folder_name', default='exp')
    parser.add_argument('--results_folder_name_pret', default='exp_supervise_pret')
    parser.add_argument('--checkpoint', default='checkpoint.pth')
    parser.add_argument('--pre_training_checkpoint', default='CP_best.pth')

    # Model
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--out_dim', default=2000, type=int)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--max_coord_box', default=3.65, type=float, help='3.65 for MP3D')
    parser.add_argument('--max_coord_scene', default=13.07, type=float, help='13.07 for MP3D')
    parser.add_argument('--crop_bounds', type=float, nargs='+', default=(0.9, 0.9))
    parser.add_argument('--max_query_size', default=5, type=int)
    parser.add_argument('--max_sample_size', default=5, type=int)

    # Optim
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--random_subscene', default=False, type=utils.bool_flag)
    parser.add_argument('--freeze_backbone', default=True, type=utils.bool_flag)

    # Loss
    parser.add_argument("--sim_coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std_coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov_coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')

    return parser


def adjust_paths(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v):
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Point Transformer Classification', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args)

    # create a directory for checkpoints
    output_dir = os.path.join(args.cp_dir, args.results_folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    args.cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(args.cat_to_idx)

    # find a mapping from the region files to their indices.
    df = pd.read_csv(args.metadata_path)
    is_accepted = df['mpcat40'].apply(lambda x: x in args.cat_to_idx)
    df = df.loc[is_accepted]
    df = df.loc[(df['split'] == 'train') | (df['split'] == 'val')]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_name_to_idx = {file_name: i for i, file_name in enumerate(file_names)}
    args.file_name_to_idx = file_name_to_idx

    # time the training
    t = time()
    train_net(args)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



