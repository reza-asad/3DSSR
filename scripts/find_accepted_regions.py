import os
import numpy as np

from scripts.helper import write_to_json


def is_region_accepted(xy_dim, pc_region, accepted_threshold=0.9):
    # consider a cube centered at the centered object with the specified xy_dim and see what percentage of the points
    # are inside that. if there ratio of the points inside is above the accepted threshold, reject the region.
    grid_extents = np.array([xy_dim, xy_dim])
    is_inside = np.abs(pc_region[:, :2]) < (grid_extents / 2.0)
    is_inside = np.sum(is_inside, axis=1) == 2

    return (np.sum(is_inside) / len(pc_region)) < accepted_threshold


def main(xy_dims, acceptance_threshold=0.9, save_results=False, final_xy_dim=None):
    crop_info = {xy_dim: {'num_rejected': 0, 'region_names': []} for xy_dim in xy_dims}
    num_total_regions = len(os.listdir(region_dir))

    region_names = os.listdir(region_dir)
    for region_name in region_names:
        pc_region = np.load(os.path.join(region_dir, region_name))
        for xy_dim in xy_dims:
            if is_region_accepted(xy_dim, pc_region, acceptance_threshold):
                crop_info[xy_dim]['region_names'].append(region_name)
            else:
                crop_info[xy_dim]['num_rejected'] += 1

    # record the ratio of accepted regions
    for xy_dim in xy_dims:
        crop_info[xy_dim]['accepted_ratio'] = 1 - crop_info[xy_dim]['num_rejected'] / num_total_regions

    # print each xy_dim and its accepted ratio:
    for xy_dim in xy_dims:
        print(xy_dim)
        print(crop_info[xy_dim]['accepted_ratio'])
        print(len(crop_info[xy_dim]['region_names']))
        print('*' * 50)

    # save accepted regions after speculating the crops info.
    if save_results:
        write_to_json(crop_info[final_xy_dim]['region_names'], results_path)


if __name__ == '__main__':
    # define paths
    dataset_name = 'matterport3d'
    data_dir = '../data/{}'.format(dataset_name)
    region_dir = os.path.join(data_dir, 'pc_regions', 'train')
    results_path = os.path.join(data_dir, 'accepted_regions.json')

    # try various xy dimensions to find the largest local and global crops that do not cover the entire scene.
    main(np.arange(0.5, 6.0, 0.5), save_results=False, final_xy_dim=2.0)
