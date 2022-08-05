import os
import argparse
import shutil

import numpy as np
import trimesh
from PIL import Image
from torch.utils.data import DataLoader

from region_dataset import Region
from models.LearningBased.region_dataset_normalized_crop import Region as RegionNorm

from scripts.helper import load_from_json
from train_3D_DINO_transformer_distributed import collate_fn
import utils
from scripts.renderer import Render
from scripts.render_scene_functions import render_single_pc


def create_img_table(table_dir, html_file_name='img_table.html', ncols=5):
    def insert_into_table(file, img_path, caption=None):
        file.write('\n')
        file.write('<td align="center" valign="center">\n')
        file.write('<img src="{}" />\n'.format(img_path))
        file.write('<br />\n')
        # add caption
        file.write('<br />\n')
        if caption is not None:
            file.write(caption)
            file.write('<br />\n')
        file.write('</td>\n')
        file.write('\n')

    html_file_path = os.path.join(table_dir, html_file_name)
    with open(html_file_path, 'w+') as f:
        # add the table
        f.write('<table width="500" border="0" cellpadding="5">\n')

        # for each scene insert its image and the global and local crops corresponding to that.
        region_dir = os.path.join(table_dir, 'pc_region')
        img_names_scene = os.listdir(region_dir)
        for img_name_scene in img_names_scene:
            # insert the region into the table
            f.write('<tr>\n')
            img_path = os.path.join('pc_region', img_name_scene)
            scene_name = img_name_scene.split('.')[0]
            insert_into_table(f, img_path, 'Region ' + '<br />\n' + scene_name)

            # insert the global and local crops
            for crop_type in ['global', 'local']:
                crop_dir = os.path.join(table_dir, crop_type)
                for img_type in ['region_crop', 'subpc']:
                    # find the local/global images of certain type (e.g., pc)
                    imgs = os.listdir(os.path.join(table_dir, crop_dir, img_type))
                    # take only the global/local imgs corresponding to the current scene.
                    cropped_img_names_scene = [img for img in imgs if img.split('*')[0] == scene_name]
                    # sort the crops by their number
                    cropped_img_names_scene = sorted(cropped_img_names_scene)

                    # insert the images for the crop type into table.
                    nrows = int(np.ceil(len(cropped_img_names_scene) / ncols))
                    for i in range(nrows):
                        f.write('<tr>\n')
                        for j in range(ncols):
                            if i * ncols + j >= len(cropped_img_names_scene):
                                break
                            img_path = os.path.join(crop_type, img_type, cropped_img_names_scene[i*ncols+j])
                            insert_into_table(f, img_path)
                        f.write('</tr>\n')

        # end the table
        f.write('</table>\n')


def save_rendered_images(results_dir, results_to_render):
    # make a directory for entire scene
    for img_type in ['pc_region']:
        os.makedirs(os.path.join(results_dir, img_type))

    # make a directory for local and global crops
    dirs = ['region_crop', 'subpc']
    for crop_type in ['local', 'global']:
        for dir_ in dirs:
            img_dir = os.path.join(results_dir, crop_type, dir_)
            os.makedirs(img_dir)

    # add images in their corresponding directory
    for results_info in results_to_render:
        # add the scene image
        scene_name = results_info['scene_name'].split('.')[0]
        for img_type in ['pc_region']:
            scene_img = results_info[img_type]
            img_path = os.path.join(results_dir, img_type, scene_name+'.png')
            scene_img.save(img_path)

        # add the local and global crops
        for crop_type in ['local', 'global']:
            for i, template in enumerate(results_info['crops'][crop_type]):
                for img_type, img in template.items():
                    img_path = os.path.join(results_dir, crop_type, img_type, scene_name+'*{}.png'.format(i))
                    img.save(img_path)


def render_cubical_multicrop(args):
    # create dataset
    if args.crop_normalized:
        dataset = RegionNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                             num_local_crops=args.local_crops_number, num_global_crops=args.global_crops_number,
                             mode='train', num_points=args.num_point, global_crop_bounds=args.global_crop_bounds,
                             local_crop_bounds=args.local_crop_bounds, cat_to_idx=args.cat_to_idx, save_crops=True)
    else:
        dataset = Region(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=args.local_crops_number,
                         num_global_crops=args.global_crops_number, mode='train', num_points=args.num_point,
                         global_crop_bounds=args.global_crop_bounds, local_crop_bounds=args.local_crop_bounds,
                         cat_to_idx=args.cat_to_idx, save_crops=True)

    # create the dataloader
    data_loader = DataLoader(dataset,
                             batch_size=8,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=collate_fn,
                             shuffle=False)

    # iterate through the data to collect all the 3D data that is going to be rendered.
    num_curr_files = 0
    for i, data in enumerate(data_loader):
        num_curr_files += len(data['file_name'])
        if num_curr_files >= args.num_files:
            break
    dataset.results_to_render = dataset.results_to_render[:args.num_files]

    # initialize the renderer.
    r = Render(rendering_kwargs)

    # render the scene and all local and global crops
    results_to_render = []
    for results_info in dataset.results_to_render:
        # populate a dictionary with the rendered images.
        results_to_render.append({'scene_name': results_info['scene_name'], 'pc_region': ...,
                                 'crops': {'global': [], 'local': []}})

        # render the point cloud region.
        pc = results_info['pc_region']
        img = render_single_pc(pc, resolution, rendering_kwargs, region=True)
        results_to_render[-1]['pc_region'] = Image.fromarray(img)

        # render the local and global crops.
        for crop_type in ['local', 'global']:
            for crop_info in results_info['crops'][crop_type]:
                template = dict(region_crop=..., subpc=...)

                # render the pc_region along with the crop.
                img = render_single_pc(pc, resolution, rendering_kwargs, region=True, with_obbox=True,
                                       obbox=crop_info['cube'])
                template['region_crop'] = Image.fromarray(img)

                # render the subpc
                img = render_single_pc(crop_info['subpc'], resolution, rendering_kwargs, region=False)
                template['subpc'] = Image.fromarray(img)

                # add the template for the corresponding crop
                results_to_render[-1]['crops'][crop_type].append(template)

    # save the rendered images.
    save_rendered_images(args.results_dir, results_to_render)

    # create an html table from the collected images.
    create_img_table(args.results_dir)


def get_args():
    parser = argparse.ArgumentParser('Rendering MultiCrop', add_help=False)

    # path parameters
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats_top4.json')
    parser.add_argument('--pc_dir', default='../../data/{}/objects_pc')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata_non_equal_full_top4.csv')
    parser.add_argument('--results_dir', default='../../results/{}/multicrop_rendering/')
    parser.add_argument('--results_folder_name', default='regions_top4')

    # cropping strategy and data type
    parser.add_argument('--data_type', default='3D')
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--cropping_strategy', default='rectangular')
    parser.add_argument('--local_crops_number', default=5, type=int)
    parser.add_argument('--global_crops_number', default=2, type=int)
    parser.add_argument('--local_crop_bounds', type=float, nargs='+', default=(0.4, 0.4))
    parser.add_argument('--global_crop_bounds', type=float, nargs='+', default=(0.7, 0.7))
    parser.add_argument('--crop_normalized', default=True, type=utils.bool_flag)
    parser.add_argument('--num_files', default=15, type=int)
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
    parser = argparse.ArgumentParser('Rendering MultiCrop', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # create a results folder name and the results directory
    args.results_dir = os.path.join(args.results_dir, args.results_folder_name)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    else:
        shutil.rmtree(args.results_dir)

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.cat_to_idx = cat_to_idx

    if args.cropping_strategy == 'rectangular':
        if args.data_type == '3D':
            # 3D cube crops.
            render_cubical_multicrop(args)
        else:
            # 2D depth images
            pass


if __name__ == '__main__':
    # set up rendering parameters
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    main()
