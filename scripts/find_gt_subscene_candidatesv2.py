import os
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.helper import load_from_json, write_to_json, render_subscene, create_img_table


def build_subscene_key(target_subscene):
    correspondence = sorted(list(target_subscene['correspondence'].keys()))
    # if the subscene is empty do not create a key and simply move on.
    if len(correspondence) == 0 and len(target_subscene['target']) == 0:
        return None
    correspondence = '-'.join(correspondence)
    key = '{}-{}-{}'.format(target_subscene['scene_name'].split('.')[0], target_subscene['target'], correspondence)

    return key


def extract_candidates(args, query_dict):
    # for each model retrieve topk retrieved results for a list of queries.
    query_result_combined = {}
    for query in args.query_list:
        # fill in the details of the query subscene.
        query_result_combined[query] = query_dict[query]
        query_result_combined[query]['target_subscenes'] = []
        seen = set()
        # take topk results from each model.
        for config_name, config_info in args.retrieved_results_paths.items():
            print('Extracting subscenes from {}'.format(config_name))
            if config_name not in args.config_list_exceptions:
                # create the path to the model's retrieved results.
                for k, v in config_info.items():
                    vars(args)[k] = v
                adjust_paths(args, exceptions=[])
                query_results_dir = os.path.join(args.cp_dir.format(args.model_name), args.results_folder_name)
                query_results_input_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'\
                    .format(args.mode, args.experiment_name)
                query_results_input_path = os.path.join(query_results_dir, query_results_input_file_name)

                # load the results of the model
                query_result = load_from_json(query_results_input_path)

                # add topk target subscenes if not already added.
                for target_subscene in query_result[query]['target_subscenes'][:args.topk]:
                    key = build_subscene_key(target_subscene)
                    if (key is not None) and (key not in seen):
                        query_result_combined[query]['target_subscenes'].append(target_subscene)
                        seen.add(key)

    # save the changes to query dict
    write_to_json(query_result_combined, args.query_dict_output_path)


def render_gt_results(args, query_results_combined):
    for query in args.query_list:
        results_info = query_results_combined[query]
        query_path = os.path.join(args.rendering_path, query, 'imgs')
        if not os.path.exists(query_path):
            try:
                os.makedirs(query_path)
            except FileExistsError:
                pass
        # render the query
        scene_name = results_info['example']['scene_name']
        query_graph = load_from_json(os.path.join(args.scene_dir, scene_name))
        q = results_info['example']['query']
        q_context = set(results_info['example']['context_objects'] + [q])

        faded_nodes = [obj for obj in query_graph.keys() if obj not in q_context]
        path = os.path.join(query_path, 'query_{}_{}.png'.format(scene_name.split('.')[0], q))
        render_subscene(graph=query_graph, objects=query_graph.keys(), highlighted_object=q_context,
                        faded_nodes=faded_nodes, path=path, model_dir=args.models_dir, colormap={},
                        with_height_offset=False)

        # render the gt candidates.
        for target_subscene in results_info['target_subscenes']:
            target_scene_name = target_subscene['scene_name']
            if '.json' not in target_scene_name:
                target_scene_name += '.json'
            target_graph = load_from_json(os.path.join(args.scene_dir, target_scene_name))
            t = target_subscene['target']
            t_context = list(target_subscene['correspondence'].keys())
            if t != '':
                t_context.append(t)

            faded_nodes = [obj for obj in target_graph.keys() if obj not in t_context]
            img_name = build_subscene_key(target_subscene)
            path = os.path.join(query_path, '{}.png'.format(img_name))
            render_subscene(graph=target_graph, objects=target_graph.keys(), highlighted_object=t_context,
                            faded_nodes=faded_nodes, path=path, model_dir=args.models_dir, colormap={},
                            with_height_offset=False)


def extract_caption(args, query_dict, q_img_name):
    for query, query_info in query_dict.items():
        q = q_img_name.split('_')[3].split('.')[0]
        scene_name = q_img_name.replace('_{}.png'.format(q), '').replace('query_', '') + '.json'
        if (query_info['example']['scene_name'] == scene_name) and (query_info['example']['query'] == q):
            # load the query scene
            query_scene = load_from_json(os.path.join(args.scene_dir, scene_name))
            cats = []
            cats.append(query_scene[q]['category'][0])
            for q_c in query_info['example']['context_objects']:
                cats.append(query_scene[q_c]['category'][0])
            cats = ' + '.join(cats)

            return cats


def save_img_pairs(args, query_dict):
    # create a directory for the combined images.
    img_pair_path = os.path.join(args.rendering_path, 'imgs_combined')
    if not os.path.exists(img_pair_path):
        try:
            os.makedirs(img_pair_path)
        except FileExistsError:
            pass

    for query in args.query_list:
        # find query and candidate image names.
        imgs_path = os.path.join(args.rendering_path, query, 'imgs')
        img_names = os.listdir(imgs_path)
        q_img_name = [e for e in img_names if 'query' in e][0]
        img_names.remove(q_img_name)

        # load the query image.
        query_img = Image.open(os.path.join(imgs_path, q_img_name))

        # create images by pairing each candidate with the query.
        offset = 100
        text_offset = 100
        for t_img_name in img_names:
            # load the target img.
            target_img = Image.open(os.path.join(imgs_path, t_img_name))
            width, height = target_img.size
            combined_img = Image.new('RGB', (width * 2 + offset, height + text_offset), color=(255, 255, 255))
            combined_img.paste(query_img, (0, 0 + text_offset))
            combined_img.paste(target_img, (width + offset, 0 + text_offset))

            # find the caption and add it as text to the image.
            caption = extract_caption(args, query_dict, q_img_name)
            # num_words = len(caption.split('+'))
            img_text = ImageDraw.Draw(combined_img)
            font = ImageFont.truetype('arial.ttf', 35)
            img_text.text((text_offset//2, 20), caption, font=font, fill=(255, 0, 0))
            # combined_img.show()
            # t=y
            # break
            combined_img.save(os.path.join(img_pair_path, '{}*{}'.format(q_img_name, t_img_name)))

    print('Number of combined images is {}'.format(len(os.listdir(img_pair_path))))


def create_metadata(args, query_dict):
    # create image url for the combined image pairs.
    img_pair_path = os.path.join(args.rendering_path, 'imgs_combined')
    prefix = 'https://myuserstudyimages.s3.us-west-2.amazonaws.com/imgs_combined/'
    img_urls = [prefix + e for e in os.listdir(img_pair_path)]

    # extract the caption for the image pairs.
    captions = []
    for img_name in os.listdir(img_pair_path):
        q_img_name = img_name.split('*')[0]
        caption = extract_caption(args, query_dict, q_img_name)
        captions.append(caption)

    # randomly shuffle the images
    np.random.seed(0)
    img_urls_and_captions = list(zip(img_urls, captions))
    np.random.shuffle(img_urls_and_captions)
    img_urls = list(list(zip(*img_urls_and_captions))[0])
    img_names = [img_url.split('/')[-1] for img_url in img_urls]
    captions = list(list(zip(*img_urls_and_captions))[1])

    print('There are {} image urls'.format(len(img_urls)))

    # select the correct chunk.
    num_rows = 0
    for chunk_id in args.chunk_ids:
        img_urls_chunk = img_urls[(chunk_id - 1) * args.chunk_size: chunk_id * args.chunk_size]
        img_names_chunk = img_names[(chunk_id - 1) * args.chunk_size: chunk_id * args.chunk_size]
        captions_chunk = captions[(chunk_id - 1) * args.chunk_size: chunk_id * args.chunk_size]

        # create a dataframe with img_urls and captions.
        df = pd.DataFrame({'image_url': img_urls_chunk, 'caption': captions_chunk, 'img_names': img_names_chunk})
        df.sort_values(by='image_url', inplace=True)
        df.to_csv(os.path.join(args.rendering_path, 'metadata_{}.csv'.format(chunk_id)), index=False)
        num_rows += len(df)

    print('Create {} data frames with total of {} records'.format(len(args.chunk_ids), num_rows))


def create_img_table_batch(args):
    # load the image names from the metadata batch.
    df_metadata = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}.csv'.format(args.batch_id)))
    img_names = df_metadata['image_url'].apply(lambda x: x.split('/')[-1]).values.tolist()

    # create img table for the current batch
    imgs_path = os.path.join(args.rendering_path, 'imgs_combined')
    create_img_table(imgs_path, 'imgs_combined', img_names, 'img_table_{}.html'.format(args.batch_id),
                     with_query_scene=False, topk=len(img_names), ncols=1, captions=['<br />\n']*len(img_names),
                     query_caption=None)

    # for query in args.query_list:
    #     # find the query img name
    #     results_info = query_results_combined[query]
    #     query_scene_name = results_info['example']['scene_name']
    #     q = results_info['example']['query']
    #     context_objects_and_q = [q] + results_info['example']['context_objects']
    #     query_img = 'query_{}_{}.png'.format(query_scene_name.split('.')[0], q)
    #     imgs_path = os.path.join(args.rendering_path, query, 'imgs')
    #
    #     # load the query scene.
    #     query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    #
    #     # add caption for the query img.
    #     query_caption = '<br />\n'
    #     cats = []
    #     for q_c in context_objects_and_q:
    #         cat = query_scene[q_c]['category'][0]
    #         cats.append(cat)
    #         # if q_c == q:
    #             # cat = '<t style="color:red">{}</t>'.format(cat)
    #     cats = ' + '.join(cats)
    #     query_caption += '<h1 style="font-size:1.5vw">{}</h1>'.format(cats)
    #     # query_caption += '{} <br />\n'.format(cats)
    #
    #     # find the target img names and add caption.
    #     imgs = []
    #     captions = []
    #     for target_subscene in results_info['target_subscenes']:
    #         # find the img name and its caption number
    #         key = build_subscene_key(target_subscene)
    #         img_name = '{}.png'.format(key)
    #         imgs.append(img_name)
    #
    #         # add caption for the image name
    #         caption = '<br />\n'
    #         caption += '{} <br />\n'.format(img_name_to_number[query][img_name])
    #         captions.append(caption)
    #
    #     # sort the images and captions by the img number
    #     imgs_numbers = [(img, captions[i], img_name_to_number[query][img]) for i, img in enumerate(imgs)]
    #     imgs_numbers = sorted(imgs_numbers, key=lambda x: x[2])
    #     imgs = list(list(zip(*imgs_numbers))[0])
    #     captions = list(list(zip(*imgs_numbers))[1])
    #
    #     # create the img table for the query
    #     create_img_table_scrollable(imgs_path, 'imgs', imgs, 'ground_truth.html', query_img, topk=len(imgs), ncols=2,
    #                                 captions=captions, query_caption=query_caption)


# def map_img_names_to_numbers(args, query_results_combined):
#     img_name_to_number = {}
#     for query in args.query_list:
#         results_info = query_results_combined[query]
#         img_names = []
#         for target_subscene in results_info['target_subscenes']:
#             key = build_subscene_key(target_subscene)
#             img_name = '{}.png'.format(key)
#             img_names.append(img_name)
#
#         # shuffle and create the map
#         np.random.seed(0)
#         np.random.shuffle(img_names)
#         img_name_to_number[query] = dict(zip(img_names, range(1, len(img_names)+1)))
#
#     return img_name_to_number


def get_args():
    parser = argparse.ArgumentParser('Compiling list of ground truth 3D subscenes', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--action', default='img_table', help='extract | render | img_table | save_img_pairs | metadata')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--query_dir', default='../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    # parser.add_argument('--query_list', default=["cabinet-18"], type=str, nargs='+')
    parser.add_argument('--query_list', default=["table-43", "curtain-7", "mirror-10", "cushion-20", "cabinet-18",
                                                 "chair-45", "cushion-33", "lighting-11", "picture-39", "lighting-46"],
                        type=str, nargs='+')
    parser.add_argument('--config_list_exceptions', default=["dino_10_config", "dino_v2_config", "csc_10_config"],
                        type=str, nargs='+')
    parser.add_argument('--retrieved_results_paths', default='retrieved_results_paths.json')
    parser.add_argument('--results_dir', default='../results/{}/')
    parser.add_argument('--results_folder_name',  default='UserStudy')
    parser.add_argument('--experiment_name', default='GroundTruthSubscenes')
    parser.add_argument('--rendering_folder_name', default='rendered_results')
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--chunk_ids', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int, nargs='+')
    parser.add_argument('--chunk_size', default=58, type=int)
    parser.add_argument('--batch_id', default=12, type=int)

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Compiling list of ground truth 3D subscenes', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # make sure to retrieve data from the requested mode
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.query_dir = os.path.join(args.query_dir, args.mode)
    args.rendering_path = os.path.join(args.results_dir, args.results_folder_name, args.rendering_folder_name,
                                       args.mode, args.experiment_name)
    args.query_results_path = os.path.join(args.results_dir, args.results_folder_name)

    # remove the rendering path if action is render and the folder already exists.
    if args.action == 'render':
        if os.path.exists(args.rendering_path):
            shutil.rmtree(args.rendering_path)

    # create the rendering path if it doesn't exist
    if not os.path.exists(args.rendering_path):
        try:
            os.makedirs(args.rendering_path)
        except FileExistsError:
            pass

    # load the query dict
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_dict = load_from_json(query_dict_input_path)

    # create the output path for the combined results from all models.
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             args.experiment_name)
    args.query_dict_output_path = os.path.join(args.query_results_path, query_output_file_name)

    # load the list of models to consider for user study
    args.retrieved_results_paths = load_from_json(args.retrieved_results_paths)

    if args.action == 'extract':
        extract_candidates(args, query_dict)
    elif args.action == 'render':
        # load the candidates from all models.
        query_results_combined = load_from_json(args.query_dict_output_path)
        render_gt_results(args, query_results_combined)
    elif args.action == 'save_img_pairs':
        save_img_pairs(args, query_dict)
    elif args.action == 'metadata':
        create_metadata(args, query_dict)
    elif args.action == 'img_table':
        # load the candidates from all models.
        create_img_table_batch(args)
    else:
        raise NotImplementedError('Action {} is not implemented'.format(args.action))


if __name__ == '__main__':
    main()
