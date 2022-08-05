import os
import argparse
import numpy as np
import torch
from chamferdist import ChamferDistance

from scripts.helper import load_from_json, render_subscene, render_single_scene, render_scene_subscene, create_img_table
from scripts.box import Box
from scripts.iou import IoU


# Supervised Point Transformer
config = {
    '3D DINO':
        {
            'cp_dir': '../results/{}',
            'results_folder_name': '3D_DINO_regions_non_equal_full_top10_seg',
            'experiment_name': '3D_DINO_point_transformer',
            'model_name': 'LearningBased'
        },
    'SUP PT':
        {
            'cp_dir': '../results/{}',
            'results_folder_name': 'region_classification_transformer_2_32_4096_non_equal_full_region_top10',
            'experiment_name': 'supervised_point_transformer',
            'model_name': 'LearningBased'
        },
    'CSC':
        {
            'cp_dir': '/home/reza/Documents/research/ContrastiveSceneContexts/data_processed_subset/matterport3d/embeddings/',
            'results_folder_name': '',
            'experiment_name': 'CSC_point_transformer',
            'model_name': 'PointTransformerSeg'
        },
    'Random':
        {
            'cp_dir': '../results/{}',
            'results_folder_name': '',
            'experiment_name': 'RandomRank',
            'model_name': 'RandomRank'
        },
    'Cat':
        {
            'cp_dir': '../results/{}',
            'results_folder_name': '',
            'experiment_name': 'CatRank',
            'model_name': 'CatRank'
        },
    'Oracle':
        {
            'cp_dir': '../results/{}',
            'results_folder_name': '',
            'experiment_name': 'OracleRank',
            'model_name': 'OracleRank'
        },
    'GK':
        {
            'cp_dir': '../results/{}',
            'results_folder_name': '',
            'experiment_name': 'GKRank',
            'model_name': 'GKRank'
        }
}


def load_pc(args, file_name):
    pc = np.load(os.path.join(args.pc_dir, file_name))

    # sample N points
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc)), args.num_points, replace=False)
    pc = np.expand_dims(pc[sampled_indices, :], axis=0)
    pc = torch.from_numpy(pc).cuda()

    return pc


def compute_chamfer_dist(args, query_scene_name, query_obj, target_scene_name, target_obj):
    chamferDist = ChamferDistance()

    # load the pc for the query aabb
    query_file_name = '-'.join([query_scene_name.split('.')[0], query_obj])
    pc_q = load_pc(args, query_file_name + '.npy')

    # load the pc for the target aabb.
    target_file_name = '-'.join([target_scene_name.split('.')[0], target_obj])
    pc_t = load_pc(args, target_file_name + '.npy')

    # compute chamfer distance.
    dist_forward = chamferDist(pc_q, pc_t)
    dist = dist_forward.detach().cpu().item()

    return dist


def translate_box(box, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    box = box.apply_transformation(transformation)

    return box


def compute_iou(box1, box2):
    # compute the iou
    iou_computer = IoU(box1, box2)
    iou = iou_computer.iou()

    return iou


def create_box_at_origin(scene, obj, box_type):
    box_vertices = np.asarray(scene[obj][box_type])
    box = Box(box_vertices)
    obj_translation = -box.translation
    box = translate_box(box, obj_translation)

    return box, obj_translation


def compute_world_coordinate_iou(query_scene, q_obj, target_scene, t_obj, target_subscene, q_translation=None,
                                 t_translation=None):
    # computing iou for anchor objects
    if q_translation is None:
        # create a box for the query object and its translation to the origin.
        box_query, q_translation = create_box_at_origin(query_scene, q_obj, box_type='aabb')

        # create a box for the target object and find its translation to the origin.
        box_target, t_translation = create_box_at_origin(target_scene, t_obj, box_type='aabb')

        # compute iou
        iou = compute_iou(box_target, box_query)

        return iou, q_translation, t_translation

    # create a box object for the context object and translated it according to the query object
    q_context_box_vertices = np.asarray(query_scene[q_obj]['aabb'])
    q_c_box = Box(q_context_box_vertices)
    q_c_box = translate_box(q_c_box, q_translation)

    # create a box object for the candidate and translated it according to the target object
    t_context_box_vertices = np.asarray(target_scene[t_obj]['aabb'])
    t_c_box = Box(t_context_box_vertices)
    t_c_box = translate_box(t_c_box, t_translation)

    # compute the IoU between the candidate and context obboxes.
    iou = compute_iou(t_c_box, q_c_box)

    return iou


def compute_corr_stats(args, query_result, cd_sim):
    # find the query cat
    query_scene_name = query_result['example']['scene_name']
    context_objects = query_result['example']['context_objects']
    query_obj = query_result['example']['query']
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    q_cat = query_scene[query_obj]['category'][0]

    # iterate through all corresponding pairs and compute stats.
    corr_stats = []
    for target_subscene in query_result['target_subscenes'][:args.topk]:
        # find the cats for the anchor objects
        target_scene_name = target_subscene['scene_name']
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))
        target_obj = target_subscene['target']
        t_cat = target_scene[target_obj]['category'][0]

        # find the CD for the anchor objects
        cd_anchor = compute_chamfer_dist(args, query_scene_name, query_obj, target_scene_name, target_obj)

        # find the IoU for the anchor objects
        iou_anchor, q_translation, t_translation = compute_world_coordinate_iou(query_scene, query_obj, target_scene,
                                                                                target_obj, target_subscene,
                                                                                q_translation=None, t_translation=None)

        stats = {'cats': [(q_cat, t_cat)], 'CD': [(cd_anchor, cd_sim[q_cat][args.cd_threshold])],
                 'IoU': [(iou_anchor, args.iou_threshold)], 'num_total': len(context_objects) + 1}
        for candidate, context_object in target_subscene['correspondence'].items():
            # populate the categories.
            q_c_cat = query_scene[context_object]['category'][0]
            t_c_cat = target_scene[candidate]['category'][0]
            stats['cats'].append((q_c_cat, t_c_cat))

            # compute the chamfer distances
            cd_context = compute_chamfer_dist(args, query_scene_name, context_object, target_scene_name, candidate)
            stats['CD'].append((cd_context, cd_sim[q_c_cat][args.cd_threshold]))

            # compute the iou
            iou_candidate = compute_world_coordinate_iou(query_scene, context_object, target_scene, candidate,
                                                         target_subscene, q_translation=q_translation,
                                                         t_translation=t_translation)
            stats['IoU'].append((iou_candidate, args.iou_threshold))

        # record stats for the target subscene
        corr_stats.append(stats)

    return corr_stats


def compute_precision(corr_stats):
    for stats in corr_stats:
        # compute precision_cat
        num_match_cat = 0
        num_match_sim = 0
        for i in range(len(stats['cats'])):
            q_cat, t_cat = stats['cats'][i]
            if stats['IoU'][i][0] > stats['IoU'][i][1]:
                if q_cat == t_cat:
                    num_match_cat += 1
                if stats['CD'][i][0] <= stats['CD'][i][1]:
                    num_match_sim += 1

        stats['precision_cat'] = '{}/{}'.format(num_match_cat, stats['num_total'])
        stats['precision_sim'] = '{}/{}'.format(num_match_sim, stats['num_total'])

    return corr_stats


def render_results(args, query_result, output_dir):
    query_scene_name = query_result['example']['scene_name']
    query_graph = load_from_json(os.path.join(args.scene_dir_raw, query_scene_name))
    q = query_result['example']['query']
    q_context = set(query_result['example']['context_objects'] + [q])

    # render the query subscene
    faded_nodes = [obj for obj in query_graph.keys() if obj not in q_context]
    path = os.path.join(output_dir, 'query_{}_{}.png'.format(query_scene_name.split('.')[0], q))
    if args.render_subscene:
        render_subscene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q], faded_nodes=faded_nodes,
                        path=path, model_dir=args.models_dir, colormap=args.colormap, with_height_offset=False)
    else:
        render_single_scene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q],
                            faded_nodes=faded_nodes, path=path, model_dir=args.models_dir, colormap=args.colormap,
                            with_height_offset=False)

    # render the topk target subscenes.
    for i, target_subscene in enumerate(query_result['target_subscenes'][:args.topk]):
        target_scene_name = target_subscene['scene_name']
        target_graph_path = os.path.join(args.scene_dir_raw, target_scene_name)
        target_graph = load_from_json(target_graph_path)
        t = target_subscene['target']
        highlighted_object = [t]
        t_context = set(list(target_subscene['correspondence'].keys()) + [t])

        # render the image
        faded_nodes = [obj for obj in target_graph.keys() if obj not in t_context]
        path = os.path.join(output_dir, 'top_{}_{}_{}.png'.format(i+1, target_scene_name.split('.')[0], t))
        if args.render_subscene:
            render_subscene(graph=target_graph, objects=target_graph.keys(), highlighted_object=highlighted_object,
                            faded_nodes=faded_nodes, path=path, model_dir=args.models_dir, colormap=args.colormap,
                            with_height_offset=False)
        else:
            render_single_scene(graph=target_graph, objects=target_graph.keys(), highlighted_object=highlighted_object,
                                faded_nodes=faded_nodes, path=path, model_dir=args.models_dir, colormap=args.colormap,
                                with_height_offset=False)


def prepare_captions(imgs, corr_stats):
    captions = []
    for i in range(len(imgs)):
        caption = '<br />\n'
        # add IoU
        for j, (q_cat, t_cat) in enumerate(corr_stats[i]['cats']):
            caption += ' IoU_{}({}, {}): {} <br />\n'.format(corr_stats[i]['IoU'][j][1], q_cat, t_cat,
                                                             np.round(corr_stats[i]['IoU'][j][0], 3))

        # add CD
        caption += '<br />\n'
        for j, (q_cat, t_cat) in enumerate(corr_stats[i]['cats']):
            caption += ' CD_{}({}, {}): {} <br />\n'.format(np.round(corr_stats[i]['CD'][j][1], 3), q_cat, t_cat,
                                                            np.round(corr_stats[i]['CD'][j][0], 3))

        # add precisions
        caption += '<br />\n'
        caption += ' Precision_cat: {} <br />\n'.format(corr_stats[i]['precision_cat'])
        caption += ' Precision_geo: {}'.format(corr_stats[i]['precision_sim'])

        captions.append(caption)

    return captions


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Render topk results with quantitative metrics', add_help=False)
    parser.add_argument('--ranking_strategy', default='3D DINO', help='3D DINO|CSC|SUP PT|Random|Cat|Oracle|GK')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='test', help='val | test')
    parser.add_argument('--scene_dir_raw', default='../data/{}/scenes')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--colormap_path', default='../data/{}/color_map.json')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--query_name', default='chair-45')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--cd_path', default='../data/{}/cd_thresholds.json')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--rendering_dir', default='../results/{}/rendered_results_with_stats')
    parser.add_argument('--topk', default=10, type=int, help='number of top results for mAP computations.')
    parser.add_argument('--iou_threshold', default=0.15)
    parser.add_argument('--cd_threshold', default='20')
    parser.add_argument('--render_subscene', action='store_true', default=True, help='render entire or subscene')

    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Render topk results with quantitative metrics', parents=[get_args()])
    args = parser.parse_args()

    # load the config for the current model
    model_config = config[args.ranking_strategy]
    for k, v in model_config.items():
        vars(args)[k] = v

    adjust_paths(args, exceptions=[])
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.scene_dir_raw = os.path.join(args.scene_dir_raw, args.mode)
    args.rendering_dir = os.path.join(args.rendering_dir, args.mode)
    args.cd_path = args.cd_path.split('.json')[0] + '_{}.json'.format(args.mode)
    args.colormap = load_from_json(args.colormap_path)

    # make a directroy for the output
    output_dir = os.path.join(args.rendering_dir, args.ranking_strategy, args.query_name, 'imgs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load the chamfer distance (CD) thresholds for query cat.
    cd_sim = load_from_json(args.cd_path)

    # load the query results.
    query_results_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.\
        format(args.mode, model_config['experiment_name'])
    query_results_path = os.path.join(args.cp_dir, args.model_name, args.results_folder_name, query_results_file_name)
    query_results = load_from_json(query_results_path)

    # compute stats for each corresponding pair (cat, IoU and CD).
    corr_stats = compute_corr_stats(args, query_results[args.query_name], cd_sim)

    # compute precision on the stats.
    corr_stats = compute_precision(corr_stats)

    # render the results.
    render_results(args, query_results[args.query_name], output_dir)

    # create img table.
    imgs = os.listdir(output_dir)
    query_img = [img for img in imgs if 'query' in img]
    imgs.remove(query_img[0])
    imgs = sorted(imgs, key=lambda x: int(x.split('_')[1]))
    captions = prepare_captions(imgs, corr_stats)
    create_img_table(output_dir, 'imgs', imgs, 'img_table.html', topk=args.topk, ncols=5, captions=captions,
                     with_query_scene=True, evaluation_plot=None, query_img=query_img[0], query_caption=None)


if __name__ == '__main__':
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    resolution = (512, 512)
    main()
