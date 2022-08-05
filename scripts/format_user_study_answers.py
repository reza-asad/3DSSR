import os
import argparse

from scripts.helper import load_from_json, write_to_json
from find_gt_subscene_candidates import map_img_names_to_letters


def process_answers(args, gt_subscene_candidates, answer):
    # find a mapping from the letters to scene names.
    img_name_to_letters = map_img_names_to_letters(args, gt_subscene_candidates)
    letters_to_img_names = {}
    for query, img_name_letter in img_name_to_letters.items():
        letters_to_img_names[query] = {}
        for img_name, letter in img_name_letter.items():
            letters_to_img_names[query][letter] = img_name

    # format the answers so it can be fed for ndcg computation.
    formatted_anwer = {}
    for query, scenes_info in answer.items():
        formatted_anwer[query] = []
        for scene_info in scenes_info:
            letter = scene_info['scene_name']
            scene_name_target = letters_to_img_names[query][letter].split('.')[0]
            house, room, t = scene_name_target.split('_')
            scene_name = '_'.join([house, room])
            info_template = {'scene_name': scene_name, 'target': t}
            match_info = {}
            for k, v in scene_info['correspondence'].items():
                if v != '':
                    match_info[v] = 1
            info_template['match_info'] = match_info
            info_template['correspondence'] = scene_info['correspondence']
            formatted_anwer[query].append(info_template)

    return formatted_anwer


def get_args():
    parser = argparse.ArgumentParser('Format answers', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--scene_dir_raw', default='../data/{}/scenes')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--colormap_path', default='../data/{}/color_map.json')
    parser.add_argument('--query_dir', default='../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    # parser.add_argument('--query_list', default=["cabinet-18"], type=str, nargs='+')
    parser.add_argument('--query_list', default=["table-43", "curtain-7", "mirror-10", "cushion-20", "cabinet-18",
                                                 "chair-45", "cushion-33", "lighting-11", "picture-39", "lighting-46"],
                        type=str, nargs='+')
    parser.add_argument('--answers_dir', default='../user_study_answers')
    parser.add_argument('--gt_subscenes_dir', default='../results/{}/GroundTruthSubscenes')
    parser.add_argument('--gt_subscenes_file_name', default='gt_subscene_candidates.json')
    parser.add_argument('--cp_dir', default='../results/{}')
    parser.add_argument('--results_folder_name',  default='OracleRankV2')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Format answers', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # load the gt subscene candidates.
    args.gt_subscenes_path = os.path.join(args.cp_dir, args.results_folder_name, args.gt_subscenes_file_name)
    gt_subscene_candidates = load_from_json(args.gt_subscenes_path)

    for file_name in os.listdir(args.answers_dir):
        # print(file_name)
        answer = load_from_json(os.path.join(args.answers_dir, file_name))
        formatted_answer = process_answers(args, gt_subscene_candidates, answer)
        write_to_json(formatted_answer, os.path.join(args.gt_subscenes_dir, file_name))


if __name__ == '__main__':
    main()
