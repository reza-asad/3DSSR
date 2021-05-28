import numpy as np
from optparse import OptionParser
from subprocess import Popen
from time import time, sleep
import ast

from scripts.render_results import main as process


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='test|val')
    parser.add_option('--model_name', dest='model_name', default='LearningBased', help='LearningBased|GKRank|CatRank|'
                                                                                       'SVDRank|RandomRank')
    parser.add_option('--experiment_name', dest='experiment_name', default='AlignRank', help='AlignRankOracle|AlignRank|'
                                                                                             'GKRank|CatRank|RandomRank|'
                                                                                             'SVDRank1D|SVDRank3D')
    parser.add_option('--topk', dest='topk', default=5, help='Number of images that is rendered for each query.')
    parser.add_option('--filter_queries', dest='filter_queries', default='["bed-33", "table-9", "sofa-28"]',
                      help='Name of the queries to render. If ["all"] is chosen all queries will be rendered')
    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()
    filter_queries = args.filter_queries.replace(', ', ',')

    # make folders for rendering each query at scene and cropped scales.
    process(1, 0, mode=args.mode, model_name=args.model_name, experiment_name=args.experiment_name,
            render=False, topk=args.topk, make_folders=True, with_img_table=False,
            filter_queries=ast.literal_eval(filter_queries))

    # render the results in parallel
    c1 = 'parallel -j5 "python3 -u render_results.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}" ::: 5 ::: ' \
         '0 1 2 3 4 ::: '
    c2 = '{mode} ::: {model_name} ::: {experiment_name} ::: True ::: {topk} ::: False ::: False ' \
         '::: {filter_queries}'.format(mode=args.mode, model_name=args.model_name, experiment_name=args.experiment_name,
                                       topk=args.topk, filter_queries=filter_queries)
    process_rendering = Popen(c1 + c2, shell=True)
    t0 = time()
    while process_rendering.poll() is None:
        print('Rendering ...')
        sleep(20)
    duration = (time() - t0) / 60
    print('Rendering Took {} minutes'.format(np.round(duration, 2)))

    # create image tables for the cropped images.
    process(1, 0, mode=args.mode, model_name=args.model_name, experiment_name=args.experiment_name,
            render=False, topk=args.topk, make_folders=False, with_img_table=True,
            filter_queries=ast.literal_eval(filter_queries))


if __name__ == '__main__':
    main()



