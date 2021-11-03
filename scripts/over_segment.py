import os
import sys
import shutil
import numpy as np


def over_segment(room_names):
    # over-segment the rooms and move them to the desired result dir.
    for room_name in room_names:
        if len(os.listdir(os.path.join(room_dir, room_name))) > 0:
            room_ply_path = os.path.join(room_dir, room_name, '{}.annotated.ply'.format(room_name))
            command = './Segmentator/segmentator {} {} {}'.format(room_ply_path, kThresh, segMinVerts)
            stream = os.popen(command)
            stream.read()

            # move the over-segmentation file to the desired results directory.
            file_name = [f for f in os.listdir(os.path.join(room_dir, room_name)) if f[-5:] == '.json'][0]
            d1 = os.path.join(room_dir, room_name, file_name)
            d2 = os.path.join(results_dir, file_name)
            shutil.move(d1, d2)


def main(num_chunks, chunk_idx, action='extract'):
    if action == 'over_segment':
        chunk_size = int(np.ceil(len(room_names) / num_chunks))
        over_segment(room_names=room_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])


if __name__ == '__main__':
    # define paths
    data_dir = '/media/reza/Large/Matterport3D_rooms'
    room_dir = os.path.join(data_dir, 'rooms')
    room_names = os.listdir(room_dir)
    results_dir = os.path.join(data_dir, 'rooms_over_segments')
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except FileExistsError:
            pass

    # define the over-segmentation parameters
    kThresh = 0.01
    segMinVerts = 20

    if len(sys.argv) == 1:
        main(1, 0, 'over_segment')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u over_segment.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: over_segment
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

