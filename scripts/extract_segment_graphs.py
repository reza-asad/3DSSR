import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
from queue import Queue
from plyfile import PlyData

from scripts.helper import load_from_json, write_to_json, visualize_labled_pc


def BFS(G, q, num_neighbours):
    queue = Queue()
    queue.put(q)
    visited = set()
    while not queue.empty():
        curr_node = queue.get()
        if (curr_node not in visited) and (len(visited) <= num_neighbours):
            for nb in G[curr_node]['neighbours']:
                queue.put(nb)
            visited.add(curr_node)

    return visited


def visualize_neighbourhood(graph, segment_id, segment_indices, vertices, num_neighbours=5):
    # find neighbours of the segment using BFS.
    nodes = BFS(graph, segment_id, num_neighbours)
    nodes.add(segment_id)

    all_vertices = []
    all_labels = []
    for node in nodes:
        I = np.array(segment_indices) == node
        all_vertices.append(vertices[I])
        all_labels.append(np.ones(np.sum(I)) * node)

    all_vertices = np.concatenate(all_vertices, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    visualize_labled_pc(all_vertices, all_labels, center_segment=segment_id)


def load_ply_mesh(ply_path):
    # load the mesh
    with open(ply_path, 'rb') as f:
        ply_ascii = PlyData.read(f)

    # collect the vertices
    vertices = np.zeros((len(ply_ascii['vertex']['x']), 3), dtype=float)
    vertices[:, 0] = ply_ascii['vertex']['x']
    vertices[:, 1] = ply_ascii['vertex']['y']
    vertices[:, 2] = ply_ascii['vertex']['z']

    # take faces
    faces = ply_ascii['face'].data

    # take the category ids
    cats = ply_ascii['vertex']['mpr40']

    return vertices, faces, cats


def add_edges(faces, segment_indices, graph):
    # iterate through each face
    for face in faces:
        face = face[0]
        # find the segment id for each vertex in the face
        face_seg_ids = [segment_indices[v] for v in face]
        face_seg_ids_count = Counter(face_seg_ids)

        # create an edge between vertices with different segment id
        if len(face_seg_ids_count) > 1:
            unique_face_seg_ids = list(face_seg_ids_count.keys())
            for i in range(len(unique_face_seg_ids)):
                for j in range(i+1, len(unique_face_seg_ids)):
                    seg_id1 = unique_face_seg_ids[i]
                    seg_id2 = unique_face_seg_ids[j]
                    graph[seg_id1]['neighbours'].add(seg_id2)
                    graph[seg_id2]['neighbours'].add(seg_id1)

    return graph


def add_acceptance_rate(segment_indices, cats, graph):
    # map each segment id to its corresponding vertex categories.
    segment_id_to_cats = {}
    for i, segment_id in enumerate(segment_indices):
        if segment_id not in segment_id_to_cats:
            segment_id_to_cats[segment_id] = [cats[i]]
        else:
            segment_id_to_cats[segment_id].append(cats[i])

    # compute the acceptance rate for the categories of each segment.
    for segment_id, cats in segment_id_to_cats.items():
        num_cats = len(cats)
        num_accepted = len([mpr40_to_mpcat40[cat] for cat in cats if cat in mpr40_to_mpcat40])
        graph[segment_id]['acceptance_rate'] = num_accepted / float(num_cats)

    return graph


def process_rooms(room_names_chunk):
    for room_name in room_names_chunk:
        room_name = room_name.split('/')[0]
        visited = set(os.listdir(results_dir))
        if room_name not in visited:
            # load the over-segmentation of the mesh.
            file_name = '{}.annotated.{}.segs.json'.format(room_name, kThreshold)
            segments = load_from_json(os.path.join(segment_dir, file_name))

            # load the mesh
            ply_path = os.path.join(room_dir, room_name, '{}.annotated.ply'.format(room_name))
            vertices, faces, cats = load_ply_mesh(ply_path)

            # populate the graph with edges.
            graph = {int(seg_id): {'neighbours': set(), 'acceptance': 0.0} for seg_id in np.unique(segments['segIndices'])}
            graph = add_edges(faces, segments['segIndices'], graph)

            # populate the graph with the acceptance rate for each segment.
            graph = add_acceptance_rate(segments['segIndices'], cats, graph)

            # visualize a segment and its neighbours
            # segment_id = segments['segIndices'][1000]
            # visualize_neighbourhood(graph, segment_id, segments['segIndices'], vertices, num_neighbours=40)
            # t=y

            # save the graph
            np.save(os.path.join(results_dir, '{}.npy'.format(room_name)), graph)

            # update the visited rooms
            visited.add(room_name)


def main(num_chunks, chunk_idx, action='extract_graphs'):
    if action == 'extract_graphs':
        # extract graphs representing the over-segmented 3D scene.
        chunk_size = int(np.ceil(len(room_names) / num_chunks))
        process_rooms(room_names_chunk=room_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])


if __name__ == '__main__':
    # set up paths and params
    kThreshold = '0.010000'
    data_dir = '../data/matterport3d'
    results_dir = os.path.join(data_dir, 'segment_graphs')
    segment_dir = os.path.join('/media/reza/Large/Matterport3D_rooms/rooms_over_segments')
    room_dir = '/media/reza/Large/Matterport3D_rooms/rooms'
    room_names = os.listdir(room_dir)

    # skip rooms with no mesh
    room_names = [os.path.join(room_name, '{}.annotated.ply'.format(room_name)) for room_name in room_names if
                  len(os.listdir(os.path.join(room_dir, room_name))) > 0]

    # find a mapping between the mpr40 id and mpcat40 accepted categories
    accepted_cats = load_from_json(os.path.join(data_dir, 'accepted_cats.json'))
    df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    # filter metadata to only include accepted cats
    is_accpeted = df['mpcat40'].apply(lambda x: x in accepted_cats)
    df = df.loc[is_accpeted]
    df = df[['mpr40', 'mpcat40']].drop_duplicates()
    mpr40_to_mpcat40 = dict(zip(df['mpr40'], df['mpcat40']))

    for folder in [results_dir, ]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except FileExistsError:
                pass

    if len(sys.argv) == 1:
        main(1, 0, 'extract_graphs')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u extract_segment_graphs.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_graphs
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
