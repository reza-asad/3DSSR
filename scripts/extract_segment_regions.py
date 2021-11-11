import os
import numpy as np
from queue import Queue

from scripts.helper import visualize_labled_pc


def filter_accepted_graph(graph):
    filtered_graph = {}
    accepted_segments = {k for k in graph.keys() if graph[k]['acceptance_rate'] > 0.0}
    for k, v in graph.items():
        if k in accepted_segments:
            filtered_graph[k] = {'neighbours': set(), 'acceptance_rate': graph[k]['acceptance_rate']}
            for nb in graph[k]['neighbours']:
                if nb in accepted_segments:
                    filtered_graph[k]['neighbours'].add(nb)

    return filtered_graph


def BFS(G, q, room_pc_labels, num_points):
    queue = Queue()
    queue.put(q)
    visited = set()
    while not queue.empty():
        curr_node = queue.get()
        num_points_segment = np.sum(room_pc_labels == curr_node)
        if (curr_node not in visited) and ((num_points - num_points_segment) >= 0):
            # sort the neighbours by the number of points they have (ascending order) and add them to the queue.
            neighbour_points = [(nb, np.sum(room_pc_labels == nb)) for nb in G[curr_node]['neighbours']]
            neighbour_points = sorted(neighbour_points, key=lambda x: x[1])
            for nb, _ in neighbour_points:
                if nb not in visited:
                    queue.put(nb)
            visited.add(curr_node)
            num_points -= num_points_segment

    return visited, num_points


def visualize_region(region, segment_id, room_pc, room_pc_labels):
    all_points_list = []
    all_labels_list = []
    for node in region:
        I = (room_pc_labels == node)
        all_points_list.append(room_pc[I, :])
        all_labels_list.append(room_pc_labels[I])

    all_points = np.concatenate(all_points_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    visualize_labled_pc(all_points, all_labels, center_segment=None)


def extract_region():
    for i, room_name in enumerate(room_names):
        # load the room and its segment labels.
        room_pc = np.load(os.path.join(room_dir, room_name))
        room_pc_labels = np.load(os.path.join(room_label_dir, room_name))

        # load the segment graph for the room.
        graph = np.load(os.path.join(graph_dir, room_name), allow_pickle=True).item()
        if filter_accepted:
            graph = filter_accepted_graph(graph)

        # randomly select a center-segment.
        segment_id = np.random.choice(list(graph.keys()))
        while graph[segment_id]['acceptance_rate'] == 0.0:
            segment_id = np.random.choice(list(graph.keys()))

        # extract the region using BFS until x% of the points are included.
        num_points = int(len(room_pc) * region_points_ratio)
        region, num_points_remaining = BFS(graph, segment_id, room_pc_labels, num_points)
        seen_ids = region
        if filter_accepted:
            ids = list(graph.keys())
            for id_ in ids:
                if id_ not in seen_ids:
                    curr_region, num_points_remaining = BFS(graph, id_, room_pc_labels, num_points_remaining)
                    region = region.union(curr_region)
                    seen_ids = seen_ids.union(region)

        # visualize the extracted region.
        visualize_region(region, segment_id, room_pc, room_pc_labels)
        # visualize_labled_pc(room_pc, room_pc_labels)


def main():
    extract_region()


if __name__ == '__main__':
    # set up paths and params
    num_regions = 10
    filter_accepted = False
    region_points_ratio = 0.3
    data_dir = '../data/matterport3d'
    room_dir = os.path.join(data_dir, 'rooms_pc')
    room_label_dir = os.path.join(data_dir, 'rooms_pc_labels')
    graph_dir = os.path.join(data_dir, 'segment_graphs')
    room_names = os.listdir(room_dir)
    np.random.seed(10)
    np.random.shuffle(room_names)
    room_names = room_names[:num_regions]

    main()
