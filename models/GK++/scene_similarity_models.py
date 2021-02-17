import os
import json
import numpy as np

from helper import read_zernike_descriptors


class GraphKernel:
    def __init__(self, voxel_dir, graph_dir, graph1_name, nth_closest_dict, mode, normalize_kernel=True, walk_length=3):
        self.voxel_dir = voxel_dir
        self.graph_dir = graph_dir
        self.graph1_name = graph1_name
        self.graph2_name = None
        self.mode = mode

        self.graph1 = self.read_graph(self.graph1_name, mode=self.mode)
        self.graph2 = None
        self.nth_closest_dict = nth_closest_dict

        self.walk_length = walk_length

        if normalize_kernel:
            self.normalize_node_kernel(self.graph1)

    def read_graph(self, graph_name, mode):
        """
        Read the graph given its name.
        :param graph_name: The json file representing the graph.
        :return: Dictionary representing the graph.
        """
        graph_path = os.path.join(self.graph_dir, mode, graph_name)
        with open(graph_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def delta_kernel(graph1, graph2, n1, n2):
        """
        Checks if two models are identical based on their file name.
        :param graph1: First dictionary graph.
        :param graph2: Second dictionary graph.
        :param n1: Node in graph1.
        :param n2: Node in graph2.
        :return: True if the file name for the two models is identical.
        """
        return graph1[n1]['file_name'] == graph2[n2]['file_name']

    @staticmethod
    def tag_kernel(graph1, graph2, n1, n2):
        """
        Compute the tag kernel between two nodes in two graphs. The cumputation is as follows: full match return 1
        , only primary tag match return 0.5, only secondary tag return 0.1, if idential models return 1
        :param graph1: First dictionary graph.
        :param graph2: Second dictionary graph.
        :param n1: Node in graph1.
        :param n2: Node in graph2.
        :return: Tag kernel between 0 and 1 where 1 means identical.
        """
        cat1 = graph1[n1]['category']
        cat2 = graph2[n2]['category']
        if len(cat1) > len(cat2):
            cat1, cat2 = cat2, cat1
        l1 = len(cat1)

        # one mesh is missing category return 0
        if l1 == 0:
            return 0
        if cat1[0] == cat2[0]:
            return 1
        else:
            return 0

    def geo_kernel(self, graph1, graph2, n1, n2):
        """
        Compute the geometry kernel between two nodes in two graphs.
        :param graph1: First dictionary graph.
        :param graph2: Second dictionary graph.
        :param n1: Node in graph1.
        :param n2: Node in graph2.
        :return: The Euclidean distance between the Zernike descriptors of the n1 and n2.
        """
        obj1 = graph1[n1]['file_name'].split('.')[0]
        obj2 = graph2[n2]['file_name'].split('.')[0]
        zernike_n1 = read_zernike_descriptors(os.path.join(self.voxel_dir, obj1 + '.inv'))
        zernike_n2 = read_zernike_descriptors(os.path.join(self.voxel_dir, obj2 + '.inv'))
        zernike_dist = np.linalg.norm((zernike_n1 - zernike_n2)**2)
        return np.exp(-(2 * zernike_dist / min(self.nth_closest_dict[obj1+'.inv'][1],
                                               self.nth_closest_dict[obj2+'.inv'][1]))**2)

    def normalize_node_kernel(self, graph):
        """
        Compute the normalization factor for each node in the graph by computing pairs of node kernels.
        :param graph: Dictioanry representing graph.
        """
        for n1, _ in graph.items():
            result = 0
            for n2, _ in graph.items():
                result += self.node_kernel(graph, graph, n1, n2, normalize=False)
            graph[n1]['normalization'] = result

    def node_kernel(self, graph1, graph2, n1, n2, normalize=True, search_query=False):
        """
        Compute the kernel between two nodes form two graphs.
        :param graph1: First dictionary graph.
        :param graph2: Second dictionary graph.
        :param n1: Node in graph1.
        :param n2: Node in graph2.
        :param normalize: If True the computed normalization terms are used in the kernel.
        :param search_query: If true the node kernel is computed for the context-based model search.
        :return: Node kernel.
        """
        sigma_n1, sigma_n2 = 1, 1
        delta_kernel = 1
        if not search_query:
            delta_kernel = self.delta_kernel(graph1, graph2, n1, n2)
            if normalize:
                sigma_n1 = 1. / graph1[n1]['normalization']
                sigma_n2 = 1. / graph2[n2]['normalization']
        elif normalize:
            sigma_n2 = 1. / graph2[n2]['normalization']
        return sigma_n1 * sigma_n2 * (0.1 * delta_kernel +
                                      0.6 * self.tag_kernel(graph1, graph2, n1, n2) +
                                      0.3 * self.geo_kernel(graph1, graph2, n1, n2))

    @staticmethod
    def edge_kernel(graph1, graph2, n1, n2, m1, m2):
        """
        This computes the edge kernel between a pair of nodes in one graph versus another.
        :param graph1: Dictionary representing graph1.
        :param graph2: Dictionary representing graph2.
        :param n1: Node 1 in graph1.
        :param n2: Node 2 in graph1.
        :param m1: Node 1 in graph2.
        :param m2: Node 2 in graph2.
        :return: True if the edge types is identical. The edge parent is ignored.
        """
        e = graph1[n1]['neighbours'][n2]
        f = graph2[m1]['neighbours'][m2]
        for edge in e:
            if ('parent' != edge) and (edge in f):
                return True
        return False

    def compute_graph_kernel(self, graph1, graph2, n1, n2, walk_length, node_kernel_seen, graph_kernel_seen,
                             search_query=False):
        """
        This computes the pth rooted walk graph kernel.
        :param graph1: Dictionary representing graph1.
        :param graph2: Dictionary representing graph2.
        :param n1: Node 1 in graph1.
        :param n2: Node 2 in graph2.
        :param walk_length: The length of the walk so far.
        :param node_kernel_seen: Dictionary containing all pairs of nodes in the two graphs for which we have computed
                                 the node kernel so far.
        :param graph_kernel_seen: Dictionary containing all pth rooted graph kernels computed so far. The key can be
                                  (n1, n2, p)
        :param search_query: If True the node kernel is computed differently.
        :return: pth rooted graph kernel starting from node n1 in graph1 and n2 in graph2.
        """
        # compute the initial node kernel
        if (n1, n2) not in node_kernel_seen:
            node_kernel_seen[(n1, n2)] = self.node_kernel(graph1, graph2, n1, n2, normalize=True,
                                                          search_query=search_query)
        result_init = node_kernel_seen[(n1, n2)]
        if (walk_length == 0) or (len(graph1[n1]['neighbours']) == 0) or (len(graph2[n2]['neighbours']) == 0):
            return result_init
        result_recursive = 0
        # check the neighbours of n1 and n2
        for r_prime in graph1[n1]['neighbours']:
            for s_prime in graph2[n2]['neighbours']:
                # add kernel to computation if the edge kernel is 1
                if self.edge_kernel(graph1, graph2, n1, r_prime, n2, s_prime):
                    if (r_prime, s_prime, walk_length-1) not in graph_kernel_seen:
                        new_kernel = self.compute_graph_kernel(graph1,
                                                               graph2,
                                                               r_prime,
                                                               s_prime,
                                                               walk_length-1,
                                                               node_kernel_seen,
                                                               graph_kernel_seen,
                                                               search_query=False)
                        graph_kernel_seen[(r_prime, s_prime, walk_length - 1)] = new_kernel
                    new_kernel = graph_kernel_seen[(r_prime, s_prime, walk_length-1)]

                    result_recursive += new_kernel
        graph_kernel_seen[(n1, n2, walk_length)] = result_init * result_recursive
        return graph_kernel_seen[(n1, n2, walk_length)]

    def compute_graph_kernel_full(self, graph1, graph2):
        """
        Compute the graph kernel between two graphs.
        :param graph1: Dictionary representing graph1.
        :param graph2: Dictionary representing graph2.
        :return: graph_kernel: The computed graph kernel.
        """
        node_kernel_seen = {}
        graph_kernel_seen = {}
        graph_kernel = 0
        for n1, _ in graph1.items():
            for n2, _ in graph2.items():
                kernel = self.compute_graph_kernel(graph1,
                                                   graph2,
                                                   n1,
                                                   n2,
                                                   self.walk_length,
                                                   node_kernel_seen,
                                                   graph_kernel_seen)
                graph_kernel += kernel
        return graph_kernel

    def compute_distance(self, normalize=True):
        """
        This computes the distance between two graphs.
        :param normalize: If True each graph kernel is normalized.
        :return: The distance between two graphs.
        """
        g1g1 = self.compute_graph_kernel_full(self.graph1, self.graph1)
        g1g2 = self.compute_graph_kernel_full(self.graph1, self.graph2)
        g2g2 = self.compute_graph_kernel_full(self.graph2, self.graph2)
        if normalize:
            g1g2_normalized = g1g2 / max(g1g1, g2g2)
            return np.sqrt(1 - 2 * g1g2_normalized + 1)
        else:
            return np.sqrt(g1g1 - 2 * g1g2 + g2g2)

    def context_based_search(self, q, training_graphs, search_query=False):
        """
        This finds the topk models in the database that have highest kernel compared to a query mdoel.
        :param q: The query model represented as a string of its id in graph1.
        :param graphs: All the graphs except graph1.
        :param topk: Integer representing the number of results returned.
        :param search_query: If True the query object is truly unknown; otherwise the function finds the corresponding
                             objects to a known object.
        :return: topk_closest: A list of the topk models and the graph they belong to along with
                               the computed kernel. T
        :return: g1_nodes_visited: All the nodes in graph1 that are vested during the random walk for computing the
                 graph kernels. Note when I compute the random walk for all other graphs compared to graph1, I return
                 the largest set of nodes visited in graph1.
        """
        topk_closest = []
        for graph_name in training_graphs:
            node_kernel_seen = {}
            graph_kernel_seen = {}
            # set up the graph and find normalization constants for the node kernel
            self.graph2_name = graph_name
            self.graph2 = self.read_graph(self.graph2_name, mode=self.mode)
            self.normalize_node_kernel(self.graph2)
            for n, _ in self.graph2.items():
                value = self.compute_graph_kernel(self.graph1,
                                                  self.graph2,
                                                  q,
                                                  n,
                                                  walk_length=self.walk_length,
                                                  node_kernel_seen=node_kernel_seen,
                                                  graph_kernel_seen=graph_kernel_seen,
                                                  search_query=search_query)

                element = (value, self.graph2_name, n)
                topk_closest.append(element)
        # sort the results
        top_results = sorted(topk_closest, reverse=True, key=lambda x: x[0])
        return top_results

    def find_cat(self, graph, node):
        cat = graph[node]['category']
        if len(cat) > 0:
            cat = cat[0]
        else:
            cat = ''
        return cat

    @staticmethod
    def find_best_constraint_candidate(node_to_kernel_candidates):
        visited = set()
        constraint_node_to_best_candidate = {}
        # sort the candidates for each constraint node using the kernel value
        for node, kernel_candidates in node_to_kernel_candidates.items():
            sorted_kernel_candidates = sorted(kernel_candidates, reverse=True, key=lambda x: x[0])
            for kernel, candidate in sorted_kernel_candidates:
                if candidate not in visited:
                    visited.add(candidate)
                    constraint_node_to_best_candidate[node] = (kernel, candidate)
                    break
        return constraint_node_to_best_candidate

    def find_combinations(self, in_list, sofar, remaining, results):
        if len(remaining) == 0:
            results.append(sofar)
            return results
        for i in range(len(in_list)):
            if len(in_list[i]) == 0:
                self.find_combinations(in_list[i + 1:], sofar, remaining[:i] + remaining[i + 1:],
                                       results)
            else:
                for j in range(len(in_list[i])):
                    self.find_combinations(in_list[i+1:], sofar + in_list[i][j:j+1], remaining[:i] + remaining[i+1:],
                                           results)
        return results

    # @staticmethod
    # def find_best_matching_subgraph(context_nodes_to_kernels_candidates, target_node, target_kernel):
    #     # for each context node pick the best candidate if it improves the average kernel
    #     subgraph_kernels = [target_kernel]
    #     best_kernel_avg = target_kernel
    #     best_other_nodes = []
    #     visited = {target_node}
    #     for context_node, kernels_candidates in context_nodes_to_kernels_candidates.items():
    #         # examine each candidate for the context object
    #         best_candidate = None
    #         best_kernel = None
    #         for kernel, candidate in kernels_candidates:
    #             if candidate not in visited:
    #                 temp_subgraph_kernels = subgraph_kernels.copy()
    #                 temp_subgraph_kernels.append(kernel)
    #                 new_kernel_avg = np.mean(temp_subgraph_kernels)
    #                 if new_kernel_avg > best_kernel_avg:
    #                     best_kernel_avg = new_kernel_avg
    #                     best_candidate = candidate
    #                     best_kernel = kernel
    #
    #         # pick the best candidate for the context object and update information
    #         if best_candidate is not None:
    #             subgraph_kernels.append(best_kernel)
    #             best_other_nodes.append(best_candidate)
    #             visited.add(best_candidate)
    #
    #     return best_kernel_avg, best_other_nodes
    @staticmethod
    def find_best_matching_subgraph(context_nodes_to_kernels_candidates, target_node, target_kernel):
        # for each context node pick the best candidate
        subgraph_kernels = [target_kernel]
        best_candidates = []
        visited = {target_node}
        for context_node, kernels_candidates in context_nodes_to_kernels_candidates.items():
            # sort the candidates by highest kernel
            sorted_kernels_candidates = sorted(kernels_candidates, reverse=True, key=lambda x: x[0])
            for kernel, candidate in sorted_kernels_candidates:
                # choose the candidate with highest kernel that is not visited.
                if candidate not in visited:
                    best_candidates.append(candidate)
                    subgraph_kernels.append(kernel)
                    visited.add(candidate)
                    break

        # compute the average kernel.
        avg_kernel = np.mean(subgraph_kernels)

        return avg_kernel, best_candidates

    def context_based_subgraph_matching(self, query_node, context_nodes, target_graphs):
        # concat context nodes and the query node
        query_and_context_nodes = context_nodes + [query_node]

        # find the best matching subgraph by computing the graph kernel
        top_results = []
        for graph_name in target_graphs:
            node_kernel_seen = {}
            graph_kernel_seen = {}
            # load the target graph
            self.graph2_name = graph_name
            self.graph2 = self.read_graph(self.graph2_name, mode=self.mode)
            self.normalize_node_kernel(self.graph2)

            # initialize the candidates for each constraint node and query node
            node_to_candidates = {node: {} for node in query_and_context_nodes}
            for node in query_and_context_nodes:
                node_to_candidates[node]['cat'] = self.find_cat(self.graph1, node)
                node_to_candidates[node]['candidates'] = []

            # find candidates for each context and query node
            for node, node_info in self.graph2.items():
                node_cat = self.find_cat(self.graph2, node)

                # build candidates for other context nodes
                for n in query_and_context_nodes:
                    if node_cat == node_to_candidates[n]['cat']:
                        node_to_candidates[n]['candidates'].append(node)

            # only proceed with subgraph matching if the query nodes matched
            if len(node_to_candidates[query_node]['candidates']) == 0:
                continue

            # for each candidate of a context node compute the graph kernel starting from the candidate versus the
            # context node.
            node_to_kernels_candidates = {node: [] for node in query_and_context_nodes}
            for q_node in query_and_context_nodes:
                for t_candidate in node_to_candidates[q_node]['candidates']:
                    value = self.compute_graph_kernel(self.graph1,
                                                      self.graph2,
                                                      q_node,
                                                      t_candidate,
                                                      walk_length=self.walk_length,
                                                      node_kernel_seen=node_kernel_seen,
                                                      graph_kernel_seen=graph_kernel_seen,
                                                      search_query=False)
                    node_to_kernels_candidates[q_node].append((value, t_candidate))

            # examine each candidates of the query node as a potential target node
            kernels_target_nodes = node_to_kernels_candidates[query_node]

            # find the best subgraph for each candidate target node
            context_nodes_to_kernels_candidates = {node: node_to_kernels_candidates[node] for node in
                                                   node_to_kernels_candidates.keys() if node != query_node}
            for target_kernel, target_node in kernels_target_nodes:
                kernel, other_nodes = self.find_best_matching_subgraph(context_nodes_to_kernels_candidates,
                                                                       target_node, target_kernel)
                element = (kernel, graph_name, target_node, other_nodes)
                top_results.append(element)

        # sort the retrieved list by the computed kernels
        top_results = sorted(top_results, reverse=True, key=lambda x: x[0])
        return top_results
