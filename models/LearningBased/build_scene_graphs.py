import os
import gc
import trimesh
import numpy as np

from scripts.helper import load_from_json, prepare_mesh_for_scene
from scripts.box import Box
from scripts.iou import IoU


class SceneGraph:
    def __init__(self, scene_dir, scene_name, models_dir, accepted_cats, enclosure_threshold=0.05,
                 obbox_expansion=1.0, context_window=512):
        # load graph and initialize neighbours
        self.graph = load_from_json(os.path.join(scene_dir, scene_name))
        self.initialize_nbs()
        self.models_dir = models_dir
        self.accepted_cats = accepted_cats

        self.enclosure_threshold = enclosure_threshold
        self.obbox_expansion = obbox_expansion
        self.context_window = context_window

    def filter_by_accepted_cats(self):
        filtered_graph = {}
        for node, node_info in self.graph.items():
            if node_info['category'][0] in self.accepted_cats:
                filtered_graph[node] = node_info
        self.graph = filtered_graph

    def initialize_nbs(self):
        for obj, obj_info in self.graph.items():
            obj_info['neighbours'] = {}

    def find_backbone_objects(self):
        backbone_objects, objects = [], []
        for obj, obj_info in self.graph.items():
            cat = obj_info['category'][0]
            if cat in self.accepted_cats:
                objects.append(obj)
            else:
                backbone_objects.append(obj)

        return backbone_objects, objects

    def compute_obbox(self, obj, with_expansion=False):
        vertices = np.asarray(self.graph[obj]['obbox'])
        obbox = Box(vertices)
        if with_expansion:
            obbox = trimesh.creation.box(obbox.scale * self.obbox_expansion, transform=obbox.transformation)
            vertices = np.asarray([obbox.centroid] + obbox.vertices.tolist())
            obbox = Box(vertices)
        return obbox

    def add_edge(self, obj1, obj2, obj1_r_obj2, obj2_r_obj1, edge1, edge2):
        def add_one_direction(o1, o2, e1, e2):
            if o2 in self.graph[o1]['neighbours']:
                if e1 not in self.graph[o1]['neighbours'][o2]:
                    self.graph[o1]['neighbours'][o2].append(e1)
            else:
                self.graph[o1]['neighbours'][o2] = [e1]

            if o1 in self.graph[o2]['neighbours']:
                if e2 not in self.graph[o2]['neighbours'][o1]:
                    self.graph[o2]['neighbours'][o1].append(e2)
            else:
                self.graph[o2]['neighbours'][o1] = [e2]

        if obj1_r_obj2:
            add_one_direction(obj1, obj2, edge1, edge2)
        elif obj2_r_obj1:
            add_one_direction(obj2, obj1, edge1, edge2)

    @staticmethod
    def check_overlap(obbox1, obbox2):
        # compute iou
        iou_computer = IoU(obbox1, obbox2)
        iou = iou_computer.iou()

        # compute intersection and the smaller volume
        volume1 = obbox1.volume
        volume2 = obbox2.volume
        intersection = iou * (volume1 + volume2) / (1 + iou)

        return intersection, volume1, volume2

    @staticmethod
    def compute_bottom_top_heights(obbox):
        # exclude the centroid of the box and sort its vertices by their height (z component)
        vertices = obbox.vertices[1:, :]
        sorted_vertices = vertices[np.argsort(vertices[:, 2])]

        # find the four vertices in the bottom and top of the obbox
        bottom_vertices = sorted_vertices[:4, 2]
        top_vertices = sorted_vertices[4:, 2]

        # compute the centroid of each four vertices
        bottom_height = np.mean(bottom_vertices)
        top_height = np.mean(top_vertices)

        return bottom_height, top_height

    @staticmethod
    def compute_1d_overlap(l1, l2):
        # assume l1 is the line to the left of l2
        if l1[0] > l2[0]:
            x1, x2 = l2
            y1, y2 = l1
        else:
            x1, x2 = l1
            y1, y2 = l2

        if y1 <= x2:
            if y2 <= x2:
                return y2 - y1
            else:
                return x2 - y1
        else:
            return 0

    def x_y_overlap(self, bottom_bbox, top_bbox):
        # compute the x-yovelap between the two bboxes
        intersection = np.zeros(2, dtype=np.float)
        top_bbox_area = np.zeros(2, dtype=np.float)
        for i in range(2):
            intersection[i] = self.compute_1d_overlap(bottom_bbox[:, i], top_bbox[:, i])
            top_bbox_area[i] = abs(top_bbox[1, i] - top_bbox[0, i])

        return np.prod(intersection / top_bbox_area) > 0.67

    def check_horizontal_support(self, mesh1, mesh2, obbox1, obbox2):
        obj1_supports, obj2_supports = False, False
        bbox1 = mesh1.bounding_box.bounds
        bbox2 = mesh2.bounding_box.bounds

        # find the centroid of the top and bottom sides of each obbox
        bottom_height1, top_height1 = self.compute_bottom_top_heights(obbox1)
        bottom_height2, top_height2 = self.compute_bottom_top_heights(obbox2)
        height1 = top_height1 - bottom_height1
        height2 = top_height2 - bottom_height2

        # check if one object is above the other and they overlap in the x-y plane
        obj2_above_obj1 = (abs(bottom_height2 - top_height1) / height1) < 0.67
        obj1_above_obj2 = (abs(bottom_height1 - top_height2) / height2) < 0.67
        if obj2_above_obj1 and self.x_y_overlap(bbox1, bbox2):
            obj1_supports = True
        elif obj1_above_obj2 and self.x_y_overlap(bbox2, bbox1):
            obj2_supports = True

        return obj1_supports, obj2_supports

    def check_enclosure(self, intersection, volume1, volume2):
        volumes = [volume1, volume2]
        obj1_encloses, obj2_encloses = False, False
        min_idx = np.argmin(volumes)
        if (abs(intersection - volumes[min_idx]) / volumes[min_idx]) < self.enclosure_threshold:
            if min_idx == 0:
                obj2_encloses = True
            else:
                obj1_encloses = True

        return obj1_encloses, obj2_encloses

    def find_main_object_relations(self, objects):
        num_objects = len(objects)
        for i in range(num_objects):
            obj1 = objects[i]
            # load the mesh
            mesh1 = prepare_mesh_for_scene(self.models_dir, self.graph, obj1)
            # load the obbox and compute its expansion.
            obbox1 = self.compute_obbox(obj1)
            obbox1_extended = self.compute_obbox(obj1, with_expansion=True)

            for j in range(i+1, num_objects):
                # load the obbox and compute its expansion
                obj2 = objects[j]
                obbox2 = self.compute_obbox(obj2)
                obbox2_extended = self.compute_obbox(obj2, with_expansion=True)

                # compute the overlap between the extended obboxes
                intersection, volume1, volume2 = self.check_overlap(obbox1_extended, obbox2_extended)

                # only bother if the obboxes overlap
                if intersection > 0:
                    # check enclosure
                    obj1_encloses, obj2_encloses = self.check_enclosure(intersection, volume1, volume2)
                    self.add_edge(obj1, obj2, obj1_encloses, obj2_encloses, 'encloses', 'enclosed')

                    # check horizontal support
                    obj1_supports, obj2_supports = False, False
                    if np.sum([obj1_encloses, obj2_encloses]) == 0:
                        # load the mesh
                        mesh2 = prepare_mesh_for_scene(self.models_dir, self.graph, obj2)
                        # check horizontal support
                        obj1_supports, obj2_supports = self.check_horizontal_support(mesh1, mesh2, obbox1, obbox2)
                        self.add_edge(obj1, obj2, obj1_supports, obj2_supports, 'supports', 'supported')

                    # if suport was not detected record a contact relationship
                    if np.sum([obj1_encloses, obj2_encloses, obj1_supports, obj2_supports]) == 0:
                        self.add_edge(obj1, obj2, True, True, 'contact', 'contact')

    def add_fc_edges(self, objects):
        for obj in objects:
            neighbours = self.graph[obj]['neighbours']
            dense_neighbours = neighbours.copy()
            obj_and_neighbours = set(neighbours.keys())
            obj_and_neighbours.add(obj)
            others = set(objects).difference(obj_and_neighbours)
            for n in others:
                dense_neighbours[n] = ['fc']
            # replace sparse with dense neighbours.
            self.graph[obj]['neighbours'] = dense_neighbours

    def build_graph(self, test_objects=None):
        # find the main objects in the scene and do not use backbone objects
        _, objects = self.find_backbone_objects()
        if test_objects is not None:
            objects = test_objects

        # find the relationship between pairs of scene objects using their obboxes
        self.find_main_object_relations(objects)

        # connect nodes that are not connected using: enclosure, support or contact.
        self.add_fc_edges(objects)
