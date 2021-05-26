import trimesh
import numpy as np
import warnings
import gc

from base_scene import BaseScene
from helper import sample_mesh


class RelationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class SceneGraph(BaseScene):
    def __init__(self, models_dir, scene_graph_dir, scene_name, model_dir_with_texture=None):
        """
        Initialize an instance of a scene
        :param models_dir: The path to the models directory
        """
        super().__init__(models_dir, scene_graph_dir, scene_name)
        self.models_dir_with_texture = model_dir_with_texture

        self.has_fake_vertical_contact = False
        self.fake_vertical_contact_counts = 0

    def add_edge(self, obj1, obj2, edge):
        """
        Add edge between two objects in the scene graph
        :param obj1: String id of object 1
        :param obj2: String id of object 2
        :param edge: Edge type
        """
        if obj2 in self.graph[obj1]['neighbours']:
            if edge not in self.graph[obj1]['neighbours'][obj2]:
                self.graph[obj1]['neighbours'][obj2].append(edge)
        else:
            self.graph[obj1]['neighbours'][obj2] = [edge]

    def build_hierarchical_scene(self, parent='0'):
        """
        This will build a scene based on the hierarchical graph built earlier. The elements are added  to the scene
        starting from a parent node in a depth first search fashion.
        :param parent: The node to start building the scene from
        """
        # use the graph and the transform matrix to bring every object to a global coordinate
        def traverse_graph(parent_obj, parent_mesh):
            # base cases with 0 children
            if len(self.graph[parent_obj]['neighbours']) == 0:
                return parent_mesh
            for neighbour in self.graph[parent_obj]['neighbours'].keys():
                neighbour_mesh = self.prepare_mesh_for_scene(neighbour, self.graph)
                parent_mesh += [neighbour_mesh]
                parent_mesh = traverse_graph(neighbour, parent_mesh)
            return parent_mesh
        if len(self.graph) == 0:
            raise ValueError('You need to populate the graph first')

        p_mesh = self.prepare_mesh_for_scene(parent, self.graph)
        scene = traverse_graph(parent, [p_mesh])
        return trimesh.Scene(scene)

    def build_scene(self, graph):
        """
        This will build a scene based on a non-hierarchical graph structure.
        """
        # use the graph and the transform matrix to bring every object to a global coordinate
        scene = []
        for node, node_info in graph.items():
            mesh = self.prepare_mesh_for_scene(node, graph)
            scene += [mesh]
        if len(graph) == 0:
            raise ValueError('You need to populate the graph first')
        return trimesh.Scene(scene)

    @staticmethod
    def compute_1d_overlap(l1, l2):
        """
        Compute the 1d intersection between two lines
        :param l1: (x1, x2) representing the beginning and end of first line
        :param l2: (y1, y2) representing the beginning and end of second line
        :return: The intersection value
        """
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
            return x2 - y1

    def compute_overlap(self, mesh1, mesh2, normal_dir=None):
        """
        Compute the overlap between the bounding boxes of two meshes in 3d or the projected overlap in 2d.
        :param mesh1: Mesh object 1
        :param mesh2: Mesh object 2
        :param normal_dir: If None check the full overlap otherwise check the 2d overlap along the normal direction.
        :return: A triplet for the computed intersection, the volume of first object and volume of the second object
                 along each dimension.
        """

        def sanity_check_volume(v, inter):
            if inter > v:
                raise ValueError('Intersection higher than volume')
            return v
        # extract the bounding boxes and compute the intersection
        bbox1 = mesh1.bounding_box.bounds
        bbox2 = mesh2.bounding_box.bounds

        if normal_dir is None:
            idxs = [0, 1, 2]
            dim = 3
        else:
            dim = 2
            direction = np.argmax(normal_dir)
            if direction == 0:
                idxs = [1, 2]
            elif direction == 1:
                idxs = [0, 2]
            else:
                idxs = [0, 1]

        intersection = np.ones(dim)
        volume1 = np.ones(dim)
        volume2 = np.ones(dim)
        for i, idx in enumerate(idxs):
            intersection[i] = self.compute_1d_overlap(bbox1[:, idx], bbox2[:, idx])
            # sanity check if the volume is 0 the intersection is also 0.
            volume1[i] = sanity_check_volume(abs(bbox1[1, idx] - bbox1[0, idx]), intersection[i])
            volume2[i] = sanity_check_volume(abs(bbox2[1, idx] - bbox2[0, idx]), intersection[i])
        return intersection, volume1, volume2

    def check_enclosure(self, obj1, obj2, overlap_percent=0.95):
        """
        Checks the enclosure of mesh1 and mesh2
        :param obj1: First mesh
        :param obj2: Second mesh
        :param overlap_percent: The overlap threshold to be count the enclosure.
        :return: True if at least 95% of the mesh1 volume is enclosed in mesh2
        """
        # load the mesh for each object
        mesh1 = self.prepare_mesh_for_scene(obj1, self.graph)
        mesh2 = self.prepare_mesh_for_scene(obj2, self.graph)

        intersection, volume1, volume2 = self.compute_overlap(mesh1, mesh2)

        # make sure you only count positive intersection
        intersection = np.maximum(intersection, 0)
        obj1_encloses_obj2, obj2_encloses_obj1 = 0, 0
        if np.prod(volume2) != 0:
            obj1_encloses_obj2 = np.prod(intersection / volume2) >= overlap_percent
        if np.prod(volume1) != 0:
            obj2_encloses_obj1 = np.prod(intersection / volume1) >= overlap_percent
        del mesh1
        del mesh2
        gc.collect()
        return obj1_encloses_obj2, obj2_encloses_obj1

    @staticmethod
    def find_angle(normal1, normal2, normal1_mag, normal2_mag, unoriented=True):
        """
        This computes the angle between two normal vectors.
        :param normal1: First normal vector.
        :param normal2: Second normal vector.
        :param normal1_mag: Magnitude of first normal vector.
        :param normal2_mag: Magnitude of second normal vector.
        :param unoriented: If true the unoriented angle is computed.
        :return: Angle between two normal vectors.
        """
        cos_theta = np.dot(normal1, normal2) / (normal1_mag * normal2_mag)
        # Underflow results in values such as 1.000000000002 which arccos can not handle. Clip cosine between -1 and 1.
        cos_theta = np.clip(cos_theta, -1, 1)
        angle_diff = np.arccos(cos_theta) * 180 / np.pi
        # print('normals: ', normal1, normal2, normal1_mag, normal2_mag)
        if unoriented:
            return min(angle_diff, 180 - angle_diff)
        else:
            return angle_diff

    def check_support_type(self, mesh1, mesh2, normal, normal_mag, angle_eps=1):
        """
        Find if mesh1 is supporting mesh2 horizontally or if the meshes have vertical contact.
        :param mesh1: Mesh representing object 1.
        :param mesh2: Mesh representing object 2.
        :param normal: The normal direction of the contact polygon.
        :param normal_mag: The magnitude of the normal vector for the contact polygon.
        :param angle_eps: The cutoff threshold to count horizontal support or vertical contact.
        :return: support: The type of support.
        """
        support = None
        gravity = np.asarray([0, 0, 1])
        angle_wrt_gravity = self.find_angle(normal, gravity, normal_mag, 1, unoriented=True)
        # print('angle_wrt_gravity: ', angle_wrt_gravity)
        if angle_wrt_gravity < angle_eps:
            cent_diff = mesh1.centroid - mesh2.centroid
            if cent_diff[2] > 0:
                support = 'supported'
            else:
                support = 'supports'
        elif abs(angle_wrt_gravity - 90) < angle_eps:
            # compute the overlap between the projected bounding boxes of the vertical contact candidates. The
            # projection is in the direction of the normal of the contact polygon.
            intersection, volume1, volume2 = self.compute_overlap(mesh1, mesh2, normal_dir=normal)
            prod_volume1 = np.prod(volume1)
            prod_volume2 = np.prod(volume2)
            overlap = 0
            if prod_volume1 != 0 and prod_volume2 != 0:
                overlap = max(np.prod(intersection/volume1), np.prod(intersection/volume2))
            elif prod_volume1 != 0:
                overlap = np.prod(intersection/volume1)
            elif prod_volume2 != 0:
                overlap = np.prod(intersection/volume2)
            if overlap > 0.1:
                support = 'vertical_contact'
            else:
                self.has_fake_vertical_contact = True
        return support

    def check_support(self, obj1, obj2, num_samples=1000, chunk_size=1000, dist_eps=0.25, angle_eps=1):
        """
        Find and examine contact polygons. If successful return the relations found e.g horizontal_support,
        vetical_contact, oblique.
        :param obj1: String id for object 1.
        :param obj2: String id for object 2.
        :param num_samples: Number of points to be sampled from obj2.
        :param chunk_size: The size of each chunk for processing the vertices of obj1.
        :param dist_eps: The distnace threshold for accepting contact polygon.
        :param angle_eps: The angle threshold between normals for accepting contact polygon.
        :return: supports: List of supports found
        """
        mesh1 = self.prepare_mesh_for_scene(obj1, self.graph)
        mesh2 = self.prepare_mesh_for_scene(obj2, self.graph)
        supports = []
        # If bounding boxes of the objects are not in contact there is no support
        intersection, _, _ = self.compute_overlap(mesh1, mesh2)
        if any(intersection < -dist_eps):
            del mesh1
            del mesh2
            gc.collect()
            return supports

        # sample 1000 points on the mesh if there are
        points2, face2_idx = sample_mesh(mesh2, count=num_samples)
        # reduce chunk_size if mesh1 is large
        if len(mesh1.faces) > 20000:
            chunk_size = 100
        # Find closest vertices in mesh2 to points on the surface of mesh1 in chunks of certain size
        if chunk_size > len(points2):
            num_chunks = 1
        else:
            num_chunks = len(points2)//chunk_size
        face_normals1 = mesh1.face_normals
        face_normals2 = mesh2.face_normals
        oblique = False
        for i in range(num_chunks):
            _, distance, triangle_id = trimesh.proximity.closest_point(mesh1, points2[i*chunk_size: (i+1) * chunk_size, ])
            arg_dist = np.argsort(distance)
            # examine the points in order of closeness to mesh1
            for j, arg_dist_idx in enumerate(arg_dist):
                # distance must be within some epsilon
                if distance[arg_dist_idx] > dist_eps:
                    break
                # compute the normals
                normal1 = face_normals1[triangle_id[arg_dist_idx], :]
                normal1_mag = np.linalg.norm(normal1)
                if normal1_mag == 0:
                    continue
                normal2 = face_normals2[face2_idx[i*chunk_size+j]]
                normal2_mag = np.linalg.norm(normal2)
                if normal2_mag == 0:
                    continue
                # check for contact polygon
                if self.find_angle(normal1, normal2, normal1_mag, normal2_mag, unoriented=True) < angle_eps:
                    oblique = True
                    # if relationship is parent check for horizontal and vertical support first
                    support = self.check_support_type(mesh1, mesh2, normal1, normal1_mag, angle_eps=angle_eps)
                    if (support is not None) and (support not in supports):
                        supports.append(support)
                    if len(supports) == 2:
                        del mesh1
                        del mesh2
                        gc.collect()
                        return supports
        del mesh1
        del mesh2
        gc.collect()
        if oblique and len(supports) == 0:
            return ['oblique']
        return supports

    def build_scene_graph(self, test_objects, num_samples, chunk_size, dist_eps, angle_eps,
                          max_num_failures=65, distance_increment=0.5, angle_increment=5,
                          distance_reset_val=6, room_cats=['room'], bad_object_cats=[]):
        """
        Build a scene graph by comparing the all pairs of objects (mesh) in the scene.
        :return: The hierarchy graph will be augmented with a set of predefined relationships; i.e enclosure,
                 horizontal_support and vertical_contact
        :param test_objects: Set of test objects to build the graph for. If this is emty the graph is built based on
               the objects in the hierarchical graph.
        :param num_samples: Number of sample points used in obj2.
        :param chunk_size: The number of vertices in obj1 to process at once.
        :param dist_eps: The distance threshold to accept a contact polygon.
        :param angle_eps: The threshold for the angle between the contact polygons.
        :param max_num_failures: Maximum number of times the script can retry comparing a pair of objects.
        :param distance_increment: How much we can relax the dist_eps after each failure in detecting a contact polygon,
                                   given a pair of objects.
        :param angle_increment: How much we can relax the angle_eps after each failure in detecting a contact polygon,
                                given a pair of objects.
        :param distance_reset_val: After how many failure attempts we should reset the dist_eps to its original value
                                   while still relaxing the angle.
        """
        if len(test_objects) == 0:
            objects = list(self.graph.keys())
        else:
            objects = test_objects

        # filter the objects and remove junks
        filtered_objects = []
        for obj in objects:
            reject = False
            for cat in bad_object_cats:
                if cat in self.graph[obj]['category']:
                    reject = True
                    break
            if not reject:
                filtered_objects.append(obj)
        i, j = 0, 1
        original_dist_eps = dist_eps
        original_angle_eps = angle_eps
        num_failure = 0
        while i < len(filtered_objects):
            while j < len(filtered_objects):
                obj1, obj2 = filtered_objects[i], filtered_objects[j]
                # print(obj1, obj2)
                # print('new_eps: ', dist_eps)
                # print('angle: ', angle_eps)
                # print('num failure: ', num_failure)
                # print(self.objects[obj1]['category'], self.objects[obj2]['category'])

                # check support and oblique relation
                supports = self.check_support(obj1, obj2, num_samples=num_samples, chunk_size=chunk_size,
                                              dist_eps=dist_eps, angle_eps=angle_eps)
                # print(supports)
                if len(supports) > 0:
                    if supports[0] == 'oblique':
                        self.add_edge(obj1, obj2, 'oblique')
                        self.add_edge(obj2, obj1, 'oblique')
                    if 'vertical_contact' in supports:
                        self.add_edge(obj1, obj2, 'vertical_contact')
                        self.add_edge(obj2, obj1, 'vertical_contact')
                    if 'supports' in supports:
                        self.add_edge(obj1, obj2, 'supports')
                        self.add_edge(obj2, obj1, 'supported')
                    if 'supported' in supports:
                        self.add_edge(obj1, obj2, 'supported')
                        self.add_edge(obj2, obj1, 'supports')

                # check if all the parent support were identified
                nbs = self.graph[obj1]['neighbours']
                if obj2 in nbs:
                    relations = nbs[obj2]
                    if 'parent' in relations:
                        if not any(relation in relations for relation in ['supports',
                                                                          'supported',
                                                                          'vertical_contact',
                                                                          'oblique']):
                            msg = 'Parent must have support relation. Check obj {} and neighbour {} in {}'\
                                .format(obj1, obj2, self.scene_name)
                            warnings.warn(msg)
                            # relax the epsilon by 1 mm
                            dist_eps += distance_increment
                            # if distance is too large there is something wrong with the 3D model, move on.
                            if dist_eps > distance_reset_val:
                                dist_eps = original_dist_eps
                                angle_eps += angle_increment
                            gc.collect()
                            num_failure += 1
                            if num_failure > max_num_failures:
                                raise RelationError('Failed to find parent relation. Check obj {} and {}'
                                                    .format(obj1, obj2))
                            continue
                        else:
                            dist_eps = original_dist_eps
                            angle_eps = original_angle_eps
                            num_failure = 0
                # check enclosure but skip it for room and wall type of objects as it trivially holds
                skip_enclosure = False
                for room_cat in room_cats:
                    if room_cat in self.graph[obj1]['category'] or room_cat in self.graph[obj2]['category']:
                        skip_enclosure = True
                        break
                if not skip_enclosure:
                    obj1_encloses_obj2, obj2_encloses_obj1 = self.check_enclosure(obj1, obj2)
                    if obj1_encloses_obj2:
                        self.add_edge(obj1, obj2, 'encloses')
                        self.add_edge(obj2, obj1, 'enclosed')
                    if obj2_encloses_obj1:
                        self.add_edge(obj2, obj1, 'encloses')
                        self.add_edge(obj1, obj2, 'enclosed')

                j += 1
                if self.has_fake_vertical_contact:
                    self.fake_vertical_contact_counts += 1
                    self.has_fake_vertical_contact = False
            i += 1
            j = i+1

    def visualize(self, parent='0', graph=None):
        """
        visualize the scene built starting from the parent node.
        :param parent: The parent node to start the visualization from.
        :param graph: If this is not None use the graph to build the visualization.
        """
        if graph is None:
            scene = self.build_hierarchical_scene(parent)
        else:
            scene = self.build_scene(graph)
        scene.show()

