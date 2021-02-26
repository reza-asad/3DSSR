import os
import trimesh
import numpy as np

from scripts.helper import load_from_json
from scripts.mesh import Mesh
from scripts.box import Box
from scripts.iou import IoU


def prepare_mesh_for_scene(obj, scene_graph):
    model_path = os.path.join(models_dir, scene_graph[obj]['file_name'])
    mesh_obj = Mesh(model_path, scene_graph[obj]['transform'])
    return mesh_obj.load(with_transform=True)


def compute_box(obj, scene_graph, ref_centroid=None):
    mesh = prepare_mesh_for_scene(obj, scene_graph)
    centroid = np.asarray(scene_graph[obj]['transform'][12:-1], dtype=np.float32)
    vertices = [centroid] + mesh.bounding_box_oriented.vertices.tolist()
    vertices = np.asarray(vertices)
    if ref_centroid is not None:
        vertices -= ref_centroid
    box = Box(vertices)
    return box


def iou_test(objects, scene_graph, scene_backbone):
    # for every pair of accepted objects compute iou and visualize the pair
    for i in range(len(objects)):
        # compute obbox of object1
        b1 = compute_box(objects[i], scene_graph)
        for j in range(i + 1, len(objects)):
            # compute obbox of object2
            b2 = compute_box(objects[j], scene_graph)

            # compute iou
            iou_computer = IoU(b1, b2)
            iou = iou_computer.iou()
            if iou > 0:
                print('Categories are: ')
                print(scene_graph[objects[i]]['category'][0], scene_graph[objects[j]]['category'][0])
                print('iou is {}'.format(iou))
                scene_pair_obj = scene_backbone.copy()
                mesh1 = prepare_mesh_for_scene(objects[i], scene_graph)
                mesh2 = prepare_mesh_for_scene(objects[j], scene_graph)
                scene_pair_obj += [mesh1.bounding_box_oriented, mesh2.bounding_box_oriented]
                scene_pair_obj = trimesh.Scene(scene_pair_obj)
                scene_pair_obj.show()
                # t=y


def invariant_iou_test(objects, scene_graph, scene_backbone):
    # compute the box of objects 2 and 3
    box2 = compute_box(objects[1], scene_graph)
    box3 = compute_box(objects[2], scene_graph)

    # compute iou in the scene coordinate frame
    iou_computer_before = IoU(box2, box3)
    iou_before = iou_computer_before.iou()

    # compute the box of object 2 and 3 with respect to the centroid of object1
    centroid1 = np.asarray(scene_graph[objects[0]]['transform'][12:-1], dtype=np.float32)
    box2 = compute_box(objects[1], scene_graph, ref_centroid=centroid1)
    box3 = compute_box(objects[2], scene_graph, ref_centroid=centroid1)
    iou_computer_after = IoU(box2, box3)
    iou_after = iou_computer_after.iou()
    print('IoU before is: {}'.format(iou_before))
    print('IoU after is: {}'.format(iou_after))

    # visualize the object pair
    mesh1 = prepare_mesh_for_scene(objects[0], scene_graph)
    mesh2 = prepare_mesh_for_scene(objects[1], scene_graph)
    mesh3 = prepare_mesh_for_scene(objects[2], scene_graph)
    scene_backbone += [mesh1.bounding_box_oriented, mesh2.bounding_box_oriented, mesh3.bounding_box_oriented]
    scene_pair_obj = trimesh.Scene(scene_backbone)
    scene_pair_obj.show()


def main():
    test_iou = False
    test_iou_invariance = True
    room_name = '1pXnuDYAj8r_room18'
    scene_graph_path = '../data/matterport3d/scene_graphs/all/{}'.format(room_name + '.json')

    # load the scene and its graph
    scene = trimesh.load('../data/matterport3d/rooms/{}/{}.annotated.ply'.format(room_name, room_name))
    scene_graph = load_from_json(scene_graph_path)
    # scene.show()

    # load objects with accepted cats
    accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')
    # determine the backbone scene and take accepted objects
    objects = []
    scene_backbone = []
    for obj, obj_info in scene_graph.items():
        cat = obj_info['category'][0]
        if cat in accepted_cats:
            objects.append(obj)
        elif cat in ['floor', 'wall', 'window', 'door']:
            mesh = prepare_mesh_for_scene(obj, scene_graph)
            scene_backbone.append(mesh)

    if test_iou:
        iou_test(objects, scene_graph, scene_backbone)
    if test_iou_invariance:
        objects = ['3', '21', '10']
        invariant_iou_test(objects, scene_graph, scene_backbone)


if __name__ == '__main__':
    models_dir = '../data/matterport3d/models'
    main()
