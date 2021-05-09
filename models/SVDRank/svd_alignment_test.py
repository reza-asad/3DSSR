import os
import numpy as np
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


def translate_obbox(obbox, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation.astype(np.float128)

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def compute_alignment_error(vertices1, vertices2):
    err = np.linalg.norm(vertices2 - vertices1)

    return err


def main():
    # define the paths and params
    scene_name = '1pXnuDYAj8r_room18.json'
    scene_graph_dir = '../../results/matterport3d/LearningBased/scene_graphs_cl/all'
    objects = ['10', '13', '6']

    # load the scene
    graph = load_from_json(os.path.join(scene_graph_dir, scene_name))

    # load the vertices for the OBB of the objects
    # obj1_vertices = np.asarray([[0., 0., 0.],
    #                             [-3, -1, 1], [3, -1, 1], [1, 1, 1], [-1, 1, 1],
    #                             [-3, -1, -1], [3, -1, -1], [1, 1, -1], [-1, 1, -1]], dtype=np.float64)
    obj1_vertices = np.asarray(graph[objects[0]]['obbox'])
    obj1_obbox = Box(obj1_vertices)
    translation = -obj1_obbox.translation

    # obj2_vertices = obj1_vertices.copy()
    # obj2_vertices[0, :2] += 30
    obj2_vertices = np.asarray(graph[objects[1]]['obbox'])
    obj2_obbox = Box(obj2_vertices)
    obj2_obbox = translate_obbox(obj2_obbox, translation)

    # theta = -np.pi/8
    # rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
    #                        [np.sin(theta), np.cos(theta), 0],
    #                        [0, 0, 1]])
    # obj3_vertices = np.dot(rotation, obj2_obbox.vertices.transpose()).transpose()
    obj3_vertices = np.asarray(graph[objects[2]]['obbox'])
    obj3_obbox = Box(obj3_vertices)
    obj3_obbox = translate_obbox(obj3_obbox, translation)

    # translate the source object
    obj1_obbox = translate_obbox(obj1_obbox, translation)

    # print(obj2_obbox.vertices)
    # print('*'*50)
    # print(obj3_obbox.vertices)

    # visualize the obboxes before the rotation
    o1 = trimesh.creation.box(obj1_obbox.scale, transform=obj1_obbox.transformation)
    o1.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")

    o2 = trimesh.creation.box(obj2_obbox.scale, transform=obj2_obbox.transformation)
    o2.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#FF0000")

    o3 = trimesh.creation.box(obj3_obbox.scale, transform=obj3_obbox.transformation)
    o3.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#00FF00")

    trimesh.Scene([o1, o2, o3]).show()

    # compute the alignment error before
    err = compute_alignment_error(obj2_obbox.vertices, obj3_obbox.vertices)
    print(err)

    # use svd to find the rotation that aligns the first obbox with the second one.
    COR = np.dot(obj2_obbox.vertices.transpose(), obj3_obbox.vertices)
    U, S, Vt = np.linalg.svd(COR)
    R = np.dot(Vt.transpose(), U.transpose())
    print(np.linalg.det(R))
    if np.linalg.det(R) < 0:
        U, S, Vt = np.linalg.svd(R)
        U[0, :] *= -1
        R = np.dot(Vt.transpose(), U.transpose())
        print(np.linalg.det(R))

    # apply the rotation to the first obbox
    obj2_vertices = np.dot(R, obj2_obbox.vertices.transpose()).transpose()
    obj2_obbox = Box(obj2_vertices)
    o2 = trimesh.creation.box(obj2_obbox.scale, transform=obj2_obbox.transformation)
    o2.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#FF0000")

    # compute the alignment error after
    err = compute_alignment_error(obj2_obbox.vertices, obj3_obbox.vertices)
    print(err)

    # visualize the obboxes after the rotation
    trimesh.Scene([o1, o2, o3]).show()


if __name__ == '__main__':
    main()
