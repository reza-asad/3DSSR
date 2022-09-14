import os
import argparse
import trimesh
import numpy as np
from PIL import Image

from scripts.helper import load_from_json, create_img_table
from scripts.renderer import Render

class2type = {
    0: "cabinet",
    1: "bed",
    2: "chair",
    3: "sofa",
    4: "table",
    5: "door",
    6: "window",
    7: "bookshelf",
    8: "picture",
    9: "counter",
    10: "desk",
    11: "curtain",
    12: "refrigerator",
    13: "showercurtrain",
    14: "toilet",
    15: "sink",
    16: "bathtub",
    17: "garbagebin",
}


def hex_to_rgb(hex):
    rgb = []
    hex = hex.lstrip('#')
    for i in (0, 2, 4):
        decimal = int(hex[i:i + 2], 16)
        rgb.append(decimal)

    return tuple(rgb)


def align_scan(meta_file, scene_mesh):
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    scene_mesh.apply_transform(axis_align_matrix)

    return scene_mesh


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


def write_bbox(corners, color, output_file):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string
    """

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                indices.append(
                    np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.03
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)


def prepare_scene_for_rendering(args, objects_info, scene_name, rot_mat):
    # load the mesh corresponding to the scene.
    scene_mesh_path = os.path.join(args.scan_dir, scene_name, '{}_vh_clean_2.ply'.format(scene_name))
    scene_mesh = trimesh.load(scene_mesh_path)

    # canonicalize the orientation of the mesh.
    meta_file = os.path.join(args.scan_dir, scene_name, scene_name + '.txt')
    scene_mesh = align_scan(meta_file, scene_mesh)

    # rotate the scene according to the gt/predicted rot mat
    transformation = np.eye(4)
    transformation[:3, :3] = rot_mat
    scene_mesh.apply_transform(transformation)

    # load the boxes and add them to the scene mesh.
    scene_with_boxes = [scene_mesh]
    for object_info in objects_info:
        # load the class id and map it to a category.
        category = class2type[object_info['class_id']]
        rgb_color = hex_to_rgb(color_map[category])

        box_corners = np.asarray(object_info['box'], dtype=np.float32)
        box_corners[..., 1] *= -1
        box_corners[..., [0, 1, 2]] = box_corners[..., [0, 2, 1]]
        inds = np.lexsort((box_corners[:, 2], box_corners[:, 1], box_corners[:, 0]))
        box_corners = box_corners[inds]
        centroid = np.mean(box_corners, axis=0)
        centroid = np.expand_dims(centroid, axis=0)
        vertices = np.concatenate([centroid, box_corners], axis=0)
        write_bbox(vertices, rgb_color, 'test.ply')
        box = trimesh.load('test.ply')
        scene_with_boxes.append(box)

    return trimesh.Scene(scene_with_boxes), len(objects_info)


def render_scene(scene_with_boxes):
    # initialize the renderer.
    r = Render(rendering_kwargs)

    # set up the camera pose and extract the dimensions of the room
    room_dimension = scene_with_boxes.extents
    camera_pose, _ = scene_with_boxes.graph[scene_with_boxes.camera.name]

    # render the pc
    img, _ = r.pyrender_render(scene_with_boxes, resolution=resolution, camera_pose=camera_pose,
                               room_dimension=room_dimension, adjust_camera_height=False, with_height_offset=False)

    return Image.fromarray(img)


def get_args():
    parser = argparse.ArgumentParser('Point Transformer Classification', add_help=False)

    parser.add_argument('--dataset', default='scannet')
    parser.add_argument('--scan_dir', default='/media/reza/Large/scannet/scans')
    parser.add_argument('--results_dir', default='../../results/{}/LayoutMatching/rendered_results')
    parser.add_argument("--experiment_name", default='two_scenes_mask_rot_att_query_equiv_two_enc_align', type=str)
    parser.add_argument('--color_map_path', default='../../data/{}/color_map.json')
    parser.add_argument('--num_queries', default=50)

    return parser


def adjust_paths(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v):
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # create a directory for results
    boxes_path = os.path.join(args.results_dir, args.experiment_name, 'query_predictions.json')
    output_dir = os.path.join(args.results_dir, args.experiment_name, 'imgs')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load the query and predicted boxes.
    query_predictions = load_from_json(boxes_path)

    # randomly take num_query results from query predictions.
    np.random.seed(0)
    n = len(query_predictions['query'])
    indices = np.random.choice(range(n), args.num_queries, replace=False)

    # render query and target scenes.
    imgs_captions = []
    for idx in indices:
        # find the scene name
        scene_name = query_predictions['scene_names'][idx]

        # determine the gt and rpedicted rot mats.
        if 'rot_mats' not in query_predictions:
            rot_mat = np.eye(3)
            pred_rot_mat = np.eye(3)
        else:
            rot_mat = np.asarray(query_predictions['rot_mats'][idx])
            pred_rot_mat = np.asarray(query_predictions['pred_rot_mats'][idx])

        # render the scene with gt boxes.
        scene_with_boxes, num_objects = prepare_scene_for_rendering(args, query_predictions['query'][idx], scene_name,
                                                                    rot_mat=rot_mat)
        img = render_scene(scene_with_boxes)
        img_name = 'gt-{}.png'.format(scene_name)
        output_path = os.path.join(output_dir, img_name)
        img.save(output_path)
        caption = '<br />\n'
        caption += 'Num objects: {} <br />\n'.format(num_objects)
        imgs_captions.append((img_name, caption))

        # render the scene with predicted boxes.
        scene_with_boxes, num_objects = prepare_scene_for_rendering(args, query_predictions['predictions'][idx],
                                                                    scene_name, rot_mat=pred_rot_mat)
        img = render_scene(scene_with_boxes)
        img_name = 'predicted-{}.png'.format(scene_name)
        output_path = os.path.join(output_dir, img_name)
        img.save(output_path)
        caption = '<br />\n'
        caption += 'Num objects: {} <br />\n'.format(num_objects)
        imgs_captions.append((img_name, caption))

    # create img table.
    imgs, captions = list(zip(*imgs_captions))
    create_img_table(output_dir, 'imgs', imgs, 'img_table.html', topk=args.num_queries*2, ncols=2, captions=captions,
                     with_query_scene=False, evaluation_plot=None, query_img=None, query_caption=None)


if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser('Point Transformer Classification', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args)

    # rendering args
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    resolution = (512, 512)
    color_map = load_from_json(args.color_map_path)

    main()


