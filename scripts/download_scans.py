import os

# download and local directories
root_dir = '/datasets/released/scannet/public/v2/scans/'
local_dir = '/media/reza/Large/ScanNetV2'

scene_names_val = open('data/scannetv2_val.txt').readlines()
scene_names_test = open('data/scannetv2_test.txt').readlines()
scene_names_train = open('data/scannetv2_train.txt').readlines()
scene_names = scene_names_val + scene_names_test + scene_names_train

for scene_name in scene_names:
    scene_name = scene_name.strip()
    decimated_scene_name = scene_name + '_vh_clean_2.ply'
    command = 'sshpass -p Ifym5tjk! rsync -rtvhP cs-gruvi-77:{}/{}/{} {}'.format(root_dir, scene_name,
                                                                                 decimated_scene_name, local_dir)
    os.system(command)


