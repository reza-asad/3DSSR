# 3DSSR

**[3DSSR: 3D SubScene Retrieval][1]**  
[Reza Asad][RA], [Manolis Savva][MS], SGP 2021


## Dependencies and Python Enviroment
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

export PYTHONPATH="${PYTHONPATH}:/path/to/the/repository"
```

## Preparing Matterport3D
1. Extract 3D object instances and save the metadata for objects (e.g oriented bounding boxes, caegory, etc).
```
cd scripts
parallel -j5 "python3 -u matterport_preprocessing.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_mesh
python3 matterport_preprocessing.py save_metadata
```
2. Save metadata for the 3D scenes (e.g link to object mesh files, transformation matrix that produces the scene, etc). Split the scenes into train/val/test based on Matterport3D's suggeted split.
```
parallel -j5 "python3 -u build_scenes_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
python3 -u build_scenes_matterport.py split_train_test_val
```
3. Extract 3D pointclouds from the objects. The pointclouds are later fed into 3D Point Capsule Networks[2] and the latent capsules are used for category prediction.
```
parallel -j5 "python3 -u extract_point_clouds.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_pc
python3 extract_point_clouds.py split_train_test_val
```

# 

## AlignRank and ALignRankOracle
To build the scene graphs:
```
parallel -j5 "python3 -u build_scene_graphs_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
python3 build_scene_graphs_matterport.py split_train_test_val
```

To train AlignRank and AlignRankOracle from scratch:

1. Train the AlignmentModule
```
python3 train_AlignmentModule.py 
```
2. Download the trained latent capsules from 
 ```
 ```
 or 
 
 Train a 3D Point Capsule Network[2] on the pointclouds extracted in step 3 of data prepration.
 
4. Train GNN for object category prediction (can skip this step for AlignRankOracle)
```

```

To run the pretrained models:
```
```

## Baselines

[1]: https://sgp2021.github.io/
[2]: https://github.com/yongheng1991/3D-point-capsule-networks
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
