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
1. Dowload the dataset from [here][1] and place it in this directory:
```
3DSSR/data/matterport3d/rooms
```
3. Extract 3D object instances and save the metadata for objects (e.g oriented bounding boxes, caegory, etc).
```
cd scripts
parallel -j5 "python3 -u matterport_preprocessing.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_mesh
python3 matterport_preprocessing.py save_metadata
```
3. Save metadata for the 3D scenes (e.g link to object mesh files, transformation matrix that produces the scene, etc). Split the scenes into train/val/test based on Matterport3D's suggeted split.
```
parallel -j5 "python3 -u build_scenes_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
python3 -u build_scenes_matterport.py split_train_test_val
```
4. Extract 3D pointclouds from the objects. The pointclouds are later fed into [3D Point Capsule Networks][2] and the latent capsules are used for category prediction.
```
parallel -j5 "python3 -u extract_point_clouds.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_pc
python3 extract_point_clouds.py split_train_test_val
```

## AlignRank and ALignRankOracle
### Scene Graph Construction
First step is to construct the scene graphs:
```
parallel -j5 "python3 -u build_scene_graphs_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
python3 build_scene_graphs_matterport.py split_train_test_val
```
### Test
To run the pretrained models for AlignRankOracle:
```
cd models/LearningBased
python3 run_AlignRank.py --experiment_name AlignRankOracle --with_cat_predictions False
```
To run the pretrained models for AlignRank:
```
cd models/LearningBased
python3 run_gnn_cat_predictions.py
python3 run_AlignRank.py --experiment_name AlignRank --with_cat_predictions True
```
### Train
To train AlignRank and AlignRankOracle from scratch follow the steps below:

1. Train the AlignmentModule
```
cd models/LearningBased
python3 train_AlignmentModule.py 
```
2. Download the trained latent capsules from [here][1] and place them in this directory:
 ```
 3DSSR/data/matterport3d/latent_caps
 ```
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train a [3D Point Capsule Network][2] on the pointclouds extracted in step 3 of data prepration.
 
3. Train GNN for object category prediction (this step can be skipped for AlignRankOracle)
```
cd models/LearningBased
python3 train_gnn.py
```
4. Run the commands that assume pretrained models (described above).


## Evaluations
1. Follow the instructions in [BASELINES.md](BASELINES.md) to run baselines and ablations.

[1]: https://sgp2021.github.io/
[2]: https://github.com/yongheng1991/3D-point-capsule-networks
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
