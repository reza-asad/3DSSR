# 3DSSR
TODO: Link to paper
[Reza Asad][RA], [Manolis Savva][MS]

<img src="https://github.com/reza-asad/3DSSR/blob/master/figures/3DSSROverview.png"/>

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
4. Extract 3D pointclouds from the objects. The pointclouds are later fed into [3D Point Capsule Networks][2] and the latent capsules are used for category prediction. This step can be skipped if you plan to use the latent capsules trained by us.
    ```
    parallel -j5 "python3 -u extract_point_clouds.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_pc
    python3 extract_point_clouds.py split_train_test_val
    ```

## PointCropRank
To train PointCrop from scratch follow the steps below:

1. To train the Alignment module run:
    ```
    cd models/LearningBased
    python3 train_AlignmentModule.py 
    ```
    The trained models will be saved in ```results/matterport3d/LearningBased/lstm_alignment```.


[1]: https://github.com/reza-asad/3DSSR
[2]: https://github.com/yongheng1991/3D-point-capsule-networks
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
