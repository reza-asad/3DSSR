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
1. Dowload the dataset and place it in this directory:
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
4. Extract 3D object-centric mesh regions.
    ```
    parallel -j5 "python3 -u extract_regions.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract
    ```
5. Sample a point cloud from each extracted 3D mesh region.
    ```
    parallel -j5 "python3 -u extract_point_clouds.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: test ::: 0 ::: 5 ::: 0 1 2 3 4
    ```

## PointCropRank
To train PointCrop from scratch follow the steps below:

1. To train the Alignment module run:
    ```
    cd models/LearningBased
    python -m torch.distributed.launch --nproc_per_node=4 train_PointCrop.py --local_crops_number 8 --global_crops_number 2 --batch_size_per_gpu 4 --num_workers 20 --results_folder_name
    PointCrop --saveckp_freq 1 --nblocks 3 --transformer_dim 256 --out_dim 2000 
    ```

[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
