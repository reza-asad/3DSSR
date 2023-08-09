# 3DSSR: 3D Subscene Retrieval
[[Link to paper]] [[video]]

<img src="https://github.com/reza-asad/3DSSR/blob/master/figures/3DSSROverview.png"/>

We tackle 3DSSR using our self-supervised point cloud encoder PointCrop.

## Dependencies and Python Enviroment
```
virtualenv 3dssr --python=python3.8
source 3dssr/bin/activate
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
4. Extract 3D object-centric mesh regions and repeat with arguments 'test' and 'val' in place of 'train'.
    ```
    parallel -j5 "python3 -u extract_regions.py {1} {2} {3} {4}" ::: train ::: 5 ::: 0 1 2 3 4 ::: extract
    ```
5. Sample point cloud from each extracted 3D mesh region and repeat with arguments 'test' and 'val' in place of 'train'.
    ```
    parallel -j5 "python3 -u extract_point_clouds.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4} --action {5}" ::: train ::: 0 ::: 5 ::: 0 1 2 3 4 ::: extract
    ```

## PointCrop
To train PointCrop from scratch:
```
cd models/LearningBased
python -m torch.distributed.launch --nproc_per_node=4 train_PointCrop.py --local_crops_number 8 --global_crops_number 2 --batch_size_per_gpu 8 --num_workers 20 --results_folder_name
PointCrop --nblocks 3 --transformer_dim 32 --out_dim 2000 
```

[Link to paper]: https://openaccess.thecvf.com/content/CVPR2023W/StruCo3D/papers/Asad_3DSSR_3D_Subscene_Retrieval_CVPRW_2023_paper.pdf 
[video]: https://www.youtube.com/watch?v=jMZFzJnu6Sk
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
