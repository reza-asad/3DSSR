## Baselines
To run each basedline model from scratch follow the instructions bellow:
### GKRank
This is an extention of [Fisher et al.][1]\'s graph kernel-based approach. 

1. Build the scene graphs:
```
parallel -j5 "python3 -u build_scene_graphs_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scene_graphs
python3 -u build_scene_graphs_matterport.py split_train_test_val
```
2. Build voxel representation of the 3D object meshes:
```
parallel -j5 "python3 -u data_processing_voxel.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: derive_zernike_features
python3 data_processing_voxel.py find_nth_closest_model
```

3. Run GKRank on the test queires:
```
```

### CatRank
Run CatRank on test queries:
```
python3 random_run.py --model_name CatRank
```

### RandomRank
```
python3 random_run.py --model_name RandomRank
```

## Ablations
To run each ablation model from scratch follow the instructions bellow:
### AlignRank[-Align]
```
python3 run_AlignRank.py --experiment_name AlignRank[-Align] --with_cat_predictions True --with_alignment False
```
### AlignRank[-GNN]
```
python3 run_AlignRank.py --experiment_name AlignRank[-GNN] --with_cat_predictions True --data-dir../../results/matterport3d/LearningBased/scene_graphs_with_predictions_linear
```

[1]: https://techmatt.github.io/pdfs/graphKernel.pdf
