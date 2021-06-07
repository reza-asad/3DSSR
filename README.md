# Project

**[Project][1]**  
[Reza Asad][RA], [Manolis Savva][MS]

<table width="500" border="0" cellpadding="5">
<tr>

<td align="center" valign="center">
<img src="https://github.com/reza-asad/reza-asad.github.io/blob/master/_publications/query_top_view.gif" />
<br />
<br />
Query Subscene (bed*, chest-of-drawers, chair, lighting)
</td>


<td align="center" valign="center">
<img src="https://github.com/reza-asad/reza-asad.github.io/blob/master/_publications/rank1_alignment_colored.gif" />
<br />
<br />
Aligning Rank 1 Target Scene
</td>
</tr>
 
<td align="center" valign="center">
<img src="https://github.com/reza-asad/reza-asad.github.io/blob/master/_publications/rank1_final.gif" />
<br />
<br />
Rank 1 Target Subscene
</td>
</tr>

</table>
 
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

## AlignRank and AlignRankOracle
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
python3 run_AlignRank.py --experiment_name AlignRank
```
### Train
To train AlignRank and AlignRankOracle from scratch follow the steps below:

1. To train the AlignmentModule run:
    ```
    cd models/LearningBased
    python3 train_AlignmentModule.py 
    ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The trained models will be saved in ```results/matterport3d/LearningBased/lstm_alignment```.

2. Download the trained latent capsules from [here][1] and place them in this directory:
    ```
    3DSSR/data/matterport3d/latent_caps
    ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or train a [3D Point Capsule Network][2] on the pointclouds extracted in step 3 of data prepration.
 
3. Train GNN for object category prediction (this step can be skipped for AlignRankOracle)
    ```
    cd models/LearningBased
    python3 train_gnn.py
    ```
4. Run the commands that assume pretrained models (described in the **Test** subsection).


## Evaluations
1. To run each baseline and ablation model from scratch follow the instructions in [BASELINES.md](BASELINES.md). Otherwise, this step can be skipped. 
2. To evaluate AlignRank against baselines and ablations run:
    ```
    cd scripts
    python3 evaluator_wrapper.py --mode test --ablations < True, False >
    ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If ```--ablations``` is True, AlignRank is evaluated against the ablations. Otherwise, AlignRank is compared against the baseline models.

3. To plot the evaluated results and compute the Area Under the Curve (AUC) run:
    ```
    python3 prepare_quantitative_results.py --ablations < True, False >
    ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If ```--ablations``` is True, the left plot compares AlignRank against ablations. Otherwise, the right plot contrasts AlignRank against the baselines.


## Rendering Results
To render the top 5 results for the queries presented in the paper run:
```
python3 render_results_wrapper.py --mode test --topk 5 --include_queries '["bed-33", "table-9", "sofa-28"]'
```
To render the top 5 results for all test queries run:
```
python3 render_results_wrapper.py --mode test --topk 5 --include_queries '["all"]'
```

[1]: https://github.com/reza-asad/3DSSR
[2]: https://github.com/yongheng1991/3D-point-capsule-networks
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
