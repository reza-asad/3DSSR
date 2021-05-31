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
1. Extract 3D object instances and save the metadata for objects (e.g oriented bounding boxes, caegory, etc):
```
cd scripts
parallel -j5 "python3 -u matterport_preprocessing.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_mesh
python3 matterport_preprocessing.py save_metadata
```
2. Save metadata for the 3D scenes (e.g link to object mesh files, transformation matrix that produces the scene, etc):
```

```


[1]: https://sgp2021.github.io/
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
