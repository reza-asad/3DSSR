# 3DSSR

**[3DSSR: 3D SubScene Retrieval][1]**  
[Reza Asad][RA], [Manolis Savva][MS], SGP 2021


## Installing Dependencies
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Preparing Matterport3D
1. Extracting 3D object instances
```
parallel -j5 "python3 -u matterport_preprocessing.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_mesh
```
2. Extracting scene metadata from Matterport3D
```

```


[1]: https://sgp2021.github.io/
[RA]: https://reza-asad.github.io/
[MS]: https://msavva.github.io/
