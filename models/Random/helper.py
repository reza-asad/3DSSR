import json


def load_from_json(path, mode='r'):
    with open(path, mode) as f:
        return json.load(f)


def write_to_json(dictionary, path, mode='w', indent=4):
    with open(path, mode) as f:
        json.dump(dictionary, f, indent=indent)
