import os
import json
from glob import glob
from pathlib import Path

dst_folder = './dst/'

def list_result_folders():
    return glob(os.path.join(dst_folder, '*/'))

def get_result_folder(key):
    return os.path.join(dst_folder, key)

def get_metadata_filepath(folder):
    return os.path.join(folder, 'metadata.json')

def get_metadata(folder):
    metadata_filepath = get_metadata_filepath(folder)
    with open(metadata_filepath, 'r', encoding='utf-8') as meta_file:
        metadata = json.load(meta_file)
    return metadata

def store_metadata(folder, metadata):
    metadata_filepath = get_metadata_filepath(folder)
    with open(metadata_filepath, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file)

def get_correction_json_filepath(folder, f):
    return os.path.join(folder, Path(f["filename"]).stem + "-correction.json")

def get_correction_json(folder, f):
    correction_json_filepath = get_correction_json_filepath(folder, f)
    with open(correction_json_filepath, 'r', encoding='utf-8') as correction_json_file:
        correction_json = json.load(correction_json_file)
    return correction_json

def store_correction_json(folder, f, new_correction_data):
    correction_json_filepath = get_correction_json_filepath(folder, f)
    with open(correction_json_filepath, 'w', encoding='utf-8') as correction_json_file:
        json.dump(new_correction_data, correction_json_file)
