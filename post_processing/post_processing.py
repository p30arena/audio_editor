import os
import json
from glob import glob
from pathlib import Path

src_folder = './src/'
dst_folder = './dst/'

def list_result_folders():
    return glob(os.path.join(dst_folder, '*/'))

def get_result_folder(key):
    return os.path.join(dst_folder, key)

def get_metadata_filepath(folder):
    return os.path.join(folder, 'metadata.json')

def get_manifest_filepath():
    return os.path.join(src_folder, 'manifest.json')

def get_manifest():
    manifest_filepath = get_manifest_filepath()
    with open(manifest_filepath, 'r', encoding='utf-8') as meta_file:
        manifest = json.load(meta_file)
    return manifest

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

def get_segments_jsonl_filepath(folder, f):
    return os.path.join(folder, Path(f["filename"]).stem + "-segments.jsonl")

def get_aligned_segments_jsonl_filepath(folder, f):
    return os.path.join(folder, Path(f["filename"]).stem + "-aligned-segments.jsonl")

def get_correction_json(folder, f):
    correction_json_filepath = get_correction_json_filepath(folder, f)
    with open(correction_json_filepath, 'r', encoding='utf-8') as correction_json_file:
        correction_json = json.load(correction_json_file)
    return correction_json

def store_correction_json(folder, f, new_correction_data):
    correction_json_filepath = get_correction_json_filepath(folder, f)
    with open(correction_json_filepath, 'w', encoding='utf-8') as correction_json_file:
        json.dump(new_correction_data, correction_json_file)

def get_segments_jsonl(folder, f):
    segments_jsonl_filepath = get_segments_jsonl_filepath(folder, f)
    with open(segments_jsonl_filepath, 'r', encoding='utf-8') as segments_jsonl_file:
        segments_jsonl = []
        for line in segments_jsonl_file.read().strip().split('\n'):
            segments_jsonl.append(json.loads(line))
    return segments_jsonl

def store_segments_jsonl(folder, f, new_segments_jsonl):
    segments_jsonl_filepath = get_segments_jsonl_filepath(folder, f)
    with open(segments_jsonl_filepath, 'w', encoding='utf-8') as segments_jsonl_file:
        for item in new_segments_jsonl:
            line = json.dumps(item)
            segments_jsonl_file.write(line)
            segments_jsonl_file.write('\n')

def get_aligned_segments_jsonl(folder, f):
    aligned_segments_jsonl_filepath = get_aligned_segments_jsonl_filepath(folder, f)
    with open(aligned_segments_jsonl_filepath, 'r', encoding='utf-8') as aligned_segments_jsonl_file:
        aligned_segments_jsonl = []
        for line in aligned_segments_jsonl_file.read().strip().split('\n'):
            aligned_segments_jsonl.append(json.loads(line))
    return aligned_segments_jsonl

def store_aligned_segments_jsonl(folder, f, new_aligned_segments_jsonl):
    aligned_segments_jsonl_filepath = get_aligned_segments_jsonl_filepath(folder, f)
    with open(aligned_segments_jsonl_filepath, 'w', encoding='utf-8') as aligned_segments_jsonl_file:
        for item in new_aligned_segments_jsonl:
            line = json.dumps(item)
            aligned_segments_jsonl_file.write(line)
            aligned_segments_jsonl_file.write('\n')
