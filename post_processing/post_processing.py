import os
import json
from glob import glob
from pathlib import Path
import difflib
import re
from Levenshtein import distance as levenshtein_distance  # Requires python-Levenshtein library

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

def find_closest_match(query_string, search_string, m_coeff = 0.5):
    matcher = difflib.SequenceMatcher(None, search_string, query_string)
    match = matcher.find_longest_match(0, len(search_string), 0, len(query_string))

    if match.size == 0 or match.size < len(query_string) * m_coeff:
        return None  # No match found

    start_index, end_index = match.a, match.a + match.size
    return search_string[start_index:end_index]

def replace_arabic_with_farsi(text):
    """Replaces common Arabic characters with their Farsi equivalents.

    Args:
        text: The input text string.

    Returns:
        The text with replaced characters.
    """

    replacements = {
        "ي": "ی",
        "ك": "ک",
        # Add more replacements as needed
    }

    for arabic, farsi in replacements.items():
        text = re.sub(re.escape(arabic), farsi, text)

    return text

def find_closest_substrings(query_string, search_string, max_distance, prefix=False):
    len_query = len(query_string)
    len_search = len(search_string)
    min_distance = None
    closest_substring = None
    closest_index = None

    for i in range(len_search - len_query + 1):
        substring = search_string[i:i + len_query + 1]
        distance = levenshtein_distance(query_string, substring)
        if prefix:
            if substring[0] != query_string[0]:
                continue
        if min_distance is None or distance < min_distance:
            min_distance = distance
            closest_substring = substring
            closest_index = i
            # Since we want the first minimal, we don't update if distance is equal
    return closest_substring