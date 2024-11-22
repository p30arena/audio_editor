import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # for importing ai

from ai import correct_transcription_st, summarize_transcription
from align import align_corrected_transcription
from post_processing import list_result_folders, get_result_folder
from post_processing import get_metadata, store_metadata, get_correction_json, store_correction_json
from post_processing import get_correction_json_filepath, get_segments_jsonl_filepath, get_aligned_segments_jsonl_filepath
from post_processing import get_correction_json_filepath, get_segments_jsonl_filepath, get_aligned_segments_jsonl_filepath
from post_processing import get_segments_jsonl, get_aligned_segments_jsonl, store_segments_jsonl, store_aligned_segments_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", required=False, help="folder key")
args = parser.parse_args()

result_folders = []

if args.key:
    result_folders = [get_result_folder(args.key)]
else:
    result_folders = list_result_folders()

for folder in result_folders:
    print("processing " + folder)
    metadata = get_metadata(folder)
    
    print(f" - has {len(metadata['files'])} files")
    for f in metadata["files"]:
        print(" - processing " + f["filename"])
        try:
            segments_jsonl = get_segments_jsonl(folder, f)
            
            transcription_text = f["text"]

            aligned_segments = align_corrected_transcription(segments_jsonl, transcription_text)

            store_aligned_segments_jsonl(folder, f, aligned_segments)
        except Exception as e:
            print(e)
    
    store_metadata(folder, metadata)
