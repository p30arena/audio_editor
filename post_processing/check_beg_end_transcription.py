import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # for importing ai

from ai import correct_transcription_st, summarize_transcription
from post_processing import list_result_folders, get_result_folder
from post_processing import get_metadata, store_metadata, get_correction_json, store_correction_json
from post_processing import get_correction_json_filepath, get_segments_jsonl_filepath, get_aligned_segments_jsonl_filepath
from post_processing import get_correction_json_filepath, get_segments_jsonl_filepath, get_aligned_segments_jsonl_filepath
from post_processing import get_segments_jsonl, get_aligned_segments_jsonl, store_segments_jsonl, store_aligned_segments_jsonl
from post_processing import get_manifest

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", required=False, help="folder key")
args = parser.parse_args()

result_folders = []

if args.key:
    result_folders = [get_result_folder(args.key)]
else:
    result_folders = list_result_folders()

manifest = get_manifest()

cnt_no_beg = 0
cnt_no_end = 0
cnt_new_ok_beg = 0
cnt_new_ok_end = 0

for folder in result_folders:
    print("processing " + folder)
    metadata = get_metadata(folder)
    
    len_files = len(metadata['files'])
    print(f" - has {len_files} files")
    for i, f in enumerate(metadata["files"]):
        print(" - processing " + f["filename"])
        try:
            transcription_text = f["text"]
            
            if i == 0:
                if not f["does_begin_with"]:
                    does_begin_with = transcription_text[:100].find(manifest['does_begin_with']) > -1
                    print("does_begin_with", transcription_text[:100])
                    print("now: ", str(does_begin_with))
                    cnt_no_beg += 1
                    if does_begin_with:
                        cnt_new_ok_beg += 1
                        f["does_begin_with"] = True
            if i == len_files - 1:
                if not f["does_end_with"]:
                    does_end_with = transcription_text[-100:].find(manifest['does_end_with']) > -1
                    print("does_end_with", transcription_text[-100:])
                    print("now: ", str(does_end_with))
                    cnt_no_end += 1
                    if does_end_with:
                        cnt_new_ok_end += 1
                        f["does_end_with"] = True
        except Exception as e:
            print(e)
    
    print('cnt_no_beg', cnt_no_beg)
    print('cnt_no_end', cnt_no_end)
    print('cnt_new_ok_beg', cnt_new_ok_beg)
    print('cnt_new_ok_end', cnt_new_ok_end)
    
    store_metadata(folder, metadata)
