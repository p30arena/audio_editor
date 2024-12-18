import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # for importing ai

from ai import correct_transcription_st, summarize_transcription
from post_processing import list_result_folders, get_result_folder, get_metadata, store_metadata, get_correction_json, store_correction_json

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
            correction_json = get_correction_json(folder, f)
            
            transcription_text = correction_json["text"]
            correction_result = correct_transcription_st(transcription_text)
            corrected_text = correction_result.corrected_transcription
            summary = summarize_transcription(corrected_text)

            new_correction_data = correction_result.model_dump()
            new_correction_data["prev_corrected_transcription"] = correction_json["correction"]["corrected_transcription"]

            store_correction_json(folder, f, new_correction_data)
            
            f["text"] = corrected_text
            f["summary"] = summary
        except Exception as e:
            print(e)
    
    store_metadata(folder, metadata)
