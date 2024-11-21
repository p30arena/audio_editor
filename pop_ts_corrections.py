# post processing, enhance transcriptions corrections
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ai import correct_transcription_st, summarize_transcription
from post_processing import list_result_folders, get_metadata, store_metadata, get_correction_json, store_correction_json

dst_folder = './dst/'

for folder in list_result_folders():
    print("processing " + folder)
    metadata = get_metadata()
    
    print(" - has " + len(metadata["files"]) + " files")
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
