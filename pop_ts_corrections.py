# post processing, enhance transcriptions corrections
import os
import json
from glob import glob
from dotenv import load_dotenv

load_dotenv()

from ai import correct_transcription_st, summarize_transcription

dst_folder = './dst/'

for folder in glob(os.path.join(dst_folder, '*/')):
    metadata_filepath = os.path.join(folder, 'metadata.json')
    if not os.path.isfile(metadata_filepath):
        print('metadata not found')
        continue
    
    with open(metadata_filepath, 'r', encoding='utf-8') as meta_file:
        metadata = json.load(meta_file)
    
    for f in metadata["files"]:
        try:
            correction_json_filepath = os.path.join(folder, f["filename"] + "-correction.json")
            with open(correction_json_filepath, 'r', encoding='utf-8') as correction_json_file:
                correction_json = json.load(correction_json_file)
            
            transcription_text = correction_json["text"]
            correction_result = correct_transcription_st(text)
            corrected_text = correction_result.corrected_transcription
            summary = summarize_transcription(corrected_text)

            new_correction_data = correction_result.model_dump()
            new_correction_data["prev_corrected_transcription"] = correction_json["correction"]["corrected_transcription"]

            with open(correction_json_filepath, 'w', encoding='utf-8') as correction_json_file:
                json.dump(new_correction_data, correction_json_file)
            
            f["text"] = corrected_text
            f["summary"] = summary
        except Exception as e:
            print(e)
    
    with open(metadata_filepath, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file)

