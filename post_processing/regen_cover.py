import argparse
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # for importing ai

from ai import gen_cover_image
from post_processing import get_result_folder, get_metadata, store_metadata, get_correction_json, store_correction_json

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", required=True, help="folder key")
args = parser.parse_args()

folder = get_result_folder(args.key)

print("processing " + folder)
metadata = get_metadata(folder)

print(f" - has {len(metadata['files'])} files")
if len(metadata["files"]) > 0:
    print(" - choosing the first file for re-generation of cover image")
    f = metadata["files"][0]
    print(" - processing " + f["filename"])
    try:
        cover_image_path = gen_cover_image(folder, metadata["title"], f["summary"])
        metadata["cover"] = cover_image_path
    except Exception as e:
        print(e)

store_metadata(folder, metadata)
