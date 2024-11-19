import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="input json file")
parser.add_argument("-b", "--filesbasepath", required=True, help="base path of audio_file")
parser.add_argument("-o", "--output", required=True, help="output manifest file, the file must exist")
args = parser.parse_args()

with open(args.input, 'r', encoding='utf-8') as in_file:
    items = json.load(in_file)

mapped_items = []

for it in items:
    title = it['title'] #str
    filename = it['audio_file'] #list

    for i, f in enumerate(filename):
        #make paths absolute
        filename[i] = os.path.join(args.filesbasepath, f)

    mapped_items.append({
        'title': title,
        'filename': filename,
    })

with open(args.output, 'r+', encoding='utf-8') as f:
    try:
        o = json.load(f)
    except Exception as e:
        print(e)
        o = {}
    
    o['items'] = mapped_items

    f.seek(0)
    json.dump(o, f)
    f.truncate()