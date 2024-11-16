import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from asr import transcribe
from ai import correct_transcription_st, summarize_transcription, gen_cover_image
from align import align_corrected_transcription

src_folder = './src/'
dst_folder = './dst/'

with open(os.path.join(src_folder, 'manifest.json'), 'r', encoding='utf-8') as f:
    manifest = json.load(f)

for item in manifest:
    title = item.get('title')
    filename = item.get('filename')
    if not title or not filename:
        print("Item missing 'title' or 'filename'. Skipping item.")
        continue

    src = os.path.join(src_folder, filename)
    base_filename = Path(filename).stem
    out_folder = os.path.join(dst_folder, title.replace(' ', '_'))  # Replace spaces with underscores
    out_audio_basename = base_filename + '.mp3'
    out_audio = os.path.join(out_folder, out_audio_basename)
    out_corrected_text = os.path.join(out_folder, base_filename + '-correction.json')
    out_transcription = os.path.join(out_folder, base_filename + '-segments.jsonl')
    out_aligned_transcription = os.path.join(out_folder, base_filename + '-aligned-segments.jsonl')

    # Check if the destination folder already exists
    if os.path.exists(out_folder):
        print(f"Destination folder '{out_folder}' already exists. Skipping item.")
        continue

    # Create the destination folder
    os.makedirs(out_folder)

    # Call the transcription function
    transcription = transcribe(src, out_audio, 'mp3')
    text = transcription["text"]
    segments = transcription["segments"]

    correction_result = correct_transcription_st(text)
    corrected_text = correction_result.corrected_transcription
    summary = summarize_transcription(corrected_text)

    aligned_segments = align_corrected_transcription(segments, corrected_text)

    with open(out_corrected_text, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'text': text, 'correction': correction_result.model_dump()}))

    cover_image_path = gen_cover_image(out_folder, title, summary)

    with open(out_transcription, 'w', encoding='utf-8') as f:
        for s in segments:
            f.write(json.dumps(s))
            f.write('\n')
    
    with open(out_aligned_transcription, 'w', encoding='utf-8') as f:
        for s in aligned_segments:
            f.write(json.dumps(s))
            f.write('\n')
    
    has_salam = corrected_text[:100].find('سلام') > -1

    # Prepare the metadata
    metadata = {
        'title': title,
        'filename': out_audio_basename,
        'has_salam': has_salam,
        'cover': 'cover.png' if cover_image_path else None,
        'text': corrected_text,
        'summary': summary,
    }

    # Save the metadata to a JSON file
    metadata_path = os.path.join(out_folder, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=4)

    print(f"Processed item '{title}' and saved output to '{out_folder}'")
