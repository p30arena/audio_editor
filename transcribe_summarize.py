import argparse
import torch
import json
import whisper
import os
from dotenv import load_dotenv
load_dotenv()

from ai import correct_transcription_st, summarize_transcription

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="input audio")
parser.add_argument("-o", "--output", required=True, help="output transcription and summary file")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("turbo", device)

transcription = whisper_model.transcribe(args.input)

text = transcription["text"]
correction_result = correct_transcription_st(text)
corrected_text = correction_result.corrected_transcription
summary = summarize_transcription(corrected_text)

with open(args.output, 'w+', encoding="utf-8") as f:
    json.dump({
        'text': text,
        'corrected_text': corrected_text,
        'summary': summary,
    }, f)