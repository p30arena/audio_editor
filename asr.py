import torch
import json
import whisper
import webrtcvad
import wave
import contextlib
import numpy as np
from pydub import AudioSegment
import torch
import os
import librosa
import soundfile as sf

whisper_model = None

def vad_segments(audio_file, aggressiveness=3):
    vad = webrtcvad.Vad(aggressiveness)
    audio, sample_rate = librosa.load(audio_file, sr=16000)
    max_offset = len(audio)
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000)
    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=frame_size).T
    segments = []
    offset = 0
    for frame in frames:
        is_speech = vad.is_speech((frame * 32767).astype(np.int16).tobytes(), sample_rate)
        if not is_speech:
            time_start = offset / sample_rate
            time_end = (offset + frame_size) / sample_rate
            segments.append((time_start, time_end))
        offset += frame_size
    return segments

def transcribe(audio_file, out_audio, out_audio_format):
    print("Detecting non-speech segments...")
    non_speech_segments = vad_segments(audio_file)
    print("Detection complete.")

    aggregated_segments = []
    if non_speech_segments:
        # Initialize the first segment
        start_time = non_speech_segments[0][0]
        end_time = non_speech_segments[0][1]

        for current_start, current_end in non_speech_segments[1:]:
            if current_start <= end_time + 0.001:  # Allow small overlap/tolerance
                # Frames are consecutive or overlapping; extend the segment
                end_time = current_end
            else:
                # Gap between frames; save the current segment and start a new one
                aggregated_segments.append((start_time, end_time))
                start_time = current_start
                end_time = current_end
        # Add the last segment
        aggregated_segments.append((start_time, end_time))

    non_speech_segments = []
    for start_time, end_time in aggregated_segments:
        duration = end_time - start_time
        if duration >= 1.0:
            non_speech_segments.append({
                'start': start_time,
                'end': end_time,
                'text': '[Non-speech sound]',
                'type': 'NonSpeech'
            })

    audio = AudioSegment.from_file(audio_file)
    segments_to_keep = []
    last_end = 0

    for section in non_speech_segments:
        start_ms = int(section['start'] * 1000)
        end_ms = int(section['end'] * 1000)
        if start_ms > last_end:
            segments_to_keep.append(audio[last_end:start_ms])
        last_end = end_ms

    if last_end < len(audio):
        segments_to_keep.append(audio[last_end:])

    edited_audio = sum(segments_to_keep)
    edited_audio.export(out_audio, format=out_audio_format)
    print(f"Audio editing complete. Edited file saved as '{out_audio}'.")

    global whisper_model
    if not whisper_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("turbo", device)

    print("Transcribing audio...")
    result = whisper_model.transcribe(
        out_audio,
        word_timestamps=True,
    )
    print("Transcription complete.")

    return result
