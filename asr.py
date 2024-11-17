import torch
import json
import whisper
import whisperx
import webrtcvad
import wave
import contextlib
import numpy as np
from pydub import AudioSegment
import torch
import os
import librosa
import soundfile as sf
import subprocess
import copy

whisper_model = None
align_models = {}

def flatten(xss):
    return [x for xs in xss for x in xs]

def noise_removal(input_audio, output_audio):
    command = [
        'ffmpeg',
        '-i', input_audio,
        '-af', 'arnndn=m=cb.rnnn',
        output_audio
    ]

    try:
        print(f'Removing Noise...')
        subprocess.run(command, check=True)
        print(f'Noise Removal Complete.')
    except subprocess.CalledProcessError as e:
        print(f'Error during conversion: {e}')
        raise e

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

def edit_vad(audio_file, out_audio, out_audio_format):
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

    return edited_audio

def avg_spoken_words_duration(words, min_word_length = 3):
    # Collect durations of words with min_word_length
    durations = []
    for entry in words:
        word = entry['word']
        if len(word.strip()) > min_word_length:
            start = entry['start']
            end = entry['end']
            duration = end - start
            durations.append(duration)

    # Compute average duration
    if durations:
        avg_duration = sum(durations) / len(durations)
    else:
        avg_duration = 0
    
    return avg_duration

def internal_transcribe(audio):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global whisper_model
    if not whisper_model:
        whisper_model = whisper.load_model("turbo", device)

    print("Transcribing audio...")
    result = whisper_model.transcribe(
        audio,
        word_timestamps=True,
    )
    global align_models
    if result["language"] not in align_models:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        align_models[result["language"]] = (model_a, metadata)
    model_a, metadata = align_models[result["language"]]
    aligned_segments = whisperx.align(result["segments"], model_a, metadata, whisperx.load_audio(audio), device, return_char_alignments=False)
    result['segments'] = aligned_segments["segments"]
    print("Transcription complete.")

    return result

def transcribe(audio_file, out_audio, out_audio_format, do_noise_removal = False, do_speedup = False):
    in_audio = audio_file
    
    if do_noise_removal:
        noise_removal(in_audio, out_audio)
        in_audio = out_audio

    edited_audio = edit_vad(in_audio, out_audio, out_audio_format)

    result = internal_transcribe(out_audio)

    if do_speedup:
        edited_audio, aligned_segments = speedup(result['segments'], edited_audio, out_audio, out_audio_format, 'speedup', 4.0)
        result['segments'] = aligned_segments
        # result = internal_transcribe(out_audio)

    return result

def speedup(segments, audio, out_audio, out_audio_format, gap_handling='keep', gap_speedup_factor=2.0):
    """
    Speeds up slow words in the audio and adjusts gaps according to the specified handling.

    Parameters:
    - segments (list): A list of segments, each containing words with 'word', 'start', and 'end' keys.
    - audio (AudioSegment): The original audio loaded using pydub.
    - out_audio (str): The file path to save the modified audio.
    - out_audio_format (str): The format to save the modified audio (e.g., 'wav', 'mp3').
    - gap_handling (str): How to handle gaps between words. Options are 'keep', 'delete', or 'speedup'.
    - gap_speedup_factor (float): The factor by which to speed up gaps if gap_handling is 'speedup'.

    Returns:
    - modified_audio (AudioSegment): The modified audio after processing.
    - copy_segments (list): The segments with adjusted timestamps.
    """
    print("Speeding-up audio...")

    min_word_length = 3
    words = flatten([s['words'] for s in segments])
    avg_duration = avg_spoken_words_duration(words)

    if avg_duration == 0:
        return None

    # Initialize variables
    modified_audio = AudioSegment.empty()
    last_end_time = 0

    modifications = {}

    for i, entry in enumerate(words):
        word = entry['word']
        start = entry['start'] * 1000  # Convert to milliseconds
        end = entry['end'] * 1000      # Convert to milliseconds
        duration_ms = end - start
        duration = duration_ms / 1000  # Duration in seconds

        gap_modif = 0
        word_modif = 0

        # Extract the audio segment for this word
        word_audio = audio[start:end]

        # Check if the word needs to be sped up
        if len(word.strip()) > min_word_length and (duration - avg_duration) >= 0.5:
            # Calculate the playback speed factor
            speed_factor = avg_duration / duration
            playback_speed = min(1 / speed_factor, 1.5)

            # Speed up the word audio
            word_audio = word_audio.speedup(playback_speed=playback_speed, chunk_size=50, crossfade=25)
            word_modif = -(duration_ms - len(word_audio))

        # # Handle any silence or gaps between words
        # if start > last_end_time:
        #     gap = audio[last_end_time:start]
        #     modified_audio += gap

        # Handle any gap before this word
        if start > last_end_time:
            gap_start = last_end_time
            gap_end = start
            gap_audio = audio[gap_start:gap_end]
            duration_ms = gap_end - gap_start

            # Handle the gap according to gap_handling parameter
            if gap_handling == 'keep' or (duration_ms / gap_speedup_factor) < 100: #does not support < 0.1f second segments
                # Keep the gap as is
                modified_audio += gap_audio
            elif gap_handling == 'delete':
                # Skip the gap
                gap_modif = -duration_ms
            elif gap_handling == 'speedup':
                # Speed up the gap audio
                gap_audio = gap_audio.speedup(playback_speed=gap_speedup_factor, chunk_size=50, crossfade=25)
                modified_audio += gap_audio
                gap_modif = -(duration_ms - len(gap_audio))
            else:
                raise ValueError(f"Invalid gap_handling value: {gap_handling}. Choose 'keep', 'delete', or 'speedup'.")

        # Append the (modified) word audio
        modified_audio += word_audio
        last_end_time = end

        if gap_modif != 0 or word_modif != 0:
            modifications[i] = {
                'index': i,
                'gap_modif': gap_modif / 1000,
                'word_modif': word_modif / 1000,
            }

    # Append any remaining audio after the last word
    if last_end_time < len(audio):
        modified_audio += audio[last_end_time:]

    # Export the modified audio
    modified_audio.export(out_audio, format=out_audio_format)

    # print(modifications)

    copy_segments = copy.deepcopy(segments)
    words_idx = 0
    cumulative_modif = 0
    # n_mods = 0
    
    for s in copy_segments:
        for w in s['words']:
            if cumulative_modif != 0:
                w['start'] += cumulative_modif
                w['end'] += cumulative_modif

            if words_idx in modifications:
                # n_mods += 1
                gap_modif = modifications[words_idx]['gap_modif']
                word_modif = modifications[words_idx]['word_modif']

                if gap_modif != 0:
                    w['start'] += gap_modif
                    w['end'] += gap_modif
                    cumulative_modif += gap_modif
                if word_modif != 0:
                    w['end'] += word_modif
                    cumulative_modif += word_modif                
            words_idx += 1
        s['start'] = s['words'][0]['start']
        s['end'] = s['words'][len(s['words']) - 1]['end']

    # print(n_mods)
    # print(len(modifications.keys()))
    print("Speed-up complete.")

    return modified_audio, copy_segments
