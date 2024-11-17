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

def flatten(xss):
    return [x for xs in xss for x in xs]

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
    global whisper_model
    if not whisper_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("turbo", device)

    print("Transcribing audio...")
    result = whisper_model.transcribe(
        audio,
        word_timestamps=True,
    )
    print("Transcription complete.")

    return result

def transcribe(audio_file, out_audio, out_audio_format, do_speedup = False):
    edited_audio = edit_vad(audio_file, out_audio, out_audio_format)

    result = internal_transcribe(out_audio)

    if do_speedup:
        edited_audio = speedup(result['segments'], edited_audio, out_audio, out_audio_format, 'speedup', 4.0)
        result = internal_transcribe(out_audio)

    return result

def speedup(segments, audio, out_audio, out_audio_format, gap_handling='keep', gap_speedup_factor=2.0):
    print("Speeding-up audio...")

    min_word_length = 3
    words = flatten([s['words'] for s in segments])
    avg_duration = avg_spoken_words_duration(words)

    if avg_duration == 0:
        return None

    # Initialize variables
    modified_audio = AudioSegment.empty()
    last_end_time = 0

    for entry in words:
        word = entry['word']
        start = entry['start'] * 1000  # Convert to milliseconds
        end = entry['end'] * 1000      # Convert to milliseconds
        duration = (end - start) / 1000  # Duration in seconds

        # Extract the audio segment for this word
        word_audio = audio[start:end]

        # Check if the word needs to be sped up
        if len(word.strip()) > min_word_length and (duration - avg_duration) >= 0.5:
            # Calculate the playback speed factor
            speed_factor = avg_duration / duration
            playback_speed = min(1 / speed_factor, 1.5)

            # Speed up the word audio
            word_audio = word_audio.speedup(playback_speed=playback_speed, chunk_size=50, crossfade=25)

        # # Handle any silence or gaps between words
        # if start > last_end_time:
        #     gap = audio[last_end_time:start]
        #     modified_audio += gap

        # Handle any gap before this word
        if start > last_end_time:
            gap_start = last_end_time
            gap_end = start
            gap_audio = audio[gap_start:gap_end]

            # Handle the gap according to gap_handling parameter
            if gap_handling == 'keep':
                # Keep the gap as is
                modified_audio += gap_audio
            elif gap_handling == 'delete':
                # Skip the gap
                pass  # Do not add the gap to modified_audio
            elif gap_handling == 'speedup':
                # Speed up the gap audio
                gap_audio = gap_audio.speedup(playback_speed=gap_speedup_factor)
                modified_audio += gap_audio
            else:
                raise ValueError(f"Invalid gap_handling value: {gap_handling}. Choose 'keep', 'delete', or 'speedup'.")

        # Append the (modified) word audio
        modified_audio += word_audio
        last_end_time = end

    # Append any remaining audio after the last word
    if last_end_time < len(audio):
        modified_audio += audio[last_end_time:]

    # Export the modified audio
    modified_audio.export(out_audio, format=out_audio_format)

    print("Speed-up complete.")

    return modified_audio
