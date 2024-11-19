import torch
import json
import whisper
import whisperx
import numpy as np
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter, normalize, compress_dynamic_range
from pydub.scipy_effects import eq
import os
import subprocess
import copy
import tempfile
import shutil
# import noisereduce as nr
# import librosa
# import soundfile as sf
# from speechbrain.pretrained import SepformerSeparation as separator
# import torchaudio

whisper_model = None
align_models = {}

def flatten(xss):
    return [x for xs in xss for x in xs]

# def use_noisereduce(audio_filepath, out_audio):
#     y, sr = librosa.load(audio_filepath, sr=None, mono=False)
#     reduced_noise = nr.reduce_noise(y=y, sr=sr)
#     sf.write(out_audio, reduced_noise, sr)

# def use_noisereduce(audio_filepath, out_audio):
#     # model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement")
#     model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement")
#     est_sources = model.separate_file(path=audio_filepath)
#     torchaudio.save(out_audio, est_sources[:, :, 0].detach().cpu(), 16000)

# afftdn removes part of speech

# def use_noisereduce(input_audio, output_audio):
#     tmp_file = None
#     if input_audio == output_audio:
#         tmp_file = tempfile.NamedTemporaryFile(delete=False)
#         shutil.copyfile(input_audio, tmp_file.name)
#     command = [
#         'ffmpeg',
#         '-i', input_audio if not tmp_file else tmp_file.name,
#         '-af', 'afftdn',
#         '-y',
#         output_audio
#     ]

#     try:
#         print(f'Removing Noise...')
#         subprocess.run(command, check=True)
#         print(f'Noise Removal Complete.')

#         if tmp_file:
#             os.remove(tmp_file.name)

#         return AudioSegment.from_file(output_audio)
#     except subprocess.CalledProcessError as e:
#         print(f'Error during conversion: {e}')
#         raise e

def enhance_audio(audio_filepath, out_audio, out_audio_format):
    # Load the audio file
    audio = AudioSegment.from_file(audio_filepath)

    # High-pass filter at 80 Hz
    audio = high_pass_filter(audio, cutoff=80)

    # Low-pass filter at 8000 Hz
    audio = low_pass_filter(audio, cutoff=8000)

    # Equalizer adjustments
    audio = eq(audio, 300, gain_dB=2.0)  # Slightly boost mid frequencies
    audio = eq(audio, 6000, gain_dB=-2.0)  # Slightly reduce high frequencies

    # Compress dynamic range
    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0,
    )

    # Normalize the audio
    audio = normalize(audio)

    # Export the processed audio
    audio.export(out_audio, out_audio_format)

    return audio

def separate_speech(input_audio, output_audio):
    tmp_file = None
    if input_audio == output_audio:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        shutil.copyfile(input_audio, tmp_file.name)
    command = [
        'ffmpeg',
        '-i', input_audio if not tmp_file else tmp_file.name,
        '-af', 'arnndn=m=cb.rnnn',
        '-y',
        output_audio
    ]

    try:
        print(f'Removing Noise...')
        subprocess.run(command, check=True)
        print(f'Noise Removal Complete.')

        if tmp_file:
            os.remove(tmp_file.name)

        return AudioSegment.from_file(output_audio)
    except subprocess.CalledProcessError as e:
        print(f'Error during conversion: {e}')
        raise e

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

def transcribe(audio_file, out_audio, out_audio_format, do_enhance = False, do_separate_speech = False, do_speedup = False, speedup_gaps = False):
    in_audio = audio_file
    edited_audio = None

    if do_enhance:
        # use_noisereduce(in_audio, out_audio)
        # in_audio = out_audio
        edited_audio = enhance_audio(in_audio, out_audio, out_audio_format)
        in_audio = out_audio

    if do_separate_speech:
        edited_audio = separate_speech(in_audio, out_audio)
        in_audio = out_audio

    result = internal_transcribe(in_audio)

    if do_speedup:
        edited_audio, aligned_segments = speedup(
            result['segments'],
            AudioSegment.from_file(in_audio) if not edited_audio else edited_audio,
            out_audio,
            out_audio_format,
            'speedup',
            1.0,
            True,
        )
        result['segments'] = aligned_segments

    return result

def speedup(segments, audio, out_audio, out_audio_format, gap_handling='keep', gap_max_seconds=1.0, enable_cdr_norm=False):
    """
    Speeds up slow words in the audio and adjusts gaps according to the specified handling.

    Parameters:
    - segments (list): A list of segments, each containing words with 'word', 'start', and 'end' keys.
    - audio (AudioSegment): The original audio loaded using pydub.
    - out_audio (str): The file path to save the modified audio.
    - out_audio_format (str): The format to save the modified audio (e.g., 'wav', 'mp3').
    - gap_handling (str): How to handle gaps between words. Options are 'keep', 'delete', or 'speedup'.
    - gap_max_seconds (float): The factor by which to speed up gaps if gap_handling is 'speedup'.
    - enable_cdr_norm (boolean): Volume Correction using compress dynamic range and normalization.

    Returns:
    - modified_audio (AudioSegment): The modified audio after processing.
    - copy_segments (list): The segments with adjusted timestamps.
    """
    print("Speeding-up audio...")

    gap_max_ms = gap_max_seconds * 1000

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
        if 'start' not in entry:
            raise ValueError("Transcription is Faulty.")

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

            factor = duration_ms / gap_max_ms

            # Handle the gap according to gap_handling parameter
            if gap_handling == 'keep' or factor <= 1.0 or (duration_ms / factor) < 100: #does not support < 0.1f second segments
                # Keep the gap as is
                modified_audio += gap_audio
            elif gap_handling == 'delete':
                # Skip the gap
                gap_modif = -duration_ms
            elif gap_handling == 'speedup':
                # Speed up the gap audio
                gap_audio = gap_audio.speedup(playback_speed=factor, chunk_size=50, crossfade=25)
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
