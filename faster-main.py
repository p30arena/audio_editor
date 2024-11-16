from faster_whisper import WhisperModel

# audio_file = "audio_2024-11-09_16-32-11.ogg"
audio_file = "audio_2024-11-12_06-41-32.ogg"

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe(audio_file, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

with open('faster-transcribe.txt', 'w', encoding='utf-8') as f:
    for segment in segments:
        print(segment.end)
        f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
