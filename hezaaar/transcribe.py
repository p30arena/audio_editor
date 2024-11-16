from hezar.models import Model

# audio_file = "audio_2024-11-09_16-32-11.ogg"
audio_file = "audio_2024-11-12_06-41-32.ogg"

whisper = Model.load("hezarai/whisper-small-fa")
transcripts = whisper.predict(audio_file)
with open('hezar-transcription.txt', 'w', encoding='utf-8') as f:
    f.write(transcripts[0]['text'])
