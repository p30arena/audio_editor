#pip install accelerate>=0.26.0

from dotenv import load_dotenv
load_dotenv()

from crisper.transcribe import transcribe_audio

# audio_file = "audio_2024-11-09_16-32-11.ogg"
audio_file = "audio_2024-11-12_06-41-32.ogg"

result = transcribe_audio(audio_file)

with open('cripser-transcription.txt', 'w', encoding='utf-8') as f:
    f.write(result['text'])
