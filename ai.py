import os
import json
import requests
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

#################
#   Functions   #
#################

def gen_cover_image(dest_folder, title, summary):
    try:
        print('Generating image...')

        prompt = f"""
Create a realistic, cover image for podcast, representing the following Title and Summary.
It should not contain any text, labels, borders, measurements nor design elements of any kind.
The image should be suitable for digital printing without any instructional or guiding elements.

\nTitle: {title}\n\nSummary: {summary}\n
"""
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="hd",
            response_format="url"
        )
        image_url = response.data[0].url

        # Download the image
        image_data = requests.get(image_url).content
        cover_image_path = os.path.join(dest_folder, 'cover.png')
        with open(cover_image_path, 'wb') as img_file:
            img_file.write(image_data)
        
        print('Generating image complete.')
    except Exception as e:
        print(e)
        cover_image_path = None  # Handle the case where image generation fails
    
    return cover_image_path

def summarize_transcription(text):
    try:
        print('Summarizing text...')

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
You are provided by a transcription of an audio.
Summarize it so it fits a line or two.
                """},
                {"role": "user", "content": text},
            ],
        )

        event = completion.choices[0].message.content

        print('Summarization complete.')

        return event
    except Exception as e:
        print(e)

def correct_transcription(text):
    try:
        print('Fixing transcription...')

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
You are given a transcription of an audio recording.
Correct any misspelled or inaccurately transcribed words while preserving the original tone, tenses, and formality/informality of the text.
Do not remove or alter words unless a clear correction is available.
                """},
                {"role": "user", "content": text},
            ],
        )

        event = completion.choices[0].message.content

        print('Fixing of transcription complete.')

        return event
    except Exception as e:
        print(e)

def correct_transcription_st(text):
    class CorrectionWord(BaseModel):
        original_word: str
        correction_word: str
    
    class Correction(BaseModel):
        corrected_transcription: str
        words: list[CorrectionWord]
    
    try:
        print('Fixing transcription...')

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
You are given a transcription of an audio recording.
Correct any misspelled or inaccurately transcribed words while preserving the original tone, tenses, and formality/informality of the text.
Do not remove or alter words unless a clear correction is available.
                """},
                {"role": "user", "content": text},
            ],
            response_format=Correction,
        )

        event = completion.choices[0].message.parsed

        print('Fixing of transcription complete.')

        return event
    except Exception as e:
        print(e)

def inspect_correction_words(allwords):
    class CorrectionWord(BaseModel):
        text_idx: int
        word_idx: int
        word: str
        is_correction: bool
        probability: float

    class TaskCorrection(BaseModel):
        words: list[CorrectionWord]
    
    try:
        print('Detecting correction words...')

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
You are provided by text indexes and words of an audio oredered by their timestamps.
Set is_correction for the words that are spoken for the purpose of correction to their previous word.
Set probability for the level of confidence of is_correction.
                """},
                {"role": "user", "content": json.dumps(allwords)},
            ],
            response_format=TaskCorrection,
        )

        event = completion.choices[0].message.parsed

        print('Detection of correction words complete.')

        return event
    except Exception as e:
        print(e)