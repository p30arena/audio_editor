import argparse
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def transcribe_audio(file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(device)
    print(torch_dtype)

    model_id = "nyrahealth/CrisperWhisper"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, language="fa")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        # return_timestamps="True",
        # batch_size=4,
        # return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(file_path, generate_kwargs={"language": "fa"})
    return result
