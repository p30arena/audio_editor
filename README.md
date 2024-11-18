### Environment Variables

Create an .env file

```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

### Installing Dependencies

Using MiniConda with Python version: 3.11.10

The **requirements.txt** file is generated using `conda list -e > requirements.txt`.

```
conda activate your_env_name
pip install -r requirements.txt
```

For whisperx:

https://github.com/SYSTRAN/faster-whisper/issues/516
```
pip install nvidia-pyindex
pip install nvidia-cudnn
python3 -m pip install nvidia-cudnn-cu11==8.9.6.50
export LD_LIBRARY_PATH=/home/ali/miniconda3/envs/nn/lib/python3.11/site-packages/nvidia/cudnn/lib
```

### Manifest

Create manifest.json

```
{
    "do_enhance": true, // remove noise, ajdust volume
    "do_separate_speech": true, // only keep spoken words using ffmpeg arnndn
    "do_speedup": true,
    "speedup_gaps": true,
    "gap_max_seconds": 1.0,
    "items": [
        {
            "title": "title/topic for audio file",
            "filename": "audio file name"
        }
    ]
}
```