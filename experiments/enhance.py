#ffmpeg -i audio_2024-11-12_06-41-32.ogg -af arnndn=m=cb.rnnn ff-test.ogg
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

# audio_file = "audio_2024-11-09_16-32-11.ogg"
audio_file = "audio_2024-11-12_06-41-32.ogg"

model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", run_opts={
    'device': 'cuda'
})

# for custom file, change path
est_sources = model.separate_file(path=audio_file)

for i in range(est_sources.shape[2]):
    torchaudio.save("source%shat.ogg" % (i), est_sources[:, :, i].detach().cpu(), 8000)
