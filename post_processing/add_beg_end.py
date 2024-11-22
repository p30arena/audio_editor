import os
import subprocess
import tempfile
import shutil

def concat(beg_audio, end_audio, input_audio, output_audio):
    tmp_file = None
    if input_audio == output_audio:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        shutil.copyfile(input_audio, tmp_file.name)
    tmp_input_audio = input_audio if not tmp_file else tmp_file.name
    files_concat = []
    if beg_audio:
        files_concat.append(beg_audio)
    files_concat.append(tmp_input_audio)
    if end_audio:
        files_concat.append(end_audio)
    
    command = [
        'ffmpeg',
        '-i', "concat:" + '|'.join(files_concat),
        '-c', 'copy',
        '-y',
        output_audio
    ]

    try:
        print(f'Concating Files...')
        subprocess.run(command, check=True)
        print(f'Files Concatenation Complete.')

        if tmp_file:
            os.remove(tmp_file.name)
    except subprocess.CalledProcessError as e:
        print(f'Error during conversion: {e}')
        raise e

# concat audio
# edit manifest: text, correction: correction and aligned segments