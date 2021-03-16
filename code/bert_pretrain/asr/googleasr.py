
'''
code from: https://github.com/googleapis/python-speech/blob/master/samples/snippets/quickstart.py
export GOOGLE_APPLICATION_CREDENTIALS=./googleasr.json
On mac, on the leo cant access to google cloud.
命令行是可以work的:
export GOOGLE_APPLICATION_CREDENTIALS=/Users/jinming/Desktop/works/tools/google/googleasr.json

gcloud ml speech recognize gs://cloud-samples-tests/speech/brooklyn.flac \
 --language-code=en-US

总是报错：安装不到最新的2.0版本
pip install --upgrade google-cloud-speech
https://github.com/googleapis/python-speech/issues/66

pip install virtualenv
virtualenv newgooglecloud
source newgooglecloud/bin/activate
newgooglecloud/bin/pip install google-cloud-speech
'''

from google.cloud import speech
import io
import os

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    client = speech.SpeechClient()
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    texts = []
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        texts.append(result.alternatives[0].transcript)
    return ' '.join(texts)
    
if __name__ == '__main__':
    # speech_file = '/Users/jinming/Downloads/005730814.wav'
    save_path = '/Users/jinming/Desktop/works/afew_wavs/val_wav2text.txt'
    wav_dir = '/Users/jinming/Desktop/works/afew_wavs/val'
    wav_names = os.listdir(wav_dir)
    all_texts = []
    for wav_name in wav_names:
        wav_path = os.path.join(wav_dir, wav_name)
        text = transcribe_file(wav_path)
        all_texts.append(wav_name + '\t' +  text + '\n')
    with open(save_path, 'w') as f:
        f.writelines(all_texts)