from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import os
import nltk
import scipy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

nltk.download('punkt')

preload_models(text_use_gpu=True,text_use_small=True,coarse_use_gpu=True,coarse_use_small=True,fine_use_gpu=True,fine_use_small=True,codec_use_gpu=True)


script = """
我有一个信念:"BlockChain is the Future!"
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)
SPEAKER = "v2/zh_speaker_9"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]

# Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
# scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
write_wav("bark_generation_zh_news.wav",rate=SAMPLE_RATE,data=np.concatenate(pieces))


  
