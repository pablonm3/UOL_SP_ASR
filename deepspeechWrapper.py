# adapted from https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
# which has a Mozzila Public License:
# https://github.com/mozilla/DeepSpeech/blob/master/LICENSE

from deepspeech import Model, version
import librosa as lr
import os
from thinkdsp import read_wave, WavFileWriter, Wave
import numpy as np


MODELS_DIR = "models"


MODELS_PER_LANG = {
    "EN": {"scorer":"/EN/deepspeech-0.9.3-models.scorer", "model":"/EN/deepspeech-0.9.3-models.pbmm"},
    "IT": {"scorer":"/IT/kenlm_it.scorer", "model":"/IT/output_graph_it.pbmm"},
    "ES": {"scorer":"/ES/kenlm_es.scorer", "model":"/ES/output_graph_es.pbmm"}
}




class ASR:
    def __init__(self, lang):
        scorer = MODELS_DIR + MODELS_PER_LANG[lang]["scorer"]
        model = MODELS_DIR + MODELS_PER_LANG[lang]["model"]
        assert os.path.exists(scorer), scorer + "not found. Perhaps you need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
        assert os.path.exists(model), model + "not found. Perhaps you need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
        self.ds = Model(model)
        self.lang = lang
        self.ds.enableExternalScorer(scorer)
        self.desired_sample_rate = self.ds.sampleRate()

    def predict(self, filename):
        assert os.path.exists(filename), filename + "does not exist"
        audio = lr.load(filename, sr=self.desired_sample_rate)[0]
        wave1 = Wave(audio, framerate=self.desired_sample_rate)
        sp = wave1.make_spectrum()
        if(self.lang=="IT"):
            sp.high_pass(50, factor=0)
            sp.low_pass(5000, factor=0)
        if(self.lang=="ES"):
            sp.high_pass(150, factor=0.3)
            sp.low_pass(3500, factor=0.2)
            amps_mean = sp.amps.mean()
            if(amps_mean < 3.1):
                sp.scale(5)
        audio = sp.make_wave().ys
        audio = (audio * 35767).astype(np.int16) # scale from -1 to 1 to +/-32767
        res = self.ds.stt(audio)
        return res
