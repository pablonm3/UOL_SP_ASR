# adapted from https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
# which has a Mozzila Public License:
# https://github.com/mozilla/DeepSpeech/blob/master/LICENSE

from deepspeech import Model, version
import librosa as lr
import numpy as np
import os

MODELS_DIR = "models"


MODELS_PER_LANG = {
    "EN": {"scorer":"/EN/deepspeech-0.9.3-models.scorer", "model":"/EN/deepspeech-0.9.3-models.pbmm"}
}

class ASR:
    def __init__(self, lang):
        scorer = MODELS_DIR + MODELS_PER_LANG[lang]["scorer"]
        model = MODELS_DIR + MODELS_PER_LANG[lang]["model"]
        assert os.path.exists(scorer), scorer + "not found. Perhaps you need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
        assert os.path.exists(model), model + "not found. Perhaps you need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
        self.ds = Model(model)
        self.ds.enableExternalScorer(scorer)
        self.desired_sample_rate = self.ds.sampleRate()

    def predict(self, filename):
        assert os.path.exists(filename), filename + "does not exist"
        audio = lr.load(filename, sr=self.desired_sample_rate)[0]
        audio = (audio * 32767).astype(np.int16) # scale from -1 to 1 to +/-32767
        res = self.ds.stt(audio)
        #res = ds.sttWithMetadata(audio, 1).transcripts[0]
        return res
