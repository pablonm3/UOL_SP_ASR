import os
from pocketsphinx import Pocketsphinx





class ASR:
    def __init__(self):
        self.ps = Pocketsphinx()

    def predict(self, filename):
        assert os.path.exists(filename), filename + "does not exist"
        self.ps.decode(audio_file=filename)
        hyp = self.ps.hypothesis()
        return hyp





