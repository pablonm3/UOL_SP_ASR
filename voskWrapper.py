import os
import queue
import vosk
import sys
import json
import wave

#model_dir = "fr-40m"
model_dir_sm = "models/EN/vosk/vosk-model-small-en-us-0.15"
model_dir_md= "models/EN/vosk/vosk-model-en-us-0.22"


if not os.path.exists(model_dir_sm):
    print ("Please download a model for your language from https://alphacephei.com/vosk/models")
    print ("and unpack as "+model_dir_sm+"' in the current folder.")
    sys.exit(0)

if not os.path.exists(model_dir_md):
    print ("Please download a model for your language from https://alphacephei.com/vosk/models")
    print ("and unpack as "+model_dir_md+"' in the current folder.")
    sys.exit(0)




class ASR:
    def __init__(self, system):
        if(system == "vosk_small"):
            self.model = vosk.Model(model_dir_sm) # load from a folder
        if(system == "vosk_md"):
            self.model = vosk.Model(model_dir_md) # load from a folder

    def predict(self, filename):
        with wave.open(filename) as wf:
            assert wf.getnchannels() == 1, "must be a mono wav"
            assert wf.getsampwidth() == 2, "must be a 16bit wav"
            assert wf.getcomptype() == "NONE", "must be PCM data"

            rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    rec.Result()
                else:
                    rec.PartialResult()
            r = json.loads(rec.FinalResult())
            return r["text"]


