import argparse
from deepspeechWrapper import ASR as DPASR

#TODO: add my recorded audio
AUDIO_FILES = {
    "EN": ["parents.wav", "checkin.wav", "what_time.wav", "suitcase.wav", "where.wav"]
}


def run(system, lang):
    acr = None
    if(system == "deepspeech"):
        acr = DPASR(lang)
    SOUNDS_DIR = "sounds/"+lang
    audio_filenames = AUDIO_FILES[lang]
    for audio_filename in audio_filenames:
        prediction = acr.predict(f"{SOUNDS_DIR}/{audio_filename}")
        print(f"prediction for {audio_filename}: {prediction}")




if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--lang", default="EN", help="language")
    argParser.add_argument("--system", default="deepspeech")
    args = argParser.parse_args()
    run(args.system, args.lang)

