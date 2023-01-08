import argparse
from deepspeechWrapper import ASR as DPASR
from voskWrapper import ASR as VOSKASR
from jiwer import wer
import jiwer
import numpy as np

#TODO: add my recorded audio
AUDIO_FILES = {
    "EN": ["parents.wav", "checkin.wav", "what_time.wav", "suitcase.wav", "where.wav", "your_sentence1.wav", "your_sentence2.wav"],
    "ES": ["parents_es.wav", "checkin_es.wav", "what_time_es.wav", "suitcase_es.wav", "where_es.wav"],
    "IT": ["parents_it.wav", "checkin_it.wav", "what_time_it.wav", "suitcase_it.wav", "where_it.wav"]

}

GROUND_TRUTHS = {
    "parents.wav": "I have lost my parents",
    "checkin.wav": "Where is the check-in desk?",
    "what_time.wav": "What time is my plane?",
    "suitcase.wav": "Please, I have lost my suitcase",
    "where.wav":"Where are the restaurants and shops?",
    "your_sentence1.wav": "I want the reimbursement of my flight",
    "your_sentence2.wav": "Have you seen my dog?",
    "parents_es.wav": "He perdido a mis padres",
    "checkin_es.wav": "¿Dónde están los mostradores?",
    "what_time_es.wav": "¿A qué hora es mi avión?",
    "suitcase_es.wav": "Por favor, he perdido mi maleta",
    "where_es.wav": "¿Dónde están los restaurantes y las tiendas?",
    "parents_it.wav": "Ho perso i miei genitori",
    "checkin_it.wav": "Dove e' il bancone?",
    "what_time_it.wav": "A che ora e’ il mio aereo?",
    "suitcase_it.wav": "Per favore, ho perso la mia valigia",
    "where_it.wav": "Dove sono i ristoranti e i negozi?",
}


def run(system, lang):
    acr = None
    if(system == "deepspeech"):
        acr = DPASR(lang)
    if(system== "vosk_small" or system== "vosk_md" ):
        acr = VOSKASR(system)
    SOUNDS_DIR = "sounds/"+lang
    audio_filenames = AUDIO_FILES[lang]
    WERs = []
    for audio_filename in audio_filenames:
        prediction = acr.predict(f"{SOUNDS_DIR}/{audio_filename}")
        ground_truth = jiwer.RemovePunctuation()(GROUND_TRUTHS[audio_filename])
        error_rate = wer(ground_truth.lower(), prediction.lower())
        WERs.append(error_rate)
        print(f'prediction for {audio_filename}: "{prediction}" - ground_truth:"{ground_truth}" -wer: {error_rate}')
    print(f"avg WER for system: {system} and lang: {lang}: ", np.mean(WERs))



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--lang", default="EN", help="language")
    argParser.add_argument("--system", default="deepspeech")
    args = argParser.parse_args()
    run(args.system, args.lang)

