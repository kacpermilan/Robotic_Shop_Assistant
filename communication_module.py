import torch
import simpleaudio
from TTS.api import TTS


class CommunicationModule:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cache_file_name = "tts_cache.wav"
        self.tts = TTS(model_name=model_name).to(device)

    def say(self, sentence):
        self.tts.tts_to_file(text=sentence, file_path=self.cache_file_name)
        wave_obj = simpleaudio.WaveObject.from_wave_file(self.cache_file_name)
        wave_obj.play()
