import torch
import pyaudio
import wave
import threading
import whisper
from TTS.api import TTS


class CommunicationModule:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tts_cache_file_name = "tts_cache.wav"
        self.stt_cache_file_name = "stt_cache.wav"
        self.tts = TTS(model_name=model_name).to(device)
        self.stt = whisper.load_model("base")
        self.audio = pyaudio.PyAudio()

    def __del__(self):
        try:
            self.audio.terminate()
        except TypeError:
            pass

    def say(self, sentence):
        self.tts.tts_to_file(text=sentence, file_path=self.tts_cache_file_name)
        say_thread = threading.Thread(target=self.__play_cached_audio)
        say_thread.start()

    def hear(self, return_queue):
        hear_thread = threading.Thread(target=self.__capture_voice_and_process, args=(return_queue,))
        hear_thread.start()

    def __play_cached_audio(self):
        wf = wave.open(self.tts_cache_file_name, 'rb')
        stream = self.audio.open(format=self.audio.get_format_from_width(wf.getsampwidth()),
                                 channels=wf.getnchannels(),
                                 rate=wf.getframerate(),
                                 output=True)
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)
        stream.stop_stream()
        stream.close()
        wf.close()

    def __capture_voice_and_process(self, return_queue):
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 16000
        chunk = 1024
        record_seconds = 2

        stream = self.audio.open(format=audio_format,
                                 channels=channels,
                                 rate=rate,
                                 input=True,
                                 frames_per_buffer=chunk)
        print("Recording...")
        frames = []

        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("Finished recording.")

        # Stop recording
        stream.stop_stream()
        stream.close()

        wave_file = wave.open(self.stt_cache_file_name, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(self.audio.get_sample_size(audio_format))
        wave_file.setframerate(rate)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

        result = self.stt.transcribe(self.stt_cache_file_name)
        return_queue.put(result["text"])
