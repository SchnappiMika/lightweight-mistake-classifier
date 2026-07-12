# -*- coding: utf-8 -*-
"""
Audio transcription module - Whisper wrapper
"""

class AudioTranscriber:
    def __init__(self, model_size="tiny"):
        self.available = False
        self.model_size = model_size
        self._try_init()

    def _try_init(self):
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            self.available = True
        except Exception as e:
            self.available = False

    def transcribe(self, audio_path):
        if not self.available:
            return None
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            return None