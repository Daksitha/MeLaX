# logic/asr_controller.py
from PySide6.QtCore import QThread, QObject, Signal, Slot
from modules.asr_vad_modules import WhisperTranscriptionThread, GoogleASRTranscriptionThread

class ASRController(QObject):
    transcriptionReady = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, recognizer="Whisper", **kwargs):
        """
        kwargs can include:
          - model, device, language (for Whisper)
          - api_key, endpoint, language, energy_threshold, record_timeout (for Google)
          - logger: a logging object, etc.
        """
        super().__init__()
        self.recognizer = recognizer
        self.worker = None
        self.thread = None
        self.kwargs = kwargs

    def start(self):
        self.stop()  # Stop previous instance if any
        self.thread = QThread()
        if self.recognizer == "Whisper":
            self.worker = WhisperTranscriptionThread(
                model=self.kwargs.get("model", "base"),
                device=self.kwargs.get("device", "cpu"),
                language=self.kwargs.get("language", "en"),
                logger=self.kwargs.get("logger", None),
                microphone_name = self.kwargs.get("microphone_name")
            )
        elif self.recognizer == "Google":
            self.worker = GoogleASRTranscriptionThread(
                api_key=self.kwargs.get("api_key", ""),
                endpoint=self.kwargs.get("endpoint", "http://www.google.com/speech-api/v2/recognize"),
                language=self.kwargs.get("language", "en-US"),
                energy_threshold=self.kwargs.get("energy_threshold", 1000),
                record_timeout=self.kwargs.get("record_timeout", 2.0),
                logger=self.kwargs.get("logger", None)
            )
        else:
            self.errorOccurred.emit("Unsupported ASR choice")
            return

        self.worker.transcription_signal.connect(self.transcriptionReady.emit)
        self.worker.moveToThread(self.thread)
        # Optionally, you can connect the thread's started signal to the worker.start method
        self.thread.started.connect(self.worker.start)
        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
