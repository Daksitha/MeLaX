# logic/tts_controller.py
from PySide6.QtCore import QThread, QObject, Signal, Slot
from modules.tts_modules import TTSWorker, GoogleTTSEngine, CoquiTTSEngine

class TTSController(QObject):
    ttsFinished = Signal(str)
    ttsError = Signal(str)

    def __init__(self, engine_choice="Google", language="en",
                 a2f_url="localhost:50051", a2f_inst="/World/audio2face/PlayerStreaming",
                 use_nlp_split=False, use_audio_streaming=True, block_until_finish=True,
                 chunk_duration=10, delay_between_chunks=4):
        super().__init__()
        self.engine_choice = engine_choice
        self.language = language
        self.a2f_url = a2f_url
        self.a2f_inst = a2f_inst
        self.use_nlp_split = use_nlp_split
        self.use_audio_streaming = use_audio_streaming
        self.block_until_finish = block_until_finish
        self.chunk_duration = chunk_duration
        self.delay_between_chunks = delay_between_chunks
        self.worker = None
        self.thread = None

    def start(self):
        self.stop()  # Stop any previous instance
        self.thread = QThread()
        if self.engine_choice == "Google":
            tts_engine = GoogleTTSEngine()
        else:
            tts_engine = CoquiTTSEngine()
        self.worker = TTSWorker(
            tts_engine=tts_engine,
            language=self.language,
            url=self.a2f_url,
            instance_name=self.a2f_inst,
            use_nlp_split=self.use_nlp_split,
            use_audio_streaming=self.use_audio_streaming,
            block_until_playback_is_finished=self.block_until_finish,
            chunk_duration=self.chunk_duration,
            delay_between_chunks=self.delay_between_chunks
        )
        self.worker.ttsFinished.connect(self.ttsFinished.emit)
        self.worker.ttsError.connect(self.ttsError.emit)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.quit()
            self.worker.wait()
            self.worker = None
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

    def send_request(self, text):
        if self.worker:
            self.worker.add_request(text)
        else:
            self.ttsError.emit("TTS Worker not started.")
