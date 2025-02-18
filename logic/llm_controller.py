# logic/llm_controller.py
from PySide6.QtCore import QThread, QObject, Signal, Slot
from modules.llm_modules import OpenAIWorker, OllamaWorker
import openai

class LLMController(QObject):
    responseReady = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, llm_choice='OpenAI', api_key=None):
        super().__init__()
        self.llm_choice = llm_choice
        self.api_key = api_key
        self.worker = None
        self.thread = None

    def start(self):
        self.stop()  # Ensure no previous worker is running
        if self.llm_choice == "OpenAI":
            if self.api_key:
                openai.api_key = self.api_key
            self.worker = OpenAIWorker(api_key=self.api_key)
        elif self.llm_choice == "LLAMA":
            self.worker = OllamaWorker(model_name="llama3.2")
        else:
            self.errorOccurred.emit("Unsupported LLM choice")
            return

        # Connect signals from the worker
        self.worker.responseReady.connect(self.handle_response)
        self.worker.errorOccurred.connect(self.handle_error)

        # Move the worker to its own thread
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

    @Slot(str)
    def handle_response(self, response):
        self.responseReady.emit(response)

    @Slot(str)
    def handle_error(self, error_message):
        self.errorOccurred.emit(error_message)

    def send_request(self, user_input, model, context, max_tokens, temperature):
        if self.worker:
            if self.llm_choice == "OpenAI":
                self.worker.add_request(model, context, max_tokens, temperature)
            elif self.llm_choice == "LLAMA":
                self.worker.add_request(context, max_tokens, temperature)
        else:
            self.errorOccurred.emit("LLM Worker not started.")
