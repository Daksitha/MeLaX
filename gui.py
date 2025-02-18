# gui.py
import sys
import os
import requests
import speech_recognition as sr
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QLineEdit
from PySide6.QtCore import Signal, Slot

# Import the UI file generated from Qt Designer
from ui_form import Ui_MainWindow
# Import your thread-safe logger
from local_logger import ThreadSafeLogger

# Import modular controllers (ensure these files exist in your logic/ folder)
from logic.llm_controller import LLMController
from logic.asr_controller import ASRController
from logic.tts_controller import TTSController
from logic.behaviour_module import BehaviorControl
from logic.human_detector_controller import HumanDetectorController


class MainWindow(QMainWindow):
    # Custom signals for inter-module communication
    transcription_signal = Signal(str)    # ASR -> GUI
    tts_request_signal = Signal(str)        # GUI -> TTS

    def __init__(self):
        super().__init__()
        # Setup UI from designer file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("MeLaX-Engine")

        # Setup logger
        self.logger = ThreadSafeLogger("interactive_gui.log")
        self.logger.log_emitted.connect(self.append_log)

        # Initialize controllers for LLM, ASR, and TTS
        self.llm_controller = None
        self.asr_controller = None
        self.tts_controller = None
        self.behavior_control = None
        self.human_detector_controller = None

        # Other state variables
        self.context = []  # stores conversation context
        self.usd_folder_path = os.path.join(os.getcwd(), "usd")
        self.headless_server_url = self.ui.a2fUrl_headless.text().strip() or "http://localhost:8011"

        # Initialize UI components
        self.setup_ui()
        self.populate_microphones()
        self.populate_languages()
        self.populate_usd_list()

        # Connect signals from controllers and UI
        self.transcription_signal.connect(self.handle_transcription)
        self.tts_request_signal.connect(self.handle_tts_request)

        # Theme setup
        self.setup_theme_selection()
        self.apply_default_theme()

        # Connect button actions for overall module control
        self.ui.startAll.clicked.connect(self.allStart)
        self.ui.stopAll.clicked.connect(self.allStop)
        self.ui.clearAll.clicked.connect(self.allClear)

        # LLM switching (and related UI group visibility)
        self.ui.LLMChocie.currentTextChanged.connect(self.switch_llm_controller)
        self.switch_llm_controller(self.ui.LLMChocie.currentText())

        # Connect additional buttons for headless server actions
        self.behavior_control = BehaviorControl(
            ui=self.ui,
            logger=self.logger,
            headless_server_url=self.ui.a2fUrl_headless.text().strip() or "http://localhost:8011",
            usd_folder_path=self.usd_folder_path
        )
        # Connect emotion selection
        self.ui.emotionsQCombo.addItems([
            "neutral", "amazement", "anger", "cheekiness", "disgust", "fear",
            "grief", "joy", "outofbreath", "pain", "sadness"
        ])
        self.ui.connectHeadlessPushbutton.clicked.connect(self.behavior_control.on_connect_button_clicked)
        self.ui.loadUsdPushbutton.clicked.connect(self.behavior_control.on_load_usd_button_clicked)
        self.ui.emotionsQCombo.currentTextChanged.connect(self.behavior_control.on_emotion_selected)



        # API key setup (for OpenAI)
        self.ui.openaiAPIKey.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.ui.openaiAPIKey.setText(self.api_key if self.api_key else "")
        self.ui.openaiAPIKey.textChanged.connect(self.update_api_key)

        # System prompt update
        self.ui.systemPromptEdit.textChanged.connect(self.update_system_prompt)
        self.update_system_prompt()

    def setup_ui(self):
        """Setup UI elements and populate combo boxes."""
        # ASR options
        self.ui.whisperModel.addItems(["tiny", "base", "small", "medium", "large"])
        self.ui.whisperDevice.addItems(["CPU", "GPU"])
        self.ui.recognizerMBOX.addItems(["Google", "Whisper"])

        # LLM options
        self.ui.LLMChocie.addItems(["OpenAI", "LLAMA"])
        self.ui.llmMBOX.addItems(["gpt-4o-mini", "gpt-3.5-turbo"])
        self.ui.llmMBOX_Llama.addItems(["llama3.2"])
        self.toggle_llm_group_visibility(self.ui.LLMChocie.currentText())

        # TTS options
        self.ui.ttsEngineCombo.addItems(["Google", "Coqui", "VoiceX"])
        self.ui.ttslanguage.addItems(["en", "en-US", "en-GB", "es", "fr"])
        self.ui.ttsSentenceSplit.addItems(["Regex", "NLP"])
        self.ui.ttsPlayback.addItems(["SinglePush", "Stream"])
        self.ui.blockUntilFinish.setChecked(True)
        self.ui.ttsChunkDuration.setRange(1, 99)
        self.ui.ttsChunkDuration.setValue(10)
        self.ui.ttsDelayBetweenChunks.setRange(0, 50)
        self.ui.ttsDelayBetweenChunks.setValue(4)
        self.ui.chunckSizeLable.setText(f"value: {self.ui.ttsChunkDuration.value()}")
        self.ui.delayLable.setText(f"value: {self.ui.ttsDelayBetweenChunks.value()}")

        # Temperature and token settings
        self.ui.temperatureOpenAI.setRange(0, 100)
        self.ui.temperatureOpenAI.setValue(70)
        self.ui.temperatureOpenAI.valueChanged.connect(self.update_temperature_label_openai)
        self.ui.temperatureLlama.setRange(0, 100)
        self.ui.temperatureLlama.setValue(70)
        self.ui.temperatureLlama.valueChanged.connect(self.update_temperature_label_llama)
        self.ui.maxTokenOpenAI.setValue(1024)

        # Google ASR options
        self.ui.googleAPI.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.ui.googleAPI.setText("AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw")
        self.ui.googleEndPoint.setText("http://www.google.com/speech-api/v2/recognize")
        self.ui.googleLanguage.addItems(["en-US", "es-ES", "fr-FR", "de-DE"])

        # Rename tabs
        self.ui.chatOutpuTab.setTabText(0, "Conversation")
        self.ui.chatOutpuTab.setTabText(1, "Log Terminal")
        self.ui.chatOutpuTab.setTabText(3, "Behaviour Log")

        self.ui.recognizerMBOX.currentTextChanged.connect(self.toggle_asr_options)
        self.toggle_asr_options("Google")

    def populate_microphones(self):
        """Populate microphone combo box."""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            self.ui.microphoneMBox.clear()
            if mic_list:
                self.ui.microphoneMBox.addItems(mic_list)
            else:
                self.ui.microphoneMBox.addItem("No microphones detected")
        except Exception as e:
            self.ui.microphoneMBox.addItem(f"Error: {e}")

    def populate_languages(self):
        """Populate language combo boxes for ASR."""
        try:
            from whisper.tokenizer import LANGUAGES
            self.ui.whisperLanguage.clear()
            for lang_name, lang_key in LANGUAGES.items():
                self.ui.whisperLanguage.addItem(lang_name, lang_key)
        except Exception as e:
            self.logger.log_error(f"Error populating Whisper languages: {e}")

    def populate_usd_list(self):
        """Populate the USD combo box with files from the usd folder."""
        self.ui.usdCombo.clear()
        if not os.path.exists(self.usd_folder_path):
            self.ui.usdCombo.addItem("No USD folder found")
            self.ui.usdCombo.setEnabled(False)
            return

        usd_files = [f for f in os.listdir(self.usd_folder_path)
                     if f.endswith((".usd", ".usda", ".usdc"))]
        if usd_files:
            self.ui.usdCombo.addItems(usd_files)
            self.ui.usdCombo.setEnabled(True)
        else:
            self.ui.usdCombo.addItem("No USD files available")
            self.ui.usdCombo.setEnabled(False)


    def toggle_asr_options(self, recognizer_type):
        if recognizer_type == "Google":
            self.ui.googleGroupBox.setVisible(True)
            self.ui.whisperGroupBox.setVisible(False)
        else:
            self.ui.googleGroupBox.setVisible(False)
            self.ui.whisperGroupBox.setVisible(True)

    def toggle_llm_group_visibility(self, llm_choice: str):
        """Toggle visibility of OpenAI and LLAMA related UI groups."""
        if llm_choice == "OpenAI":
            self.ui.openai_group.setVisible(True)
            self.ui.llama_group.setVisible(False)
        elif llm_choice == "LLAMA":
            self.ui.openai_group.setVisible(False)
            self.ui.llama_group.setVisible(True)
        else:
            self.ui.openai_group.setVisible(False)
            self.ui.llama_group.setVisible(False)

    def setup_theme_selection(self):
        """Optionally load and select themes (QSS files) from a stylesheets folder."""
        self.stylesheets_folder = Path(__file__).parent / "stylesheets"

    def apply_default_theme(self):
        """Apply the default dark theme if available."""
        if not self.stylesheets_folder.exists():
            return
        for f in self.stylesheets_folder.glob("*.qss"):
            if f.stem.lower().startswith("dark"):
                with open(f, "r", encoding="utf-8") as file:
                    self.setStyleSheet(file.read())
                return

    def apply_dark_theme(self):
        """Apply a dark theme."""
        if not self.stylesheets_folder.exists():
            return
        for f in self.stylesheets_folder.glob("*.qss"):
            if f.stem.lower().startswith("dark"):
                with open(f, "r", encoding="utf-8") as file:
                    self.setStyleSheet(file.read())
                return

    def apply_light_theme(self):
        """Apply a light theme."""
        if not self.stylesheets_folder.exists():
            return
        for f in self.stylesheets_folder.glob("*.qss"):
            if f.stem.lower().startswith("light"):
                with open(f, "r", encoding="utf-8") as file:
                    self.setStyleSheet(file.read())
                return

    def update_temperature_label_openai(self, value: int):
        """Update the temperature label for OpenAI."""
        temperature = value / 100.0
        self.ui.llmTemperatureLable.setText(f"{temperature:.2f}")

    def update_temperature_label_llama(self, value: int):
        """Update the temperature label for Llama."""
        temperature = value / 100.0
        self.ui.TemperatureLablellama.setText(f"{temperature:.2f}")

    def update_api_key(self, text):
        """Update API key for OpenAI."""
        self.api_key = text.strip()

    def update_system_prompt(self):
        """Update system prompt and reset conversation context."""
        new_prompt = self.ui.systemPromptEdit.toPlainText().strip()
        if new_prompt:
            self.logger.log_info(f"System prompt updated: {new_prompt}")
            self.context = [{"role": "developer", "content": [{"type": "text", "text": new_prompt}]}]
            self.ui.contextBrowserOpenAI.clear()
            self.ui.contextBrowserOpenAI.append(f"<b>Developer:</b> {new_prompt}")
        else:
            default_prompt = "You are a helpful and knowledgeable assistant that answers questions short and clear."
            self.ui.systemPromptEdit.setText(default_prompt)
            self.logger.log_info("System prompt was empty. Resetting to default.")
            self.context = [{"role": "developer", "content": [{"type": "text", "text": default_prompt}]}]
            self.ui.contextBrowserOpenAI.clear()
            self.ui.contextBrowserOpenAI.append(f"<b>Developer:</b> {default_prompt}")

    def switch_llm_controller(self, llm_choice: str):
        """Switch the LLM controller based on the user's selection."""
        self.logger.log_info(f"Switching LLM to: {llm_choice}")
        # Stop previous LLM controller if active
        if self.llm_controller:
            self.llm_controller.stop()
        # Create a new LLM controller based on the selection
        if llm_choice == "OpenAI":
            self.ui.openai_group.setVisible(True)
            self.ui.llama_group.setVisible(False)
            api_key = self.ui.openaiAPIKey.text().strip()
            self.llm_controller = LLMController(llm_choice="OpenAI", api_key=api_key)
        elif llm_choice == "LLAMA":
            self.ui.openai_group.setVisible(False)
            self.ui.llama_group.setVisible(True)
            self.llm_controller = LLMController(llm_choice="LLAMA")
        else:
            self.logger.log_error("Unsupported LLM choice")
            return

        self.llm_controller.responseReady.connect(self.display_llm_response)
        self.llm_controller.errorOccurred.connect(self.display_llm_error)
        self.llm_controller.start()

    @Slot(str)
    def display_llm_response(self, response):
        """Display LLM response and trigger TTS."""
        self.ui.contextBrowserOpenAI.append(f"<b>Assistant:</b> {response}")
        self.tts_request_signal.emit(response)

    @Slot(str)
    def display_llm_error(self, error_message):
        """Display error message from LLM controller."""
        QMessageBox.critical(self, "LLM Error", error_message)
        self.logger.log_error(f"LLM Error: {error_message}")

    def start_asr(self):
        """Start ASR controller based on selected recognizer."""

        selected_mic_name = self.ui.microphoneMBox.currentText()

        if self.asr_controller:
            self.asr_controller.stop()
        asr_choice = self.ui.recognizerMBOX.currentText()
        if asr_choice == "Whisper":
            selected_model = self.ui.whisperModel.currentText()
            device_txt = self.ui.whisperDevice.currentText()
            lang_key = self.ui.whisperLanguage.currentData()
            device_map = {"CPU": "cpu", "GPU": "cuda"}
            device = device_map.get(device_txt, "cpu")
            self.asr_controller = ASRController(
                recognizer="Whisper",
                model=selected_model,
                device=device,
                language=lang_key,
                logger=self.logger,
                microphone_name=selected_mic_name
            )
        elif asr_choice == "Google":
            api_key = self.ui.googleAPI.text().strip() or "YOUR_DEFAULT_API_KEY"
            endpoint = self.ui.googleEndPoint.text().strip() or "http://www.google.com/speech-api/v2/recognize"
            lang = self.ui.googleLanguage.currentText()
            self.asr_controller = ASRController(
                recognizer="Google",
                api_key=api_key,
                endpoint=endpoint,
                language=lang,
                energy_threshold=1000,
                record_timeout=2.0,
                logger=self.logger,
                microphone_name=selected_mic_name
            )
        else:
            self.logger.log_info("Unsupported ASR choice. No thread started.")
            return
        self.asr_controller.transcriptionReady.connect(self.transcription_signal.emit)
        self.asr_controller.start()

    def stop_asr(self):
        """Stop ASR controller if active."""
        if self.asr_controller:
            self.asr_controller.stop()
            self.asr_controller = None

    def start_tts(self):
        """Start TTS controller based on selected options."""
        if self.tts_controller:
            self.tts_controller.stop()
        engine_choice = self.ui.ttsEngineCombo.currentText()
        lang_choice = self.ui.ttslanguage.currentText()
        a2f_url = self.ui.a2fUrl.text().strip() or "localhost:50051"
        a2f_inst = self.ui.a2fInstanceName.text().strip() or "/World/audio2face/PlayerStreaming"
        split_choice = self.ui.ttsSentenceSplit.currentText()
        use_nlp = (split_choice == "NLP")
        playback = self.ui.ttsPlayback.currentText()
        use_streaming = (playback == "Stream")
        block_until = self.ui.blockUntilFinish.isChecked()
        chunk_dur = self.ui.ttsChunkDuration.value()
        delay_chunks = self.ui.ttsDelayBetweenChunks.value()
        self.tts_controller = TTSController(
            engine_choice=engine_choice,
            language=lang_choice,
            a2f_url=a2f_url,
            a2f_inst=a2f_inst,
            use_nlp_split=use_nlp,
            use_audio_streaming=use_streaming,
            block_until_finish=block_until,
            chunk_duration=chunk_dur,
            delay_between_chunks=delay_chunks
        )
        self.tts_controller.ttsFinished.connect(self.handle_tts_finished)
        self.tts_controller.ttsError.connect(self.handle_tts_error)
        self.tts_controller.start()

    def stop_tts(self):
        """Stop TTS controller if active."""
        if self.tts_controller:
            self.tts_controller.stop()
            self.tts_controller = None
    def start_behaviour(self):
        """
        Minimal start function for behaviour (headless server/emotion control).
        Since BehaviorControl is stateless in this design, we just log that it is ready.
        """
        if self.behavior_control:
            self.logger.log_info("Behaviour control is ready.")
        else:
            self.logger.log_error("Behaviour control is not configured.")

    def stop_behaviour(self):
        """
        Minimal stop function for behaviour.
        For this module, we simply clear the instance.
        """
        if self.behavior_control:
            self.logger.log_info("Stopping behaviour control.")
            self.behavior_control = None
        else:
            self.logger.log_info("Behaviour control is not running.")

    def start_human_detection(self):
        """Starts the human detection controller."""
        # If the human detector instance is None (or has been stopped), reinitialize it.
        if self.human_detector_controller is None:
            self.logger.log_info("Reinitializing human detector.")
            self.human_detector_controller = HumanDetectorController(
                camera_index=0,               # Use default webcam (adjust if needed)
                detection_time_threshold=2.0, # Trigger if human is detected > 2 seconds
                logger=self.logger,           # Pass your logger instance
                debug_mode=True               # Enable visualization for debugging
            )
            self.human_detector_controller.humanDetected.connect(self.on_human_detected)
        self.logger.log_info("Starting human detection.")
        self.human_detector_controller.start()

    def stop_human_detection(self):
        """Stops the human detection controller."""
        if self.human_detector_controller:
            self.logger.log_info("Stopping human detection.")
            self.human_detector_controller.stop()
            # Optionally, set to None so that a fresh instance is created on next start
            self.human_detector_controller = None
        else:
            self.logger.log_info("Human detector is not running.")

    def allStart(self):
        """Start all modules: LLM, ASR, TTS, behaviour and human detection."""
        self.logger.log_info("Starting all modules.")
        self.switch_llm_controller(self.ui.LLMChocie.currentText())
        self.start_asr()
        self.start_tts()
        self.start_behaviour()
        self.start_human_detection()
        self.logger.log_info("All modules started.")

    def allStop(self):
        """Stop all modules: LLM, ASR, TTS, behaviour and human detection."""
        self.logger.log_info("Stopping all modules.")
        if self.llm_controller:
            self.llm_controller.stop()
            self.llm_controller = None
        self.stop_asr()
        self.stop_tts()
        self.stop_behaviour()
        self.stop_human_detection()
        self.logger.log_info("All modules stopped.")

    def allClear(self):
        """Stop all modules and reset UI settings to default."""
        self.allStop()
        self.ui.contextBrowserOpenAI.clear()
        self.ui.systemPromptEdit.setText("You are a helpful assistant.")
        # Reset other UI fields as needed
        self.ui.ttsEngineCombo.setCurrentIndex(0)
        self.ui.ttslanguage.setCurrentIndex(0)
        self.ui.a2fUrl.setText("localhost:50051")
        self.ui.a2fInstanceName.setText("/World/audio2face/PlayerStreaming")
        self.ui.ttsSentenceSplit.setCurrentIndex(0)
        self.ui.ttsPlayback.setCurrentIndex(0)
        self.ui.blockUntilFinish.setChecked(True)
        self.ui.ttsChunkDuration.setValue(10)
        self.ui.ttsDelayBetweenChunks.setValue(4)
        self.ui.recognizerMBOX.setCurrentIndex(0)
        self.ui.whisperModel.setCurrentIndex(0)
        self.ui.whisperDevice.setCurrentIndex(0)
        self.ui.googleLanguage.setCurrentIndex(0)
        self.ui.microphoneMBox.setCurrentIndex(0)
        self.ui.LLMChocie.setCurrentIndex(0)
        self.ui.llmMBOX.setCurrentIndex(0)
        self.ui.temperatureOpenAI.setValue(70)
        self.ui.maxTokenOpenAI.setValue(1024)
        self.logger.log_info("All settings and context reset.")

    @Slot(str)
    def handle_transcription(self, user_input):
        """Handle ASR transcription, display user input, and forward to LLM."""
        self.logger.log_info(f"ASR recognized: {user_input}")
        self.ui.contextBrowserOpenAI.append(f"<b>User:</b> {user_input}")
        self.send_to_llm(user_input)

    def send_to_llm(self, user_input):
        """Forward user input to the active LLM controller."""
        temperature = self.ui.temperatureOpenAI.value() / 100.0
        max_tokens = self.ui.maxTokenOpenAI.value()
        if self.llm_controller:
            if self.ui.openai_group.isVisible():
                selected_model = self.ui.llmMBOX.currentText()
                context = [{"role": "user", "content": user_input}]
                self.llm_controller.send_request(user_input, selected_model, context, max_tokens, temperature)
            elif self.ui.llama_group.isVisible():
                messages = [{"role": "user", "content": user_input}]
                self.llm_controller.send_request(user_input, None, messages, max_tokens, temperature)
        else:
            self.logger.log_error("LLM controller not active.")

    @Slot(str)
    def handle_tts_request(self, text):
        """Send text to TTS controller for audio synthesis."""
        if self.tts_controller:
            self.tts_controller.send_request(text)
        else:
            self.logger.log_error("TTS controller not active.")

    @Slot(str)
    def handle_tts_finished(self, msg):
        """Handle notification that TTS finished playback."""
        self.logger.log_info(f"TTS finished: {msg}")

    @Slot(str)
    def handle_tts_error(self, error_message):
        """Display TTS error message."""
        QMessageBox.critical(self, "TTS Error", error_message)
        self.logger.log_error(f"TTS error: {error_message}")

    @Slot(str)
    def append_log(self, message: str):
        """Append log messages to the log output widget."""
        self.ui.logOutput.append(message)

    @Slot()
    def on_human_detected(self):
        self.logger.log_info("Human detected continuously for 2 seconds; starting conversation flow.")
        # TODO: Insert logic to start the conversation flow or any other desired action.

    def closeEvent(self, event):
        """Handle application close event by stopping all threads."""
        self.logger.log_info("Application closing; stopping threads.")
        self.allStop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
