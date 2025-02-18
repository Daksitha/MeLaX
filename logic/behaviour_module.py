# behaviour_control.py
import os
import requests
from PySide6.QtWidgets import QToolButton

class BehaviorControl:
    def __init__(self, ui, logger, headless_server_url: str, usd_folder_path: str):
        """
        Initialize the behavior control module.

        Args:
            ui: The UI instance (to access buttons and other widgets).
            logger: A logger instance for logging messages.
            headless_server_url (str): The base URL for the headless server.
            usd_folder_path (str): Path to the folder containing USD files.
        """
        self.ui = ui
        self.logger = logger
        self.headless_server_url = headless_server_url or "http://localhost:8011"
        self.usd_folder_path = usd_folder_path

    def check_server_status(self) -> bool:
        """Check the status of the headless server."""
        try:
            url = f"{self.headless_server_url}/status"
            self.logger.log_info(f"Checking server status: {url}")
            response = requests.get(url, timeout=2, headers={"accept": "application/json"})
            self.logger.log_info(f"Status check response: {response.text}, {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data == "OK":
                    return True
        except requests.exceptions.RequestException as e:
            self.logger.log_error(f"Error checking server status: {e}")
        return False

    def update_server_status(self, connected: bool):
        """Update the UI button appearance based on headless server connection status."""
        if connected:
            self.ui.connectHeadlessPushbutton.setStyleSheet("QToolButton { background-color: green; }")
        else:
            self.ui.connectHeadlessPushbutton.setStyleSheet("QToolButton { background-color: red; }")

    def on_connect_button_clicked(self):
        """Handle connection attempt to headless server."""
        self.headless_server_url = self.ui.a2fUrl_headless.text().strip() or "http://localhost:8011"
        connected = self.check_server_status()
        self.update_server_status(connected)
        if connected:
            self.logger.log_info("Headless server connected.")
        else:
            self.logger.log_error("Unable to connect to headless server.")

    def on_load_usd_button_clicked(self):
        """Attempt to load the selected USD file if the headless server is connected."""
        connected = self.check_server_status()
        if not connected:
            self.logger.log_error("Server not connected, cannot load USD.")
            return
        success = self.load_selected_usd()
        if success:
            self.ui.loadUsdPushbutton.setStyleSheet("QToolButton { background-color: green; }")
        else:
            self.ui.loadUsdPushbutton.setStyleSheet("QToolButton { background-color: red; }")

    def load_selected_usd(self) -> bool:
        """Load the USD file selected in the combo box via the headless server."""
        selected_usd = self.ui.usdCombo.currentText()
        if not selected_usd or selected_usd.startswith("No USD"):
            self.logger.log_error("Invalid USD selection.")
            return False
        usd_path = os.path.join(self.usd_folder_path, selected_usd)
        try:
            payload = {"file_name": usd_path}
            response = requests.post(
                f"{self.headless_server_url}/A2F/USD/Load",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                json=payload
            )
            if response.status_code == 200 and response.json().get("status") == "OK":
                self.logger.log_info(f"USD loaded successfully: {usd_path}")
                return True
            else:
                self.logger.log_error("Failed to load USD.")
        except requests.exceptions.RequestException as e:
            self.logger.log_error(f"Error loading USD: {e}")
        return False

    def on_emotion_selected(self, emotion: str):
        """Send a request to set the selected emotion (and reset others) via the headless server."""
        emotions = [
            "amazement", "anger", "cheekiness", "disgust", "fear",
            "grief", "joy", "outofbreath", "pain", "sadness"
        ]
        payload = {
            "a2f_instance": "/World/audio2face/CoreFullface",
            "emotions": {e: 0 for e in emotions}
        }
        if emotion != "neutral":
            payload["emotions"][emotion] = 1
        try:
            self.logger.log_info(f"Emotion payload: {payload}")
            response = requests.post(
                f"{self.headless_server_url}/A2F/A2E/SetEmotionByName",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                json=payload
            )
            if response.status_code == 422:
                self.logger.log_error(f"Validation error: {response.text}")
                return
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") == "OK":
                    self.logger.log_info(f"Emotion set successfully: {emotion}, Message: {response_data.get('message')}")
                else:
                    self.logger.log_error(f"Unexpected response: {response.text}")
            else:
                self.logger.log_error(f"Failed to set emotion: {emotion}, Response: {response.text}")
        except requests.exceptions.RequestException as e:
            self.logger.log_error(f"Error setting emotion: {e}")
