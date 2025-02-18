# logic/human_detector_controller.py
from modules.human_detection_modules import HumanDetectorThread, YOLOv5HumanDetectorThread
from PySide6.QtCore import QObject, Signal


class HumanDetectorController(QObject):
    """
    The controller manages the human detection thread and exposes its signal.
    You can connect the `humanDetected` signal to a slot in your GUI to trigger conversation
    flow or other actions.
    """
    humanDetected = Signal()

    def __init__(self, camera_index=0, detection_time_threshold=2.0, parent=None, logger=None, debug_mode=False):
        """
        Args:
            camera_index (int): Webcam index.
            detection_time_threshold (float): Duration in seconds that a human must be continuously
                                                detected to trigger the signal.
            parent: Optional parent QObject.
            logger: Optional logger instance.
        """
        super().__init__(parent)
        self.logger = logger
        self.camera_index = camera_index
        self.detection_time_threshold = detection_time_threshold
        # self.thread = HumanDetectorThread(
        #     camera_index=self.camera_index,
        #     detection_time_threshold=self.detection_time_threshold,
        #     logger=self.logger
        # )
        # # When the thread emits humanDetected, re-emit it from the controller.
        # self.thread.humanDetected.connect(self.humanDetected.emit)
        self.thread = YOLOv5HumanDetectorThread(
            camera_index=camera_index,
            detection_time_threshold=detection_time_threshold,
            debug=debug_mode,
            logger=logger
        )
        # Expose the thread's signal.
        self.humanDetected = self.thread.humanDetected

    def start(self):
        """Start the human detection thread."""
        if self.logger:
            self.logger.log_info("Starting HumanDetectorController.")
        self.thread.start()

    def stop(self):
        """Stop the human detection thread."""
        if self.logger:
            self.logger.log_info("Stopping HumanDetectorController.")
        self.thread.stop()
