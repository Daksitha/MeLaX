import cv2
import time
from PySide6.QtCore import QThread, QObject, Signal
import torch
import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

class HumanDetectorThread(QThread):
    """
    This thread continuously captures frames from the webcam, detects human presence
    using OpenCV’s HOG people detector, and emits a signal if a human is continuously
    present for longer than the specified threshold.
    """
    humanDetected = Signal()  # Signal emitted when a human is detected for long enough

    def __init__(self, camera_index=0, detection_time_threshold=2.0, parent=None, logger=None):
        """
        Args:
            camera_index (int): The index of the webcam to use.
            detection_time_threshold (float): How long (in seconds) a human must be continuously
                                                detected before triggering the signal.
            parent: Optional parent for the QThread.
            logger: Optional logger object with a log_info/log_error interface.
        """
        super().__init__(parent)
        self.camera_index = camera_index
        self.detection_time_threshold = detection_time_threshold
        self.logger = logger
        self.running = True

        # Initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            if self.logger:
                self.logger.log_error("HumanDetectorThread: Cannot open webcam")
            return

        human_start_time = None
        human_emitted = False

        if self.logger:
            self.logger.log_info("HumanDetectorThread started.")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.logger:
                    self.logger.log_error("HumanDetectorThread: Cannot read frame from webcam")
                break

            # Resize frame for faster processing (optional)
            frame_resized = cv2.resize(frame, (640, 480))
            # Detect humans in the frame
            rects, _ = self.hog.detectMultiScale(frame_resized, winStride=(8, 8))
            if len(rects) > 0:
                # At least one human found in the frame
                if human_start_time is None:
                    human_start_time = time.time()
                    human_emitted = False
                    if self.logger:
                        self.logger.log_info("Human detected; starting timer.")
                else:
                    elapsed = time.time() - human_start_time
                    if elapsed >= self.detection_time_threshold and not human_emitted:
                        if self.logger:
                            self.logger.log_info("Human detected for more than %.1f seconds. Emitting signal." % self.detection_time_threshold)
                        self.humanDetected.emit()
                        human_emitted = True  # Prevent continuous emissions during the same detection session
            else:
                # No human in frame; reset timer and flag
                if human_start_time is not None and self.logger:
                    self.logger.log_info("Human no longer detected; resetting timer.")
                human_start_time = None
                human_emitted = False

            self.msleep(50)  # Sleep for 50ms to avoid high CPU usage

        cap.release()
        if self.logger:
            self.logger.log_info("HumanDetectorThread stopped.")

    def stop(self):
        """Stop the human detection thread."""
        self.running = False
        self.quit()
        self.wait()


class YOLOv5HumanDetectorThread(QThread):
    """
    A QThread that uses a YOLOv5 model to detect humans in the webcam feed.
    If a person is detected continuously for longer than a specified threshold,
    it emits the `humanDetected` signal. When debug is enabled, it displays the
    video feed with bounding boxes and labels.
    """
    humanDetected = Signal()

    def __init__(self, camera_index=0, detection_time_threshold=2.0, debug=False, parent=None, logger=None):
        """
        Args:
            camera_index (int): The webcam index to use.
            detection_time_threshold (float): Seconds a person must be continuously detected.
            debug (bool): If True, show the video feed with drawn detections.
            logger: Optional logger instance with methods like log_info/log_error.
        """
        super().__init__(parent)
        self.camera_index = camera_index
        self.detection_time_threshold = detection_time_threshold
        self.debug = debug
        self.logger = logger
        self.running = True

        # Load the YOLOv5 detection model from PyTorch Hub.
        # Using the pretrained 'yolov5s' model.
        if self.logger:
            self.logger.log_info("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.names = self.model.names  # Mapping of class indices to names

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            if self.logger:
                self.logger.log_error("YOLOv5HumanDetectorThread: Cannot open webcam")
            return

        human_start_time = None
        human_emitted = False

        if self.logger:
            self.logger.log_info("YOLOv5HumanDetectorThread started.")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.logger:
                    self.logger.log_error("YOLOv5HumanDetectorThread: Cannot read frame")
                break

            # Run inference with YOLOv5.
            # YOLOv5 model automatically handles BGR images (as provided by OpenCV).
            results = self.model(frame)
            # Get detections: each row is [x_min, y_min, x_max, y_max, confidence, class]
            detections = results.xyxy[0]

            person_detected = False

            # Loop over detections
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                conf = conf.item()
                cls = int(cls.item())
                label = f"{self.names[cls]} {conf:.2f}"

                # Check if detection is a person.
                if self.names[cls].lower() == "person":
                    person_detected = True

                if self.debug:
                    # Draw bounding box and label on the frame.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if self.debug:
                cv2.imshow("YOLOv5 Human Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            # Timing logic: start a timer if a person is detected.
            if person_detected:
                if human_start_time is None:
                    human_start_time = time.time()
                    human_emitted = False
                    if self.logger:
                        self.logger.log_info("Person detected; starting timer.")
                else:
                    elapsed = time.time() - human_start_time
                    if elapsed >= self.detection_time_threshold and not human_emitted:
                        if self.logger:
                            self.logger.log_info(f"Person detected for more than {self.detection_time_threshold:.1f} seconds. Emitting signal.")
                        self.humanDetected.emit()
                        human_emitted = True  # Avoid repeated emissions during same detection session.
            else:
                if human_start_time is not None and self.logger:
                    self.logger.log_info("Person no longer detected; resetting timer.")
                human_start_time = None
                human_emitted = False

            self.msleep(50)

        cap.release()
        if self.debug:
            cv2.destroyAllWindows()
        if self.logger:
            self.logger.log_info("YOLOv5HumanDetectorThread stopped.")

    def stop(self):
        """Stop the human detection thread."""
        self.running = False
        self.quit()
        self.wait()

