import cv2
import logging
import time

class Camera:
    def __init__(self, device_id, width, height, fps):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def initialize(self):
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.device_id}")
                return False
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error(f"Failed to read frame from {self.device_id}")
                return False
            self.logger.info(f"Camera initialized: {self.device_id} ({self.width}x{self.height} @ {self.fps}fps)")
            return True
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    def get_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(f"Failed to capture frame from {self.device_id}")
                return None
            return frame
        except Exception as e:
            self.logger.error(f"Camera read error: {e}")
            return None

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("Camera connection closed")