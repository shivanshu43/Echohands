import cv2

from src.utils.config import CAMERA_INDEX

class Camera:
    
    # Handles webcam initialization and frame capture.
    

    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        """
        Open the webcam.
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

    def get_frame(self):
        """
        Capture a single frame.
        """
        if self.cap is None:
            raise RuntimeError("Camera has not been started.")

        success, frame = self.cap.read()

        if not success:
            return None

        frame = cv2.flip(frame, 1)
        return frame

    def stop(self):
        """
        Release the webcam.
        """
        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()