import time
import cv2

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor
from src.core.predictor import Predictor
from src.core.dynamic_predictor import DynamicPredictor
from src.core.recognition_controller import RecognitionController
from src.core.word_builder import WordBuilder

from src.utils.config import WINDOW_NAME


def main():

    # ==========================================
    # Initialize components
    # ==========================================

    camera = Camera()

    detector = HandDetector()

    processor = LandmarkProcessor()

    static_predictor = Predictor()

    dynamic_predictor = DynamicPredictor()

    controller = RecognitionController(
        static_predictor,
        dynamic_predictor,
        static_confidence_threshold=0.60,
    )

    word_builder = WordBuilder()

    # ==========================================
    # Display variables
    # ==========================================

    last_prediction = "None"

    last_confidence = 0.0

    # ==========================================
    # Gesture emission protection
    #
    # Prevents repeated gesture addition during
    # controller state transitions.
    # ==========================================

    gesture_consumed = False

    previous_mode = controller.NONE

    # ==========================================
    # Hand-entry protection
    #
    # Prevents accidental first gesture
    # duplication when the hand enters.
    # ==========================================

    hand_was_present = False

    waiting_for_hand_initialization = False

    recognition_ready = False

    # ==========================================
    # Space key tracking
    # ==========================================

    last_space_time = 0.0

    double_space_interval = 0.5

    # ==========================================
    # Start camera
    # ==========================================

    camera.start()

    print(
        "\n========== Sign Language Recognition ==========\n"
    )

    print("Static gestures : A-Y + 0-9")

    print("Dynamic gestures: J / Z")

    print("Press SPACE to add a space.")

    print(
        "Press SPACE twice quickly to clear text."
    )

    print(
        "Press BACKSPACE to remove last character."
    )

    print("Press 'Q' to exit.\n")

    try:

        while True:

            # ==========================================
            # Get frame
            # ==========================================

            frame = camera.get_frame()

            if frame is None:

                print(
                    "Failed to capture frame."
                )

                break

            # ==========================================
            # Detect hand
            # ==========================================

            results = detector.detect(
                frame
            )

            # ==========================================
            # Extract features
            # ==========================================

            features = (
                processor.extract_features(
                    results
                )
            )

            # ==========================================
            # Hand entry / exit detection
            # ==========================================

            hand_present = features is not None

            if (
                hand_present
                and not hand_was_present
            ):

                waiting_for_hand_initialization = True

                recognition_ready = False

                gesture_consumed = True

            elif (
                not hand_present
                and hand_was_present
            ):

                waiting_for_hand_initialization = False

                recognition_ready = False

                gesture_consumed = False

            hand_was_present = hand_present

            # ==========================================
            # Recognition Controller
            # ==========================================

            result = controller.update(
                features
            )

            prediction = result[
                "prediction"
            ]

            confidence = result[
                "confidence"
            ]

            mode = result[
                "mode"
            ]

            sequence_complete = result[
                "sequence_complete"
            ]

            # ==========================================
            # Hand initialization completed
            # ==========================================

            if (
                waiting_for_hand_initialization
                and mode == controller.STATIC
            ):

                waiting_for_hand_initialization = False

                recognition_ready = True

                # Consume the gesture already present
                # during initialization.

                gesture_consumed = True

            # ==========================================
            # Reset gesture permission only when the
            # controller enters NONE.
            # ==========================================

            if (
                recognition_ready
                and mode == controller.NONE
                and previous_mode != controller.NONE
            ):

                gesture_consumed = False

            # ==========================================
            # Add recognized gesture only once
            # ==========================================

            if (
                prediction is not None
                and recognition_ready
                and not waiting_for_hand_initialization
                and not gesture_consumed
            ):

                word_builder.add(
                    prediction
                )

                gesture_consumed = True

                last_prediction = prediction

                last_confidence = confidence

            # ==========================================
            # Update prediction display
            # ==========================================

            elif prediction is not None:

                last_prediction = prediction

                last_confidence = confidence

            # ==========================================
            # Store current mode
            # ==========================================

            previous_mode = mode

            # ==========================================
            # Draw hand landmarks
            # ==========================================

            frame = detector.draw(
                frame,
                results
            )

            # ==========================================
            # User-friendly Beta status
            # ==========================================

            if not hand_present:

                status = "No Hand Detected"

                display_confidence = None

            elif (
                waiting_for_hand_initialization
                or mode == controller.INITIALIZING
            ):

                status = "Initializing"

                display_confidence = None

            elif (
                recognition_ready
                and mode == controller.NONE
                and gesture_consumed
            ):

                status = "Gesture Locked"

                display_confidence = (
                    last_confidence
                )

            elif (
                recognition_ready
                and mode == controller.STATIC
            ):

                status = "Ready"

                display_confidence = (
                    last_confidence
                )

            else:

                status = "Recognizing"

                display_confidence = (
                    last_confidence
                )

            # ==========================================
            # Confidence text
            # ==========================================

            if display_confidence is None:

                confidence_text = (
                    "Confidence: --"
                )

            else:

                confidence_text = (
                    f"Confidence: "
                    f"{display_confidence * 100:.1f}%"
                )

            # ==========================================
            # Clean Beta UI
            # Top-left information panel
            # ==========================================

            overlay = frame.copy()

            # Semi-transparent dark gray panel

            cv2.rectangle(
                overlay,
                (10, 10),
                (250, 85),
                (55, 55, 55),
                -1,
            )

            # Blend panel with webcam frame

            cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0,
                frame,
            )

            # ==========================================
            # UI text styling
            # ==========================================

            UI_COLOR = (
                255,
                255,
                255,
            )

            UI_FONT = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            UI_FONT_SCALE = 0.45

            UI_THICKNESS = 1

            # ------------------------------------------
            # Mode
            # ------------------------------------------

            cv2.putText(
                frame,
                f"Mode: {mode}",
                (20, 30),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # ------------------------------------------
            # Confidence
            # ------------------------------------------

            cv2.putText(
                frame,
                confidence_text,
                (20, 52),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # ------------------------------------------
            # Status
            # ------------------------------------------

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 74),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # ==========================================
            # Current recognized text
            #
            # Displayed at bottom center.
            # ==========================================

            current_text = (
                word_builder.get_text()
            )

            if current_text:

                display_text = current_text

            elif last_prediction != "None":

                display_text = last_prediction

            else:

                display_text = ""

            # ==========================================
            # Draw bottom-center text
            # ==========================================

            if display_text:

                font = (
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                font_scale = 1.0

                thickness = 2

                (
                    text_width,
                    text_height,
                ), baseline = cv2.getTextSize(
                    display_text,
                    font,
                    font_scale,
                    thickness,
                )

                frame_height, frame_width = (
                    frame.shape[:2]
                )

                # Center text horizontally

                text_x = int(
                    (
                        frame_width
                        - text_width
                    )
                    / 2
                )

                text_y = (
                    frame_height
                    - 35
                )

                # --------------------------------------
                # Draw dark outline
                #
                # This keeps the white text visible
                # against both light and dark backgrounds.
                # --------------------------------------

                cv2.putText(
                    frame,
                    display_text,
                    (
                        text_x,
                        text_y,
                    ),
                    font,
                    font_scale,
                    (
                        40,
                        40,
                        40,
                    ),
                    5,
                    cv2.LINE_AA,
                )

                # --------------------------------------
                # Draw main white text
                # --------------------------------------

                cv2.putText(
                    frame,
                    display_text,
                    (
                        text_x,
                        text_y,
                    ),
                    font,
                    font_scale,
                    (
                        255,
                        255,
                        255,
                    ),
                    thickness,
                    cv2.LINE_AA,
                )

            # ==========================================
            # Show frame
            # ==========================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # ==========================================
            # Keyboard input
            # ==========================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------
            # Quit
            # ------------------------------------------

            if key in [
                ord("q"),
                ord("Q"),
            ]:

                break

            # ------------------------------------------
            # SPACE
            # ------------------------------------------

            elif key == 32:

                current_time = time.time()

                if (
                    last_space_time > 0
                    and (
                        current_time
                        - last_space_time
                        <= double_space_interval
                    )
                ):

                    word_builder.clear()

                    last_space_time = 0.0

                else:

                    word_builder.space()

                    last_space_time = current_time

            # ------------------------------------------
            # BACKSPACE
            # ------------------------------------------

            elif key in [
                8,
                127,
            ]:

                word_builder.backspace()

                last_space_time = 0.0

            # ------------------------------------------
            # Any other key resets
            # double-space timing
            # ------------------------------------------

            elif key != 255:

                last_space_time = 0.0

    finally:

        detector.close()

        camera.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()