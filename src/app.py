import os

# ==========================================================
# Reduce TensorFlow startup noise
#
# Must be set BEFORE TensorFlow is imported.
# ==========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from pathlib import Path

import cv2

from src.utils.startup_ui import StartupUI
from src.utils.model_manager import (
    load_manifest,
    prepare_models,
)

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor
from src.core.predictor import Predictor
from src.core.dynamic_predictor import DynamicPredictor
from src.core.recognition_controller import (
    RecognitionController
)
from src.core.word_builder import WordBuilder

from src.utils.config import WINDOW_NAME


def main():

    # ==========================================================
    # STARTUP UI
    # ==========================================================

    startup_ui = StartupUI()

    startup_ui.set_status(
        "Initializing EchoHands...",
        "Please wait..."
    )

    try:

        # ======================================================
        # Load manifest
        # ======================================================

        startup_ui.set_status(
            "Loading configuration...",
            "Reading model manifest"
        )

        manifest_path = Path(
            "manifest_test.json"
        )

        manifest = load_manifest(
            manifest_path
        )

        # ======================================================
        # Prepare model package
        # ======================================================

        model_dir = prepare_models(
            manifest,
            startup_ui=startup_ui
        )

        # ======================================================
        # Models verified
        # ======================================================

        startup_ui.set_status(
            "Models verified.",
            "SHA-256 integrity checks passed"
        )

        startup_ui.set_progress(
            1.0
        )

        # Give the user a moment to see
        # the successful verification.

        cv2.waitKey(300)

        # ======================================================
        # Model paths
        # ======================================================

        static_model_path = (
            model_dir
            / "random_forest.pkl"
        )

        static_encoder_path = (
            model_dir
            / "label_encoder.pkl"
        )

        dynamic_model_path = (
            model_dir
            / "dynamic_lstm.keras"
        )

        dynamic_encoder_path = (
            model_dir
            / "dynamic_label_encoder.npy"
        )

        # ======================================================
        # Initialize recognition engine
        # ======================================================

        startup_ui.set_status(
            "Initializing recognition engine...",
            "Loading AI models"
        )

        static_predictor = Predictor(
            model_path=static_model_path,
            encoder_path=static_encoder_path,
        )

        dynamic_predictor = DynamicPredictor(
            model_path=dynamic_model_path,
            encoder_path=dynamic_encoder_path,
        )

        controller = RecognitionController(
            static_predictor,
            dynamic_predictor,
            static_confidence_threshold=0.60,
        )

        word_builder = WordBuilder()

        # ======================================================
        # Initialize camera and vision components
        # ======================================================

        startup_ui.set_status(
            "Starting camera...",
            "Initializing hand recognition"
        )

        camera = Camera()

        detector = HandDetector()

        processor = LandmarkProcessor()

        # ======================================================
        # Startup complete
        # ======================================================

        startup_ui.complete()

        cv2.waitKey(700)

        startup_ui.close()

        # ======================================================
        # Display variables
        # ======================================================

        last_prediction = "None"

        last_confidence = 0.0

        # ======================================================
        # Gesture emission protection
        # ======================================================

        gesture_consumed = False

        previous_mode = controller.NONE

        # ======================================================
        # Hand-entry protection
        # ======================================================

        hand_was_present = False

        waiting_for_hand_initialization = False

        recognition_ready = False

        # ======================================================
        # Space key tracking
        # ======================================================

        last_space_time = 0.0

        double_space_interval = 0.5

        # ======================================================
        # Start camera
        # ======================================================

        camera.start()

        print(
            "\n========== Sign Language Recognition ==========\n"
        )

        print(
            "Static gestures : A-Y + 0-9"
        )

        print(
            "Dynamic gestures: J / Z"
        )

        print(
            "Press SPACE to add a space."
        )

        print(
            "Press SPACE twice quickly to clear text."
        )

        print(
            "Press BACKSPACE to remove last character."
        )

        print(
            "Press 'Q' to exit.\n"
        )

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

                hand_present = (
                    features is not None
                )

                if (
                    hand_present
                    and not hand_was_present
                ):

                    waiting_for_hand_initialization = (
                        True
                    )

                    recognition_ready = False

                    gesture_consumed = True

                elif (
                    not hand_present
                    and hand_was_present
                ):

                    waiting_for_hand_initialization = (
                        False
                    )

                    recognition_ready = False

                    gesture_consumed = False

                hand_was_present = (
                    hand_present
                )

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

                    waiting_for_hand_initialization = (
                        False
                    )

                    recognition_ready = True

                    gesture_consumed = True

                # ==========================================
                # Reset gesture permission only when
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

                    last_prediction = (
                        prediction
                    )

                    last_confidence = (
                        confidence
                    )

                # ==========================================
                # Update prediction display
                # ==========================================

                elif prediction is not None:

                    last_prediction = (
                        prediction
                    )

                    last_confidence = (
                        confidence
                    )

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
                # User-friendly status
                # ==========================================

                if not hand_present:

                    status = (
                        "No Hand Detected"
                    )

                    display_confidence = None

                elif (
                    waiting_for_hand_initialization
                    or mode == controller.INITIALIZING
                ):

                    status = (
                        "Initializing"
                    )

                    display_confidence = None

                elif (
                    recognition_ready
                    and mode == controller.NONE
                    and gesture_consumed
                ):

                    status = (
                        "Gesture Locked"
                    )

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
                # UI panel
                # ==========================================

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (250, 85),
                    (55, 55, 55),
                    -1,
                )

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
                # ==========================================

                current_text = (
                    word_builder.get_text()
                )

                if current_text:

                    display_text = (
                        current_text
                    )

                elif last_prediction != "None":

                    display_text = (
                        last_prediction
                    )

                else:

                    display_text = ""

                # ==========================================
                # Bottom-center text
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
                    ), baseline = (
                        cv2.getTextSize(
                            display_text,
                            font,
                            font_scale,
                            thickness,
                        )
                    )

                    frame_height, frame_width = (
                        frame.shape[:2]
                    )

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
                    # Dark outline
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
                    # White text
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

                # ==========================================
                # Quit
                # ==========================================

                if key in [
                    ord("q"),
                    ord("Q"),
                ]:

                    break

                # ==========================================
                # SPACE
                # ==========================================

                elif key == 32:

                    current_time = (
                        time.time()
                    )

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

                        last_space_time = (
                            current_time
                        )

                # ==========================================
                # BACKSPACE
                # ==========================================

                elif key in [
                    8,
                    127,
                ]:

                    word_builder.backspace()

                    last_space_time = 0.0

                # ==========================================
                # Other key
                # ==========================================

                elif key != 255:

                    last_space_time = 0.0

        finally:

            detector.close()

            camera.stop()

            cv2.destroyAllWindows()

    except Exception as error:

        # ==================================================
        # Startup/runtime failure
        # ==================================================

        startup_ui.set_status(
            "EchoHands could not start.",
            str(error)
        )

        startup_ui.set_progress(
            0.0
        )

        print(
            "\nEchoHands startup failed:"
        )

        print(
            error
        )

        input(
            "\nPress ENTER to exit..."
        )

        startup_ui.close()

        raise


if __name__ == "__main__":

    main()