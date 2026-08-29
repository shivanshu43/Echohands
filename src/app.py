import time
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import WINDOW_NAME
from src.utils.model_manager import (
    load_manifest,
    prepare_models,
)
from src.utils.startup_ui import StartupUI


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT / "model_manifest.json"
)

SIGN_GUIDE_DIR = (
    PROJECT_ROOT / "sign description"
)

SIGN_GUIDE_IMAGE = (
    SIGN_GUIDE_DIR / "sign letters.png"
)

# ==========================================================
# MANIFEST
# ==========================================================

def find_manifest():

    if MANIFEST_PATH.exists():
        return MANIFEST_PATH

    raise FileNotFoundError(
        "EchoHands production model manifest "
        "could not be found.\n\n"
        f"Expected location:\n"
        f"{MANIFEST_PATH}\n\n"
        "Please make sure model_manifest.json "
        "exists in the EchoHands project root."
    )


# ==========================================================
# SIGN GUIDE IMAGE DISCOVERY
# ==========================================================

def find_sign_images():

    if not SIGN_GUIDE_DIR.exists():
        return []

    if not SIGN_GUIDE_IMAGE.exists():
        return []

    return [SIGN_GUIDE_IMAGE]

# ==========================================================
# SIGN GUIDE
# ==========================================================

def create_sign_guide():
    """Create the sign-guide window without blocking recognition."""

    image_paths = find_sign_images()

    if not image_paths:
        print(
            "Sign guide: image not found.\n"
            f"Expected:\n{SIGN_GUIDE_IMAGE}"
        )
        return False

    image = cv2.imread(str(image_paths[0]))

    if image is None:
        print(
            "Sign guide: unable to load image.\n"
            f"File:\n{SIGN_GUIDE_IMAGE}"
        )
        return False

    guide_window = "EchoHands - Sign Guide"

    cv2.namedWindow(guide_window, cv2.WINDOW_NORMAL)

    screen_width, screen_height = get_screen_size()

    # Approximately the same visual proportion as the requested
    # fullscreen screenshot, while preserving the image aspect ratio.
    target_width = max(400, int(screen_width * 0.24))
    target_height = max(300, int(screen_height * 0.38))

    image_height, image_width = image.shape[:2]

    if image_width <= 0 or image_height <= 0:
        cv2.destroyWindow(guide_window)
        return False

    scale = min(
        target_width / image_width,
        target_height / image_height,
    )

    display_width = max(1, int(image_width * scale))
    display_height = max(1, int(image_height * scale))

    display = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )

    header_height = 50

    canvas = cv2.copyMakeBorder(
        display,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )

    cv2.putText(
        canvas,
        "SIGN GUIDE",
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "G / Q / ESC to close",
        (14, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(guide_window, canvas)
    cv2.resizeWindow(guide_window, canvas.shape[1], canvas.shape[0])

    guide_x = max(0, screen_width - canvas.shape[1] - 35)
    guide_y = 35

    try:
        cv2.moveWindow(guide_window, guide_x, guide_y)
    except Exception:
        pass

    return True


def close_sign_guide():
    """Safely close the sign-guide window."""
    try:
        cv2.destroyWindow("EchoHands - Sign Guide")
    except Exception:
        pass


def sign_guide_is_open():
    """Return True while the guide window is visible."""
    try:
        return (
            cv2.getWindowProperty(
                "EchoHands - Sign Guide",
                cv2.WND_PROP_VISIBLE,
            ) >= 1
        )
    except Exception:
        return False


# ==========================================================
# SCREEN SIZE
# ==========================================================

def get_screen_size():

    try:

        import ctypes

        user32 = ctypes.windll.user32

        width = (
            user32.GetSystemMetrics(0)
        )

        height = (
            user32.GetSystemMetrics(1)
        )

        if (
            width > 0
            and height > 0
        ):

            return (
                width,
                height,
            )

    except Exception:
        pass

    return (
        1536,
        864,
    )


# ==========================================================
# WINDOW SIZE
# ==========================================================

def calculate_window_size():

    screen_width, screen_height = (
        get_screen_size()
    )

    # ------------------------------------------------------
    # Keep the application comfortable without occupying
    # the entire desktop.
    # ------------------------------------------------------

    target_width = int(
        screen_width * 0.75
    )

    max_width = int(
        screen_width * 0.80
    )

    max_height = int(
        screen_height * 0.78
    )

    target_width = min(
        target_width,
        max_width,
    )

    # Maintain 16:9.
    target_height = int(
        target_width * 9 / 16
    )

    if target_height > max_height:

        target_height = max_height

        target_width = int(
            target_height * 16 / 9
        )

    return (
        target_width,
        target_height,
    )


# ==========================================================
# ASPECT-RATIO SAFE RESIZE
# ==========================================================

def resize_frame_preserve_aspect(
    frame,
    target_width,
    target_height,
):

    if frame is None:
        return None

    source_height, source_width = (
        frame.shape[:2]
    )

    if (
        source_width <= 0
        or source_height <= 0
    ):

        return frame

    # ------------------------------------------------------
    # Do not distort the camera image.
    # ------------------------------------------------------

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(
        1,
        int(source_width * scale),
    )

    resized_height = max(
        1,
        int(source_height * scale),
    )

    # ------------------------------------------------------
    # Prefer INTER_AREA when shrinking.
    # ------------------------------------------------------

    if scale < 1.0:

        interpolation = (
            cv2.INTER_AREA
        )

    else:

        interpolation = (
            cv2.INTER_LINEAR
        )

    resized = cv2.resize(
        frame,
        (
            resized_width,
            resized_height,
        ),
        interpolation=interpolation,
    )

    # ------------------------------------------------------
    # Letterbox.
    # ------------------------------------------------------

    canvas = np.zeros(
        (
            target_height,
            target_width,
            3,
        ),
        dtype=frame.dtype,
    )

    x_offset = (
        target_width
        - resized_width
    ) // 2

    y_offset = (
        target_height
        - resized_height
    ) // 2

    canvas[
        y_offset:
        y_offset + resized_height,
        x_offset:
        x_offset + resized_width,
    ] = resized

    return canvas


# ==========================================================
# DISPLAY GEOMETRY
# ==========================================================

def calculate_display_geometry(
    source_width,
    source_height,
    display_width,
    display_height,
):

    scale = min(
        display_width / source_width,
        display_height / source_height,
    )

    rendered_width = int(
        source_width * scale
    )

    rendered_height = int(
        source_height * scale
    )

    offset_x = (
        display_width
        - rendered_width
    ) // 2

    offset_y = (
        display_height
        - rendered_height
    ) // 2

    return (
        scale,
        offset_x,
        offset_y,
        rendered_width,
        rendered_height,
    )


# ==========================================================
# SIGN GUIDE BUTTON GEOMETRY
# ==========================================================

def get_info_button_geometry(
    frame_width,
):

    radius = 20

    center_x = (
        frame_width - 38
    )

    center_y = 38

    return (
        center_x,
        center_y,
        radius,
    )


# ==========================================================
# DRAW INFO BUTTON
# ==========================================================

def draw_info_button(
    frame,
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    (
        center_x,
        center_y,
        radius,
    ) = get_info_button_geometry(
        frame_width
    )

    # ------------------------------------------------------
    # Background
    # ------------------------------------------------------

    overlay = frame.copy()

    cv2.circle(
        overlay,
        (
            center_x,
            center_y,
        ),
        radius,
        (
            45,
            45,
            45,
        ),
        -1,
        cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        0.82,
        frame,
        0.18,
        0,
        frame,
    )

    # ------------------------------------------------------
    # Border
    # ------------------------------------------------------

    cv2.circle(
        frame,
        (
            center_x,
            center_y,
        ),
        radius,
        (
            155,
            155,
            155,
        ),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------
    # "i"
    # ------------------------------------------------------

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    text = "i"

    (
        text_width,
        text_height,
    ), _ = cv2.getTextSize(
        text,
        font,
        0.65,
        2,
    )

    text_x = (
        center_x
        - text_width // 2
    )

    text_y = (
        center_y
        + text_height // 2
        + 1
    )

    cv2.putText(
        frame,
        text,
        (
            text_x,
            text_y,
        ),
        font,
        0.65,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return frame


# ==========================================================
# INFO BUTTON CLICK TEST
# ==========================================================

def info_button_clicked(
    display_x,
    display_y,
    source_width,
    source_height,
    display_width,
    display_height,
):

    (
        scale,
        offset_x,
        offset_y,
        rendered_width,
        rendered_height,
    ) = calculate_display_geometry(
        source_width,
        source_height,
        display_width,
        display_height,
    )

    # ------------------------------------------------------
    # Convert displayed mouse coordinates back into
    # source-frame coordinates.
    # ------------------------------------------------------

    source_x = (
        display_x
        - offset_x
    ) / scale

    source_y = (
        display_y
        - offset_y
    ) / scale

    (
        center_x,
        center_y,
        radius,
    ) = get_info_button_geometry(
        source_width
    )

    hit_radius = (
        radius + 8
    )

    distance_squared = (
        (source_x - center_x) ** 2
        + (source_y - center_y) ** 2
    )

    return (
        distance_squared
        <= hit_radius ** 2
    )


# ==========================================================
# BOTTOM CONTROL BAR
# ==========================================================

def draw_control_bar(
    frame,
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    # ------------------------------------------------------
    # Compact bar
    # ------------------------------------------------------

    bar_height = 38

    bar_y2 = (
        frame_height - 10
    )

    bar_y1 = (
        bar_y2 - bar_height
    )

    bar_x1 = 18

    bar_x2 = (
        frame_width - 18
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            bar_x1,
            bar_y1,
        ),
        (
            bar_x2,
            bar_y2,
        ),
        (
            30,
            30,
            30,
        ),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame,
    )

    # ------------------------------------------------------
    # Compact controls
    # ------------------------------------------------------

    controls = [
        (
            "SPACE",
            "Space",
            54,
        ),
        (
            "SPACE x2",
            "Clear",
            62,
        ),
        (
            "BACKSPACE",
            "Delete",
            74,
        ),
        (
            "G",
            "Guide",
            24,
        ),
        (
            "Q",
            "Exit",
            24,
        ),
    ]

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.30

    gap = 15

    control_sizes = []

    total_width = 0

    # ------------------------------------------------------
    # Calculate everything first.
    #
    # This guarantees Q cannot overflow.
    # ------------------------------------------------------

    for (
        key_text,
        action_text,
        key_width,
    ) in controls:

        (
            action_width,
            _,
        ), _ = cv2.getTextSize(
            action_text,
            font,
            font_scale,
            1,
        )

        width = (
            key_width
            + 6
            + action_width
        )

        control_sizes.append(
            (
                key_text,
                action_text,
                key_width,
                width,
            )
        )

        total_width += width

    total_width += (
        gap
        * (
            len(control_sizes) - 1
        )
    )

    available_width = (
        bar_x2
        - bar_x1
        - 20
    )

    # ------------------------------------------------------
    # Smaller spacing if necessary.
    # ------------------------------------------------------

    if total_width > available_width:

        gap = 9

        total_width = sum(
            item[3]
            for item in control_sizes
        ) + (
            gap
            * (
                len(control_sizes) - 1
            )
        )

    # ------------------------------------------------------
    # Final emergency scale reduction.
    # ------------------------------------------------------

    if total_width > available_width:

        font_scale = 0.27

        control_sizes = []

        total_width = 0

        for (
            key_text,
            action_text,
            key_width,
        ) in controls:

            (
                action_width,
                _,
            ), _ = cv2.getTextSize(
                action_text,
                font,
                font_scale,
                1,
            )

            width = (
                key_width
                + 5
                + action_width
            )

            control_sizes.append(
                (
                    key_text,
                    action_text,
                    key_width,
                    width,
                )
            )

            total_width += width

        total_width += (
            gap
            * (
                len(control_sizes) - 1
            )
        )

    # ------------------------------------------------------
    # Center.
    # ------------------------------------------------------

    current_x = max(
        bar_x1 + 10,
        int(
            (
                frame_width
                - total_width
            ) / 2
        ),
    )

    key_height = 22

    for index, (
        key_text,
        action_text,
        key_width,
        width,
    ) in enumerate(
        control_sizes
    ):

        key_y1 = (
            bar_y1
            + (
                bar_height
                - key_height
            ) // 2
        )

        key_y2 = (
            key_y1
            + key_height
        )

        # --------------------------------------------------
        # Key background
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (
                current_x,
                key_y1,
            ),
            (
                current_x + key_width,
                key_y2,
            ),
            (
                55,
                55,
                55,
            ),
            -1,
        )

        # --------------------------------------------------
        # Key border
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (
                current_x,
                key_y1,
            ),
            (
                current_x + key_width,
                key_y2,
            ),
            (
                95,
                95,
                95,
            ),
            1,
        )

        # --------------------------------------------------
        # Key label
        # --------------------------------------------------

        (
            key_text_width,
            key_text_height,
        ), _ = cv2.getTextSize(
            key_text,
            font,
            font_scale,
            1,
        )

        key_text_x = (
            current_x
            + (
                key_width
                - key_text_width
            ) // 2
        )

        key_text_y = (
            key_y1
            + (
                key_height
                + key_text_height
            ) // 2
        )

        cv2.putText(
            frame,
            key_text,
            (
                key_text_x,
                key_text_y,
            ),
            font,
            font_scale,
            (
                245,
                245,
                245,
            ),
            1,
            cv2.LINE_AA,
        )

        # --------------------------------------------------
        # Action text
        # --------------------------------------------------

        action_x = (
            current_x
            + key_width
            + 6
        )

        action_y = (
            key_y1
            + 16
        )

        cv2.putText(
            frame,
            action_text,
            (
                action_x,
                action_y,
            ),
            font,
            font_scale,
            (
                205,
                205,
                205,
            ),
            1,
            cv2.LINE_AA,
        )

        current_x += (
            width
            + gap
        )

    return frame


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # STARTUP UI
    # ======================================================

    startup = StartupUI()

    startup.start()

    def initialize_application():

        startup.set_task(
            "Starting EchoHands..."
        )

        startup.set_status(
            "Preparing the sign language recognition system."
        )

        startup.set_task(
            "Loading model configuration..."
        )

        startup.set_status(
            "Checking the EchoHands recognition model package."
        )

        manifest_path = (
            find_manifest()
        )

        manifest = (
            load_manifest(
                manifest_path
            )
        )

        startup.set_task(
            "Preparing recognition models..."
        )

        startup.set_status(
            "Checking the local cache first. "
            "Missing models will be downloaded and verified automatically."
        )

        model_directory = (
            prepare_models(
                manifest,
                startup=startup
            )
        )

        startup.set_task(
            "Loading recognition engine..."
        )

        startup.set_status(
            "Loading the verified recognition models. "
            "Please wait while the engine initializes."
        )

        from src.core.camera import Camera
        from src.core.hand_detector import HandDetector
        from src.core.landmark_processor import LandmarkProcessor
        from src.core.predictor import Predictor
        from src.core.dynamic_predictor import DynamicPredictor
        from src.core.recognition_controller import RecognitionController
        from src.core.word_builder import WordBuilder

        random_forest_path = (
            model_directory
            / manifest["models"]["random_forest"]["filename"]
        )

        label_encoder_path = (
            model_directory
            / manifest["models"]["label_encoder"]["filename"]
        )

        dynamic_lstm_path = (
            model_directory
            / manifest["models"]["dynamic_lstm"]["filename"]
        )

        dynamic_label_encoder_path = (
            model_directory
            / manifest["models"]["dynamic_label_encoder"]["filename"]
        )

        # --------------------------------------------------
        # Static predictor
        # --------------------------------------------------

        static_predictor = Predictor(
            model_path=str(
                random_forest_path
            ),
            encoder_path=str(
                label_encoder_path
            ),
        )

        # --------------------------------------------------
        # Dynamic predictor
        # --------------------------------------------------

        startup.set_task(
            "Loading dynamic gesture model..."
        )

        startup.set_status(
            "Initializing J / Z dynamic gesture recognition."
        )

        dynamic_predictor = DynamicPredictor(
            model_path=str(
                dynamic_lstm_path
            ),
            encoder_path=str(
                dynamic_label_encoder_path
            ),
        )

        # --------------------------------------------------
        # Recognition system
        # --------------------------------------------------

        startup.set_task(
            "Initializing recognition system..."
        )

        startup.set_status(
            "Preparing hand detection, landmark processing, "
            "and recognition control."
        )

        camera = Camera()

        detector = HandDetector()

        processor = LandmarkProcessor()

        controller = RecognitionController(
            static_predictor,
            dynamic_predictor,
            static_confidence_threshold=0.60,
        )

        word_builder = WordBuilder()

        # --------------------------------------------------
        # Camera
        # --------------------------------------------------

        startup.set_task(
            "Starting camera..."
        )

        startup.set_status(
            "Opening the webcam. Please allow camera access "
            "if Windows asks."
        )

        camera.start()

        return (
            manifest,
            model_directory,
            camera,
            detector,
            processor,
            controller,
            word_builder,
        )

    # ======================================================
    # INITIALIZATION
    # ======================================================

    try:

        (
            manifest,
            model_directory,
            camera,
            detector,
            processor,
            controller,
            word_builder,
        ) = startup.run_worker(
            initialize_application
        )

        startup.set_task(
            "EchoHands is ready."
        )

        startup.set_status(
            "All models are ready. Starting recognition..."
        )

        startup.finish(
            "EchoHands is ready."
        )

    except Exception:

        raise

    # ======================================================
    # STATE
    # ======================================================

    last_prediction = "None"

    last_confidence = 0.0

    gesture_consumed = False

    previous_mode = (
        controller.NONE
    )

    hand_was_present = False

    waiting_for_hand_initialization = False

    recognition_ready = False

    last_space_time = 0.0

    double_space_interval = 0.5

    # ------------------------------------------------------
    # Sign guide request
    # ------------------------------------------------------

    sign_guide_requested = False

    sign_guide_window_exists = False
    guide_open = False

    # ======================================================
    # WINDOW
    # ======================================================

    display_width, display_height = (
        calculate_window_size()
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        display_width,
        display_height
    )

    # ======================================================
    # MOUSE CALLBACK
    # ======================================================

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param,
    ):

        nonlocal sign_guide_requested

        if (
            event
            != cv2.EVENT_LBUTTONUP
        ):

            return

        if param is None:
            return

        source_width = (
            param["source_width"]
        )

        source_height = (
            param["source_height"]
        )

        display_width = (
            param["display_width"]
        )

        display_height = (
            param["display_height"]
        )

        if info_button_clicked(
            x,
            y,
            source_width,
            source_height,
            display_width,
            display_height,
        ):

            sign_guide_requested = True

    mouse_state = {
        "source_width": 0,
        "source_height": 0,
        "display_width": display_width,
        "display_height": display_height,
    }

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
        mouse_state,
    )

    # ======================================================
    # CAMERA LOOP
    # ======================================================

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
        "Press G or click the info button for sign guide."
    )

    print(
        "Press Q to exit.\n"
    )

    try:

        while True:

            # ==================================================
            # GET FRAME
            # ==================================================

            frame = (
                camera.get_frame()
            )

            if frame is None:

                print(
                    "Failed to capture frame."
                )

                break

            source_height, source_width = (
                frame.shape[:2]
            )

            # --------------------------------------------------
            # Keep the mouse callback state current.
            # The callback itself is registered once below.
            # --------------------------------------------------

            mouse_state["source_width"] = source_width
            mouse_state["source_height"] = source_height

            # ==================================================
            # DETECT HAND
            # ==================================================

            results = detector.detect(
                frame
            )

            # ==================================================
            # FEATURES
            # ==================================================

            features = (
                processor.extract_features(
                    results
                )
            )

            # ==================================================
            # HAND PRESENCE
            # ==================================================

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

                recognition_ready = (
                    False
                )

                gesture_consumed = (
                    True
                )

            elif (
                not hand_present
                and hand_was_present
            ):

                waiting_for_hand_initialization = (
                    False
                )

                recognition_ready = (
                    False
                )

                gesture_consumed = (
                    False
                )

            hand_was_present = (
                hand_present
            )

            # ==================================================
            # CONTROLLER
            # ==================================================

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

            # ==================================================
            # INITIALIZATION COMPLETE
            # ==================================================

            if (
                waiting_for_hand_initialization
                and mode
                == controller.STATIC
            ):

                waiting_for_hand_initialization = (
                    False
                )

                recognition_ready = (
                    True
                )

                gesture_consumed = (
                    True
                )

            # ==================================================
            # RESET GESTURE PERMISSION
            # ==================================================

            if (
                recognition_ready
                and mode
                == controller.NONE
                and previous_mode
                != controller.NONE
            ):

                gesture_consumed = (
                    False
                )

            # ==================================================
            # COMMIT GESTURE
            # ==================================================

            if (
                prediction is not None
                and recognition_ready
                and not waiting_for_hand_initialization
                and not gesture_consumed
            ):

                word_builder.add(
                    prediction
                )

                gesture_consumed = (
                    True
                )

                last_prediction = (
                    prediction
                )

                last_confidence = (
                    confidence
                )

            elif prediction is not None:

                last_prediction = (
                    prediction
                )

                last_confidence = (
                    confidence
                )

            previous_mode = (
                mode
            )

            # ==================================================
            # DRAW LANDMARKS
            # ==================================================

            frame = detector.draw(
                frame,
                results
            )

            # ==================================================
            # STATUS
            # ==================================================

            if not hand_present:

                status = (
                    "No Hand Detected"
                )

                display_confidence = (
                    None
                )

            elif (
                waiting_for_hand_initialization
                or mode
                == controller.INITIALIZING
            ):

                status = (
                    "Initializing"
                )

                display_confidence = (
                    None
                )

            elif (
                recognition_ready
                and mode
                == controller.NONE
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
                and mode
                == controller.STATIC
            ):

                status = (
                    "Ready"
                )

                display_confidence = (
                    last_confidence
                )

            elif (
                mode
                == controller.DYNAMIC_CANDIDATE
            ):

                status = (
                    "Detecting movement"
                )

                display_confidence = (
                    None
                )

            elif (
                mode
                == controller.DYNAMIC
            ):

                status = (
                    "Recognizing dynamic gesture"
                )

                display_confidence = (
                    None
                )

            else:

                status = (
                    "Recognizing"
                )

                display_confidence = (
                    last_confidence
                )

            # ==================================================
            # CONFIDENCE
            # ==================================================

            if display_confidence is None:

                confidence_text = (
                    "Confidence: --"
                )

            else:

                confidence_text = (
                    f"Confidence: "
                    f"{display_confidence * 100:.1f}%"
                )

            # ==================================================
            # PREDICTION
            # ==================================================

            if prediction is None:

                prediction_text = (
                    "Prediction: --"
                )

            else:

                prediction_text = (
                    f"Prediction: {prediction}"
                )

            # ==================================================
            # TOP LEFT PANEL
            # ==================================================

            overlay = frame.copy()

            # Smaller than previous version.
            panel_width = 305
            panel_height = 98

            cv2.rectangle(
                overlay,
                (
                    10,
                    10,
                ),
                (
                    panel_width,
                    panel_height,
                ),
                (
                    45,
                    45,
                    45,
                ),
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

            UI_FONT = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            UI_FONT_SCALE = 0.38

            UI_THICKNESS = 1

            UI_COLOR = (
                255,
                255,
                255,
            )

            # --------------------------------------------------
            # Mode
            # --------------------------------------------------

            cv2.putText(
                frame,
                f"Mode: {mode}",
                (
                    20,
                    29,
                ),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # --------------------------------------------------
            # Prediction
            # --------------------------------------------------

            cv2.putText(
                frame,
                prediction_text,
                (
                    20,
                    51,
                ),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            cv2.putText(
                frame,
                confidence_text,
                (
                    20,
                    73,
                ),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # --------------------------------------------------
            # Status
            # --------------------------------------------------

            cv2.putText(
                frame,
                f"Status: {status}",
                (
                    20,
                    93,
                ),
                UI_FONT,
                UI_FONT_SCALE,
                UI_COLOR,
                UI_THICKNESS,
                cv2.LINE_AA,
            )

            # ==================================================
            # CURRENT TEXT
            # ==================================================

            current_text = (
                word_builder.get_text()
            )

            if current_text:

                font = (
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                font_scale = 1.0

                thickness = 2

                (
                    text_width,
                    text_height,
                ), _ = cv2.getTextSize(
                    current_text,
                    font,
                    font_scale,
                    thickness,
                )

                frame_height, frame_width = (
                    frame.shape[:2]
                )

                text_x = int(
                    (
                        frame_width
                        - text_width
                    ) / 2
                )

                # Keep text above the control bar.
                text_y = (
                    frame_height
                    - 72
                )

                # --------------------------------------------------
                # Outline
                # --------------------------------------------------

                cv2.putText(
                    frame,
                    current_text,
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

                # --------------------------------------------------
                # Main text
                # --------------------------------------------------

                cv2.putText(
                    frame,
                    current_text,
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

            # ==================================================
            # INFO BUTTON
            # ==================================================

            frame = draw_info_button(
                frame
            )

            # ==================================================
            # CONTROL BAR
            # ==================================================

            frame = draw_control_bar(
                frame
            )

            # ==================================================
            # DISPLAY FRAME
            # ==================================================

            display_frame = (
                resize_frame_preserve_aspect(
                    frame,
                    display_width,
                    display_height,
                )
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame
            )

            # ==================================================
            # SIGN GUIDE CLICK
            # ==================================================

            if sign_guide_requested:

                sign_guide_requested = False

                if not sign_guide_window_exists:
                    guide_open = True
                    sign_guide_window_exists = create_sign_guide()
                    guide_open = False

            # --------------------------------------------------
            # Keep recognition running while the guide is open.
            # --------------------------------------------------

            if sign_guide_window_exists:

                if not sign_guide_is_open():
                    sign_guide_window_exists = False
                    guide_open = False

            # ==================================================
            # KEYBOARD
            # ==================================================

            key = (
                cv2.waitKeyEx(1)
            )

            if key == -1:

                key = 255

            else:

                key = (
                    key & 0xFF
                )

            # ==================================================
            # QUIT
            # ==================================================

            if key in [
                ord("q"),
                ord("Q"),
            ]:

                if sign_guide_window_exists:
                    close_sign_guide()
                    sign_guide_window_exists = False

                break

            # ==================================================
            # SIGN GUIDE
            # ==================================================

            elif key in [
                ord("g"),
                ord("G"),
            ]:

                if sign_guide_window_exists:

                    close_sign_guide()
                    sign_guide_window_exists = False
                    guide_open = False

                else:

                    guide_open = True
                    sign_guide_window_exists = create_sign_guide()
                    guide_open = False

            # ==================================================
            # SPACE
            # ==================================================

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

                    last_space_time = (
                        0.0
                    )

                else:

                    word_builder.space()

                    last_space_time = (
                        current_time
                    )

            # ==================================================
            # BACKSPACE
            # ==================================================

            elif key in [
                8,
                127,
            ]:

                word_builder.backspace()

                if not word_builder.get_text():

                    last_prediction = (
                        "None"
                    )

                    last_confidence = (
                        0.0
                    )

                last_space_time = (
                    0.0
                )

            # ==================================================
            # OTHER KEY
            # ==================================================

            elif key != 255:

                last_space_time = (
                    0.0
                )

    finally:

        detector.close()

        camera.stop()

        cv2.destroyAllWindows()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()