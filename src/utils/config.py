from pathlib import Path


# ==========================================================
# Application
# ==========================================================

APP_NAME = "EchoHands"


# ==========================================================
# Camera
# ==========================================================

CAMERA_INDEX = 0

WINDOW_NAME = (
    "EchoHands - ASL Recognition"
)


# ==========================================================
# Dataset Generator
# ==========================================================

DATASET_WINDOW_NAME = (
    "Dataset Generator"
)


# ==========================================================
# Display
# ==========================================================

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720


# ==========================================================
# Model configuration
#
# These paths are retained for compatibility.
# The application itself uses the active model package
# returned by model_manager.py.
# ==========================================================

RANDOM_FOREST_MODEL_PATH = (
    "models/random_forest.pkl"
)

LABEL_ENCODER_PATH = (
    "models/label_encoder.pkl"
)


# ==========================================================
# Sign assistance
#
# Change this path ONLY if your sign image has a
# different filename/location.
# ==========================================================

SIGN_LETTERS_IMAGE_PATH = (
    Path("Assets")
    / "sign_guide"
    / "sign letters.png"
)