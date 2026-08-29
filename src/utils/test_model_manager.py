from pathlib import Path

from src.utils.model_manager import (
    download_from_google_drive,
    validate_downloaded_model,
)


FILE_ID = "1IRimkqzOuCgzl1PQ-ZxRu2R9DkThtFjy"

EXPECTED_SIZE = 136

EXPECTED_SHA256 = (
    "07a22b63a139601426c1d87c2816a851ab17f00335bd12d75c31c7af4894cf82"
)

OUTPUT_FILE = Path("test_model.npy")


print("Downloading model...")

download_from_google_drive(
    FILE_ID,
    OUTPUT_FILE
)

print("Download completed.")

print("Validating model...")

validate_downloaded_model(
    OUTPUT_FILE,
    EXPECTED_SIZE,
    EXPECTED_SHA256
)

print("Model validation successful.")
