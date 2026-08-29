from pathlib import Path

from src.utils.model_manager import (
    load_manifest,
    prepare_models,
)


MANIFEST_PATH = Path("manifest_test.json")


print("Loading manifest...")

manifest = load_manifest(MANIFEST_PATH)

print("Preparing models...")

model_dir = prepare_models(manifest)

print()
print("Active model directory:")
print(model_dir)