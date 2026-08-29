from pathlib import Path

import os
import json
import hashlib
import re
import urllib.parse
import urllib.request
import time
from datetime import datetime, timezone


APP_NAME = "EchoHands"


# ==========================================================
# Cache
# ==========================================================

def get_cache_root() -> Path:

    if os.name == "nt":

        base_dir = Path(
            os.environ.get(
                "LOCALAPPDATA",
                Path.home()
                / "AppData"
                / "Local"
            )
        )

        return (
            base_dir
            / APP_NAME
            / "models"
        )

    return (
        Path.home()
        / ".cache"
        / APP_NAME
        / "models"
    )


def ensure_cache_root() -> Path:

    cache_root = get_cache_root()

    cache_root.mkdir(
        parents=True,
        exist_ok=True
    )

    return cache_root


def get_version_dir(
    version: str
) -> Path:

    return (
        get_cache_root()
        / version
    )


def get_cache_state_path() -> Path:

    return (
        get_cache_root()
        / "cache_state.json"
    )


def load_cache_state() -> dict:

    state_path = (
        get_cache_state_path()
    )

    if not state_path.exists():

        return {}

    try:

        with state_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {}


def save_cache_state(
    state: dict
) -> None:

    cache_root = (
        ensure_cache_root()
    )

    state_path = (
        cache_root
        / "cache_state.json"
    )

    with state_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            indent=4
        )


# ==========================================================
# Manifest
# ==========================================================

def load_manifest(
    manifest_path: Path
) -> dict:

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest file not found: "
            f"{manifest_path}"
        )

    try:

        with manifest_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            manifest = json.load(file)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Manifest contains invalid JSON: "
            f"{manifest_path}"
        ) from error

    if not isinstance(
        manifest,
        dict
    ):

        raise ValueError(
            "Manifest root must be "
            "a JSON object."
        )

    return manifest


def validate_manifest(
    manifest: dict
) -> None:

    required_fields = [
        "manifest_version",
        "model_version",
        "published_at",
        "expires_at",
        "models",
    ]

    for field in required_fields:

        if field not in manifest:

            raise ValueError(
                f"Manifest is missing "
                f"required field: {field}"
            )

    if not isinstance(
        manifest["models"],
        dict
    ):

        raise ValueError(
            "Manifest 'models' must "
            "be an object."
        )

    required_models = [
        "random_forest",
        "label_encoder",
        "dynamic_lstm",
        "dynamic_label_encoder",
    ]

    for model_name in required_models:

        if model_name not in (
            manifest["models"]
        ):

            raise ValueError(
                f"Manifest is missing "
                f"required model: "
                f"{model_name}"
            )

        model_info = (
            manifest["models"][
                model_name
            ]
        )

        required_model_fields = [
            "filename",
            "file_id",
            "url",
            "sha256",
            "size_bytes",
        ]

        for field in required_model_fields:

            if field not in model_info:

                raise ValueError(
                    f"Model '{model_name}' "
                    f"is missing field: "
                    f"{field}"
                )


# ==========================================================
# Google Drive download
# ==========================================================

def download_from_google_drive(
    file_id: str,
    destination: Path,
    progress_callback=None,
    status_callback=None,
    max_retries: int = 3,
) -> None:
    """
    Stream a Google Drive file to disk.

    progress_callback(downloaded_bytes, total_bytes) is called as
    chunks arrive. status_callback(message) is used for the small
    periods where Google Drive is preparing/confirming the download.
    """
    base_url = "https://drive.google.com/uc"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    opener = urllib.request.build_opener()

    def notify(message):
        if status_callback is not None:
            status_callback(message)

    params = urllib.parse.urlencode({
        "export": "download",
        "id": file_id,
    })

    initial_url = f"{base_url}?{params}"

    # ----------------------------------------------------------
    # Prepare the Google Drive download URL.
    # ----------------------------------------------------------

    notify(
        "Connecting to the model server. "
        "The first connection may take a few seconds."
    )

    request = urllib.request.Request(
        initial_url,
        headers=headers,
    )

    try:
        with opener.open(
            request,
            timeout=60,
        ) as response:

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
            )

            if "text/html" not in content_type.lower():
                download_url = initial_url

            else:
                notify(
                    "Google Drive is preparing the model download. "
                    "This short pause is expected for large files."
                )

                html = response.read().decode(
                    "utf-8",
                    errors="ignore",
                )

                confirm_match = re.search(
                    r'name="confirm"\s+value="([^"]+)"',
                    html,
                )

                uuid_match = re.search(
                    r'name="uuid"\s+value="([^"]+)"',
                    html,
                )

                if not confirm_match:
                    raise RuntimeError(
                        "Google Drive confirmation token not found."
                    )

                if not uuid_match:
                    raise RuntimeError(
                        "Google Drive confirmation UUID not found."
                    )

                download_params = urllib.parse.urlencode({
                    "id": file_id,
                    "export": "download",
                    "confirm": confirm_match.group(1),
                    "uuid": uuid_match.group(1),
                })

                download_url = (
                    "https://drive.usercontent.google.com/download?"
                    + download_params
                )

    except Exception as error:
        raise RuntimeError(
            "Unable to connect to the model server: "
            f"{error}"
        ) from error

    partial_path = Path(
        str(destination) + ".part"
    )

    chunk_size = 1024 * 1024
    last_error = None

    # ----------------------------------------------------------
    # Stream the actual file.
    # ----------------------------------------------------------

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            if partial_path.exists():
                partial_path.unlink()

            notify(
                "Starting the model download. "
                "Download time depends on your internet speed "
                "and connection quality."
            )

            request = urllib.request.Request(
                download_url,
                headers=headers,
            )

            with opener.open(
                request,
                timeout=60,
            ) as response:

                content_type = (
                    response.headers.get(
                        "Content-Type",
                        ""
                    )
                )

                if "text/html" in content_type.lower():
                    raise RuntimeError(
                        "Google Drive returned HTML instead "
                        "of the model file."
                    )

                content_length = (
                    response.headers.get(
                        "Content-Length"
                    )
                )

                try:
                    total_bytes = (
                        int(content_length)
                        if content_length
                        else None
                    )
                except ValueError:
                    total_bytes = None

                downloaded_bytes = 0

                with partial_path.open(
                    "wb"
                ) as file:

                    while True:

                        chunk = response.read(
                            chunk_size
                        )

                        if not chunk:
                            break

                        file.write(
                            chunk
                        )

                        downloaded_bytes += (
                            len(chunk)
                        )

                        if progress_callback is not None:
                            progress_callback(
                                downloaded_bytes,
                                total_bytes,
                            )

                if (
                    total_bytes is not None
                    and downloaded_bytes
                    != total_bytes
                ):
                    raise ConnectionError(
                        "Incomplete download: "
                        f"{downloaded_bytes} of "
                        f"{total_bytes} bytes received."
                    )

            partial_path.replace(
                destination
            )

            return

        except Exception as error:

            last_error = error

            try:
                if partial_path.exists():
                    partial_path.unlink()
            except OSError:
                pass

            if attempt >= max_retries:
                raise RuntimeError(
                    "Google Drive download failed after "
                    f"{max_retries} attempts: {error}"
                ) from error

            notify(
                "The download connection paused unexpectedly. "
                f"Retrying ({attempt}/{max_retries})..."
            )

            time.sleep(
                1.5 * attempt
            )

    raise RuntimeError(
        f"Download failed: {last_error}"
    )


# ==========================================================
# SHA-256
# ==========================================================

def calculate_sha256(
    file_path: Path
) -> str:

    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# ==========================================================
# Model validation
# ==========================================================

def validate_downloaded_model(
    file_path: Path,
    expected_size: int,
    expected_sha256: str
) -> None:

    if not file_path.exists():

        raise RuntimeError(
            f"Downloaded file does not "
            f"exist: {file_path}"
        )

    actual_size = (
        file_path.stat().st_size
    )

    if actual_size != expected_size:

        raise RuntimeError(
            f"File size mismatch for "
            f"{file_path.name}: "
            f"expected {expected_size}, "
            f"got {actual_size}"
        )

    actual_sha256 = (
        calculate_sha256(
            file_path
        )
    )

    if (
        actual_sha256.lower()
        != expected_sha256.lower()
    ):

        raise RuntimeError(
            f"SHA-256 mismatch for "
            f"{file_path.name}: "
            f"expected {expected_sha256}, "
            f"got {actual_sha256}"
        )


# ==========================================================
# Metadata
# ==========================================================

def save_package_metadata(
    version_dir: Path,
    version: str
) -> None:

    installed_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata = {
        "version": version,
        "installed_at": installed_at,
    }

    metadata_path = (
        version_dir
        / "cache_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


# ==========================================================
# Download package
# ==========================================================

def download_model_package(
    manifest: dict,
    startup=None
) -> Path:
    """
    Download and validate all models in the manifest.

    The overall progress bar is based on actual bytes across the
    complete package. It never resets to 25/50/75% between models.
    """
    validate_manifest(
        manifest
    )

    version = manifest["model_version"]

    version_dir = get_version_dir(
        version
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    models = manifest["models"]
    model_names = list(models.keys())

    total_models = len(model_names)

    # Total bytes of the complete model package.
    total_package_bytes = sum(
        int(models[name]["size_bytes"])
        for name in model_names
    )

    completed_package_bytes = 0
    completed_models = 0

    if startup is not None:
        startup.initialize_models(
            [
                models[name]["filename"]
                for name in model_names
            ]
        )

        startup.set_overall_progress(
            0,
            total_package_bytes,
            0,
            total_models
        )

    for model_name in model_names:

        model_info = models[model_name]

        filename = model_info["filename"]
        file_id = model_info["file_id"]
        expected_size = int(
            model_info["size_bytes"]
        )
        expected_sha256 = model_info["sha256"]

        destination = (
            version_dir / filename
        )

        if startup is not None:
            startup.set_model_status(
                filename,
                "connecting"
            )

            startup.set_task(
                "Preparing model download..."
            )

            startup.set_status(
                "Connecting to the model server. "
                "The first connection may take a few seconds."
            )

        print(
            f"Downloading {filename}..."
        )

        download_started = time.monotonic()
        last_progress_time = download_started
        last_progress_bytes = 0

        def report_status(message):
            if startup is not None:
                startup.set_status(
                    message
                )

        def report_download(
            downloaded,
            response_total
        ):
            nonlocal last_progress_time
            nonlocal last_progress_bytes

            now = time.monotonic()

            elapsed = (
                now - last_progress_time
            )

            if elapsed >= 0.2:
                speed = (
                    downloaded
                    - last_progress_bytes
                ) / elapsed

                last_progress_time = now
                last_progress_bytes = downloaded

            else:
                total_elapsed = max(
                    now - download_started,
                    0.001
                )

                speed = (
                    downloaded
                    / total_elapsed
                )

            # The manifest size is authoritative for package-wide
            # progress. This also keeps progress working if the
            # server omits Content-Length.
            current_total = (
                expected_size
            )

            overall_downloaded = (
                completed_package_bytes
                + min(
                    downloaded,
                    expected_size
                )
            )

            if startup is not None:
                startup.set_download_progress(
                    filename,
                    downloaded,
                    current_total,
                    overall_downloaded,
                    total_package_bytes,
                    speed
                )

        if startup is not None:
            startup.set_model_status(
                filename,
                "connecting"
            )

        download_from_google_drive(
            file_id,
            destination,
            progress_callback=report_download,
            status_callback=report_status,
        )

        if startup is not None:
            startup.set_model_status(
                filename,
                "downloaded"
            )

            startup.set_task(
                "Verifying model integrity..."
            )

            startup.set_status(
                f"Checking {filename} using its "
                "SHA-256 fingerprint. "
                "This can briefly pause while the file "
                "is being checked."
            )

            startup.set_model_status(
                filename,
                "verifying"
            )

        print(
            f"Validating {filename}..."
        )

        validate_downloaded_model(
            destination,
            expected_size,
            expected_sha256
        )

        print(
            f"{filename} validated successfully."
        )

        completed_package_bytes += (
            expected_size
        )

        completed_models += 1

        if startup is not None:
            startup.set_model_status(
                filename,
                "verified"
            )

            # Keep the bar at the actual cumulative byte position.
            startup.set_overall_progress(
                completed_package_bytes,
                total_package_bytes,
                completed_models,
                total_models
            )

    save_package_metadata(
        version_dir,
        version
    )

    if startup is not None:
        startup.set_overall_progress(
            total_package_bytes,
            total_package_bytes,
            total_models,
            total_models
        )

        startup.set_task(
            "Model package ready."
        )

        startup.set_status(
            "All recognition models have been "
            "downloaded and verified successfully."
        )

    print(
        "All models downloaded and "
        "validated successfully."
    )

    return version_dir


# ==========================================================
# Expiration
# ==========================================================

CACHE_EXPIRATION_DAYS = 90


def is_cache_expired(
    version_dir: Path
) -> bool:

    metadata_path = (
        version_dir
        / "cache_metadata.json"
    )

    if not metadata_path.exists():

        return True

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )

        installed_at = (
            metadata.get(
                "installed_at"
            )
        )

        if not installed_at:

            return True

        installed_time = (
            datetime.fromisoformat(
                installed_at
            )
        )

        if installed_time.tzinfo is None:

            installed_time = (
                installed_time.replace(
                    tzinfo=timezone.utc
                )
            )

        age = (
            datetime.now(
                timezone.utc
            )
            - installed_time
        )

        return (
            age.days
            >= CACHE_EXPIRATION_DAYS
        )

    except Exception:

        return True


# ==========================================================
# Cached package validation
# ==========================================================

def is_cached_package_valid(
    manifest: dict
) -> bool:

    validate_manifest(
        manifest
    )

    version = (
        manifest["model_version"]
    )

    version_dir = (
        get_version_dir(
            version
        )
    )

    if not version_dir.exists():

        return False

    if is_cache_expired(
        version_dir
    ):

        return False

    for model_name, model_info in (
        manifest["models"].items()
    ):

        filename = (
            model_info["filename"]
        )

        expected_size = (
            model_info["size_bytes"]
        )

        expected_sha256 = (
            model_info["sha256"]
        )

        model_path = (
            version_dir
            / filename
        )

        if not model_path.exists():

            return False

        try:

            validate_downloaded_model(
                model_path,
                expected_size,
                expected_sha256
            )

        except RuntimeError:

            return False

    return True


# ==========================================================
# Prepare models
# ==========================================================

def prepare_models(
    manifest: dict,
    startup=None
) -> Path:

    validate_manifest(
        manifest
    )

    version = (
        manifest["model_version"]
    )

    models = (
        manifest["models"]
    )

    if startup is not None:

        startup.initialize_models(
            [
                info["filename"]
                for info in models.values()
            ]
        )

        startup.set_task(
            "Checking recognition models..."
        )

        startup.set_status(
            "Looking for a valid local "
            "model package."
        )

    print(
        f"Checking EchoHands model package "
        f"{version}..."
    )

    if is_cached_package_valid(
        manifest
    ):

        print(
            "Cached model package is valid."
        )

        if startup is not None:

            for model_info in (
                models.values()
            ):

                startup.set_model_status(
                    model_info["filename"],
                    "cached"
                )

            total_cached_bytes = sum(
                int(info["size_bytes"])
                for info in models.values()
            )

            startup.set_overall_progress(
                total_cached_bytes,
                total_cached_bytes,
                len(models),
                len(models)
            )

            startup.set_task(
                "Recognition models are ready."
            )

            startup.set_status(
                "A valid model package was "
                "found on this computer. "
                "No download is required."
            )

        save_cache_state({
            "active_version": version,
            "status": "ready",
            "installed_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )
        })

        return get_version_dir(
            version
        )

    print(
        "Cached model package is "
        "missing or invalid."
    )

    if startup is not None:

        startup.set_task(
            "Downloading recognition models..."
        )

        startup.set_status(
            "This is a one-time setup. "
            "The models may take a few minutes "
            "to download and verify."
        )

    print(
        "Downloading model package..."
    )

    package_dir = (
        download_model_package(
            manifest,
            startup=startup
        )
    )

    save_cache_state({
        "active_version": version,
        "status": "ready",
        "installed_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    })

    if startup is not None:

        startup.set_task(
            "Model package ready."
        )

        startup.set_status(
            "All recognition models have "
            "been downloaded and verified."
        )

    print(
        "Model package prepared successfully."
    )

    return package_dir


# ==========================================================
# Cleanup expired versions
# ==========================================================

def cleanup_expired_versions(
    active_version=None
) -> None:

    cache_root = (
        get_cache_root()
    )

    if not cache_root.exists():

        return

    for version_dir in (
        cache_root.iterdir()
    ):

        if not version_dir.is_dir():

            continue

        if (
            active_version is not None
            and version_dir.name
            == active_version
        ):

            continue

        if is_cache_expired(
            version_dir
        ):

            print(
                "Removing expired "
                f"model cache: "
                f"{version_dir.name}"
            )

            import shutil

            shutil.rmtree(
                version_dir,
                ignore_errors=True
            )