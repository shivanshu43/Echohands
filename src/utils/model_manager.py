from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import json
import hashlib
import re
import urllib.parse
import urllib.request
import shutil


APP_NAME = "EchoHands"

MODEL_CACHE_EXPIRATION_DAYS = 90


# ==========================================================
# CACHE PATHS
# ==========================================================


def get_cache_root() -> Path:
    """
    Return the platform-appropriate local cache directory
    for EchoHands.
    """

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

    cache_root = (
        get_cache_root()
    )

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


def get_version_metadata_path(
    version: str
) -> Path:

    return (
        get_version_dir(version)
        / "cache_metadata.json"
    )


# ==========================================================
# CACHE STATE
# ==========================================================


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
# VERSION METADATA
# ==========================================================


def save_version_metadata(
    version: str,
    installed_at: str
) -> None:

    version_dir = (
        get_version_dir(version)
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = (
        version_dir
        / "cache_metadata.json"
    )

    metadata = {
        "version": version,
        "installed_at": installed_at,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


def load_version_metadata(
    version: str
) -> dict:

    metadata_path = (
        get_version_metadata_path(
            version
        )
    )

    if not metadata_path.exists():

        return {}

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {}


# ==========================================================
# CACHE EXPIRATION
# ==========================================================


def is_cache_expired(
    version: str
) -> bool:

    metadata = (
        load_version_metadata(
            version
        )
    )

    installed_at = (
        metadata.get(
            "installed_at"
        )
    )

    if not installed_at:

        return True

    try:

        installed_time = (
            datetime.fromisoformat(
                installed_at
            )
        )

    except ValueError:

        return True

    if installed_time.tzinfo is None:

        installed_time = (
            installed_time.replace(
                tzinfo=timezone.utc
            )
        )

    expiration_time = (
        installed_time
        + timedelta(
            days=MODEL_CACHE_EXPIRATION_DAYS
        )
    )

    return (
        datetime.now(timezone.utc)
        >= expiration_time
    )


# ==========================================================
# CLEANUP
# ==========================================================


def cleanup_expired_models(
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

        version = (
            version_dir.name
        )

        if (
            active_version is not None
            and version == active_version
        ):

            continue

        if is_cache_expired(
            version
        ):

            print(
                f"Removing expired model cache: "
                f"{version}"
            )

            try:

                shutil.rmtree(
                    version_dir
                )

            except OSError as error:

                print(
                    f"Warning: failed to remove "
                    f"cache {version}: {error}"
                )


# ==========================================================
# MANIFEST
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

            manifest = json.load(
                file
            )

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
            "Manifest root must be a JSON object."
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
            "Manifest 'models' must be an object."
        )

    required_models = [
        "random_forest",
        "label_encoder",
        "dynamic_lstm",
        "dynamic_label_encoder",
    ]

    for model_name in required_models:

        if (
            model_name
            not in manifest["models"]
        ):

            raise ValueError(
                f"Manifest is missing "
                f"required model: {model_name}"
            )

        model_info = (
            manifest["models"][model_name]
        )

        if not isinstance(
            model_info,
            dict
        ):

            raise ValueError(
                f"Model '{model_name}' "
                f"must be an object."
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
                    f"is missing field: {field}"
                )


# ==========================================================
# GOOGLE DRIVE DOWNLOAD
# ==========================================================


def download_from_google_drive(
    file_id: str,
    destination: Path,
    startup_ui=None,
    model_name="",
    model_index=0,
    total_models=1,
) -> None:

    base_url = (
        "https://drive.google.com/uc"
    )

    params = urllib.parse.urlencode({
        "export": "download",
        "id": file_id
    })

    request = urllib.request.Request(
        f"{base_url}?{params}",
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        request
    ) as response:

        content_type = (
            response.headers.get(
                "Content-Type",
                ""
            )
        )

        if (
            "text/html"
            not in content_type.lower()
        ):

            _download_response(
                response,
                destination,
                startup_ui,
                model_name,
                model_index,
                total_models,
            )

            return

        data = response.read()

    html = data.decode(
        "utf-8",
        errors="ignore"
    )

    confirm_match = re.search(
        r'name="confirm"\s+value="([^"]+)"',
        html
    )

    uuid_match = re.search(
        r'name="uuid"\s+value="([^"]+)"',
        html
    )

    if not confirm_match:

        raise RuntimeError(
            "Google Drive confirmation "
            "token not found."
        )

    if not uuid_match:

        raise RuntimeError(
            "Google Drive confirmation UUID "
            "not found."
        )

    confirm = (
        confirm_match.group(1)
    )

    uuid = (
        uuid_match.group(1)
    )

    download_params = (
        urllib.parse.urlencode({
            "id": file_id,
            "export": "download",
            "confirm": confirm,
            "uuid": uuid
        })
    )

    download_url = (
        "https://drive.usercontent.google.com/download"
        f"?{download_params}"
    )

    download_request = (
        urllib.request.Request(
            download_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )
    )

    with urllib.request.urlopen(
        download_request
    ) as response:

        final_content_type = (
            response.headers.get(
                "Content-Type",
                ""
            )
        )

        if (
            "text/html"
            in final_content_type.lower()
        ):

            raise RuntimeError(
                "Google Drive returned HTML "
                "instead of the model."
            )

        _download_response(
            response,
            destination,
            startup_ui,
            model_name,
            model_index,
            total_models,
        )


# ==========================================================
# DOWNLOAD RESPONSE
# ==========================================================


def _download_response(
    response,
    destination: Path,
    startup_ui,
    model_name,
    model_index,
    total_models,
) -> None:

    total_size = (
        response.headers.get(
            "Content-Length"
        )
    )

    try:

        total_size = int(
            total_size
        )

    except (
        TypeError,
        ValueError
    ):

        total_size = 0

    downloaded = 0

    chunk_size = (
        1024 * 1024
    )

    with destination.open(
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

            downloaded += len(
                chunk
            )

            if total_size > 0:

                file_progress = (
                    downloaded
                    / total_size
                )

            else:

                file_progress = 0.0

            file_progress = max(
                0.0,
                min(
                    file_progress,
                    1.0
                )
            )

            package_progress = (
                (
                    model_index
                    + file_progress
                )
                / total_models
            )

            if startup_ui is not None:

                startup_ui.set_model_status(
                    model_name,
                    "Downloading",
                    file_progress
                )

                startup_ui.set_progress(
                    package_progress
                )


# ==========================================================
# SHA-256 WITH UI PROGRESS
# ==========================================================


def calculate_sha256(
    file_path: Path,
    startup_ui=None,
    model_name=""
) -> str:
    """
    Calculate SHA-256 while continuing to
    update the startup UI.

    This prevents the startup window from
    appearing frozen while large files are
    being verified.
    """

    sha256 = hashlib.sha256()

    total_size = (
        file_path.stat().st_size
    )

    processed = 0

    chunk_size = (
        1024 * 1024
    )

    with file_path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:

                break

            sha256.update(
                chunk
            )

            processed += len(
                chunk
            )

            if total_size > 0:

                progress = (
                    processed
                    / total_size
                )

            else:

                progress = 1.0

            progress = max(
                0.0,
                min(
                    progress,
                    1.0
                )
            )

            # ----------------------------------------------
            # Keep startup UI alive during hashing.
            # ----------------------------------------------

            if startup_ui is not None:

                startup_ui.set_model_status(
                    model_name,
                    "Verifying",
                    progress
                )

                startup_ui.set_status(
                    "Verifying model integrity...",
                    (
                        f"Checking {model_name}. "
                        f"This may take a moment for larger files."
                    )
                )

    return (
        sha256.hexdigest()
    )


# ==========================================================
# MODEL VALIDATION
# ==========================================================


def validate_downloaded_model(
    file_path: Path,
    expected_size: int,
    expected_sha256: str,
    startup_ui=None,
) -> None:

    if not file_path.exists():

        raise RuntimeError(
            f"Downloaded file does not exist: "
            f"{file_path}"
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
            file_path,
            startup_ui=startup_ui,
            model_name=file_path.name
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
# DOWNLOAD MODEL PACKAGE
# ==========================================================


def download_model_package(
    manifest: dict,
    startup_ui=None
) -> Path:

    validate_manifest(
        manifest
    )

    version = (
        manifest["model_version"]
    )

    version_dir = (
        get_version_dir(version)
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    models = (
        manifest["models"]
    )

    total_models = len(
        models
    )

    # ------------------------------------------------------
    # Initialize startup UI
    # ------------------------------------------------------

    if startup_ui is not None:

        startup_ui.initialize_models(
            [
                model_info["filename"]
                for model_info
                in models.values()
            ]
        )

        startup_ui.set_status(
            "Downloading recognition models...",
            (
                "EchoHands is downloading the models "
                "required for sign language recognition."
            )
        )

    print(
        f"Preparing EchoHands model package "
        f"{version}..."
    )

    # ------------------------------------------------------
    # Download each model
    # ------------------------------------------------------

    for index, (
        model_name,
        model_info
    ) in enumerate(
        models.items()
    ):

        filename = (
            model_info["filename"]
        )

        file_id = (
            model_info["file_id"]
        )

        expected_size = (
            model_info["size_bytes"]
        )

        expected_sha256 = (
            model_info["sha256"]
        )

        destination = (
            version_dir
            / filename
        )

        # --------------------------------------------------
        # Downloading
        # --------------------------------------------------

        if startup_ui is not None:

            startup_ui.set_model_status(
                filename,
                "Downloading",
                0.0
            )

            startup_ui.set_status(
                "Downloading recognition models...",
                (
                    f"Downloading model "
                    f"{index + 1} of "
                    f"{total_models}."
                )
            )

        print(
            f"Downloading {filename}..."
        )

        download_from_google_drive(
            file_id,
            destination,
            startup_ui=startup_ui,
            model_name=filename,
            model_index=index,
            total_models=total_models,
        )

        # --------------------------------------------------
        # Download finished
        # --------------------------------------------------

        if startup_ui is not None:

            startup_ui.set_model_status(
                filename,
                "Verifying",
                0.0
            )

            startup_ui.set_status(
                "Verifying model integrity...",
                (
                    f"The download of {filename} "
                    f"is complete. Checking the file."
                )
            )

        print(
            f"Validating {filename}..."
        )

        # --------------------------------------------------
        # Validate
        #
        # The UI remains alive inside calculate_sha256().
        # --------------------------------------------------

        validate_downloaded_model(
            destination,
            expected_size,
            expected_sha256,
            startup_ui=startup_ui
        )

        print(
            f"{filename} validated successfully."
        )

        # --------------------------------------------------
        # Verified
        # --------------------------------------------------

        if startup_ui is not None:

            startup_ui.set_model_status(
                filename,
                "Verified",
                1.0
            )

            startup_ui.set_progress(
                (
                    index + 1
                )
                / total_models
            )

            startup_ui.set_status(
                "Model verified.",
                (
                    f"{filename} was downloaded "
                    f"and verified successfully."
                )
            )

    # ------------------------------------------------------
    # Installation timestamp
    # ------------------------------------------------------

    installed_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_version_metadata(
        version,
        installed_at
    )

    print(
        "All models downloaded and "
        "validated successfully."
    )

    # ------------------------------------------------------
    # Final UI
    # ------------------------------------------------------

    if startup_ui is not None:

        startup_ui.set_progress(
            1.0
        )

        startup_ui.set_status(
            "Models verified.",
            (
                "All recognition models were "
                "downloaded and verified successfully."
            )
        )

    return version_dir


# ==========================================================
# CACHE VALIDATION
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
        get_version_dir(version)
    )

    if not version_dir.exists():

        return False

    if is_cache_expired(
        version
    ):

        return False

    for (
        model_name,
        model_info
    ) in manifest["models"].items():

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
# PREPARE MODELS
# ==========================================================


def prepare_models(
    manifest: dict,
    startup_ui=None
) -> Path:

    validate_manifest(
        manifest
    )

    version = (
        manifest["model_version"]
    )

    # ------------------------------------------------------
    # Initialize UI
    # ------------------------------------------------------

    if startup_ui is not None:

        startup_ui.initialize_models(
            [
                model_info["filename"]
                for model_info
                in manifest["models"].values()
            ]
        )

        startup_ui.set_status(
            "Checking model package...",
            (
                "Checking whether EchoHands already "
                "has the required recognition models."
            )
        )

    print(
        f"Checking EchoHands model package "
        f"{version}..."
    )

    # ======================================================
    # VALID CACHE
    # ======================================================

    if is_cached_package_valid(
        manifest
    ):

        print(
            "Cached model package is valid."
        )

        if startup_ui is not None:

            for model_info in (
                manifest["models"].values()
            ):

                startup_ui.set_model_status(
                    model_info["filename"],
                    "Verified",
                    1.0
                )

            startup_ui.set_progress(
                1.0
            )

            startup_ui.set_status(
                "Models verified.",
                (
                    "Using the previously downloaded "
                    "model package."
                )
            )

        installed_at = (
            load_version_metadata(
                version
            ).get(
                "installed_at"
            )
        )

        save_cache_state({
            "active_version": version,
            "status": "ready",
            "installed_at": installed_at,
        })

        cleanup_expired_models(
            active_version=version
        )

        return get_version_dir(
            version
        )

    # ======================================================
    # CACHE MISSING / INVALID
    # ======================================================

    print(
        "Cached model package is "
        "missing or invalid."
    )

    print(
        "Downloading model package..."
    )

    if startup_ui is not None:

        startup_ui.set_status(
            "Setting up EchoHands for first use...",
            (
                "The required recognition models "
                "need to be downloaded."
            )
        )

    # ------------------------------------------------------
    # Download
    # ------------------------------------------------------

    package_dir = (
        download_model_package(
            manifest,
            startup_ui=startup_ui
        )
    )

    # ------------------------------------------------------
    # Installation timestamp
    # ------------------------------------------------------

    installed_at = (
        load_version_metadata(
            version
        ).get(
            "installed_at"
        )
    )

    # ------------------------------------------------------
    # Save active cache state
    # ------------------------------------------------------

    save_cache_state({
        "active_version": version,
        "status": "ready",
        "installed_at": installed_at,
    })

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    cleanup_expired_models(
        active_version=version
    )

    print(
        "Model package prepared successfully."
    )

    return package_dir