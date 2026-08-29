from pathlib import Path
from datetime import datetime, timezone
import os
import json
import hashlib
import re
import urllib.parse
import urllib.request


APP_NAME = "EchoHands"

# Number of days after which an inactive cached model package
# becomes eligible for cleanup.
MODEL_CACHE_EXPIRATION_DAYS = 90


def get_cache_root() -> Path:
    """
    Return the platform-appropriate local cache directory
    for EchoHands.
    """

    if os.name == "nt":
        base_dir = Path(
            os.environ.get(
                "LOCALAPPDATA",
                Path.home() / "AppData" / "Local"
            )
        )

        return base_dir / APP_NAME / "models"

    return Path.home() / ".cache" / APP_NAME / "models"


def ensure_cache_root() -> Path:
    """
    Create the EchoHands model cache directory if it
    does not exist.
    """

    cache_root = get_cache_root()

    cache_root.mkdir(
        parents=True,
        exist_ok=True
    )

    return cache_root


def get_version_dir(version: str) -> Path:
    """
    Return the cache directory for a specific model version.
    """

    return get_cache_root() / version


def get_cache_state_path() -> Path:
    """
    Return the path of the global cache state file.
    """

    return get_cache_root() / "cache_state.json"


def load_cache_state() -> dict:
    """
    Load the current model cache state.

    Returns an empty dictionary if no state exists yet
    or if the state file cannot be read.
    """

    state_path = get_cache_state_path()

    if not state_path.exists():
        return {}

    try:
        with state_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_cache_state(state: dict) -> None:
    """
    Save the global model cache state.
    """

    cache_root = ensure_cache_root()

    state_path = cache_root / "cache_state.json"

    with state_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            indent=4
        )


def get_version_metadata_path(version: str) -> Path:
    """
    Return the metadata path for a specific model version.
    """

    return (
        get_version_dir(version)
        / "cache_metadata.json"
    )


def save_version_metadata(
    version: str,
    installed_at: str | None = None
) -> None:
    """
    Save metadata describing when a model version
    was installed.
    """

    version_dir = get_version_dir(version)

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if installed_at is None:
        installed_at = (
            datetime.now(timezone.utc)
            .isoformat()
        )

    metadata = {
        "version": version,
        "installed_at": installed_at
    }

    metadata_path = get_version_metadata_path(
        version
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


def load_version_metadata(version: str) -> dict:
    """
    Load metadata for a specific model version.

    Returns an empty dictionary if metadata does not exist
    or cannot be read.
    """

    metadata_path = get_version_metadata_path(
        version
    )

    if not metadata_path.exists():
        return {}

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        if not isinstance(metadata, dict):
            return {}

        return metadata

    except (json.JSONDecodeError, OSError):
        return {}


def load_manifest(manifest_path: Path) -> dict:
    """
    Load and parse a local manifest file.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}"
        )

    try:

        with manifest_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            manifest = json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Manifest contains invalid JSON: "
            f"{manifest_path}"
        ) from exc

    if not isinstance(manifest, dict):

        raise ValueError(
            "Manifest root must be a JSON object."
        )

    return manifest


def validate_manifest(manifest: dict) -> None:
    """
    Validate the basic structure of the EchoHands
    model manifest.
    """

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
                f"Manifest is missing required field: "
                f"{field}"
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

        if model_name not in manifest["models"]:

            raise ValueError(
                f"Manifest is missing required model: "
                f"{model_name}"
            )

        model_info = manifest["models"][model_name]

        if not isinstance(model_info, dict):

            raise ValueError(
                f"Model '{model_name}' must be an object."
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
                    f"Model '{model_name}' is missing "
                    f"field: {field}"
                )


def download_from_google_drive(
    file_id: str,
    destination: Path
) -> None:
    """
    Download a Google Drive file.

    Handles both normal downloads and Google's
    large-file virus-scan confirmation page.
    """

    base_url = "https://drive.google.com/uc"

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

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        data = response.read()

    # Normal direct download.
    if "text/html" not in content_type.lower():

        destination.write_bytes(data)

        return

    # Google returned an HTML confirmation page.
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
            "Google Drive confirmation token "
            "not found."
        )

    if not uuid_match:

        raise RuntimeError(
            "Google Drive confirmation UUID "
            "not found."
        )

    confirm = confirm_match.group(1)
    uuid = uuid_match.group(1)

    download_params = urllib.parse.urlencode({
        "id": file_id,
        "export": "download",
        "confirm": confirm,
        "uuid": uuid
    })

    download_url = (
        "https://drive.usercontent.google.com/download"
        f"?{download_params}"
    )

    download_request = urllib.request.Request(
        download_url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        download_request
    ) as response:

        final_content_type = response.headers.get(
            "Content-Type",
            ""
        )

        final_data = response.read()

    if "text/html" in final_content_type.lower():

        raise RuntimeError(
            "Google Drive returned HTML instead "
            "of the model."
        )

    destination.write_bytes(
        final_data
    )


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while chunk := file.read(
            1024 * 1024
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


def validate_downloaded_model(
    file_path: Path,
    expected_size: int,
    expected_sha256: str
) -> None:
    """
    Validate a downloaded model using its expected
    file size and SHA-256 hash.
    """

    if not file_path.exists():

        raise RuntimeError(
            f"Downloaded file does not exist: "
            f"{file_path}"
        )

    actual_size = file_path.stat().st_size

    if actual_size != expected_size:

        raise RuntimeError(
            f"File size mismatch for "
            f"{file_path.name}: "
            f"expected {expected_size}, "
            f"got {actual_size}"
        )

    actual_sha256 = calculate_sha256(
        file_path
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


def download_model_package(
    manifest: dict
) -> Path:
    """
    Download and validate all models described
    by the manifest.

    Returns the directory containing the
    validated model package.
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

    print(
        f"Preparing EchoHands model package "
        f"{version}..."
    )

    for model_name, model_info in (
        manifest["models"].items()
    ):

        filename = model_info["filename"]
        file_id = model_info["file_id"]

        expected_size = (
            model_info["size_bytes"]
        )

        expected_sha256 = (
            model_info["sha256"]
        )

        destination = (
            version_dir / filename
        )

        print(
            f"Downloading {filename}..."
        )

        download_from_google_drive(
            file_id,
            destination
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

    print(
        "All models downloaded and "
        "validated successfully."
    )

    # Save per-version installation metadata
    # only after every model has passed validation.
    save_version_metadata(
        version
    )

    return version_dir


def is_cached_package_valid(
    manifest: dict
) -> bool:
    """
    Check whether the model package described
    by the manifest already exists locally
    and passes validation.
    """

    validate_manifest(
        manifest
    )

    version = manifest["model_version"]

    version_dir = get_version_dir(
        version
    )

    if not version_dir.exists():
        return False

    for model_name, model_info in (
        manifest["models"].items()
    ):

        filename = model_info["filename"]

        expected_size = (
            model_info["size_bytes"]
        )

        expected_sha256 = (
            model_info["sha256"]
        )

        model_path = (
            version_dir / filename
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


def is_cache_expired(
    state: dict
) -> bool:
    """
    Determine whether a cached model package
    has exceeded the configured cache lifetime.

    Returns True when the timestamp is missing,
    invalid, or older than the configured
    expiration period.
    """

    installed_at = state.get(
        "installed_at"
    )

    if not installed_at:
        return True

    try:

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

    except ValueError:

        return True

    age = (
        datetime.now(timezone.utc)
        - installed_time
    )

    return (
        age.days
        >= MODEL_CACHE_EXPIRATION_DAYS
    )


def cleanup_expired_models(
    active_version: str
) -> None:
    """
    Remove expired inactive model versions
    from the local cache.

    The active version is NEVER deleted.

    Versions without valid metadata are left
    untouched so that cleanup does not
    accidentally remove an unknown package.
    """

    cache_root = get_cache_root()

    if not cache_root.exists():
        return

    for version_dir in cache_root.iterdir():

        if not version_dir.is_dir():
            continue

        version = version_dir.name

        # Never delete the currently active version.
        if version == active_version:
            continue

        metadata = load_version_metadata(
            version
        )

        if not metadata:
            continue

        if is_cache_expired(
            metadata
        ):

            print(
                f"Removing expired model cache: "
                f"{version}"
            )

            for file_path in (
                version_dir.iterdir()
            ):

                if file_path.is_file():

                    try:
                        file_path.unlink()

                    except OSError as exc:

                        print(
                            f"Could not remove "
                            f"{file_path.name}: "
                            f"{exc}"
                        )

            try:

                version_dir.rmdir()

            except OSError as exc:

                print(
                    f"Could not remove cache "
                    f"directory {version}: "
                    f"{exc}"
                )


def prepare_models(
    manifest: dict
) -> Path:
    """
    Prepare the required EchoHands model package.

    Reuses the local cache when a valid package
    exists.

    Downloads and validates the package when
    it is missing or invalid.

    Saves cache state after successful
    preparation.

    Cleans up expired inactive versions.

    Returns the directory containing the
    active model package.
    """

    validate_manifest(
        manifest
    )

    version = manifest["model_version"]

    print(
        f"Checking EchoHands model package "
        f"{version}..."
    )

    state = load_cache_state()

    if is_cached_package_valid(
        manifest
    ):

        print(
            "Cached model package is valid."
        )

        installed_at = state.get(
            "installed_at"
        )

        # If the global state doesn't have a
        # timestamp yet, try the version metadata.
        if not installed_at:

            metadata = load_version_metadata(
                version
            )

            installed_at = metadata.get(
                "installed_at"
            )

        # Existing cache migration:
        # If this is an older package that does
        # not have metadata, record the current
        # time as the point at which the package
        # was first recognized by this version
        # of the manager.
        if not installed_at:

            installed_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

        save_cache_state({
            "active_version": version,
            "status": "ready",
            "installed_at": installed_at
        })

        # Ensure this version also has metadata.
        if not load_version_metadata(
            version
        ):

            save_version_metadata(
                version,
                installed_at
            )

        # Clean expired inactive versions,
        # while protecting the active version.
        cleanup_expired_models(
            active_version=version
        )

        return get_version_dir(
            version
        )

    print(
        "Cached model package is missing "
        "or invalid."
    )

    print(
        "Downloading model package..."
    )

    package_dir = download_model_package(
        manifest
    )

    installed_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_cache_state({
        "active_version": version,
        "status": "ready",
        "installed_at": installed_at
    })

    # The package has been completely downloaded
    # and validated, so it is now safe to clean
    # old inactive versions.
    cleanup_expired_models(
        active_version=version
    )

    print(
        "Model package prepared successfully."
    )

    return package_dir