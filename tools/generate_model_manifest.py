"""
EchoHands Model Manifest Generator

Owner/maintainer tool only.

Workflow:
    1. Put the four required model files inside a versioned Google Drive folder.
    2. Run this script with the model version and folder URL.
    3. The script verifies the required files, downloads temporary copies,
       calculates SHA-256 hashes and sizes, and generates model_manifest.json.
    4. If model_manifest.json already exists, the script asks whether to:
         Y = delete the old manifest
         N = keep it and rename it to manifest_old.json
         C = cancel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow


# ==========================================================
# Configuration
# ==========================================================

APP_NAME = "EchoHands"

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

REQUIRED_MODELS = {
    "random_forest": "random_forest.pkl",
    "label_encoder": "label_encoder.pkl",
    "dynamic_lstm": "dynamic_lstm.keras",
    "dynamic_label_encoder": "dynamic_label_encoder.npy",
}

MANIFEST_VERSION = 1
CACHE_EXPIRATION_DAYS = 90

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_CREDENTIALS = SCRIPT_DIR / "credentials.json"
DEFAULT_TOKEN = SCRIPT_DIR / "token.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "model_manifest.json"


# ==========================================================
# Version helpers
# ==========================================================

def normalize_version(version: str) -> str:
    """
    Normalize versions such as:
        1.0.0
        v1.0.0

    Returns:
        v1.0.0
    """
    version = str(version).strip()

    if version.startswith("v"):
        version = version[1:]

    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(
            "Invalid model version. Expected format like 1.0.0 or v1.0.0."
        )

    return f"v{version}"


# ==========================================================
# Google Drive helpers
# ==========================================================

def extract_folder_id(folder_url_or_id: str) -> str:
    """
    Extract a Google Drive folder ID from:
        - a normal Drive folder URL
        - a URL containing ?id=...
        - a raw folder ID
    """
    value = str(folder_url_or_id).strip()

    # Raw ID.
    if re.fullmatch(r"[-\w]{10,}", value):
        return value

    parsed = urlparse(value)

    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]

    match = re.search(r"/folders/([-\w]+)", parsed.path)
    if match:
        return match.group(1)

    raise ValueError(
        "Could not extract a Google Drive folder ID from the supplied value.\n"
        "Provide the folder URL in the form:\n"
        "https://drive.google.com/drive/folders/FOLDER_ID"
    )


def authenticate(credentials_path: Path, token_path: Path):
    """
    Authenticate the owner with Google Drive read-only access.
    """
    credentials_path = Path(credentials_path)
    token_path = Path(token_path)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Google OAuth credentials were not found:\n"
            f"{credentials_path}\n\n"
            "Download your OAuth client credentials as credentials.json "
            "and place them there."
        )

    creds = None

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(token_path),
                SCOPES,
            )
        except Exception:
            print("Existing token.json could not be read. Re-authorizing...")

    if creds and creds.expired and creds.refresh_token:
        print("Refreshing Google authorization...")
        creds.refresh(Request())

    if not creds or not creds.valid:
        print()
        print("Google authorization is required.")
        print("A browser window may open for the owner account.")
        print()

        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path),
            SCOPES,
        )

        creds = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        creds.to_json(),
        encoding="utf-8",
    )

    return build(
        "drive",
        "v3",
        credentials=creds,
    )


def get_folder_name(service, folder_id: str) -> str:
    metadata = service.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,trashed",
        supportsAllDrives=True,
    ).execute()

    if metadata.get("trashed"):
        raise RuntimeError("The supplied Google Drive folder is in the trash.")

    if metadata.get("mimeType") != "application/vnd.google-apps.folder":
        raise RuntimeError("The supplied ID is not a Google Drive folder.")

    return metadata["name"]


def list_folder_files(service, folder_id: str) -> list[dict]:
    """
    List direct children of the supplied Drive folder.
    """
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=(
                f"'{folder_id}' in parents "
                "and trashed = false"
            ),
            spaces="drive",
            fields=(
                "nextPageToken,"
                "files(id,name,mimeType,size)"
            ),
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files.extend(response.get("files", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


# ==========================================================
# Model validation
# ==========================================================

def validate_required_files(files: list[dict]) -> dict[str, dict]:
    """
    Verify that exactly the four required filenames can be found.

    Extra files are allowed, but are ignored.
    """
    by_name = {}

    for file_info in files:
        name = file_info.get("name")
        if name:
            by_name[name] = file_info

    missing = []
    found = {}

    for model_name, filename in REQUIRED_MODELS.items():
        file_info = by_name.get(filename)

        if file_info is None:
            missing.append(filename)
        else:
            found[model_name] = file_info

    if missing:
        print()
        print("ERROR: Required model files are missing:")
        for filename in missing:
            print(f"  - {filename}")

        raise RuntimeError(
            "The Drive folder does not contain all four required model files."
        )

    return found


# ==========================================================
# Hash / download helpers
# ==========================================================

def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def download_drive_file(
    service,
    file_id: str,
    destination: Path,
) -> None:
    """
    Download a Drive file to a temporary local path.

    This is used only by the owner-side manifest generator so that
    the exact SHA-256 and byte size can be recorded.
    """
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

    with destination.open("wb") as file:
        downloader = MediaIoBaseDownload(
            file,
            request,
            chunksize=1024 * 1024,
        )

        done = False

        while not done:
            _, done = downloader.next_chunk()


# ==========================================================
# Manifest generation
# ==========================================================

def build_manifest(
    service,
    model_version: str,
    folder_id: str,
    model_files: dict[str, dict],
) -> dict:
    """
    Download each required model temporarily, calculate its SHA-256
    and size, then build the complete manifest.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=CACHE_EXPIRATION_DAYS)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "model_version": model_version,
        "published_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "models": {},
    }

    with tempfile.TemporaryDirectory(
        prefix="echohands_manifest_"
    ) as temp_dir:
        temp_root = Path(temp_dir)

        total = len(model_files)

        for index, (model_name, file_info) in enumerate(
            model_files.items(),
            start=1,
        ):
            filename = REQUIRED_MODELS[model_name]
            file_id = file_info["id"]
            destination = temp_root / filename

            print(
                f"[{index}/{total}] Downloading {filename} "
                "temporarily for verification..."
            )

            download_drive_file(
                service,
                file_id,
                destination,
            )

            size_bytes = destination.stat().st_size
            sha256 = calculate_sha256(destination)

            manifest["models"][model_name] = {
                "filename": filename,
                "file_id": file_id,
                "url": (
                    "https://drive.google.com/uc"
                    f"?export=download&id={file_id}"
                ),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }

            print(
                f"      Size: {size_bytes:,} bytes"
            )
            print(
                f"      SHA-256: {sha256}"
            )

    return manifest


# ==========================================================
# Existing manifest handling
# ==========================================================

def handle_existing_manifest(output_path: Path) -> None:
    """
    Ask what to do when model_manifest.json already exists.

    Y = delete old manifest
    N = rename old manifest to manifest_old.json
    """
    if not output_path.exists():
        return

    print()
    print("=" * 60)
    print("Existing manifest found")
    print("=" * 60)
    print()
    print(f"Existing file: {output_path}")
    print()
    print(
        "Yes, Delete it (press Y)"
    )
    print(
        "No, keep it and rename it as manifest_old.json (press N)"
    )
    print()

    while True:
        choice = input("Choice [Y/N]: ").strip().lower()

        if choice == "y":
            output_path.unlink()
            print()
            print("Old model_manifest.json deleted.")
            return

        if choice == "n":
            backup_path = output_path.with_name("manifest_old.json")

            if backup_path.exists():
                print()
                print(
                    "ERROR: manifest_old.json already exists."
                )
                print(
                    "The old manifest will NOT be overwritten."
                )
                print()
                print(
                    "Remove or rename manifest_old.json and run "
                    "the generator again."
                )
                raise RuntimeError(
                    "Cannot create manifest_old.json because it already exists."
                )

            output_path.rename(backup_path)

            print()
            print(
                "Old model_manifest.json renamed to "
                "manifest_old.json."
            )
            return

        print("Please press Y or N.")


def write_manifest(
    manifest: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )
        file.write("\n")


# ==========================================================
# CLI
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an EchoHands model_manifest.json "
            "from a Google Drive model-version folder."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Model package version, e.g. 1.0.0 or v1.0.0",
    )

    parser.add_argument(
        "--folder-url",
        required=True,
        help="Google Drive URL of the versioned model folder.",
    )

    parser.add_argument(
        "--credentials",
        default=str(DEFAULT_CREDENTIALS),
        help=(
            "Path to Google OAuth credentials.json. "
            "Defaults to tools/credentials.json."
        ),
    )

    parser.add_argument(
        "--token",
        default=str(DEFAULT_TOKEN),
        help=(
            "Path to Google OAuth token.json. "
            "Defaults to tools/token.json."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output manifest path. "
            "Defaults to project-root/model_manifest.json."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        model_version = normalize_version(args.version)

        folder_id = extract_folder_id(
            args.folder_url
        )

        credentials_path = Path(
            args.credentials
        ).resolve()

        token_path = Path(
            args.token
        ).resolve()

        output_path = Path(
            args.output
        ).resolve()

        print()
        print("=" * 60)
        print("EchoHands Model Manifest Generator")
        print("=" * 60)
        print()
        print(f"Model version : {model_version}")
        print(f"Folder ID     : {folder_id}")
        print(f"Output        : {output_path}")
        print()

        service = authenticate(
            credentials_path,
            token_path,
        )

        folder_name = get_folder_name(
            service,
            folder_id,
        )

        print(f"Drive folder  : {folder_name}")
        print()

        print("Reading files from Google Drive...")
        files = list_folder_files(
            service,
            folder_id,
        )

        print(
            f"Found {len(files)} item(s) in the folder."
        )

        model_files = validate_required_files(
            files
        )

        print()
        print("All four required model files were found:")
        for model_name, filename in REQUIRED_MODELS.items():
            file_info = model_files[model_name]
            print(
                f"  [OK] {filename} "
                f"(Drive ID: {file_info['id']})"
            )

        print()
        print("Calculating SHA-256 hashes and file sizes...")
        print(
            "Temporary downloaded copies will be removed "
            "automatically after generation."
        )
        print()

        manifest = build_manifest(
            service=service,
            model_version=model_version,
            folder_id=folder_id,
            model_files=model_files,
        )

        print()
        print("=" * 60)
        print("Model files verified successfully.")
        print("=" * 60)

        # IMPORTANT:
        # Only ask about the old manifest after the new manifest
        # has been completely built and all four files have been
        # downloaded and hashed successfully.
        handle_existing_manifest(
            output_path
        )

        write_manifest(
            manifest,
            output_path,
        )

        print()
        print("=" * 60)
        print("Manifest generated successfully")
        print("=" * 60)
        print()
        print(f"File: {output_path}")
        print()
        print(
            f"Model package version: {model_version}"
        )
        print(
            "Required model files: 4"
        )
        print()
        print(
            "Review the manifest before committing/publishing it."
        )
        print()

        return 0

    except KeyboardInterrupt:
        print()
        print("Operation cancelled.")
        return 1

    except Exception as error:
        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print()
        print(error)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
