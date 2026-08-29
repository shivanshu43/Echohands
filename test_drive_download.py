import re
import urllib.parse
import urllib.request


def download_from_google_drive(file_id: str, destination: str) -> None:
    """
    Download a Google Drive file, including large files that
    require the virus-scan confirmation step.
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

    with urllib.request.urlopen(request) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()

    # Direct file download.
    if "text/html" not in content_type.lower():
        with open(destination, "wb") as file:
            file.write(data)

        print("Downloaded directly.")
        return

    # Google returned the large-file confirmation page.
    html = data.decode("utf-8", errors="ignore")

    print("Google Drive returned a confirmation page.")

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
            "Could not find the Google Drive confirmation token."
        )

    if not uuid_match:
        raise RuntimeError(
            "Could not find the Google Drive UUID."
        )

    confirm = confirm_match.group(1)
    uuid = uuid_match.group(1)

    print("Confirmation token found.")
    print("UUID found.")

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

    with urllib.request.urlopen(download_request) as response:
        final_content_type = response.headers.get(
            "Content-Type",
            ""
        )

        final_data = response.read()

    if "text/html" in final_content_type.lower():
        raise RuntimeError(
            "Google Drive returned HTML instead of the model."
        )

    with open(destination, "wb") as file:
        file.write(final_data)

    print("Downloaded successfully after confirmation.")


if __name__ == "__main__":
    FILE_ID = "1Dq9dnB5Y5aF-Tv10pXeu5RN_Z1GWpZE_"
    OUTPUT_FILE = "drive_test_random_forest.pkl"

    download_from_google_drive(
        FILE_ID,
        OUTPUT_FILE
    )