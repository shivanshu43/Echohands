import os
import subprocess
import sys
import tempfile
import time
import textwrap
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

SIGN_DESCRIPTION_FILE = (
    SIGN_GUIDE_DIR / "sign alphabet description.txt"
)

ASSETS_DIR = PROJECT_ROOT / "Assets"

SIGN_DESCRIPTION_ICON = (
    SIGN_GUIDE_DIR / "sign_description_icon.png"
)

VIDEO_TUTORIAL_ICON = (
    SIGN_GUIDE_DIR / "video_tutorial_icon.png"
)

# Replace this with the YouTube tutorial URL you want to ship
# with the public Beta release.
YOUTUBE_VIDEO_URL = "https://youtu.be/6_gXiBe9y9A?si=LckB1iLPO6Bpk-hQ"
_video_tutorial_process = None
_video_tutorial_status = ""
_video_tutorial_status_until = 0.0
_video_tutorial_ready_marker = None
_video_tutorial_opening_started = 0.0
_description_process = None

_description_lines = []
_description_scroll = 0
_description_max_scroll = 0

SIGN_GUIDE_WINDOW = "EchoHands - Recognition Help"
SIGN_DESCRIPTION_WINDOW = "EchoHands - Gesture Description"

# Mouse hitboxes used by the non-blocking sign-guide window.
_sign_guide_hitboxes = {}

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
# SIGN GUIDE / RECOGNITION HELP
# ==========================================================

def _wrap_text(text, width):
    """Wrap text for compact OpenCV display."""
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _find_guide_icon(candidates):
    """Find a guide icon in the sign-description or Assets folders."""
    search_dirs = (SIGN_GUIDE_DIR, ASSETS_DIR)

    for directory in search_dirs:
        for filename in candidates:
            path = directory / filename
            if path.exists():
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    return image

    return None


def _make_document_icon(size=78):
    """Create a clean fallback document icon for the text resource."""
    icon = np.zeros((size, size, 4), dtype=np.uint8)

    # Transparent background.
    icon[:, :, 3] = 0

    margin = 9
    x1, y1 = margin, margin
    x2, y2 = size - margin, size - margin

    # White document body.
    cv2.rectangle(
        icon,
        (x1, y1),
        (x2, y2),
        (235, 235, 235, 255),
        -1,
        cv2.LINE_AA,
    )

    # Folded corner.
    fold = 15
    pts = np.array(
        [
            [x2 - fold, y1],
            [x2, y1],
            [x2, y1 + fold],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(
        icon,
        [pts],
        (165, 165, 165, 255),
    )

    # Simple TXT mark.
    cv2.putText(
        icon,
        "TXT",
        (x1 + 5, y1 + 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (45, 45, 45, 255),
        1,
        cv2.LINE_AA,
    )

    return icon


def _load_guide_icon(path, fallback_text):
    """Load a guide icon from the requested path, if available."""
    if path.exists():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image

    return None


def _make_video_fallback_icon(size=78):
    """Create a clean fallback play icon when no video asset is present."""
    fallback = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(
        fallback,
        (size // 2, size // 2),
        31,
        (55, 55, 55, 255),
        -1,
        cv2.LINE_AA,
    )
    pts = np.array(
        [[31, 23], [31, 55], [55, 39]],
        dtype=np.int32,
    )
    cv2.fillPoly(
        fallback,
        [pts],
        (235, 235, 235, 255),
    )
    return fallback


def _resize_icon(image, size=72):
    """Resize an icon to a square display area."""
    if image is None:
        return None

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(size / width, size / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )


def _place_image(canvas, image, center_x, center_y):
    """Place BGR/BGRA image on a BGR canvas."""
    if image is None:
        return

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = x1 + width
    y2 = y1 + height

    canvas_height, canvas_width = canvas.shape[:2]

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = width - max(0, x2 - canvas_width)
    src_y2 = height - max(0, y2 - canvas_height)

    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    source = image[src_y1:src_y2, src_x1:src_x2]

    if source.shape[2] == 4:
        alpha = source[:, :, 3:4].astype(np.float32) / 255.0
        foreground = source[:, :, :3].astype(np.float32)
        background = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        blended = foreground * alpha + background * (1.0 - alpha)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
    else:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = source[:, :, :3]


def _draw_centered_text(canvas, text, y, font_scale, color, thickness=1):
    """Draw one centered line of text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    x = max(8, (canvas.shape[1] - width) // 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_centered_text_at_x(canvas, text, center_x, y, font_scale, color, thickness=1):
    """Draw one line centered around a specific x-coordinate."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, _), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    x = int(center_x - width / 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def create_sign_guide():
    """Create the non-blocking recognition-help window."""
    global _sign_guide_hitboxes

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

    screen_width, screen_height = get_screen_size()

    # Give the reference image most of the panel width while keeping
    # a small, consistent margin around it.
    guide_width = min(600, max(540, int(screen_width * 0.31)))
    guide_height = min(860, max(800, int(screen_height * 0.88)))

    canvas = np.full(
        (guide_height, guide_width, 3),
        30,
        dtype=np.uint8,
    )

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------
    cv2.rectangle(
        canvas,
        (0, 0),
        (guide_width - 1, 54),
        (38, 38, 38),
        -1,
    )

    cv2.putText(
        canvas,
        "RECOGNITION HELP",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "G / ESC to close",
        (16, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # ----------------------------------------------------------
    # Help message
    # ----------------------------------------------------------
    help_text = (
        "Having trouble getting a sign recognized? You may be "
        "experiencing low-confidence predictions because the "
        "gesture may not exactly match the intended sign."
    )

    threshold_text = (
        "EchoHands registers a letter only when model confidence "
        "is above 60%."
    )

    resource_text = (
        "If a sign is difficult to recognize, use the resources "
        "below to check the expected gesture and common mistakes."
    )

    y = 82
    text_width = 68

    for line in _wrap_text(help_text, text_width):
        _draw_centered_text(
            canvas,
            line,
            y,
            0.36,
            (225, 225, 225),
            1,
        )
        y += 18

    y += 5
    for line in _wrap_text(threshold_text, text_width):
        _draw_centered_text(
            canvas,
            line,
            y,
            0.36,
            (245, 245, 245),
            1,
        )
        y += 18

    y += 5
    for line in _wrap_text(resource_text, text_width):
        _draw_centered_text(
            canvas,
            line,
            y,
            0.33,
            (185, 185, 185),
            1,
        )
        y += 17

    # ----------------------------------------------------------
    # Sign reference image — nearly full panel width
    # ----------------------------------------------------------
    image_top = y + 10
    horizontal_margin = 18
    available_width = guide_width - (horizontal_margin * 2)
    image_height, image_width = image.shape[:2]

    scale = available_width / image_width
    display_width = max(1, int(image_width * scale))
    display_height = max(1, int(image_height * scale))

    display = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    image_x = (guide_width - display_width) // 2
    image_y = image_top

    cv2.rectangle(
        canvas,
        (image_x - 2, image_y - 2),
        (image_x + display_width + 1, image_y + display_height + 1),
        (75, 75, 75),
        1,
    )

    canvas[
        image_y:image_y + display_height,
        image_x:image_x + display_width,
    ] = display

    # ----------------------------------------------------------
    # Resource icons
    # ----------------------------------------------------------
    icon_slot = 82
    icon_y = image_y + display_height + 62

    left_center_x = guide_width // 4
    right_center_x = (guide_width * 3) // 4

    description_icon = _load_guide_icon(
        SIGN_DESCRIPTION_ICON,
        "TXT",
    )
    if description_icon is None:
        description_icon = _find_guide_icon(
            [
                "sign_description_icon.png",
                "gesture_description_icon.png",
                "txt_icon.png",
                "text_icon.png",
                "description_icon.png",
                "txt.png",
            ]
        )
    if description_icon is None:
        description_icon = _make_document_icon()

    video_icon = _load_guide_icon(
        VIDEO_TUTORIAL_ICON,
        "PLAY",
    )
    if video_icon is None:
        video_icon = _find_guide_icon(
            [
                "video_tutorial_icon.png",
                "youtube_icon.png",
                "youtube.png",
                "play_icon.png",
            ]
        )
    if video_icon is None:
        video_icon = _make_video_fallback_icon()

    description_icon = _resize_icon(description_icon, icon_slot)
    video_icon = _resize_icon(video_icon, icon_slot)

    _place_image(canvas, description_icon, left_center_x, icon_y)
    _place_image(canvas, video_icon, right_center_x, icon_y)

    # Smaller, consistent click rings so the icons do not look
    # oversized relative to the reference image.
    icon_radius = 46
    cv2.circle(
        canvas,
        (left_center_x, icon_y),
        icon_radius,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        (right_center_x, icon_y),
        icon_radius,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    # ----------------------------------------------------------
    # Resource labels — each label is centered under its own icon
    # ----------------------------------------------------------
    label_y = icon_y + 62

    description_lines = [
        "Avoid common mistakes",
        "for better confidence percentage",
    ]

    video_lines = [
        "Gesture performance",
        "video tutorial",
    ]

    for index, line in enumerate(description_lines):
        _draw_centered_text_at_x(
            canvas,
            line,
            left_center_x,
            label_y + index * 17,
            0.31,
            (210, 210, 210),
            1,
        )

    for index, line in enumerate(video_lines):
        _draw_centered_text_at_x(
            canvas,
            line,
            right_center_x,
            label_y + index * 17,
            0.31,
            (210, 210, 210),
            1,
        )

    # ----------------------------------------------------------
    # Tutorial status
    # ----------------------------------------------------------
    if _video_tutorial_status:
        _draw_centered_text(
            canvas,
            _video_tutorial_status,
            guide_height - 18,
            0.31,
            (205, 205, 205),
            1,
        )

    _sign_guide_hitboxes = {
        "description": (
            left_center_x - 62,
            icon_y - 62,
            left_center_x + 62,
            label_y + 34,
        ),
        "video": (
            right_center_x - 62,
            icon_y - 62,
            right_center_x + 62,
            label_y + 34,
        ),
    }

    cv2.namedWindow(
        SIGN_GUIDE_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        SIGN_GUIDE_WINDOW,
        sign_guide_mouse_callback,
    )

    cv2.imshow(
        SIGN_GUIDE_WINDOW,
        canvas,
    )

    cv2.resizeWindow(
        SIGN_GUIDE_WINDOW,
        guide_width,
        guide_height,
    )

    guide_x = max(
        0,
        screen_width - guide_width - 35,
    )
    guide_y = 35

    try:
        cv2.moveWindow(
            SIGN_GUIDE_WINDOW,
            guide_x,
            guide_y,
        )
    except Exception:
        pass

    return True

def close_sign_guide():
    """Safely close the recognition-help window and text resource."""
    global _sign_guide_hitboxes
    _sign_guide_hitboxes = {}

    try:
        cv2.destroyWindow(SIGN_GUIDE_WINDOW)
    except Exception:
        pass

    close_sign_description()


def sign_guide_is_open():
    """Return True while the recognition-help window is visible."""
    try:
        return (
            cv2.getWindowProperty(
                SIGN_GUIDE_WINDOW,
                cv2.WND_PROP_VISIBLE,
            ) >= 1
        )
    except Exception:
        return False


def _guide_hitbox_contains(x, y, box):
    x1, y1, x2, y2 = box
    return (
        x1 <= x <= x2
        and y1 <= y <= y2
    )


def create_sign_description():
    """Open the sign-alphabet description in a small scrollable window."""
    global _description_process

    if _description_process is not None:
        if _description_process.poll() is None:
            return True
        _description_process = None

    if not SIGN_DESCRIPTION_FILE.exists():
        print(
            "Sign alphabet description: file not found.\n"
            f"Expected:\n{SIGN_DESCRIPTION_FILE}"
        )
        return False

    # Use a separate Tkinter process so the text window has a real native
    # Text widget + Scrollbar and remains fully interactive without blocking
    # the OpenCV recognition loop.
    helper_code = """
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

file_path = Path(sys.argv[1])

try:
    text = file_path.read_text(encoding='utf-8')
except Exception as exc:
    text = f'Unable to read the sign alphabet description file.\\n\\n{exc}'

root = tk.Tk()
root.title('EchoHands - Sign Alphabet Description')
root.geometry('700x560')
root.minsize(520, 360)
root.configure(bg='#1e1e1e')

try:
    root.attributes('-topmost', False)
except Exception:
    pass

header = tk.Frame(root, bg='#262626', height=62)
header.pack(fill='x')
header.pack_propagate(False)

tk.Label(
    header,
    text='SIGN ALPHABET DESCRIPTION',
    font=('Segoe UI', 14),
    fg='#ffffff',
    bg='#262626',
    anchor='w',
).pack(fill='x', padx=18, pady=(10, 0))

tk.Label(
    header,
    text='Scroll to read  |  Close the window when finished',
    font=('Segoe UI', 9),
    fg='#aaaaaa',
    bg='#262626',
    anchor='w',
).pack(fill='x', padx=18, pady=(1, 0))

content = tk.Frame(root, bg='#1e1e1e')
content.pack(fill='both', expand=True, padx=14, pady=14)

scrollbar = ttk.Scrollbar(content, orient='vertical')
scrollbar.pack(side='right', fill='y')

text_widget = tk.Text(
    content,
    wrap='word',
    yscrollcommand=scrollbar.set,
    bg='#1e1e1e',
    fg='#e1e1e1',
    insertbackground='#ffffff',
    selectbackground='#4a4a4a',
    selectforeground='#ffffff',
    relief='flat',
    borderwidth=0,
    highlightthickness=0,
    font=('Segoe UI', 10),
    padx=8,
    pady=6,
)
text_widget.pack(side='left', fill='both', expand=True)
scrollbar.config(command=text_widget.yview)

text_widget.insert('1.0', text)

# Keep the document read-only without disabling the Text widget. A disabled
# Text widget can prevent normal focus/scroll interaction on some Windows Tk
# builds. Blocking edit keys preserves the native scrollbar and scrolling.
def block_edit(event):
    return 'break'

text_widget.bind('<KeyPress>', block_edit)
text_widget.bind('<Control-KeyPress>', block_edit)
text_widget.bind('<Control-Shift-KeyPress>', block_edit)

def on_mousewheel(event):
    if event.delta:
        steps = max(1, abs(int(event.delta / 120)))
        direction = -1 if event.delta > 0 else 1
        text_widget.yview_scroll(direction * steps, 'units')
    return 'break'

# Bind scrolling to the text, its container, and the root window. This makes
# the mouse wheel work even when the pointer is over the window background.
text_widget.bind('<MouseWheel>', on_mousewheel)
content.bind('<MouseWheel>', on_mousewheel)
root.bind('<MouseWheel>', on_mousewheel)

# Linux-style wheel events.
text_widget.bind('<Button-4>', lambda event: (text_widget.yview_scroll(-3, 'units'), 'break')[1])
text_widget.bind('<Button-5>', lambda event: (text_widget.yview_scroll(3, 'units'), 'break')[1])
content.bind('<Button-4>', lambda event: (text_widget.yview_scroll(-3, 'units'), 'break')[1])
content.bind('<Button-5>', lambda event: (text_widget.yview_scroll(3, 'units'), 'break')[1])
root.bind('<Button-4>', lambda event: (text_widget.yview_scroll(-3, 'units'), 'break')[1])
root.bind('<Button-5>', lambda event: (text_widget.yview_scroll(3, 'units'), 'break')[1])

# Keep keyboard navigation available.
text_widget.focus_set()
root.bind('<Escape>', lambda event: root.destroy())
text_widget.yview_moveto(0.0)

root.mainloop()
""".strip()

    try:
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            )

        _description_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                helper_code,
                str(SIGN_DESCRIPTION_FILE),
            ],
            creationflags=creationflags,
        )
        return True
    except Exception as exc:
        _description_process = None
        print(
            "Sign alphabet description: unable to open text window.\n"
            f"Error: {exc}"
        )
        return False


def close_sign_description():
    """Safely close the separate sign-description window."""
    global _description_process

    if _description_process is None:
        return

    try:
        if _description_process.poll() is None:
            _description_process.terminate()
            try:
                _description_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _description_process.kill()
    except Exception:
        pass
    finally:
        _description_process = None


def sign_description_is_open():
    """Return True while the sign-description process is running."""
    global _description_process

    if _description_process is None:
        return False

    if _description_process.poll() is None:
        return True

    _description_process = None
    return False


def poll_video_tutorial_status():
    """Clear temporary tutorial status messages automatically."""
    global _video_tutorial_status
    global _video_tutorial_status_until
    global _video_tutorial_ready_marker
    global _video_tutorial_opening_started

    now = time.monotonic()

    if _video_tutorial_ready_marker:
        marker = Path(_video_tutorial_ready_marker)
        if marker.exists():
            try:
                marker.unlink()
            except Exception:
                pass
            _video_tutorial_ready_marker = None
            _video_tutorial_opening_started = 0.0
            _video_tutorial_status = ""
            _video_tutorial_status_until = 0.0
            create_sign_guide()
            return

    # If pywebview does not emit its ``shown`` event on a particular backend,
    # do not leave the opening message stuck forever. Once the child process
    # has remained alive for a few seconds, the tutorial window has had ample
    # time to initialize, so the status can safely be cleared.
    if (
        _video_tutorial_status == "Opening gesture tutorial..."
        and _video_tutorial_opening_started > 0.0
        and now - _video_tutorial_opening_started >= 5.0
    ):
        _video_tutorial_status = ""
        _video_tutorial_opening_started = 0.0
        _video_tutorial_status_until = 0.0
        create_sign_guide()
        return

    # If the tutorial process died before its native window was shown, report
    # a launch failure rather than leaving the opening message on screen.
    if (
        _video_tutorial_status == "Opening gesture tutorial..."
        and _video_tutorial_process is not None
        and _video_tutorial_process.poll() is not None
    ):
        _video_tutorial_status = "Unable to open gesture tutorial."
        _video_tutorial_opening_started = 0.0
        _video_tutorial_status_until = now + 3.0
        create_sign_guide()
        return

    # Temporary messages such as "already open" should disappear on their
    # own instead of waiting for another video-button click.
    if (
        _video_tutorial_status
        and _video_tutorial_status_until > 0
        and now >= _video_tutorial_status_until
    ):
        _video_tutorial_status = ""
        _video_tutorial_status_until = 0.0
        create_sign_guide()


def open_video_tutorial():
    """Open the configured YouTube tutorial without opening duplicates."""
    global _video_tutorial_process
    global _video_tutorial_status
    global _video_tutorial_status_until
    global _video_tutorial_ready_marker
    global _video_tutorial_opening_started

    # If the previous tutorial process is still alive, do not create another
    # video window. Give the user a short-lived confirmation instead.
    if _video_tutorial_process is not None:
        if _video_tutorial_process.poll() is None:
            _video_tutorial_status = "Gesture tutorial is already open."
            _video_tutorial_opening_started = 0.0
            _video_tutorial_status_until = time.monotonic() + 1.8
            create_sign_guide()
            return True

        _video_tutorial_process = None
        _video_tutorial_ready_marker = None
        _video_tutorial_opening_started = 0.0

    if not YOUTUBE_VIDEO_URL:
        _video_tutorial_status = "Gesture tutorial URL is not configured."
        _video_tutorial_status_until = time.monotonic() + 3.0
        create_sign_guide()
        print("Gesture tutorial: no YouTube URL has been configured yet.")
        return False

    # Show immediate feedback while the separate webview process starts.
    _video_tutorial_status = "Opening gesture tutorial..."
    _video_tutorial_status_until = 0.0
    _video_tutorial_opening_started = time.monotonic()
    create_sign_guide()

    marker_path = (
        Path(tempfile.gettempdir())
        / f"echohands_tutorial_{os.getpid()}_{int(time.time() * 1000)}.ready"
    )
    _video_tutorial_ready_marker = str(marker_path)

    helper_code = """
import sys
from pathlib import Path
import webview

url = sys.argv[1]
marker = Path(sys.argv[2])

window = webview.create_window(
    'EchoHands - Gesture Tutorial',
    url,
    width=760,
    height=500,
    resizable=True,
)

# This fires when the native webview window has actually been shown.
def on_shown():
    try:
        marker.write_text('shown', encoding='utf-8')
    except Exception:
        pass

window.events.shown += on_shown
webview.start()
""".strip()

    try:
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            )

        _video_tutorial_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                helper_code,
                YOUTUBE_VIDEO_URL,
                str(marker_path),
            ],
            creationflags=creationflags,
        )

        return True

    except Exception as exc:
        _video_tutorial_process = None
        _video_tutorial_ready_marker = None
        _video_tutorial_opening_started = 0.0
        _video_tutorial_status = "Unable to open gesture tutorial."
        _video_tutorial_status_until = time.monotonic() + 3.0
        create_sign_guide()
        print(
            "Gesture tutorial: unable to open embedded video window.\n"
            f"Error: {exc}"
        )
        return False

def handle_sign_guide_click(x, y):
    """Handle clicks on the two recognition-help resources."""
    if _guide_hitbox_contains(
        x,
        y,
        _sign_guide_hitboxes.get(
            "description",
            (-1, -1, -1, -1),
        ),
    ):
        if not sign_description_is_open():
            create_sign_description()
        return True

    if _guide_hitbox_contains(
        x,
        y,
        _sign_guide_hitboxes.get(
            "video",
            (-1, -1, -1, -1),
        ),
    ):
        open_video_tutorial()
        return True

    return False




def sign_guide_mouse_callback(event, x, y, flags, param):
    """Mouse callback for the two clickable guide resources."""
    if event != cv2.EVENT_LBUTTONUP:
        return

    handle_sign_guide_click(x, y)

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

        if event != cv2.EVENT_LBUTTONUP:
            return

        # Clicks on the main interface open/toggle the help panel.
        if param is None:
            return

        source_width = param["source_width"]
        source_height = param["source_height"]
        display_width = param["display_width"]
        display_height = param["display_height"]

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

            # Remove the tutorial opening message as soon as the webview
            # process reports that its native window has been shown.
            poll_video_tutorial_status()

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
                    close_sign_description()

            # --------------------------------------------------
            # Resource window may be closed independently.
            # --------------------------------------------------

            # ==================================================
            # KEYBOARD
            # ==================================================

            key = (
                cv2.waitKeyEx(1)
            )

            # Process tutorial status again immediately after OpenCV handles
            # the current mouse/keyboard event. This guarantees short-lived
            # messages such as 'already open' expire without another click.
            poll_video_tutorial_status()

            if key == -1:

                key = 255

            else:

                key = (
                    key & 0xFF
                )

            # ==================================================
            # ESC / GUIDE RESOURCE CLOSE
            # ==================================================

            if key == 27:

                if sign_guide_window_exists:
                    close_sign_guide()
                    sign_guide_window_exists = False
                    guide_open = False

                else:
                    close_sign_description()

                continue

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