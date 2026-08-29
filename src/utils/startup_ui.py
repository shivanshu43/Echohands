import cv2
import numpy as np


class StartupUI:

    WINDOW_NAME = "EchoHands - Starting"

    WIDTH = 1000
    HEIGHT = 720

    def __init__(self):

        self.current_status = (
            "Initializing EchoHands..."
        )

        self.current_detail = (
            "Preparing the application."
        )

        self.progress = 0.0

        self.phase = "initializing"

        # --------------------------------------------------
        # Lightweight ASCII animation.
        #
        # Avoids Unicode characters such as ✓ or ◐,
        # which were previously appearing as ???.
        # --------------------------------------------------

        self.spinner_index = 0

        self.spinner_frames = [
            "[..]",
            "[...]",
            "[. .]",
            "[...]",
        ]

        self.model_rows = {}

        self.total_models = 0

        self.completed_models = 0

        self.current_model = ""

        self.window_created = False

    # ==========================================================
    # INITIALIZE MODEL LIST
    # ==========================================================

    def initialize_models(
        self,
        model_names
    ):

        self.model_rows = {}

        self.total_models = len(
            model_names
        )

        self.completed_models = 0

        for name in model_names:

            self.model_rows[name] = {
                "status": "Waiting",
                "progress": 0.0,
            }

        self.show()

    # ==========================================================
    # MAIN DISPLAY
    # ==========================================================

    def show(self):

        frame = np.zeros(
            (
                self.HEIGHT,
                self.WIDTH,
                3
            ),
            dtype=np.uint8
        )

        frame[:] = (
            24,
            24,
            24
        )

        # ======================================================
        # HEADER
        # ======================================================

        self._put_text(
            frame,
            "EchoHands",
            (65, 65),
            1.25,
            2
        )

        self._put_text(
            frame,
            "Sign Language Recognition",
            (68, 98),
            0.52,
            1
        )

        # ======================================================
        # MAIN STATUS
        # ======================================================

        self._put_text(
            frame,
            self.current_status,
            (65, 150),
            0.70,
            2
        )

        # ------------------------------------------------------
        # Animated spinner
        # ------------------------------------------------------

        spinner = (
            self.spinner_frames[
                self.spinner_index
                % len(self.spinner_frames)
            ]
        )

        self.spinner_index += 1

        self._put_text(
            frame,
            spinner,
            (875, 150),
            0.50,
            1
        )

        # ======================================================
        # DETAIL
        # ======================================================

        self._put_text(
            frame,
            self.current_detail,
            (65, 185),
            0.43,
            1
        )

        # ======================================================
        # INFORMATION MESSAGE
        # ======================================================

        if self.phase == "downloading":

            self._put_text(
                frame,
                "First-time setup: EchoHands is downloading the recognition models.",
                (65, 215),
                0.40,
                1
            )

            self._put_text(
                frame,
                "This may take a few minutes depending on your internet speed.",
                (65, 240),
                0.40,
                1
            )

        elif self.phase == "verifying":

            self._put_text(
                frame,
                "The download is complete. EchoHands is checking the file integrity.",
                (65, 215),
                0.40,
                1
            )

            self._put_text(
                frame,
                "Large model files may take a moment to verify. Please wait.",
                (65, 240),
                0.40,
                1
            )

        elif self.phase == "checking":

            self._put_text(
                frame,
                "Checking your existing model package before starting recognition.",
                (65, 215),
                0.40,
                1
            )

        elif self.phase == "loading":

            self._put_text(
                frame,
                "The models are ready. Initializing the recognition engine.",
                (65, 215),
                0.40,
                1
            )

        elif self.phase == "camera":

            self._put_text(
                frame,
                "Recognition is ready. Starting the camera and hand detector.",
                (65, 215),
                0.40,
                1
            )

        elif self.phase == "ready":

            self._put_text(
                frame,
                "All required components are ready.",
                (65, 215),
                0.40,
                1
            )

        # ======================================================
        # SEPARATOR
        # ======================================================

        cv2.line(
            frame,
            (65, 275),
            (935, 275),
            (65, 65, 65),
            1
        )

        # ======================================================
        # MODEL SETUP
        # ======================================================

        self._put_text(
            frame,
            "MODEL SETUP",
            (65, 310),
            0.46,
            1
        )

        # ======================================================
        # MODEL ROWS
        # ======================================================

        row_y = 350

        row_height = 47

        for index, (
            name,
            info
        ) in enumerate(
            self.model_rows.items()
        ):

            y = (
                row_y
                + index * row_height
            )

            status = info[
                "status"
            ]

            progress = info[
                "progress"
            ]

            # --------------------------------------------------
            # Status indicator
            # --------------------------------------------------

            if status == "Verified":

                symbol = "[OK]"

            elif status in (
                "Downloading",
                "Verifying"
            ):

                symbol = "[>>]"

            else:

                symbol = "[  ]"

            self._put_text(
                frame,
                symbol,
                (65, y),
                0.40,
                1
            )

            # --------------------------------------------------
            # Model name
            # --------------------------------------------------

            self._put_text(
                frame,
                name,
                (145, y),
                0.40,
                1
            )

            # --------------------------------------------------
            # Model status
            # --------------------------------------------------

            if status == "Downloading":

                status_text = (
                    f"Downloading "
                    f"{int(progress * 100)}%"
                )

            elif status == "Verifying":

                # Show verification progress as well.
                #
                # This makes it obvious that the application
                # is not frozen while SHA-256 is calculated.

                status_text = (
                    f"Downloaded  |  "
                    f"Verifying {int(progress * 100)}%"
                )

            elif status == "Verified":

                status_text = (
                    "Downloaded  |  Verified"
                )

            else:

                status_text = (
                    "Waiting"
                )

            self._put_text(
                frame,
                status_text,
                (650, y),
                0.38,
                1
            )

        # ======================================================
        # OVERALL PROGRESS
        # ======================================================

        progress_title_y = 555

        self._put_text(
            frame,
            "OVERALL PROGRESS",
            (
                65,
                progress_title_y
            ),
            0.43,
            1
        )

        # ======================================================
        # PROGRESS BAR
        # ======================================================

        bar_x = 65

        bar_y = 575

        bar_width = 870

        bar_height = 18

        # Background
        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (
                65,
                65,
                65
            ),
            -1
        )

        progress_width = int(
            bar_width
            * max(
                0.0,
                min(
                    self.progress,
                    1.0
                )
            )
        )

        if progress_width > 0:

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y
                ),
                (
                    bar_x + progress_width,
                    bar_y + bar_height
                ),
                (
                    245,
                    245,
                    245
                ),
                -1
            )

        # ======================================================
        # PROGRESS INFORMATION
        # ======================================================

        percentage = int(
            self.progress * 100
        )

        self._put_text(
            frame,
            f"{percentage}%",
            (
                65,
                620
            ),
            0.43,
            1
        )

        self._put_text(
            frame,
            (
                f"{self.completed_models}/"
                f"{self.total_models} models verified"
            ),
            (
                180,
                620
            ),
            0.43,
            1
        )

        # ======================================================
        # CURRENT OPERATION
        #
        # This replaces the old CURRENT TASK section.
        # It is more useful because it explains the pause
        # during verification.
        # ======================================================

        operation_y = 655

        if self.phase == "verifying":

            operation_text = (
                "Verifying model integrity"
            )

        elif self.phase == "downloading":

            operation_text = (
                "Downloading recognition models"
            )

        elif self.phase == "checking":

            operation_text = (
                "Checking installed models"
            )

        elif self.phase == "loading":

            operation_text = (
                "Initializing recognition engine"
            )

        elif self.phase == "camera":

            operation_text = (
                "Starting camera"
            )

        elif self.phase == "ready":

            operation_text = (
                "Startup complete"
            )

        else:

            operation_text = (
                "Preparing EchoHands"
            )

        self._put_text(
            frame,
            operation_text,
            (
                65,
                operation_y
            ),
            0.40,
            1
        )

        # ------------------------------------------------------
        # Animated indicator next to operation
        # ------------------------------------------------------

        self._put_text(
            frame,
            spinner,
            (
                355,
                operation_y
            ),
            0.40,
            1
        )

        # ======================================================
        # FOOTER
        # ======================================================

        self._put_text(
            frame,
            "Please keep this window open while EchoHands prepares the recognition system.",
            (
                65,
                695
            ),
            0.36,
            1
        )

        # ======================================================
        # DISPLAY
        # ======================================================

        cv2.imshow(
            self.WINDOW_NAME,
            frame
        )

        cv2.waitKey(1)

        self.window_created = True

    # ==========================================================
    # SET GENERAL STATUS
    # ==========================================================

    def set_status(
        self,
        status,
        detail=""
    ):

        self.current_status = status

        self.current_detail = detail

        status_lower = (
            status.lower()
        )

        if "download" in status_lower:

            self.phase = "downloading"

        elif "verif" in status_lower:

            self.phase = "verifying"

        elif (
            "recognition"
            in status_lower
        ):

            self.phase = "loading"

        elif "camera" in status_lower:

            self.phase = "camera"

        elif "ready" in status_lower:

            self.phase = "ready"

        elif "check" in status_lower:

            self.phase = "checking"

        else:

            self.phase = "initializing"

        self.show()

    # ==========================================================
    # SET MODEL STATUS
    # ==========================================================

    def set_model_status(
        self,
        model_name,
        status,
        progress=0.0
    ):

        if model_name not in self.model_rows:

            self.model_rows[
                model_name
            ] = {
                "status": "Waiting",
                "progress": 0.0,
            }

            self.total_models = len(
                self.model_rows
            )

        self.model_rows[
            model_name
        ]["status"] = status

        self.model_rows[
            model_name
        ]["progress"] = max(
            0.0,
            min(
                float(progress),
                1.0
            )
        )

        self.current_model = (
            model_name
        )

        self.completed_models = sum(
            1
            for info
            in self.model_rows.values()
            if info["status"] == "Verified"
        )

        self.show()

    # ==========================================================
    # SET OVERALL PROGRESS
    # ==========================================================

    def set_progress(
        self,
        progress
    ):

        self.progress = max(
            0.0,
            min(
                float(progress),
                1.0
            )
        )

        self.show()

    # ==========================================================
    # COMPLETE STARTUP
    # ==========================================================

    def complete(self):

        self.progress = 1.0

        self.phase = "ready"

        self.current_status = (
            "EchoHands is ready."
        )

        self.current_detail = (
            "All models have been downloaded and verified successfully."
        )

        for info in (
            self.model_rows.values()
        ):

            info["status"] = "Verified"

            info["progress"] = 1.0

        self.completed_models = (
            self.total_models
        )

        self.current_model = (
            "Startup complete."
        )

        self.show()

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(self):

        try:

            cv2.destroyWindow(
                self.WINDOW_NAME
            )

        except cv2.error:

            pass

        self.window_created = False

    # ==========================================================
    # TEXT HELPER
    # ==========================================================

    def _put_text(
        self,
        frame,
        text,
        position,
        scale,
        thickness
    ):

        cv2.putText(
            frame,
            str(text),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (
                245,
                245,
                245
            ),
            thickness,
            cv2.LINE_AA
        )