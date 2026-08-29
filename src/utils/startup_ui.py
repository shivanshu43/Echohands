import queue
import threading
from pathlib import Path
import tkinter as tk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


class StartupUI:

    SPINNER_FRAMES = [
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    ]

    CONTENT_WIDTH = 700

    def __init__(self):

        self.root = None
        self.running = False

        self.spinner_index = 0

        self.task_label = None
        self.info_label = None
        self.spinner_label = None

        self.progress_canvas = None
        self.progress_bar = None
        self.progress_text = None

        self.models_frame = None

        self.banner_image = None

        self.model_rows = {}

        self.total_models = 0
        self.completed_models = 0

        self.current_progress = 0.0

        self._ui_queue = queue.Queue()

        self._worker_thread = None
        self._worker_result = None
        self._worker_error = None
        self._worker_done = False

        self._main_thread_id = threading.get_ident()

    # ==========================================================
    # START
    # ==========================================================

    def start(self):

        if self.running:
            return

        self.root = tk.Tk()

        self.root.title(
            "EchoHands - Starting"
        )

        self.root.configure(
            bg="#151515"
        )

        self.root.resizable(
            False,
            False
        )

        width = 780
        height = 590

        screen_width = (
            self.root.winfo_screenwidth()
        )

        screen_height = (
            self.root.winfo_screenheight()
        )

        position_x = (
            screen_width - width
        ) // 2

        position_y = (
            screen_height - height
        ) // 2

        self.root.geometry(
            f"{width}x{height}+"
            f"{position_x}+{position_y}"
        )

        main = tk.Frame(
            self.root,
            bg="#151515"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=38,
            pady=24
        )

        # ======================================================
        # BANNER
        # ======================================================

        self._create_banner(
            main
        )

        # ======================================================
        # CURRENT TASK
        # ======================================================

        task_frame = tk.Frame(
            main,
            bg="#151515"
        )

        task_frame.pack(
            fill="x",
            pady=(8, 0)
        )

        self.task_label = tk.Label(
            task_frame,
            text="Starting EchoHands...",
            font=(
                "Segoe UI Semibold",
                17
            ),
            fg="#ffffff",
            bg="#151515",
            anchor="w",
            justify="left",
            wraplength=self.CONTENT_WIDTH - 50
        )

        self.task_label.pack(
            side="left",
            fill="x",
            expand=True
        )

        self.spinner_label = tk.Label(
            task_frame,
            text="⠋",
            font=(
                "Segoe UI",
                19
            ),
            fg="#ffffff",
            bg="#151515"
        )

        self.spinner_label.pack(
            side="right"
        )

        # ======================================================
        # INFORMATION
        # ======================================================

        self.info_label = tk.Label(
            main,
            text=(
                "Preparing EchoHands. "
                "Please wait..."
            ),
            font=(
                "Segoe UI",
                10
            ),
            fg="#aaaaaa",
            bg="#151515",
            justify="left",
            anchor="w",
            wraplength=self.CONTENT_WIDTH - 20
        )

        self.info_label.pack(
            fill="x",
            pady=(8, 14)
        )

        # ======================================================
        # DIVIDER
        # ======================================================

        tk.Frame(
            main,
            height=1,
            bg="#383838"
        ).pack(
            fill="x",
            pady=(0, 14)
        )

        # ======================================================
        # MODEL SETUP TITLE
        # ======================================================

        tk.Label(
            main,
            text="MODEL SETUP",
            font=(
                "Segoe UI Semibold",
                10
            ),
            fg="#bbbbbb",
            bg="#151515",
            anchor="w"
        ).pack(
            fill="x",
            pady=(0, 6)
        )

        # ======================================================
        # MODEL LIST
        # ======================================================

        self.models_frame = tk.Frame(
            main,
            bg="#151515"
        )

        self.models_frame.pack(
            fill="x"
        )

        # ======================================================
        # OVERALL PROGRESS TITLE
        # ======================================================

        tk.Label(
            main,
            text="OVERALL DOWNLOAD PROGRESS",
            font=(
                "Segoe UI Semibold",
                10
            ),
            fg="#bbbbbb",
            bg="#151515",
            anchor="w"
        ).pack(
            fill="x",
            pady=(14, 6)
        )

        # ======================================================
        # PROGRESS BAR
        # ======================================================

        self.progress_canvas = tk.Canvas(
            main,
            width=self.CONTENT_WIDTH,
            height=12,
            bg="#343434",
            highlightthickness=0
        )

        self.progress_canvas.pack(
            fill="x"
        )

        self.progress_bar = (
            self.progress_canvas.create_rectangle(
                0,
                0,
                0,
                12,
                fill="#eeeeee",
                outline=""
            )
        )

        # ======================================================
        # PROGRESS TEXT
        # ======================================================

        self.progress_text = tk.Label(
            main,
            text="0.0%    0 / 0 models verified",
            font=(
                "Segoe UI",
                9
            ),
            fg="#aaaaaa",
            bg="#151515",
            anchor="w"
        )

        self.progress_text.pack(
            fill="x",
            pady=(6, 0)
        )

        # ======================================================
        # ACTIVATE UI
        # ======================================================

        self.running = True

        self._animate_spinner()

        self.root.after(
            50,
            self._process_queue
        )

        self.root.update_idletasks()
        self.root.update()

    # ==========================================================
    # BANNER
    # ==========================================================

    def _create_banner(
        self,
        parent
    ):

        banner_path = (
            Path(__file__).resolve().parents[2]
            / "Assets"
            / "banner.png"
        )

        if not banner_path.exists():
            return

        try:

            if Image and ImageTk:

                image = Image.open(
                    banner_path
                ).convert(
                    "RGBA"
                )

                target_width = (
                    self.CONTENT_WIDTH
                )

                if image.width != target_width:

                    target_height = max(
                        1,
                        round(
                            image.height
                            * target_width
                            / image.width
                        )
                    )

                    image = image.resize(
                        (
                            target_width,
                            target_height
                        ),
                        Image.LANCZOS
                    )

                self.banner_image = (
                    ImageTk.PhotoImage(
                        image
                    )
                )

            else:

                self.banner_image = (
                    tk.PhotoImage(
                        file=str(
                            banner_path
                        )
                    )
                )

            banner_label = tk.Label(
                parent,
                image=self.banner_image,
                bg="#151515",
                bd=0,
                highlightthickness=0
            )

            banner_label.pack(
                fill="x",
                pady=(0, 2)
            )

        except Exception:

            self.banner_image = None

    # ==========================================================
    # SPINNER
    # ==========================================================

    def _animate_spinner(self):

        if (
            not self.running
            or not self.root
        ):
            return

        try:

            self.spinner_label.config(
                text=self.SPINNER_FRAMES[
                    self.spinner_index
                ]
            )

            self.spinner_index = (
                self.spinner_index + 1
            ) % len(
                self.SPINNER_FRAMES
            )

            self.root.after(
                100,
                self._animate_spinner
            )

        except tk.TclError:

            self.running = False

    # ==========================================================
    # UI QUEUE
    # ==========================================================

    def _process_queue(self):

        if (
            not self.running
            or not self.root
        ):
            return

        try:

            while True:

                func, args, kwargs = (
                    self._ui_queue.get_nowait()
                )

                func(
                    *args,
                    **kwargs
                )

        except queue.Empty:

            pass

        except tk.TclError:

            self.running = False

            return

        self.root.after(
            50,
            self._process_queue
        )

    # ==========================================================
    # THREAD-SAFE UI CALL
    # ==========================================================

    def _call_ui(
        self,
        func,
        *args,
        **kwargs
    ):

        if (
            threading.get_ident()
            == self._main_thread_id
        ):

            func(
                *args,
                **kwargs
            )

        elif self.running:

            self._ui_queue.put(
                (
                    func,
                    args,
                    kwargs
                )
            )

    # ==========================================================
    # WORKER
    # ==========================================================

    def run_worker(
        self,
        target
    ):

        if not self.running:
            self.start()

        self._worker_result = None
        self._worker_error = None
        self._worker_done = False

        def worker():

            try:

                self._worker_result = (
                    target()
                )

            except Exception as error:

                self._worker_error = error

            finally:

                self._worker_done = True

        self._worker_thread = (
            threading.Thread(
                target=worker,
                name="EchoHandsStartup",
                daemon=True
            )
        )

        self._worker_thread.start()

        self.root.after(
            50,
            self._check_worker
        )

        self.root.mainloop()

        if self._worker_error is not None:

            raise self._worker_error

        return self._worker_result

    # ==========================================================
    # WORKER CHECK
    # ==========================================================

    def _check_worker(self):

        if not self.running:
            return

        if not self._worker_done:

            self.root.after(
                50,
                self._check_worker
            )

            return

        if self._worker_error is not None:

            self._show_error(
                str(
                    self._worker_error
                )
            )

            self.root.after(
                1800,
                self.close
            )

            return

        self.finish()

    # ==========================================================
    # ERROR
    # ==========================================================

    def _show_error(
        self,
        message
    ):

        self.task_label.config(
            text="EchoHands could not start."
        )

        self.spinner_label.config(
            text="!"
        )

        self.info_label.config(
            text=message,
            wraplength=self.CONTENT_WIDTH - 20
        )

        self._update_window()

    # ==========================================================
    # TASK
    # ==========================================================

    def set_task(
        self,
        message
    ):

        self._call_ui(
            self._set_task,
            message
        )

    def _set_task(
        self,
        message
    ):

        if self.running:

            self.task_label.config(
                text=message
            )

    # ==========================================================
    # STATUS
    # ==========================================================

    def set_status(
        self,
        message
    ):

        self._call_ui(
            self._set_status,
            message
        )

    def _set_status(
        self,
        message
    ):

        if self.running:

            self.info_label.config(
                text=message
            )

    # ==========================================================
    # MODEL INITIALIZATION
    # ==========================================================

    def initialize_models(
        self,
        model_names
    ):

        self._call_ui(
            self._initialize_models,
            model_names
        )

    def _initialize_models(
        self,
        model_names
    ):

        if not self.running:
            return

        for widget in (
            self.models_frame.winfo_children()
        ):

            widget.destroy()

        self.model_rows = {}

        self.total_models = (
            len(model_names)
        )

        self.completed_models = 0

        for model_name in model_names:

            row = tk.Frame(
                self.models_frame,
                bg="#151515"
            )

            row.pack(
                fill="x",
                pady=3
            )

            state = tk.Label(
                row,
                text="[ ]",
                font=(
                    "Consolas",
                    10
                ),
                fg="#777777",
                bg="#151515",
                width=5,
                anchor="w"
            )

            state.pack(
                side="left"
            )

            name = tk.Label(
                row,
                text=model_name,
                font=(
                    "Consolas",
                    10
                ),
                fg="#dddddd",
                bg="#151515",
                anchor="w"
            )

            name.pack(
                side="left"
            )

            status = tk.Label(
                row,
                text="Waiting",
                font=(
                    "Segoe UI",
                    9
                ),
                fg="#888888",
                bg="#151515",
                anchor="e"
            )

            status.pack(
                side="right"
            )

            self.model_rows[
                model_name
            ] = {
                "state": state,
                "name": name,
                "status": status,
            }

        self._update_progress()

    # ==========================================================
    # MODEL STATUS
    # ==========================================================

    def set_model_status(
        self,
        model_name,
        status
    ):

        self._call_ui(
            self._set_model_status,
            model_name,
            status
        )

    def _set_model_status(
        self,
        model_name,
        status
    ):

        if (
            not self.running
            or model_name
            not in self.model_rows
        ):

            return

        row = self.model_rows[
            model_name
        ]

        states = {

            "waiting": (
                "[ ]",
                "Waiting"
            ),

            "preparing": (
                "[>]",
                "Preparing..."
            ),

            "connecting": (
                "[>]",
                "Connecting..."
            ),

            "downloading": (
                "[>]",
                "Downloading..."
            ),

            "downloaded": (
                "[✓]",
                "Downloaded"
            ),

            "verifying": (
                "[>]",
                "Verifying integrity..."
            ),

            "verified": (
                "[✓]",
                "Verified"
            ),

            "cached": (
                "[✓]",
                "Ready"
            ),

            "error": (
                "[!]",
                "Failed"
            ),
        }

        state_text, status_text = (
            states.get(
                status,
                (
                    "[ ]",
                    status
                )
            )
        )

        row["state"].config(
            text=state_text,
            fg=(
                "#ffffff"
                if status != "waiting"
                else "#777777"
            )
        )

        row["status"].config(
            text=status_text,
            fg=(
                "#ffffff"
                if status != "waiting"
                else "#888888"
            )
        )

        if status == "verified":

            self.completed_models = min(
                self.completed_models + 1,
                self.total_models
            )

        self._update_progress()

    # ==========================================================
    # DYNAMIC BYTE PROGRESS
    # ==========================================================

    def set_download_progress(
        self,
        filename,
        downloaded,
        file_total,
        overall_downloaded,
        overall_total,
        speed_bps=0.0
    ):

        self._call_ui(
            self._set_download_progress,
            filename,
            downloaded,
            file_total,
            overall_downloaded,
            overall_total,
            speed_bps
        )

    def _set_download_progress(
        self,
        filename,
        downloaded,
        file_total,
        overall_downloaded,
        overall_total,
        speed_bps
    ):

        if not self.running:
            return

        file_percentage = (
            downloaded
            / file_total
            * 100
            if file_total
            else 0.0
        )

        overall_percentage = (
            overall_downloaded
            / overall_total
            * 100
            if overall_total
            else 0.0
        )

        self.current_progress = (
            overall_percentage
        )

        self._update_progress_value(
            overall_percentage
        )

        if speed_bps > 0:

            speed_text = (
                self._format_bytes(
                    speed_bps
                )
                + "/s"
            )

        else:

            speed_text = (
                "calculating speed..."
            )

        self.progress_text.config(
            text=(
                f"{overall_percentage:.1f}%    "
                f"{self.completed_models} / "
                f"{self.total_models} models verified"
            )
        )

        # IMPORTANT:
        # Keep this to exactly two lines.
        # Do not add the internet-speed/footer message here.
        self.info_label.config(
            text=(
                f"Downloading {filename}\n"
                f"{self._format_bytes(downloaded)} / "
                f"{self._format_bytes(file_total)} "
                f"({file_percentage:.1f}%)  •  "
                f"{speed_text}"
            )
        )

    # ==========================================================
    # OVERALL BYTE PROGRESS
    # ==========================================================

    def set_overall_progress(
        self,
        downloaded_bytes,
        total_bytes,
        completed_models=None,
        total_models=None
    ):

        self._call_ui(
            self._set_overall_progress,
            downloaded_bytes,
            total_bytes,
            completed_models,
            total_models
        )

    def _set_overall_progress(
        self,
        downloaded_bytes,
        total_bytes,
        completed_models=None,
        total_models=None
    ):

        if not self.running:
            return

        if total_bytes:

            percentage = (
                downloaded_bytes
                / total_bytes
                * 100
            )

        else:

            percentage = 0.0

        self.current_progress = (
            percentage
        )

        self._update_progress_value(
            percentage
        )

        if completed_models is not None:

            self.completed_models = (
                completed_models
            )

        if total_models is not None:

            self.total_models = (
                total_models
            )

        self.progress_text.config(
            text=(
                f"{percentage:.1f}%    "
                f"{self.completed_models} / "
                f"{self.total_models} models verified"
            )
        )

    # ==========================================================
    # FORMAT BYTES
    # ==========================================================

    def _format_bytes(
        self,
        value
    ):

        value = float(
            value or 0
        )

        for unit in (
            "B",
            "KB",
            "MB",
            "GB"
        ):

            if (
                value < 1024
                or unit == "GB"
            ):

                return (
                    f"{value:.1f} {unit}"
                )

            value /= 1024

        return (
            f"{value:.1f} GB"
        )

    # ==========================================================
    # LEGACY MODEL PROGRESS
    # ==========================================================

    def set_progress(
        self,
        completed,
        total
    ):

        self._call_ui(
            self._set_progress,
            completed,
            total
        )

    def _set_progress(
        self,
        completed,
        total
    ):

        self.completed_models = (
            completed
        )

        self.total_models = (
            total
        )

        try:

            self.progress_text.config(
                text=(
                    f"{self.current_progress:.1f}%    "
                    f"{self.completed_models} / "
                    f"{self.total_models} "
                    "models verified"
                )
            )

        except tk.TclError:

            pass

    # ==========================================================
    # PROGRESS BAR VALUE
    # ==========================================================

    def _update_progress_value(
        self,
        percentage
    ):

        try:

            percentage = max(
                0,
                min(
                    100,
                    percentage
                )
            )

            width = (
                self.CONTENT_WIDTH
                * percentage
                / 100
            )

            self.progress_canvas.coords(
                self.progress_bar,
                0,
                0,
                width,
                12
            )

        except tk.TclError:

            pass

    # ==========================================================
    # UPDATE PROGRESS
    # ==========================================================

    def _update_progress(self):

        try:

            self.progress_text.config(
                text=(
                    f"{self.current_progress:.1f}%    "
                    f"{self.completed_models} / "
                    f"{self.total_models} "
                    "models verified"
                )
            )

        except tk.TclError:

            pass

    # ==========================================================
    # FINISH
    # ==========================================================

    def finish(
        self,
        message="EchoHands is ready."
    ):

        if not self.running:
            return

        self.task_label.config(
            text=message
        )

        self.spinner_label.config(
            text="✓"
        )

        self.info_label.config(
            text=(
                "Startup completed. "
                "Launching recognition..."
            )
        )

        self.root.after(
            450,
            self.close
        )

    # ==========================================================
    # UPDATE WINDOW
    # ==========================================================

    def _update_window(self):

        if (
            not self.running
            or not self.root
        ):

            return

        try:

            self.root.update_idletasks()
            self.root.update()

        except tk.TclError:

            self.running = False

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(self):

        if not self.running:
            return

        self.running = False

        try:

            self.root.quit()
            self.root.destroy()

        except tk.TclError:

            pass