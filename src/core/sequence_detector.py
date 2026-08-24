import numpy as np


class SequenceDetector:

    IDLE = "IDLE"
    RECORDING = "RECORDING"

    def __init__(
        self,
        start_threshold=0.020,
        stop_threshold=0.008,
        start_frames=3,
        stop_frames=8,
        min_sequence_frames=15,
        max_sequence_frames=80,
    ):

        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold

        self.start_frames = start_frames
        self.stop_frames = stop_frames

        self.min_sequence_frames = min_sequence_frames
        self.max_sequence_frames = max_sequence_frames

        self.state = self.IDLE

        self.previous_features = None

        self.sequence = []

        self.movement_counter = 0
        self.stationary_counter = 0

    # ==========================================================
    # Motion calculation
    # ==========================================================

    def _calculate_motion(self, features):

        if (
            features is None
            or self.previous_features is None
        ):

            return 0.0

        current = np.asarray(
            features,
            dtype=np.float32
        )

        previous = np.asarray(
            self.previous_features,
            dtype=np.float32
        )

        return float(
            np.mean(
                np.abs(
                    current - previous
                )
            )
        )

    # ==========================================================
    # Force dynamic recording
    #
    # Used when RecognitionController has already confirmed
    # that the user's motion is likely a dynamic gesture.
    # ==========================================================

    def start_recording(self, initial_frames=None):

        self.state = self.RECORDING

        self.sequence = []

        self.movement_counter = 0
        self.stationary_counter = 0

        if initial_frames is not None:

            for frame in initial_frames:

                self.sequence.append(
                    np.asarray(
                        frame,
                        dtype=np.float32
                    ).copy()
                )

    # ==========================================================
    # Update
    # ==========================================================

    def update(self, features):

        """
        Process one frame.

        Returns:
            completed_sequence when gesture finishes.
            None otherwise.
        """

        if features is None:

            self.previous_features = None

            return None

        motion = self._calculate_motion(
            features
        )

        # ======================================================
        # IDLE
        # ======================================================

        if self.state == self.IDLE:

            if motion >= self.start_threshold:

                self.movement_counter += 1

            else:

                self.movement_counter = 0

            if (
                self.movement_counter
                >= self.start_frames
            ):

                self.state = self.RECORDING

                self.sequence = []

                self.stationary_counter = 0

                self.sequence.append(
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

        # ======================================================
        # RECORDING
        # ======================================================

        elif self.state == self.RECORDING:

            self.sequence.append(
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            if motion <= self.stop_threshold:

                self.stationary_counter += 1

            else:

                self.stationary_counter = 0

            # --------------------------------------------------
            # Maximum sequence length
            # --------------------------------------------------

            if (
                len(self.sequence)
                >= self.max_sequence_frames
            ):

                completed_sequence = (
                    self.sequence.copy()
                )

                self.reset()

                return completed_sequence

            # --------------------------------------------------
            # Gesture stopped
            # --------------------------------------------------

            if (
                self.stationary_counter
                >= self.stop_frames
            ):

                if (
                    len(self.sequence)
                    >= self.min_sequence_frames
                ):

                    completed_sequence = (
                        self.sequence.copy()
                    )

                    self.reset()

                    return completed_sequence

                else:

                    self.reset()

        self.previous_features = np.asarray(
            features,
            dtype=np.float32
        ).copy()

        return None

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.state = self.IDLE

        self.previous_features = None

        self.sequence = []

        self.movement_counter = 0

        self.stationary_counter = 0

    # ==========================================================
    # Getters
    # ==========================================================

    def get_state(self):

        return self.state

    def get_sequence_length(self):

        return len(
            self.sequence
        )

    def get_motion(self, features):

        return self._calculate_motion(
            features
        )