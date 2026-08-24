import numpy as np

from src.core.sequence_detector import SequenceDetector


class RecognitionController:

    NONE = "NONE"
    INITIALIZING = "INITIALIZING"
    STATIC = "STATIC"
    DYNAMIC_CANDIDATE = "DYNAMIC_CANDIDATE"
    DYNAMIC = "DYNAMIC"

    def __init__(
        self,
        static_predictor,
        dynamic_predictor,
        dynamic_confidence_threshold=0.85,

        # Static confidence benchmark
        static_confidence_threshold=0.60,

        # Dynamic motion threshold
        movement_threshold=0.035,

        # Consecutive movement frames required
        # before confirming dynamic movement
        dynamic_start_frames=6,

        # Maximum low-motion gap allowed
        # during dynamic candidate detection
        dynamic_gap_frames=3,

        # Static prediction confirmation
        static_prediction_frames=4,

        # Locked gesture release threshold
        unlock_threshold=0.035,

        # Consecutive changed frames required
        # before unlocking a gesture
        unlock_change_frames=5,

        # Frames to ignore whenever a hand appears
        initialization_frames=8,
    ):

        self.static_predictor = static_predictor
        self.dynamic_predictor = dynamic_predictor

        self.sequence_detector = SequenceDetector(
            start_threshold=movement_threshold
        )

        # ======================================================
        # Thresholds
        # ======================================================

        self.dynamic_confidence_threshold = (
            dynamic_confidence_threshold
        )

        self.static_confidence_threshold = (
            static_confidence_threshold
        )

        self.movement_threshold = (
            movement_threshold
        )

        self.dynamic_start_frames = (
            dynamic_start_frames
        )

        self.dynamic_gap_frames = (
            dynamic_gap_frames
        )

        self.static_prediction_frames = (
            static_prediction_frames
        )

        self.unlock_threshold = (
            unlock_threshold
        )

        self.unlock_change_frames = (
            unlock_change_frames
        )

        self.initialization_frames = (
            initialization_frames
        )

        # ======================================================
        # Main state
        # ======================================================

        self.mode = self.NONE

        self.previous_features = None

        # ======================================================
        # Hand initialization
        # ======================================================

        self.initialization_count = 0

        # ======================================================
        # Gesture locking
        # ======================================================

        self.locked_features = None

        self.unlock_change_count = 0

        # ======================================================
        # Static recognition tracking
        # ======================================================

        self.static_candidate = None

        self.static_candidate_count = 0

        # ======================================================
        # Dynamic candidate tracking
        # ======================================================

        self.dynamic_motion_count = 0

        self.dynamic_gap_count = 0

        self.dynamic_candidate_frames = []

        # ======================================================
        # Last recognition
        # ======================================================

        self.last_prediction = None

        self.last_confidence = 0.0

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
    # Change from locked gesture
    # ==========================================================

    def _calculate_change_from_locked(
        self,
        features
    ):

        if (
            features is None
            or self.locked_features is None
        ):

            return 0.0

        current = np.asarray(
            features,
            dtype=np.float32
        )

        locked = np.asarray(
            self.locked_features,
            dtype=np.float32
        )

        return float(
            np.mean(
                np.abs(
                    current - locked
                )
            )
        )

    # ==========================================================
    # Reset static tracking
    # ==========================================================

    def _reset_static_tracking(self):

        self.static_candidate = None

        self.static_candidate_count = 0

    # ==========================================================
    # Reset dynamic candidate
    # ==========================================================

    def _reset_dynamic_candidate(self):

        self.dynamic_motion_count = 0

        self.dynamic_gap_count = 0

        self.dynamic_candidate_frames = []

    # ==========================================================
    # Reset initialization
    # ==========================================================

    def _reset_initialization(self):

        self.initialization_count = 0

    # ==========================================================
    # Reset gesture state
    # ==========================================================

    def _reset_gesture_state(self):

        self.sequence_detector.reset()

        self._reset_static_tracking()

        self._reset_dynamic_candidate()

        self.unlock_change_count = 0

    # ==========================================================
    # Enter initialization
    #
    # Called whenever a new hand enters the camera
    # after a NO HAND state.
    # ==========================================================

    def _start_initialization(self, features):

        self._reset_gesture_state()

        self.locked_features = None

        self.initialization_count = 1

        self.mode = self.INITIALIZING

        self.previous_features = (
            np.asarray(
                features,
                dtype=np.float32
            ).copy()
        )

    # ==========================================================
    # Update
    # ==========================================================

    def update(self, features):

        # ======================================================
        # NO HAND
        #
        # The user has removed their hand from the camera.
        #
        # Completely reset recognition state so that when
        # the hand enters again, initialization starts again.
        # ======================================================

        if features is None:

            self.previous_features = None

            self.locked_features = None

            self._reset_initialization()

            self._reset_gesture_state()

            self.mode = self.NONE

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.NONE,
                "sequence_complete": False,
            }

        # ======================================================
        # HAND JUST ENTERED
        #
        # If the previous frame had no hand, then
        # previous_features is None.
        #
        # Start initialization and do not recognize anything yet.
        # ======================================================

        if (
            self.mode == self.NONE
            and self.previous_features is None
        ):

            self._start_initialization(
                features
            )

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.INITIALIZING,
                "sequence_complete": False,
            }

        # ======================================================
        # INITIALIZING
        #
        # Ignore frames while the hand enters and settles.
        #
        # This happens:
        #
        # NO HAND
        #     ↓
        # HAND ENTERS
        #     ↓
        # INITIALIZING
        #     ↓
        # STATIC
        #
        # ======================================================

        if self.mode == self.INITIALIZING:

            self.initialization_count += 1

            self.previous_features = (
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            if (
                self.initialization_count
                < self.initialization_frames
            ):

                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "mode": self.INITIALIZING,
                    "sequence_complete": False,
                }

            # --------------------------------------------------
            # Hand has settled
            # --------------------------------------------------

            self.initialization_count = 0

            self._reset_static_tracking()

            self._reset_dynamic_candidate()

            self.previous_features = (
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            self.mode = self.STATIC

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.STATIC,
                "sequence_complete": False,
            }

        # ======================================================
        # Calculate motion
        # ======================================================

        motion = self._calculate_motion(
            features
        )

        # ======================================================
        # NONE
        #
        # Wait for a locked gesture to genuinely change.
        # ======================================================

        if self.mode == self.NONE:

            # --------------------------------------------------
            # No locked gesture
            # --------------------------------------------------

            if self.locked_features is None:

                self.mode = self.STATIC

                self._reset_static_tracking()

                self._reset_dynamic_candidate()

            # --------------------------------------------------
            # Gesture is locked
            # --------------------------------------------------

            else:

                gesture_change = (
                    self._calculate_change_from_locked(
                        features
                    )
                )

                # ----------------------------------------------
                # Same gesture or small landmark jitter
                # ----------------------------------------------

                if (
                    gesture_change
                    < self.unlock_threshold
                ):

                    self.unlock_change_count = 0

                    self.previous_features = (
                        np.asarray(
                            features,
                            dtype=np.float32
                        ).copy()
                    )

                    return {
                        "prediction": None,
                        "confidence": 0.0,
                        "mode": self.NONE,
                        "sequence_complete": False,
                    }

                # ----------------------------------------------
                # Possible new gesture
                # ----------------------------------------------

                self.unlock_change_count += 1

                if (
                    self.unlock_change_count
                    < self.unlock_change_frames
                ):

                    self.previous_features = (
                        np.asarray(
                            features,
                            dtype=np.float32
                        ).copy()
                    )

                    return {
                        "prediction": None,
                        "confidence": 0.0,
                        "mode": self.NONE,
                        "sequence_complete": False,
                    }

                # ----------------------------------------------
                # Sustained change confirmed
                # ----------------------------------------------

                self.locked_features = None

                self.unlock_change_count = 0

                self._reset_gesture_state()

                self.mode = self.STATIC

        # ======================================================
        # STATIC
        # ======================================================

        if self.mode == self.STATIC:

            # --------------------------------------------------
            # Significant movement detected
            # --------------------------------------------------

            if (
                motion
                >= self.movement_threshold
            ):

                self.mode = (
                    self.DYNAMIC_CANDIDATE
                )

                self.dynamic_motion_count = 1

                self.dynamic_gap_count = 0

                self.dynamic_candidate_frames = [
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                ]

                self.previous_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "mode": self.DYNAMIC_CANDIDATE,
                    "sequence_complete": False,
                }

            # --------------------------------------------------
            # Static prediction
            # --------------------------------------------------

            prediction, confidence = (
                self.static_predictor.predict(
                    features
                )
            )

            # --------------------------------------------------
            # Track stable prediction
            # --------------------------------------------------

            if (
                prediction
                == self.static_candidate
            ):

                self.static_candidate_count += 1

            else:

                self.static_candidate = prediction

                self.static_candidate_count = 1

            self.last_prediction = prediction

            self.last_confidence = confidence

            # --------------------------------------------------
            # Static gesture confirmed
            # --------------------------------------------------

            if (
                self.static_candidate_count
                >= self.static_prediction_frames
                and confidence
                >= self.static_confidence_threshold
            ):

                self.locked_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                self.unlock_change_count = 0

                self._reset_dynamic_candidate()

                self.mode = self.NONE

                self.previous_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "mode": self.NONE,
                    "sequence_complete": False,
                }

            self.previous_features = (
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            return {
                "prediction": prediction,
                "confidence": confidence,
                "mode": self.STATIC,
                "sequence_complete": False,
            }

        # ======================================================
        # DYNAMIC CANDIDATE
        # ======================================================

        if self.mode == self.DYNAMIC_CANDIDATE:

            self.dynamic_candidate_frames.append(
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            if (
                motion
                >= self.movement_threshold
            ):

                self.dynamic_motion_count += 1

                self.dynamic_gap_count = 0

            else:

                self.dynamic_gap_count += 1

                self.dynamic_motion_count = 0

            # --------------------------------------------------
            # Genuine dynamic movement confirmed
            # --------------------------------------------------

            if (
                self.dynamic_motion_count
                >= self.dynamic_start_frames
            ):

                self.sequence_detector.start_recording(
                    self.dynamic_candidate_frames
                )

                self._reset_dynamic_candidate()

                self._reset_static_tracking()

                self.mode = self.DYNAMIC

                self.previous_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "mode": self.DYNAMIC,
                    "sequence_complete": False,
                }

            # --------------------------------------------------
            # Movement stopped before confirmation
            # --------------------------------------------------

            if (
                self.dynamic_gap_count
                >= self.dynamic_gap_frames
            ):

                self._reset_dynamic_candidate()

                self._reset_static_tracking()

                self.mode = self.STATIC

            self.previous_features = (
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.mode,
                "sequence_complete": False,
            }

        # ======================================================
        # DYNAMIC
        # ======================================================

        if self.mode == self.DYNAMIC:

            completed_sequence = (
                self.sequence_detector.update(
                    features
                )
            )

            # --------------------------------------------------
            # Still recording
            # --------------------------------------------------

            if completed_sequence is None:

                self.previous_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "mode": self.DYNAMIC,
                    "sequence_complete": False,
                }

            # --------------------------------------------------
            # Dynamic sequence completed
            # --------------------------------------------------

            prediction, confidence = (
                self.dynamic_predictor.predict(
                    completed_sequence
                )
            )

            # --------------------------------------------------
            # Valid dynamic gesture
            # --------------------------------------------------

            if (
                prediction is not None
                and confidence
                >= self.dynamic_confidence_threshold
            ):

                self.last_prediction = prediction

                self.last_confidence = confidence

                self.locked_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                self.unlock_change_count = 0

                self._reset_gesture_state()

                self.mode = self.NONE

                self.previous_features = (
                    np.asarray(
                        features,
                        dtype=np.float32
                    ).copy()
                )

                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "mode": self.NONE,
                    "sequence_complete": True,
                }

            # --------------------------------------------------
            # Invalid dynamic sequence
            # --------------------------------------------------

            self._reset_gesture_state()

            self.mode = self.STATIC

            self.previous_features = (
                np.asarray(
                    features,
                    dtype=np.float32
                ).copy()
            )

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.STATIC,
                "sequence_complete": False,
            }

        # ======================================================
        # FALLBACK
        # ======================================================

        self._reset_gesture_state()

        self.mode = self.STATIC

        self.previous_features = (
            np.asarray(
                features,
                dtype=np.float32
            ).copy()
        )

        return {
            "prediction": None,
            "confidence": 0.0,
            "mode": self.STATIC,
            "sequence_complete": False,
        }

    # ==========================================================
    # Getters
    # ==========================================================

    def get_mode(self):

        return self.mode

    def get_sequence_length(self):

        return (
            self.sequence_detector
            .get_sequence_length()
        )

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.sequence_detector.reset()

        self.mode = self.NONE

        self.previous_features = None

        self.locked_features = None

        self.unlock_change_count = 0

        self.initialization_count = 0

        self._reset_static_tracking()

        self._reset_dynamic_candidate()

        self.last_prediction = None

        self.last_confidence = 0.0