import numpy as np
from collections import deque

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

        # ======================================================
        # Confidence
        # ======================================================

        dynamic_confidence_threshold=0.85,
        static_confidence_threshold=0.60,

        # Strong static prediction protects against accidental
        # dynamic activation.
        dynamic_static_guard_threshold=0.70,

        # ======================================================
        # Static smoothing
        # ======================================================

        smoothing_window=5,

        # ======================================================
        # Initialization
        # ======================================================

        initialization_frames=8,

        # ======================================================
        # Dynamic movement
        # ======================================================

        # Raw landmark movement threshold.
        #
        # This is intentionally not the old 0.045 threshold that
        # was applied to the smoothed 70-feature representation.
        dynamic_motion_threshold=0.030,

        # Number of consecutive meaningful movement observations
        # required before entering DYNAMIC.
        dynamic_start_frames=6,

        # Short natural gaps are tolerated.
        dynamic_gap_frames=3,

        # Recent movement window.
        dynamic_window_frames=10,

        # Number of moving observations required in that window.
        dynamic_required_motion_frames=6,

        # ======================================================
        # Static prediction stability
        # ======================================================

        static_prediction_frames=4,

        # ======================================================
        # Gesture lock
        # ======================================================

        # Number of frames required for a DIFFERENT static label
        # to prove that the user actually changed gesture.
        unlock_label_frames=5,

        # ======================================================
        # M / N stabilization
        # ======================================================

        mn_confidence_threshold=0.45,
        mn_window_frames=8,
        mn_required_votes=5,
        mn_confirmation_frames=3,

        # ======================================================
        # Dynamic lock escape
        # ======================================================

        # Require more evidence when trying to leave a committed
        # static gesture through physical movement.
        locked_dynamic_window_frames=12,
        locked_dynamic_required_frames=8,
    ):

        self.static_predictor = static_predictor
        self.dynamic_predictor = dynamic_predictor

        # ======================================================
        # Sequence detector
        # ======================================================

        self.sequence_detector = SequenceDetector(
            start_threshold=dynamic_motion_threshold
        )

        # ======================================================
        # Configuration
        # ======================================================

        self.dynamic_confidence_threshold = (
            dynamic_confidence_threshold
        )

        self.static_confidence_threshold = (
            static_confidence_threshold
        )

        self.dynamic_static_guard_threshold = (
            dynamic_static_guard_threshold
        )

        self.smoothing_window = (
            smoothing_window
        )

        self.initialization_frames = (
            initialization_frames
        )

        self.dynamic_motion_threshold = (
            dynamic_motion_threshold
        )

        self.dynamic_start_frames = (
            dynamic_start_frames
        )

        self.dynamic_gap_frames = (
            dynamic_gap_frames
        )

        self.dynamic_window_frames = (
            dynamic_window_frames
        )

        self.dynamic_required_motion_frames = (
            dynamic_required_motion_frames
        )

        self.static_prediction_frames = (
            static_prediction_frames
        )

        self.unlock_label_frames = (
            unlock_label_frames
        )

        self.mn_confidence_threshold = (
            mn_confidence_threshold
        )

        self.mn_window_frames = (
            mn_window_frames
        )

        self.mn_required_votes = (
            mn_required_votes
        )

        self.mn_confirmation_frames = (
            mn_confirmation_frames
        )

        self.locked_dynamic_window_frames = (
            locked_dynamic_window_frames
        )

        self.locked_dynamic_required_frames = (
            locked_dynamic_required_frames
        )

        # ======================================================
        # Main state
        # ======================================================

        self.mode = self.NONE

        # ======================================================
        # Feature state
        # ======================================================

        self.previous_raw_features = None
        self.previous_features = None

        self.feature_history = deque(
            maxlen=self.smoothing_window
        )

        # Recent raw landmark movement values.
        self.raw_motion_history = deque(
            maxlen=self.dynamic_window_frames
        )

        # ======================================================
        # Initialization
        # ======================================================

        self.initialization_count = 0

        # ======================================================
        # Gesture locking
        # ======================================================

        self.locked_features = None

        # ------------------------------------------------------
        # THIS is the important new state.
        #
        # The identity of the committed gesture is stored
        # separately from its feature representation.
        # ------------------------------------------------------

        self.locked_label = None

        # Different-label evidence while locked.
        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        # Dynamic movement evidence while locked.
        self.locked_motion_history = deque(
            maxlen=self.locked_dynamic_window_frames
        )

        # ======================================================
        # Static prediction
        # ======================================================

        self.static_candidate = None
        self.static_candidate_count = 0

        # ======================================================
        # M / N history
        # ======================================================

        self.mn_prediction_history = deque(
            maxlen=self.mn_window_frames
        )

        self.mn_stable_prediction = None
        self.mn_stable_count = 0

        # ======================================================
        # Dynamic candidate
        # ======================================================

        self.dynamic_motion_count = 0
        self.dynamic_gap_count = 0

        self.dynamic_candidate_frames = []

        self.dynamic_motion_history = deque(
            maxlen=self.dynamic_window_frames
        )

        # ======================================================
        # Last visible prediction
        # ======================================================

        self.last_prediction = None
        self.last_confidence = 0.0

    # ==========================================================
    # Static feature smoothing
    # ==========================================================

    def _smooth_features(
        self,
        features
    ):

        current = np.asarray(
            features,
            dtype=np.float32
        )

        self.feature_history.append(
            current.copy()
        )

        stacked = np.stack(
            list(self.feature_history),
            axis=0
        )

        return np.mean(
            stacked,
            axis=0
        ).astype(
            np.float32
        )

    # ==========================================================
    # Raw landmark motion
    # ==========================================================

    def _calculate_raw_motion(
        self,
        current_features
    ):

        if (
            current_features is None
            or self.previous_raw_features is None
        ):

            return 0.0

        current = np.asarray(
            current_features,
            dtype=np.float32
        )

        previous = np.asarray(
            self.previous_raw_features,
            dtype=np.float32
        )

        if current.shape != previous.shape:

            return 0.0

        # ------------------------------------------------------
        # First 42 values are the normalized landmark coordinates.
        # The remaining 28 values are geometric features.
        #
        # Dynamic movement uses only the landmark coordinates.
        # ------------------------------------------------------

        if (
            current.ndim != 1
            or current.shape[0] < 42
            or previous.shape[0] < 42
        ):

            return 0.0

        current_landmarks = current[:42]
        previous_landmarks = previous[:42]

        difference = np.abs(
            current_landmarks
            - previous_landmarks
        )

        # Median is deliberately used instead of mean.
        #
        # A few noisy landmarks should not dominate the entire
        # motion measurement.
        return float(
            np.median(difference)
        )

    # ==========================================================
    # Locked-feature difference
    #
    # This is NO LONGER used to unlock a gesture.
    #
    # It is retained only as diagnostic/internal information.
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

        if current.shape != locked.shape:

            return 0.0

        return float(
            np.mean(
                np.abs(
                    current - locked
                )
            )
        )

    # ==========================================================
    # Static candidate
    # ==========================================================

    def _update_static_candidate(
        self,
        prediction
    ):

        if (
            prediction is not None
            and prediction == self.static_candidate
        ):

            self.static_candidate_count += 1

        else:

            self.static_candidate = prediction
            self.static_candidate_count = 1

    # ==========================================================
    # M/N history
    # ==========================================================

    def _update_mn_history(
        self,
        prediction,
        confidence
    ):

        self.mn_prediction_history.append(
            (
                prediction,
                float(confidence)
            )
        )

        valid = [
            item
            for item in self.mn_prediction_history
            if (
                item[0] in ("M", "N")
                and item[1]
                >= self.mn_confidence_threshold
            )
        ]

        if not valid:

            self.mn_stable_prediction = None
            self.mn_stable_count = 0

            return

        m_votes = sum(
            1
            for label, _ in valid
            if label == "M"
        )

        n_votes = sum(
            1
            for label, _ in valid
            if label == "N"
        )

        if m_votes == 0 and n_votes == 0:

            self.mn_stable_prediction = None
            self.mn_stable_count = 0

            return

        if m_votes >= n_votes:

            winner = "M"
            winner_votes = m_votes

        else:

            winner = "N"
            winner_votes = n_votes

        if (
            winner
            == self.mn_stable_prediction
        ):

            self.mn_stable_count += 1

        else:

            self.mn_stable_prediction = winner
            self.mn_stable_count = 1

        if (
            winner_votes
            < self.mn_required_votes
        ):

            return

    # ==========================================================
    # M/N candidate
    # ==========================================================

    def _get_mn_candidate(
        self
    ):

        valid = [
            item
            for item in self.mn_prediction_history
            if (
                item[0] in ("M", "N")
                and item[1]
                >= self.mn_confidence_threshold
            )
        ]

        if not valid:

            return None, 0.0, 0

        m_votes = sum(
            1
            for label, _ in valid
            if label == "M"
        )

        n_votes = sum(
            1
            for label, _ in valid
            if label == "N"
        )

        if m_votes >= n_votes:

            label = "M"
            votes = m_votes

        else:

            label = "N"
            votes = n_votes

        confidences = [
            confidence
            for prediction, confidence
            in valid
            if prediction == label
        ]

        confidence = (
            float(np.mean(confidences))
            if confidences
            else 0.0
        )

        return (
            label,
            confidence,
            votes
        )

    # ==========================================================
    # M/N stable
    # ==========================================================

    def _is_stable_mn(
        self
    ):

        (
            label,
            confidence,
            votes
        ) = self._get_mn_candidate()

        if label is None:

            return False

        return (
            votes
            >= self.mn_required_votes
            and confidence
            >= self.mn_confidence_threshold
            and self.mn_stable_count
            >= self.mn_confirmation_frames
        )

    # ==========================================================
    # Reset dynamic candidate
    # ==========================================================

    def _reset_dynamic_candidate(
        self
    ):

        self.dynamic_motion_count = 0
        self.dynamic_gap_count = 0

        self.dynamic_candidate_frames = []

        self.dynamic_motion_history.clear()

    # ==========================================================
    # Reset static tracking
    # ==========================================================

    def _reset_static_tracking(
        self
    ):

        self.static_candidate = None
        self.static_candidate_count = 0

        self.mn_prediction_history.clear()

        self.mn_stable_prediction = None
        self.mn_stable_count = 0

    # ==========================================================
    # Reset gesture state
    # ==========================================================

    def _reset_gesture_state(
        self
    ):

        self.sequence_detector.reset()

        self._reset_static_tracking()

        self._reset_dynamic_candidate()

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        self.locked_motion_history.clear()

    # ==========================================================
    # Initialize new hand
    # ==========================================================

    def _start_initialization(
        self,
        raw_features
    ):

        self._reset_gesture_state()

        self.locked_features = None
        self.locked_label = None

        self.feature_history.clear()

        smoothed = self._smooth_features(
            raw_features
        )

        self.initialization_count = 1

        self.mode = self.INITIALIZING

        self.previous_raw_features = (
            raw_features.copy()
        )

        self.previous_features = (
            smoothed.copy()
        )

    # ==========================================================
    # Response
    # ==========================================================

    def _response(
        self,
        prediction=None,
        confidence=0.0,
        mode=None,
        sequence_complete=False,
    ):

        if mode is None:

            mode = self.mode

        return {
            "prediction": prediction,
            "confidence": float(
                confidence
            ),
            "mode": mode,
            "sequence_complete": (
                sequence_complete
            ),
        }

    # ==========================================================
    # LOCKED STATE
    # ==========================================================

    def _handle_locked_state(
        self,
        raw_features,
        smoothed_features,
        raw_motion
    ):

        prediction, confidence = (
            self.static_predictor.predict(
                smoothed_features
            )
        )

        self.last_prediction = prediction
        self.last_confidence = float(
            confidence
        )

        # ------------------------------------------------------
        # Always expose the current model prediction.
        #
        # This is important for debugging M/N and other gestures.
        # ------------------------------------------------------

        # ======================================================
        # 1. SAME LABEL
        # ======================================================
        #
        # This is the most important fix.
        #
        # If the model still sees the same committed gesture,
        # movement of the hand does NOT unlock it.
        # ======================================================

        if (
            self.locked_label is not None
            and prediction == self.locked_label
        ):

            self.unlock_candidate = None
            self.unlock_candidate_count = 0

        # ======================================================
        # 2. DIFFERENT STATIC LABEL
        # ======================================================

        else:

            if prediction is not None:

                if (
                    prediction
                    == self.unlock_candidate
                ):

                    self.unlock_candidate_count += 1

                else:

                    self.unlock_candidate = (
                        prediction
                    )

                    self.unlock_candidate_count = 1

            else:

                self.unlock_candidate = None
                self.unlock_candidate_count = 0

            # --------------------------------------------------
            # Do NOT unlock from one bad RF frame.
            # --------------------------------------------------

            if (
                self.unlock_candidate is not None
                and self.unlock_candidate_count
                >= self.unlock_label_frames
            ):

                self.locked_features = None
                self.locked_label = None

                self.unlock_candidate = None
                self.unlock_candidate_count = 0

                self.locked_motion_history.clear()

                self._reset_static_tracking()
                self._reset_dynamic_candidate()

                self.mode = self.STATIC

                self.previous_raw_features = (
                    raw_features.copy()
                )

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=prediction,
                    confidence=confidence,
                    mode=self.STATIC
                )

        # ======================================================
        # 3. DYNAMIC ESCAPE
        # ======================================================
        #
        # A committed static gesture should NOT be unlocked by
        # feature difference.
        #
        # A genuine sustained dynamic trajectory is different:
        # it is allowed to escape the static lock.
        # ======================================================

        moving = (
            raw_motion
            >= self.dynamic_motion_threshold
        )

        self.locked_motion_history.append(
            moving
        )

        recent_movement = sum(
            1
            for value
            in self.locked_motion_history
            if value
        )

        # ------------------------------------------------------
        # Only consider dynamic escape when physical movement
        # is sustained.
        #
        # A few noisy frames cannot do it.
        # ------------------------------------------------------

        if (
            recent_movement
            >= self.locked_dynamic_required_frames
        ):

            # Start dynamic candidate.
            self._reset_dynamic_candidate()

            self.dynamic_candidate_frames = [
                smoothed_features.copy()
            ]

            self.dynamic_motion_count = 1

            self.dynamic_motion_history.append(
                True
            )

            self.mode = (
                self.DYNAMIC_CANDIDATE
            )

            self.previous_raw_features = (
                raw_features.copy()
            )

            self.previous_features = (
                smoothed_features.copy()
            )

            return self._response(
                prediction=prediction,
                confidence=confidence,
                mode=self.DYNAMIC_CANDIDATE
            )

        # ======================================================
        # Remain locked.
        # ======================================================

        self.previous_raw_features = (
            raw_features.copy()
        )

        self.previous_features = (
            smoothed_features.copy()
        )

        return self._response(
            prediction=prediction,
            confidence=confidence,
            mode=self.NONE
        )

    # ==========================================================
    # UPDATE
    # ==========================================================

    def update(
        self,
        features
    ):

        # ======================================================
        # NO HAND
        # ======================================================

        if features is None:

            self.previous_raw_features = None
            self.previous_features = None

            self.locked_features = None
            self.locked_label = None

            self.feature_history.clear()

            self.locked_motion_history.clear()

            self._reset_initialization()
            self._reset_gesture_state()

            self.mode = self.NONE

            return self._response(
                mode=self.NONE
            )

        raw_features = np.asarray(
            features,
            dtype=np.float32
        )

        # ======================================================
        # SMOOTHED STATIC FEATURES
        # ======================================================

        smoothed_features = (
            self._smooth_features(
                raw_features
            )
        )

        # ======================================================
        # RAW MOVEMENT
        # ======================================================

        raw_motion = (
            self._calculate_raw_motion(
                raw_features
            )
        )

        self.raw_motion_history.append(
            raw_motion
        )

        # ======================================================
        # NEW HAND
        # ======================================================

        if (
            self.mode == self.NONE
            and self.previous_raw_features is None
        ):

            self._start_initialization(
                raw_features
            )

            return self._response(
                mode=self.INITIALIZING
            )

        # ======================================================
        # INITIALIZATION
        # ======================================================

        if (
            self.mode
            == self.INITIALIZING
        ):

            self.initialization_count += 1

            self.previous_raw_features = (
                raw_features.copy()
            )

            self.previous_features = (
                smoothed_features.copy()
            )

            if (
                self.initialization_count
                < self.initialization_frames
            ):

                return self._response(
                    mode=self.INITIALIZING
                )

            self.initialization_count = 0

            self._reset_static_tracking()
            self._reset_dynamic_candidate()

            self.mode = self.STATIC

            return self._response(
                mode=self.STATIC
            )

        # ======================================================
        # LOCKED GESTURE
        # ======================================================

        if (
            self.mode == self.NONE
            and self.locked_label is not None
        ):

            return self._handle_locked_state(
                raw_features,
                smoothed_features,
                raw_motion
            )

        # ======================================================
        # STATIC
        # ======================================================

        if self.mode == self.STATIC:

            prediction, confidence = (
                self.static_predictor.predict(
                    smoothed_features
                )
            )

            self.last_prediction = prediction
            self.last_confidence = float(
                confidence
            )

            # --------------------------------------------------
            # Normal static stability
            # --------------------------------------------------

            self._update_static_candidate(
                prediction
            )

            # --------------------------------------------------
            # M/N temporal stabilization
            # --------------------------------------------------

            self._update_mn_history(
                prediction,
                confidence
            )

            (
                mn_prediction,
                mn_confidence,
                mn_votes
            ) = self._get_mn_candidate()

            mn_stable = (
                self._is_stable_mn()
            )

            display_prediction = prediction
            display_confidence = float(
                confidence
            )

            if mn_stable:

                display_prediction = (
                    mn_prediction
                )

                display_confidence = max(
                    float(confidence),
                    float(mn_confidence)
                )

                self.last_prediction = (
                    display_prediction
                )

                self.last_confidence = (
                    display_confidence
                )

            # --------------------------------------------------
            # Dynamic candidate
            #
            # Static mode requires real raw landmark movement.
            # --------------------------------------------------

            self.dynamic_motion_history.append(
                raw_motion
                >= self.dynamic_motion_threshold
            )

            recent_movement = sum(
                1
                for value
                in self.dynamic_motion_history
                if value
            )

            strong_static = (
                confidence
                >= self.dynamic_static_guard_threshold
            )

            # --------------------------------------------------
            # Strong static prediction + weak movement:
            # stay static.
            # --------------------------------------------------

            if (
                strong_static
                and recent_movement
                < self.dynamic_required_motion_frames
            ):

                self._reset_dynamic_candidate()

            # --------------------------------------------------
            # Genuine sustained movement:
            # begin dynamic candidate.
            # --------------------------------------------------

            elif (
                recent_movement
                >= self.dynamic_required_motion_frames
            ):

                self.mode = (
                    self.DYNAMIC_CANDIDATE
                )

                self.dynamic_motion_count = (
                    recent_movement
                )

                self.dynamic_gap_count = 0

                self.dynamic_candidate_frames = [
                    smoothed_features.copy()
                ]

                self.previous_raw_features = (
                    raw_features.copy()
                )

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=display_prediction,
                    confidence=display_confidence,
                    mode=self.DYNAMIC_CANDIDATE
                )

            # --------------------------------------------------
            # Static commit
            # --------------------------------------------------

            mn_confirmed = (
                mn_stable
                and mn_votes
                >= self.mn_required_votes
                and mn_confidence
                >= self.mn_confidence_threshold
            )

            normal_static_confirmed = (
                self.static_candidate_count
                >= self.static_prediction_frames
                and confidence
                >= self.static_confidence_threshold
            )

            if (
                mn_confirmed
                or normal_static_confirmed
            ):

                self.locked_features = (
                    smoothed_features.copy()
                )

                self.locked_label = (
                    display_prediction
                )

                self.unlock_candidate = None
                self.unlock_candidate_count = 0

                self.locked_motion_history.clear()

                self._reset_dynamic_candidate()

                self.mode = self.NONE

                self.previous_raw_features = (
                    raw_features.copy()
                )

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=display_prediction,
                    confidence=display_confidence,
                    mode=self.NONE
                )

            self.previous_raw_features = (
                raw_features.copy()
            )

            self.previous_features = (
                smoothed_features.copy()
            )

            return self._response(
                prediction=display_prediction,
                confidence=display_confidence,
                mode=self.STATIC
            )

        # ======================================================
        # DYNAMIC CANDIDATE
        # ======================================================

        if (
            self.mode
            == self.DYNAMIC_CANDIDATE
        ):

            self.dynamic_candidate_frames.append(
                smoothed_features.copy()
            )

            prediction, confidence = (
                self.static_predictor.predict(
                    smoothed_features
                )
            )

            self.last_prediction = prediction
            self.last_confidence = float(
                confidence
            )

            self._update_mn_history(
                prediction,
                confidence
            )

            (
                mn_prediction,
                mn_confidence,
                mn_votes
            ) = self._get_mn_candidate()

            mn_stable = (
                self._is_stable_mn()
            )

            display_prediction = prediction
            display_confidence = float(
                confidence
            )

            if mn_stable:

                display_prediction = (
                    mn_prediction
                )

                display_confidence = max(
                    float(confidence),
                    float(mn_confidence)
                )

            moving = (
                raw_motion
                >= self.dynamic_motion_threshold
            )

            self.dynamic_motion_history.append(
                moving
            )

            if moving:

                self.dynamic_motion_count += 1
                self.dynamic_gap_count = 0

            else:

                self.dynamic_gap_count += 1

                if (
                    self.dynamic_gap_count
                    > self.dynamic_gap_frames
                ):

                    self._reset_dynamic_candidate()

                    self._reset_static_tracking()

                    self.mode = self.STATIC

                    self.previous_raw_features = (
                        raw_features.copy()
                    )

                    self.previous_features = (
                        smoothed_features.copy()
                    )

                    return self._response(
                        prediction=display_prediction,
                        confidence=display_confidence,
                        mode=self.STATIC
                    )

            recent_movement = sum(
                1
                for value
                in self.dynamic_motion_history
                if value
            )

            # --------------------------------------------------
            # A strong static pose can cancel an immature
            # dynamic transition.
            # --------------------------------------------------

            if (
                confidence
                >= self.dynamic_static_guard_threshold
                and recent_movement
                < self.dynamic_required_motion_frames
            ):

                self._reset_dynamic_candidate()

                self.mode = self.STATIC

                self.previous_raw_features = (
                    raw_features.copy()
                )

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=display_prediction,
                    confidence=display_confidence,
                    mode=self.STATIC
                )

            # --------------------------------------------------
            # Confirm dynamic.
            # --------------------------------------------------

            if (
                self.dynamic_motion_count
                >= self.dynamic_start_frames
                and recent_movement
                >= self.dynamic_required_motion_frames
            ):

                self.sequence_detector.start_recording(
                    self.dynamic_candidate_frames
                )

                self._reset_dynamic_candidate()
                self._reset_static_tracking()

                # The old committed static identity must no longer
                # suppress this dynamic gesture.
                self.locked_features = None
                self.locked_label = None

                self.mode = self.DYNAMIC

                self.previous_raw_features = (
                    raw_features.copy()
                )

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    mode=self.DYNAMIC
                )

            self.previous_raw_features = (
                raw_features.copy()
            )

            self.previous_features = (
                smoothed_features.copy()
            )

            return self._response(
                prediction=display_prediction,
                confidence=display_confidence,
                mode=self.DYNAMIC_CANDIDATE
            )

        # ======================================================
        # DYNAMIC
        # ======================================================

        if self.mode == self.DYNAMIC:

            completed_sequence = (
                self.sequence_detector.update(
                    smoothed_features
                )
            )

            self.previous_raw_features = (
                raw_features.copy()
            )

            self.previous_features = (
                smoothed_features.copy()
            )

            if completed_sequence is None:

                return self._response(
                    mode=self.DYNAMIC
                )

            prediction, confidence = (
                self.dynamic_predictor.predict(
                    completed_sequence
                )
            )

            self.last_prediction = prediction
            self.last_confidence = float(
                confidence
            )

            if (
                prediction is not None
                and confidence
                >= self.dynamic_confidence_threshold
            ):

                self.locked_features = (
                    smoothed_features.copy()
                )

                self.locked_label = prediction

                self.unlock_candidate = None
                self.unlock_candidate_count = 0

                self._reset_gesture_state()

                self.mode = self.NONE

                return self._response(
                    prediction=prediction,
                    confidence=confidence,
                    mode=self.NONE,
                    sequence_complete=True
                )

            # --------------------------------------------------
            # Failed dynamic prediction.
            # Return to static without locking an invalid result.
            # --------------------------------------------------

            self._reset_gesture_state()

            self.locked_features = None
            self.locked_label = None

            self.mode = self.STATIC

            return self._response(
                mode=self.STATIC
            )

        # ======================================================
        # FALLBACK
        # ======================================================

        self._reset_gesture_state()

        self.locked_features = None
        self.locked_label = None

        self.mode = self.STATIC

        self.previous_raw_features = (
            raw_features.copy()
        )

        self.previous_features = (
            smoothed_features.copy()
        )

        return self._response(
            mode=self.STATIC
        )

    # ==========================================================
    # INITIALIZATION RESET
    # ==========================================================

    def _reset_initialization(
        self
    ):

        self.initialization_count = 0

    # ==========================================================
    # GETTERS
    # ==========================================================

    def get_mode(
        self
    ):

        return self.mode

    def get_sequence_length(
        self
    ):

        return (
            self.sequence_detector
            .get_sequence_length()
        )

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(
        self
    ):

        self.sequence_detector.reset()

        self.mode = self.NONE

        self.previous_raw_features = None
        self.previous_features = None

        self.locked_features = None
        self.locked_label = None

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        self.initialization_count = 0

        self.feature_history.clear()

        self.raw_motion_history.clear()
        self.locked_motion_history.clear()

        self._reset_static_tracking()
        self._reset_dynamic_candidate()

        self.last_prediction = None
        self.last_confidence = 0.0
