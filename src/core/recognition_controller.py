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

        # ======================================================
        # Static prediction
        # ======================================================

        static_prediction_frames=4,

        # M/N commonly have lower confidence than easier
        # static signs. Do not accept a single weak frame.
        mn_confidence_threshold=0.40,
        mn_window_frames=8,
        mn_required_votes=5,

        # ======================================================
        # Dynamic trajectory
        # ======================================================

        # Minimum frame-to-frame movement of the index fingertip
        # relative to the wrist.
        dynamic_step_threshold=0.012,

        # Minimum total travelled distance before dynamic mode
        # can start.
        dynamic_path_threshold=0.075,

        # Minimum displacement from the point where the candidate
        # started. Prevents pure back-and-forth jitter from being
        # considered a gesture.
        dynamic_net_displacement_threshold=0.030,

        # Number of meaningful trajectory steps required.
        dynamic_start_frames=5,

        # Maximum low-motion gap while building a trajectory.
        dynamic_gap_frames=3,

        # Number of recent trajectory samples examined.
        dynamic_window_frames=10,

        # ======================================================
        # Static lock
        # ======================================================

        # A different static prediction must persist before
        # unlocking the committed gesture.
        unlock_change_frames=5,

        # ======================================================
        # Hand initialization
        # ======================================================

        initialization_frames=8,
    ):

        self.static_predictor = static_predictor
        self.dynamic_predictor = dynamic_predictor

        # ======================================================
        # Sequence detector
        # ======================================================

        self.sequence_detector = SequenceDetector(
            start_threshold=dynamic_step_threshold
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

        self.static_prediction_frames = (
            static_prediction_frames
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

        self.dynamic_step_threshold = (
            dynamic_step_threshold
        )

        self.dynamic_path_threshold = (
            dynamic_path_threshold
        )

        self.dynamic_net_displacement_threshold = (
            dynamic_net_displacement_threshold
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

        # ======================================================
        # Previous features
        # ======================================================

        self.previous_features = None

        # ======================================================
        # Feature smoothing
        # ======================================================

        self.feature_history = deque(
            maxlen=5
        )

        # ======================================================
        # Hand initialization
        # ======================================================

        self.initialization_count = 0

        # ======================================================
        # Gesture lock
        # ======================================================

        self.locked_features = None

        # IMPORTANT:
        # The committed label is the identity of the lock.
        self.locked_label = None

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        # ======================================================
        # Static prediction tracking
        # ======================================================

        self.static_candidate = None
        self.static_candidate_count = 0

        # ======================================================
        # M/N tracking
        # ======================================================

        self.mn_history = deque(
            maxlen=self.mn_window_frames
        )

        # ======================================================
        # Dynamic trajectory tracking
        # ======================================================

        self.dynamic_motion_count = 0
        self.dynamic_gap_count = 0

        self.dynamic_candidate_frames = []

        self.dynamic_motion_history = deque(
            maxlen=self.dynamic_window_frames
        )

        # Index fingertip trajectory.
        self.dynamic_start_point = None
        self.dynamic_current_point = None

        self.dynamic_path_length = 0.0
        self.dynamic_net_displacement = 0.0

        # ======================================================
        # Last prediction
        # ======================================================

        self.last_prediction = None
        self.last_confidence = 0.0

    # ==========================================================
    # FEATURE SMOOTHING
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
    # RECONSTRUCT NORMALIZED LANDMARKS
    #
    # The first 42 features are:
    #
    # x0, y0, x1, y1, ... x20, y20
    #
    # They are already wrist-relative and scale-normalized by
    # LandmarkProcessor.
    # ==========================================================

    def _get_landmarks(
        self,
        features
    ):

        if features is None:

            return None

        values = np.asarray(
            features,
            dtype=np.float32
        )

        if (
            values.ndim != 1
            or values.shape[0] < 42
        ):

            return None

        landmarks = values[:42].reshape(
            21,
            2
        )

        return landmarks

    # ==========================================================
    # INDEX FINGERTIP POSITION
    #
    # Landmark 8 = index fingertip.
    #
    # Because the feature representation is wrist-relative,
    # this gives us the index-finger trajectory relative to
    # the hand itself.
    #
    # This is much more useful for J/Z than measuring changes
    # across all 70 features.
    # ==========================================================

    def _get_index_tip(
        self,
        features
    ):

        landmarks = self._get_landmarks(
            features
        )

        if landmarks is None:

            return None

        return landmarks[8].copy()

    # ==========================================================
    # TRAJECTORY UPDATE
    # ==========================================================

    def _update_trajectory(
        self,
        features
    ):

        point = self._get_index_tip(
            features
        )

        if point is None:

            return {
                "step": 0.0,
                "moving": False,
                "path": self.dynamic_path_length,
                "net": self.dynamic_net_displacement,
            }

        # ------------------------------------------------------
        # First trajectory point
        # ------------------------------------------------------

        if self.dynamic_current_point is None:

            self.dynamic_current_point = (
                point.copy()
            )

            if self.dynamic_start_point is None:

                self.dynamic_start_point = (
                    point.copy()
                )

            return {
                "step": 0.0,
                "moving": False,
                "path": self.dynamic_path_length,
                "net": self.dynamic_net_displacement,
            }

        # ------------------------------------------------------
        # Frame-to-frame index fingertip movement
        # ------------------------------------------------------

        step = float(
            np.linalg.norm(
                point
                - self.dynamic_current_point
            )
        )

        moving = (
            step
            >= self.dynamic_step_threshold
        )

        # ------------------------------------------------------
        # Only meaningful movement contributes to path length.
        #
        # Tiny movements are treated as jitter rather than
        # accumulating indefinitely.
        # ------------------------------------------------------

        if moving:

            self.dynamic_path_length += step

            self.dynamic_motion_count += 1

            self.dynamic_gap_count = 0

        else:

            self.dynamic_gap_count += 1

        # ------------------------------------------------------
        # Current displacement from trajectory start.
        # ------------------------------------------------------

        if self.dynamic_start_point is not None:

            self.dynamic_net_displacement = float(
                np.linalg.norm(
                    point
                    - self.dynamic_start_point
                )
            )

        self.dynamic_current_point = (
            point.copy()
        )

        self.dynamic_motion_history.append(
            moving
        )

        # Keep trajectory state bounded.
        if (
            len(self.dynamic_motion_history)
            > self.dynamic_window_frames
        ):

            self.dynamic_motion_history.popleft()

        return {
            "step": step,
            "moving": moving,
            "path": self.dynamic_path_length,
            "net": self.dynamic_net_displacement,
        }

    # ==========================================================
    # RESET TRAJECTORY
    # ==========================================================

    def _reset_trajectory(
        self
    ):

        self.dynamic_motion_count = 0
        self.dynamic_gap_count = 0

        self.dynamic_candidate_frames = []

        self.dynamic_motion_history.clear()

        self.dynamic_start_point = None
        self.dynamic_current_point = None

        self.dynamic_path_length = 0.0
        self.dynamic_net_displacement = 0.0

    # ==========================================================
    # STATIC RESET
    # ==========================================================

    def _reset_static_tracking(
        self
    ):

        self.static_candidate = None
        self.static_candidate_count = 0

        self.mn_history.clear()

    # ==========================================================
    # GESTURE STATE RESET
    # ==========================================================

    def _reset_gesture_state(
        self
    ):

        self.sequence_detector.reset()

        self._reset_static_tracking()

        self._reset_trajectory()

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

    # ==========================================================
    # INITIALIZATION RESET
    # ==========================================================

    def _reset_initialization(
        self
    ):

        self.initialization_count = 0

    # ==========================================================
    # RESPONSE
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
    # LOCK GESTURE
    # ==========================================================

    def _lock_gesture(
        self,
        label,
        features
    ):

        self.locked_label = label

        self.locked_features = (
            np.asarray(
                features,
                dtype=np.float32
            ).copy()
        )

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        self._reset_trajectory()

    # ==========================================================
    # UNLOCK GESTURE
    # ==========================================================

    def _unlock_gesture(
        self
    ):

        self.locked_label = None
        self.locked_features = None

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        self._reset_trajectory()

    # ==========================================================
    # UPDATE DIFFERENT-LABEL CANDIDATE
    #
    # A locked gesture is NOT released because of feature
    # distance anymore.
    #
    # The model must consistently identify a different gesture.
    # ==========================================================

    def _update_unlock_candidate(
        self,
        prediction
    ):

        if (
            prediction is None
            or prediction == self.locked_label
        ):

            self.unlock_candidate = None
            self.unlock_candidate_count = 0

            return False

        if (
            prediction
            == self.unlock_candidate
        ):

            self.unlock_candidate_count += 1

        else:

            self.unlock_candidate = prediction
            self.unlock_candidate_count = 1

        return (
            self.unlock_candidate_count
            >= self.unlock_change_frames
        )

    # ==========================================================
    # M/N HISTORY
    # ==========================================================

    def _update_mn_history(
        self,
        prediction,
        confidence
    ):

        if prediction not in (
            "M",
            "N"
        ):

            self.mn_history.clear()

            return

        self.mn_history.append(
            (
                prediction,
                float(confidence)
            )
        )

    # ==========================================================
    # GET STABLE M/N
    # ==========================================================

    def _get_stable_mn(
        self
    ):

        if not self.mn_history:

            return None, 0.0, 0

        valid = [
            item
            for item in self.mn_history
            if item[1]
            >= self.mn_confidence_threshold
        ]

        if not valid:

            return None, 0.0, 0

        m_votes = sum(
            1
            for label, _
            in valid
            if label == "M"
        )

        n_votes = sum(
            1
            for label, _
            in valid
            if label == "N"
        )

        if m_votes >= n_votes:

            winner = "M"
            votes = m_votes

        else:

            winner = "N"
            votes = n_votes

        if (
            votes
            < self.mn_required_votes
        ):

            return None, 0.0, votes

        confidences = [
            confidence
            for label, confidence
            in valid
            if label == winner
        ]

        average_confidence = (
            float(
                np.mean(
                    confidences
                )
            )
            if confidences
            else 0.0
        )

        return (
            winner,
            average_confidence,
            votes
        )

    # ==========================================================
    # UPDATE STATIC CANDIDATE
    # ==========================================================

    def _update_static_candidate(
        self,
        prediction
    ):

        if (
            prediction is not None
            and prediction
            == self.static_candidate
        ):

            self.static_candidate_count += 1

        else:

            self.static_candidate = prediction
            self.static_candidate_count = 1

    # ==========================================================
    # START DYNAMIC CANDIDATE
    # ==========================================================

    def _start_dynamic_candidate(
        self,
        features
    ):

        self.mode = (
            self.DYNAMIC_CANDIDATE
        )

        self.dynamic_candidate_frames = [
            np.asarray(
                features,
                dtype=np.float32
            ).copy()
        ]

        # The first point is the origin of this candidate.
        index_tip = self._get_index_tip(
            features
        )

        self.dynamic_start_point = (
            None
            if index_tip is None
            else index_tip.copy()
        )

        self.dynamic_current_point = (
            None
            if index_tip is None
            else index_tip.copy()
        )

        self.dynamic_path_length = 0.0
        self.dynamic_net_displacement = 0.0

        self.dynamic_motion_count = 0
        self.dynamic_gap_count = 0

        self.dynamic_motion_history.clear()

    # ==========================================================
    # CANCEL DYNAMIC CANDIDATE
    # ==========================================================

    def _cancel_dynamic_candidate(
        self
    ):

        self._reset_trajectory()

        self.mode = self.STATIC

    # ==========================================================
    # LOCKED STATE
    # ==========================================================

    def _handle_locked_state(
        self,
        features,
        trajectory
    ):

        prediction, confidence = (
            self.static_predictor.predict(
                features
            )
        )

        self.last_prediction = prediction
        self.last_confidence = float(
            confidence
        )

        # ------------------------------------------------------
        # SAME LABEL
        #
        # Movement does not unlock the gesture.
        # ------------------------------------------------------

        if (
            self.locked_label is not None
            and prediction == self.locked_label
        ):

            self.unlock_candidate = None
            self.unlock_candidate_count = 0

        else:

            # --------------------------------------------------
            # A different static label must persist.
            # --------------------------------------------------

            if self._update_unlock_candidate(
                prediction
            ):

                self._unlock_gesture()

                self.mode = self.STATIC

                self.previous_features = (
                    features.copy()
                )

                return self._response(
                    prediction=prediction,
                    confidence=confidence,
                    mode=self.STATIC
                )

        # ------------------------------------------------------
        # Dynamic escape from a locked static gesture.
        #
        # A trajectory must be substantial enough to represent
        # an intentional dynamic sign.
        # ------------------------------------------------------

        if (
            trajectory["moving"]
            and trajectory["path"]
            >= self.dynamic_path_threshold
            and trajectory["net"]
            >= self.dynamic_net_displacement_threshold
        ):

            self._start_dynamic_candidate(
                features
            )

            self.previous_features = (
                features.copy()
            )

            return self._response(
                prediction=prediction,
                confidence=confidence,
                mode=self.DYNAMIC_CANDIDATE
            )

        self.previous_features = (
            features.copy()
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

            self.previous_features = None

            self.locked_features = None
            self.locked_label = None

            self.feature_history.clear()

            self._reset_initialization()
            self._reset_gesture_state()

            self.mode = self.NONE

            return self._response(
                mode=self.NONE
            )

        current_features = np.asarray(
            features,
            dtype=np.float32
        )

        # ======================================================
        # HAND ENTERED
        # ======================================================

        if (
            self.mode == self.NONE
            and self.previous_features is None
        ):

            self.feature_history.clear()

            self._reset_gesture_state()

            self.locked_features = None
            self.locked_label = None

            self.feature_history.append(
                current_features.copy()
            )

            self.initialization_count = 1

            self.mode = (
                self.INITIALIZING
            )

            self.previous_features = (
                current_features.copy()
            )

            return self._response(
                mode=self.INITIALIZING
            )

        # ======================================================
        # SMOOTH
        # ======================================================

        smoothed_features = (
            self._smooth_features(
                current_features
            )
        )

        # ======================================================
        # INITIALIZATION
        # ======================================================

        if (
            self.mode
            == self.INITIALIZING
        ):

            self.initialization_count += 1

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
            self._reset_trajectory()

            self.mode = self.STATIC

            return self._response(
                mode=self.STATIC
            )

        # ======================================================
        # TRAJECTORY UPDATE
        #
        # This is intentionally based on the normalized index
        # fingertip rather than all 70 features.
        # ======================================================

        trajectory = (
            self._update_trajectory(
                smoothed_features
            )
        )

        # ======================================================
        # LOCKED STATIC GESTURE
        # ======================================================

        if (
            self.mode == self.NONE
            and self.locked_label is not None
        ):

            return self._handle_locked_state(
                smoothed_features,
                trajectory
            )

        # ======================================================
        # NONE WITHOUT LOCK
        # ======================================================

        if (
            self.mode == self.NONE
            and self.locked_label is None
        ):

            self.mode = self.STATIC

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
            ) = self._get_stable_mn()

            # --------------------------------------------------
            # NORMAL static stability
            # --------------------------------------------------

            self._update_static_candidate(
                prediction
            )

            # --------------------------------------------------
            # DYNAMIC GATE
            #
            # IMPORTANT:
            #
            # Do NOT enter dynamic mode just because one frame
            # moved.
            #
            # First establish a meaningful index-finger path.
            # --------------------------------------------------

            trajectory_ready = (
                trajectory["path"]
                >= self.dynamic_path_threshold
                and trajectory["net"]
                >= self.dynamic_net_displacement_threshold
                and self.dynamic_motion_count
                >= self.dynamic_start_frames
            )

            # --------------------------------------------------
            # Strong M/N/static prediction protection.
            #
            # A confident static prediction with only ordinary
            # movement remains static.
            #
            # We do NOT permanently block dynamic recognition.
            # A sufficiently strong trajectory can still pass.
            # --------------------------------------------------

            if (
                prediction in ("M", "N")
                and confidence
                >= self.mn_confidence_threshold
                and not trajectory_ready
            ):

                self._reset_trajectory()

            elif trajectory_ready:

                self._start_dynamic_candidate(
                    smoothed_features
                )

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=prediction,
                    confidence=confidence,
                    mode=self.DYNAMIC_CANDIDATE
                )

            # --------------------------------------------------
            # M/N confirmed
            # --------------------------------------------------

            if (
                mn_prediction is not None
                and mn_votes
                >= self.mn_required_votes
            ):

                self._lock_gesture(
                    mn_prediction,
                    smoothed_features
                )

                self._reset_static_tracking()

                self.mode = self.NONE

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=mn_prediction,
                    confidence=mn_confidence,
                    mode=self.NONE
                )

            # --------------------------------------------------
            # Normal static confirmed
            # --------------------------------------------------

            if (
                self.static_candidate_count
                >= self.static_prediction_frames
                and confidence
                >= self.static_confidence_threshold
            ):

                self._lock_gesture(
                    prediction,
                    smoothed_features
                )

                self._reset_static_tracking()

                self.mode = self.NONE

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=prediction,
                    confidence=confidence,
                    mode=self.NONE
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
        # DYNAMIC CANDIDATE
        # ======================================================

        if (
            self.mode
            == self.DYNAMIC_CANDIDATE
        ):

            self.dynamic_candidate_frames.append(
                smoothed_features.copy()
            )

            # --------------------------------------------------
            # Static predictor remains active during candidate.
            #
            # This lets a static M/N pose cancel the candidate
            # instead of blindly proceeding toward J/Z.
            # --------------------------------------------------

            static_prediction, static_confidence = (
                self.static_predictor.predict(
                    smoothed_features
                )
            )

            self.last_prediction = (
                static_prediction
            )

            self.last_confidence = (
                float(static_confidence)
            )

            # --------------------------------------------------
            # Update trajectory using the current frame.
            # --------------------------------------------------

            trajectory = (
                self._update_trajectory(
                    smoothed_features
                )
            )

            # --------------------------------------------------
            # If movement has stopped before enough trajectory
            # evidence exists, cancel dynamic detection.
            # --------------------------------------------------

            if (
                not trajectory["moving"]
            ):

                self.dynamic_gap_count += 1

            else:

                self.dynamic_gap_count = 0

            # --------------------------------------------------
            # Strong static prediction + insufficient trajectory
            # means this was almost certainly pose jitter.
            # --------------------------------------------------

            if (
                static_prediction in ("M", "N")
                and static_confidence
                >= self.mn_confidence_threshold
                and (
                    trajectory["path"]
                    < self.dynamic_path_threshold
                    or trajectory["net"]
                    < self.dynamic_net_displacement_threshold
                )
            ):

                self._cancel_dynamic_candidate()

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=static_prediction,
                    confidence=static_confidence,
                    mode=self.STATIC
                )

            # --------------------------------------------------
            # Movement stopped for too long.
            # --------------------------------------------------

            if (
                self.dynamic_gap_count
                > self.dynamic_gap_frames
            ):

                self._cancel_dynamic_candidate()

                self._reset_static_tracking()

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=static_prediction,
                    confidence=static_confidence,
                    mode=self.STATIC
                )

            # --------------------------------------------------
            # Confirm genuine trajectory.
            # --------------------------------------------------

            trajectory_ready = (
                trajectory["path"]
                >= self.dynamic_path_threshold
                and trajectory["net"]
                >= self.dynamic_net_displacement_threshold
                and self.dynamic_motion_count
                >= self.dynamic_start_frames
            )

            if trajectory_ready:

                # --------------------------------------------------
                # Feed the accumulated candidate frames to the
                # existing sequence detector.
                # --------------------------------------------------

                self.sequence_detector.start_recording(
                    self.dynamic_candidate_frames
                )

                self._reset_trajectory()

                self._reset_static_tracking()

                # Dynamic recognition is now independent of any
                # previous static lock.
                self.locked_features = None
                self.locked_label = None

                self.unlock_candidate = None
                self.unlock_candidate_count = 0

                self.mode = self.DYNAMIC

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=static_prediction,
                    confidence=static_confidence,
                    mode=self.DYNAMIC
                )

            self.previous_features = (
                smoothed_features.copy()
            )

            return self._response(
                prediction=static_prediction,
                confidence=static_confidence,
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

            self.previous_features = (
                smoothed_features.copy()
            )

            if completed_sequence is None:

                return self._response(
                    mode=self.DYNAMIC
                )

            # --------------------------------------------------
            # LSTM prediction
            # --------------------------------------------------

            prediction, confidence = (
                self.dynamic_predictor.predict(
                    completed_sequence
                )
            )

            self.last_prediction = prediction
            self.last_confidence = float(
                confidence
            )

            # --------------------------------------------------
            # Valid J/Z
            # --------------------------------------------------

            if (
                prediction is not None
                and confidence
                >= self.dynamic_confidence_threshold
            ):

                self._lock_gesture(
                    prediction,
                    smoothed_features
                )

                self._reset_gesture_state()

                self.mode = self.NONE

                self.previous_features = (
                    smoothed_features.copy()
                )

                return self._response(
                    prediction=prediction,
                    confidence=confidence,
                    mode=self.NONE,
                    sequence_complete=True
                )

            # --------------------------------------------------
            # Invalid dynamic sequence.
            # Return safely to static.
            # --------------------------------------------------

            self._reset_gesture_state()

            self.locked_features = None
            self.locked_label = None

            self.mode = self.STATIC

            self.previous_features = (
                smoothed_features.copy()
            )

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

        self.previous_features = (
            smoothed_features.copy()
        )

        return self._response(
            mode=self.STATIC
        )

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

        self.previous_features = None

        self.locked_features = None
        self.locked_label = None

        self.unlock_candidate = None
        self.unlock_candidate_count = 0

        self.initialization_count = 0

        self.feature_history.clear()

        self._reset_static_tracking()
        self._reset_trajectory()

        self.last_prediction = None
        self.last_confidence = 0.0
