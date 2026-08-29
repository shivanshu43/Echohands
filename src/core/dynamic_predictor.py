import numpy as np
import tensorflow as tf


class DynamicPredictor:

    TARGET_FRAMES = 40

    FEATURES_PER_FRAME = 70

    def __init__(
        self,
        model_path,
        encoder_path,
    ):

        self.model_path = model_path

        self.encoder_path = encoder_path

        # ==================================================
        # Load dynamic LSTM model
        # ==================================================

        self.model = (
            tf.keras.models.load_model(
                self.model_path
            )
        )

        # ==================================================
        # Load dynamic label encoder
        # ==================================================

        self.classes = np.load(
            self.encoder_path,
            allow_pickle=True
        )

    # ======================================================
    # Resample sequence
    # ======================================================

    def _resample_sequence(
        self,
        sequence
    ):

        original_frames = (
            sequence.shape[0]
        )

        if (
            original_frames
            == self.TARGET_FRAMES
        ):

            return sequence

        old_indices = np.linspace(
            0,
            original_frames - 1,
            original_frames
        )

        new_indices = np.linspace(
            0,
            original_frames - 1,
            self.TARGET_FRAMES
        )

        resampled = np.zeros(
            (
                self.TARGET_FRAMES,
                self.FEATURES_PER_FRAME
            ),
            dtype=np.float32
        )

        for feature_index in range(
            self.FEATURES_PER_FRAME
        ):

            resampled[
                :,
                feature_index
            ] = np.interp(
                new_indices,
                old_indices,
                sequence[
                    :,
                    feature_index
                ]
            )

        return resampled

    # ======================================================
    # Prediction
    # ======================================================

    def predict(
        self,
        sequence
    ):

        if sequence is None:

            return None, 0.0

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.ndim != 2:

            raise ValueError(
                "Sequence must have shape "
                f"(frames, "
                f"{self.FEATURES_PER_FRAME})."
            )

        if (
            sequence.shape[1]
            != self.FEATURES_PER_FRAME
        ):

            raise ValueError(
                f"Expected "
                f"{self.FEATURES_PER_FRAME} "
                f"features per frame, "
                f"got "
                f"{sequence.shape[1]}."
            )

        if sequence.shape[0] < 2:

            return None, 0.0

        # ==================================================
        # Normalize sequence length
        # ==================================================

        sequence = (
            self._resample_sequence(
                sequence
            )
        )

        # ==================================================
        # Add batch dimension
        # ==================================================

        sequence = sequence.reshape(
            1,
            self.TARGET_FRAMES,
            self.FEATURES_PER_FRAME
        )

        # ==================================================
        # LSTM prediction
        # ==================================================

        prediction = (
            self.model.predict(
                sequence,
                verbose=0
            )
        )

        predicted_index = int(
            np.argmax(
                prediction[0]
            )
        )

        predicted_class = (
            self.classes[
                predicted_index
            ]
        )

        confidence = float(
            prediction[
                0
            ][
                predicted_index
            ]
        )

        return (
            predicted_class,
            confidence
        )