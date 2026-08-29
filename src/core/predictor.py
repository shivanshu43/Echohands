import joblib
import numpy as np


class Predictor:

    def __init__(
        self,
        model_path,
        encoder_path,
    ):

        self.model_path = model_path

        self.encoder_path = encoder_path

        self.model = self.load_model()

        self.label_encoder = (
            self.load_label_encoder()
        )

        # ======================================================
        # Diagnostic information
        #
        # Does NOT change predict() return values.
        # ======================================================

        self.last_top_predictions = []

    # ======================================================
    # Load Random Forest
    # ======================================================

    def load_model(self):

        try:

            return joblib.load(
                self.model_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Failed to load model from "
                f"'{self.model_path}': {error}"
            )

    # ======================================================
    # Load Label Encoder
    # ======================================================

    def load_label_encoder(self):

        try:

            return joblib.load(
                self.encoder_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Failed to load label encoder from "
                f"'{self.encoder_path}': {error}"
            )

    # ======================================================
    # Prediction
    # ======================================================

    def predict(
        self,
        features
    ):

        if features is None:

            self.last_top_predictions = []

            return None, 0.0

        try:

            features = np.asarray(
                features,
                dtype=np.float32
            )

            if features.ndim != 1:

                raise ValueError(
                    "Features must be a 1D array."
                )

            features = features.reshape(
                1,
                -1
            )

            predicted_class = (
                self.model.predict(
                    features
                )[0]
            )

            probabilities = (
                self.model.predict_proba(
                    features
                )[0]
            )

            confidence = float(
                probabilities[
                    predicted_class
                ]
            )

            predicted_label = (
                self.label_encoder
                .inverse_transform(
                    [predicted_class]
                )[0]
            )

            # ==================================================
            # Store top-3 model predictions for diagnostics.
            #
            # This does NOT modify the existing return format.
            # ==================================================

            ranked_indices = np.argsort(
                probabilities
            )[::-1]

            top_predictions = []

            for class_index in ranked_indices[:3]:

                label = (
                    self.label_encoder
                    .inverse_transform(
                        [class_index]
                    )[0]
                )

                probability = float(
                    probabilities[
                        class_index
                    ]
                )

                top_predictions.append(
                    (
                        str(label),
                        probability
                    )
                )

            self.last_top_predictions = (
                top_predictions
            )

            return (
                predicted_label,
                confidence
            )

        except Exception as error:

            raise RuntimeError(
                f"Prediction failed: {error}"
            )