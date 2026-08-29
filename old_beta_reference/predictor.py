import joblib
import numpy as np

from src.utils.config import (
    RANDOM_FOREST_MODEL_PATH,
    LABEL_ENCODER_PATH,
)


class Predictor:

    def __init__(self):

        self.model = self.load_model()
        self.label_encoder = self.load_label_encoder()

    def load_model(self):

        try:

            return joblib.load(RANDOM_FOREST_MODEL_PATH)

        except Exception as error:

            raise RuntimeError(
                f"Failed to load model from "
                f"'{RANDOM_FOREST_MODEL_PATH}': {error}"
            )

    def load_label_encoder(self):

        try:

            return joblib.load(LABEL_ENCODER_PATH)

        except Exception as error:

            raise RuntimeError(
                f"Failed to load label encoder from "
                f"'{LABEL_ENCODER_PATH}': {error}"
            )

    def predict(self, features):

        if features is None:
            return None, 0.0

        try:

            features = np.asarray(
                features,
                dtype=np.float32
            )

            if features.ndim != 1:
                raise ValueError("Features must be a 1D array.")

            features = features.reshape(1, -1)

            predicted_class = self.model.predict(features)[0]

            probabilities = self.model.predict_proba(features)[0]

            confidence = probabilities[predicted_class]

            predicted_label = self.label_encoder.inverse_transform(
                [predicted_class]
            )[0]

            return predicted_label, confidence

        except Exception as error:

            raise RuntimeError(
                f"Prediction failed: {error}"
            )