import math
import numpy as np


class LandmarkProcessor:

    # ============================================================
    # Basic distance
    # ============================================================

    def _distance(self, p1, p2):

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # ============================================================
    # Angle ABC
    # B = middle point
    # ============================================================

    def _angle(self, a, b, c):

        ba = np.array([
            a[0] - b[0],
            a[1] - b[1]
        ])

        bc = np.array([
            c[0] - b[0],
            c[1] - b[1]
        ])

        denominator = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denominator == 0:

            return 0.0

        cosine = (
            np.dot(ba, bc) /
            denominator
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return float(
            np.arccos(cosine)
        )

    # ============================================================
    # Extract features
    # ============================================================

    def extract_features(self, results):

        if (
            results is None
            or not results.multi_hand_landmarks
        ):

            return None

        hand = results.multi_hand_landmarks[0]

        # ========================================================
        # Store landmarks as (x, y)
        # ========================================================

        landmarks = []

        for landmark in hand.landmark:

            landmarks.append([
                landmark.x,
                landmark.y
            ])

        # ========================================================
        # Existing normalization
        #
        # Same normalization used by the original 42-feature
        # pipeline.
        # ========================================================

        wrist = landmarks[0]

        wrist_x = wrist[0]
        wrist_y = wrist[1]

        max_distance = 0.0

        for landmark in landmarks:

            dx = landmark[0] - wrist_x
            dy = landmark[1] - wrist_y

            distance = math.sqrt(
                dx * dx +
                dy * dy
            )

            if distance > max_distance:

                max_distance = distance

        if max_distance == 0:

            max_distance = 1.0

        # ========================================================
        # 42 ORIGINAL FEATURES
        # ========================================================

        features = []

        for landmark in landmarks:

            features.append(
                (landmark[0] - wrist_x)
                / max_distance
            )

            features.append(
                (landmark[1] - wrist_y)
                / max_distance
            )

        # ========================================================
        # 28 GEOMETRIC FEATURES
        # ========================================================

        # --------------------------------------------------------
        # 1. Thumb tip → fingertip distances
        # --------------------------------------------------------

        thumb_tip = landmarks[4]

        fingertip_indices = [
            8,   # Index
            12,  # Middle
            16,  # Ring
            20   # Pinky
        ]

        for index in fingertip_indices:

            features.append(
                self._distance(
                    thumb_tip,
                    landmarks[index]
                )
                / max_distance
            )

        # --------------------------------------------------------
        # 2. Fingertip → fingertip distances
        # --------------------------------------------------------

        fingertip_pairs = [
            (8, 12),
            (12, 16),
            (16, 20),
            (8, 16),
            (8, 20),
            (12, 20)
        ]

        for a, b in fingertip_pairs:

            features.append(
                self._distance(
                    landmarks[a],
                    landmarks[b]
                )
                / max_distance
            )

        # --------------------------------------------------------
        # 3. Thumb joint → finger base distances
        # --------------------------------------------------------

        thumb_points = [
            2,
            3
        ]

        finger_points = [
            6,
            10,
            14,
            18
        ]

        for thumb_index in thumb_points:

            for finger_index in finger_points:

                features.append(
                    self._distance(
                        landmarks[thumb_index],
                        landmarks[finger_index]
                    )
                    / max_distance
                )

        # --------------------------------------------------------
        # 4. Finger joint angles
        # --------------------------------------------------------

        angle_triplets = [

            # Index
            (5, 6, 7),
            (6, 7, 8),

            # Middle
            (9, 10, 11),
            (10, 11, 12),

            # Ring
            (13, 14, 15),
            (14, 15, 16),

            # Pinky
            (17, 18, 19),
            (18, 19, 20),

            # Thumb
            (1, 2, 3),
            (2, 3, 4)
        ]

        for a, b, c in angle_triplets:

            features.append(
                self._angle(
                    landmarks[a],
                    landmarks[b],
                    landmarks[c]
                )
            )

        # ========================================================
        # Safety check
        # ========================================================

        if len(features) != 70:

            raise ValueError(
                f"Expected 70 features, "
                f"got {len(features)}"
            )

        return features
    
    def add_geometric_features(self, original_features):

            original_features = np.asarray(
                original_features,
                dtype=np.float32
            )

            if original_features.shape[0] != 42:    
                raise ValueError(
                    f"Expected 42 original features, "
                    f"got {original_features.shape[0]}"
                )

            # Reconstruct 21 normalized landmarks
            landmarks = []

            for i in range(21):

                landmarks.append([
                    float(original_features[i * 2]),
                    float(original_features[i * 2 + 1])
                ])

            # --------------------------------------------------------
            # Scale
            # --------------------------------------------------------

            wrist = landmarks[0]

            max_distance = 0.0

            for landmark in landmarks:

                distance = self._distance(
                    wrist,
                    landmark
                )

                if distance > max_distance:
                    max_distance = distance

            if max_distance == 0:

                max_distance = 1.0

            geometric_features = []

            # --------------------------------------------------------
            # 1. Thumb tip → fingertip distances
            # --------------------------------------------------------

            thumb_tip = landmarks[4]

            fingertip_indices = [
                8,
                12,
                16,
                20
            ]

            for index in fingertip_indices:

                geometric_features.append(
                    self._distance(
                        thumb_tip,
                        landmarks[index]
                    ) / max_distance
                )

            # --------------------------------------------------------
            # 2. Fingertip → fingertip distances
            # --------------------------------------------------------

            fingertip_pairs = [
                (8, 12),
                (12, 16),
                (16, 20),
                (8, 16),
                (8, 20),
                (12, 20)
            ]

            for a, b in fingertip_pairs:

                geometric_features.append(
                    self._distance(
                        landmarks[a],
                        landmarks[b]
                    ) / max_distance
                )

            # --------------------------------------------------------
            # 3. Thumb joint → finger base distances
            # --------------------------------------------------------

            thumb_points = [
                2,
                3
            ]

            finger_points = [
                6,
                10,
                14,
                18
            ]

            for thumb_index in thumb_points:

                for finger_index in finger_points:

                    geometric_features.append(
                        self._distance(
                            landmarks[thumb_index],
                            landmarks[finger_index]
                        ) / max_distance
                    )

            # --------------------------------------------------------
            # 4. Finger joint angles
            # --------------------------------------------------------

            angle_triplets = [

                (5, 6, 7),
                (6, 7, 8),

                (9, 10, 11),
                (10, 11, 12),

                (13, 14, 15),
                (14, 15, 16),

                (17, 18, 19),
                (18, 19, 20),

                (1, 2, 3),
                (2, 3, 4)
            ]

            for a, b, c in angle_triplets:

                geometric_features.append(
                    self._angle(
                        landmarks[a],
                        landmarks[b],
                        landmarks[c]
                    )
                )

            if len(geometric_features) != 28:

                raise ValueError(
                    f"Expected 28 geometric features, "
                    f"got {len(geometric_features)}"
                )

            return np.concatenate([
                original_features,
                np.asarray(
                    geometric_features,
                    dtype=np.float32
                )
            ])
    
    