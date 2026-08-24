import cv2
import mediapipe as mp


class HandDetector:

    # Detects hands and draws custom multicolor landmarks.

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):

        # Converts BGR webcam frame to RGB for MediaPipe.

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = self.hands.process(
            rgb_frame
        )

        return results

    def draw(self, frame, results):

        # Draws custom multicolor hand landmarks.

        if not results.multi_hand_landmarks:
            return frame

        height, width, _ = frame.shape

        # ==================================================
        # COLORS
        # OpenCV uses BGR format
        # ==================================================

        RED = (0, 0, 255)

        # Index finger
        PURPLE = (180, 0, 180)

        # Middle finger
        YELLOW = (0, 255, 255)

        # Ring finger
        GREEN = (0, 255, 0)

        # Pinky finger
        BLUE = (255, 120, 0)

        # Thumb
        CREAM = (190, 220, 230)

        # Palm connections
        PALM = (180, 180, 180)

        # ==================================================
        # FINGER CONNECTION GROUPS
        # Each complete finger uses one consistent color
        # ==================================================

        finger_connections = [

            # Thumb
            (
                [
                    (1, 2),
                    (2, 3),
                    (3, 4),
                ],
                CREAM,
            ),

            # Index finger
            (
                [
                    (5, 6),
                    (6, 7),
                    (7, 8),
                ],
                PURPLE,
            ),

            # Middle finger
            (
                [
                    (9, 10),
                    (10, 11),
                    (11, 12),
                ],
                YELLOW,
            ),

            # Ring finger
            (
                [
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ],
                GREEN,
            ),

            # Pinky finger
            (
                [
                    (17, 18),
                    (18, 19),
                    (19, 20),
                ],
                BLUE,
            ),
        ]

        # ==================================================
        # PALM CONNECTIONS
        # ==================================================

        palm_connections = [

            (0, 1),
            (0, 5),
            (5, 9),
            (9, 13),
            (13, 17),
            (17, 0),

        ]

        # ==================================================
        # LANDMARK COLOR GROUPS
        # ==================================================

        landmark_colors = {

            # Wrist
            0: RED,

            # Thumb - cream
            1: CREAM,
            2: CREAM,
            3: CREAM,
            4: CREAM,

            # Index - purple
            5: RED,
            6: PURPLE,
            7: PURPLE,
            8: PURPLE,

            # Middle - yellow
            9: RED,
            10: YELLOW,
            11: YELLOW,
            12: YELLOW,

            # Ring - green
            13: RED,
            14: GREEN,
            15: GREEN,
            16: GREEN,

            # Pinky - blue
            17: RED,
            18: BLUE,
            19: BLUE,
            20: BLUE,
        }

        # ==================================================
        # DRAW EACH DETECTED HAND
        # ==================================================

        for hand_landmarks in results.multi_hand_landmarks:

            # ----------------------------------------------
            # Draw finger connections
            # ----------------------------------------------

            for connections, color in finger_connections:

                for start_id, end_id in connections:

                    start = hand_landmarks.landmark[
                        start_id
                    ]

                    end = hand_landmarks.landmark[
                        end_id
                    ]

                    start_x = int(
                        start.x * width
                    )

                    start_y = int(
                        start.y * height
                    )

                    end_x = int(
                        end.x * width
                    )

                    end_y = int(
                        end.y * height
                    )

                    cv2.line(
                        frame,
                        (start_x, start_y),
                        (end_x, end_y),
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # ----------------------------------------------
            # Draw palm connections
            # ----------------------------------------------

            for start_id, end_id in palm_connections:

                start = hand_landmarks.landmark[
                    start_id
                ]

                end = hand_landmarks.landmark[
                    end_id
                ]

                start_x = int(
                    start.x * width
                )

                start_y = int(
                    start.y * height
                )

                end_x = int(
                    end.x * width
                )

                end_y = int(
                    end.y * height
                )

                cv2.line(
                    frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    PALM,
                    2,
                    cv2.LINE_AA,
                )

            # ----------------------------------------------
            # Draw landmark points
            # ----------------------------------------------

            for landmark_id, landmark in enumerate(
                hand_landmarks.landmark
            ):

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                color = landmark_colors[
                    landmark_id
                ]

                # Dark outline
                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (40, 40, 40),
                    -1,
                    cv2.LINE_AA,
                )

                # Colored landmark
                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

        return frame

    def close(self):

        # Releases MediaPipe resources.

        self.hands.close()