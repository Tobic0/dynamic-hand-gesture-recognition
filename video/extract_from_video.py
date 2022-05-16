import os
import itertools
import cv2
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def process_landmarks(landmarks):
    landmark_list = []
    # Get id and coordinate of each landmark
    for i, landmark in enumerate(landmarks.landmark):
        landmark_list.append(np.array([landmark.x, landmark.y, landmark.z]))

    return landmark_list


PATH = os.path.join('gestures_data')

actions = np.array(['hello', 'pick-up', 'stop'])
# Number of videos recorded per gesture
no_sequences = 10
# Number of frames for each video
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence)))
        except:
            pass


cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                success, image = cap.read()
                image = cv2.flip(image, 1)
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to pass by reference
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                h, w, _ = image.shape
                h_landmarks = []

                # Check if at least a hand is currently detected
                if results.multi_hand_landmarks:
                    # Get all hand landmarks and handedness of detected hands
                    # ZIP: takes iterables (can be zero or more), aggregates them in a tuple, and returns it.
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        h_landmarks = process_landmarks(hand_landmarks)
                        # Transform into numpy array
                        h_landmarks = np.array(h_landmarks)

                        # Transform landmarks to absolute position (coordinates in pixel)
                        for i in range(len(h_landmarks)):
                            h_landmarks[i][0] = h_landmarks[i][0] * w
                            h_landmarks[i][1] = h_landmarks[i][1] * h

                        # Transform landmarks to relative position with respect to wrist keypoint (keypoint 0)
                        for i in range(len(h_landmarks)):
                            h_landmarks[i][0] = h_landmarks[i][0] - h_landmarks[0][0]
                            h_landmarks[i][1] = h_landmarks[i][1] - h_landmarks[0][1]
                            h_landmarks[i][2] = h_landmarks[i][2] - h_landmarks[0][2]

                        # Convert landmarks list into a one-dimensional list of length 63 (21*3)
                        h_landmarks = np.array(h_landmarks).flatten()

                        # Get max & min values of landmarks list in order to apply normalization
                        max_value = max(list(map(abs, h_landmarks)))  # Return max value of list
                        min_value = min(list(map(abs, h_landmarks)))  # Return min value of list

                        # Normalization of landmarks list using linear scaling
                        h_landmarks = np.array(list(map(lambda value: (value - min_value) / (max_value - min_value), h_landmarks)))
                        # print(h_landmarks)
                        print("np array: ", h_landmarks.shape)

                        # Draw hand landmarks
                        for idx, hand_handedness in enumerate(results.multi_handedness):
                            mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[idx],
                                                      mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                                      mp_drawing_styles.get_default_hand_connections_style())

                if frame_num == 0:
                    cv2.putText(image, 'Starting collection', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('MediaPipe Hands', image)

                npy_path = os.path.join(PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, np.array(h_landmarks))
                # cv2.putText(image, "'ESC' to exit", (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

                # Wait until ESC key is pressed to close program
                if cv2.waitKey(10) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()
