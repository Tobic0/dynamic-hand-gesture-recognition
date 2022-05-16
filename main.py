from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

actions = np.array(['hello', 'pick-up', 'stop'])
# Number of videos recorded per gesture
no_sequences = 5
# Number of frames for each video
sequence_length = 30


def process_landmarks(landmarks):
    landmark_list = []
    # Get id and coordinate of each landmark
    for i, landmark in enumerate(landmarks.landmark):
        landmark_list.append(np.array([landmark.x, landmark.y, landmark.z]))

    return landmark_list


def main():
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(sequence_length, 63)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('gesture.h5')

    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
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
                    h_landmarks = np.array(list(map(lambda value: (value - min_value) / (max_value - min_value),
                                                    h_landmarks)))
                    sequence.insert(0, h_landmarks)
                    sequence = sequence[:30]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(res, " -> ", actions[np.argmax(res)])

                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw hand landmarks
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[idx],
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

            cv2.putText(image, "'ESC' to exit", (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            cv2.imshow('Dynamic hand gesture recognition', image)
            # Wait until ESC key os pressed to close program
            if cv2.waitKey(10) & 0xFF == 27:
                break
    cap.release()


if __name__ == "__main__":
    main()
