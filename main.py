from tensorflow import keras
import numpy as np
import cv2
import mediapipe as mp
import landmarks_transformer as lt
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Read gesture classes defined in file CLASSES.txt
actions = np.array(open("CLASSES.txt").read().splitlines())
# Number of frames for each video
sequence_length = 30


def main():
    # Load keras classification model in order to predict gesture
    model = keras.models.load_model('gesture.h5')

    sequence = []
    sentence = []
    threshold = 0.85

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            start_time = time.time()
            success, image = cap.read()
            image = cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to pass by reference
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if at least a hand is currently detected
            if results.multi_hand_landmarks:
                # Get all hand landmarks and handedness of detected hands
                # ZIP: takes iterables (can be zero or more), aggregates them in a tuple, and returns it.
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Transform and normalize hand landmarks
                    h_landmarks = lt.process_landmarks(hand_landmarks, image)

                    # Add last detected landmarks to sequence list and get only last 30 landmarks
                    sequence.insert(0, h_landmarks)
                    sequence = sequence[:30]

                    # Check if sequence length is 30, as classification is trained on 30 frames
                    if len(sequence) == 30:
                        # Count how many zeros arrays are contained in sequence
                        count = 0
                        for s in sequence:
                            if np.count_nonzero(s) == 0:
                                count += 1
                        # If there are more than _ zero arrays don't predict, as the prediction might not be accurate
                        if count < 10:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]

                            if (res[np.argmax(res)] > threshold) & (actions[np.argmax(res)] != "general"):
                                # Check if sentence has at least one action predicted, if the current action is not
                                # the same as the last action, append it, otherwise just append
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                                # Print prediction for each class for current sequence of landmarks
                                # print("threshold: ", res, " -> ", actions[np.argmax(res)])

                            # Reduce the sentence length always to 5
                            if len(sentence) > 5:
                                sentence = sentence[-5:]

                    # print("sentence: ", sentence)
                    # Print on screen the current sentence
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw hand landmarks on screen
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[idx],
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())
            else:
                # If no hand is detected generate an array with 63 zeros and insert it in sequence
                zeroArray = np.zeros(63)
                sequence.insert(0, zeroArray)
                sequence = sequence[:30]

            cv2.putText(image, "'ESC' to exit", (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            cv2.imshow('Dynamic hand gesture recognition', image)
            # Wait until ESC key os pressed to close program
            if cv2.waitKey(10) & 0xFF == 27:
                break
            print("FPS: ", 1.0 / (time.time() - start_time))
    cap.release()


if __name__ == "__main__":
    main()
