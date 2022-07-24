import os
import cv2
import numpy as np
import mediapipe as mp
import landmarks_extractor as le


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

PATH = os.path.join('gestures_data')

# Read gesture classes defined in file CLASSES.txt
actions = np.array(open("../CLASSES.txt").read().splitlines())
# Number of videos recorded per gesture
no_sequences = 10
# Number of frames for each video
sequence_length = 30

recording = False

for action in actions:
    try:
        os.makedirs(os.path.join(PATH, action))
    except:
        pass


# Function to draw on opencv image the options menu
def draw_options(_image, op=0):
    if op == 0:
        for a in range(1, actions.size+1):
            cv2.putText(_image, str(a) + ") Record new key-points for action " + actions[a - 1], (5, 20 + (20 * a)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(_image, str(a)+") Record new key-points for action "+actions[a-1], (5, 20 + (20 * a)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    elif 1 <= op <= actions.size:
        cv2.putText(_image, "Current action: " + actions[op - 1], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)
        cv2.putText(_image, "Current action: "+actions[op-1], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(_image, "Press 0 to go back", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)
        cv2.putText(_image, "Press 0 to go back", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    else:
        cv2.putText(_image, "test", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


# Function for selecting the option from list by pressing on keyboard
def select_options(_key, _opt):
    # 0 = 48, 4 = 51
    if 48 <= _key <= (48 + actions.size):
        option = _key-48
        print("key: ", option)
        return option
    elif (1 <= _opt <= actions.size) & (_key == 32):
        global recording
        recording = True
        return _opt
    else:
        return _opt


# Current option type selected
opt = 0
cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        key = cv2.waitKey(10)

        # Wait until ESC key is pressed to close program
        if key == 27:
            break
        opt = select_options(key, opt)

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

        if recording:
            for sequence in range(no_sequences):
                # Create folder for sequence number
                try:
                    os.makedirs(os.path.join(PATH, actions[opt - 1],
                                             str(len(os.listdir(PATH + "/" + actions[opt - 1])))))
                except:
                    pass
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
                        # Draw hand landmarks
                        for idx, hand_handedness in enumerate(results.multi_handedness):
                            mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[idx],
                                                      mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                                      mp_drawing_styles.get_default_hand_connections_style())

                            # Get all hand landmarks and handedness of detected hands
                            # ZIP: takes iterables (can be zero or more), aggregates them in a tuple, and returns it.
                            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                h_landmarks = le.process_landmarks(hand_landmarks, image)
                    else:
                        h_landmarks = np.zeros(63)

                    if frame_num == 0:
                        cv2.putText(image, 'Starting collection', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions[opt - 1],
                                                                                             sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.waitKey(2500)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions[opt - 1],
                                                                                             sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('Dynamic hand gesture recognition', image)
                    # Wait until ESC key is pressed to close program
                    if cv2.waitKey(10) & 0xFF == 27:
                        break

                    npy_path = os.path.join(PATH, actions[opt - 1],
                                            str(len(os.listdir(PATH + "/" + actions[opt - 1])) - 1), str(frame_num))
                    np.save(npy_path, np.array(h_landmarks))
                    print(str(frame_num), ": ", h_landmarks)
            recording = False
            print("Finished recording!")

        draw_options(image, opt)

        cv2.putText(image, "'ESC' to exit", (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Dynamic hand gesture recognition', image)

    cap.release()
    cv2.destroyAllWindows()
