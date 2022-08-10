## @package extract_from_video
# Documentation for video landmark extraction.
# This program is used to extract the coordinates of the landmarks generated by Google's MediaPipe framework
# from the augmented videos generated with video_augmentation. The data extracted is then saved in PATH location.
import os
import cv2
import numpy as np
import mediapipe as mp
import landmarks_transformer as lt


## MediaPipe solution drawing utils
mp_drawing = mp.solutions.drawing_utils
## MediaPipe drawing styles for detected hands
mp_drawing_styles = mp.solutions.drawing_styles
## Define which MediaPipe solution to use
mp_hands = mp.solutions.hands

## Papth where to output the hand landmarks
PATH = os.path.join('gestures_data')
## Path of folder containing the augmented videos used for landmark extraction
PATH_VIDEO_AUGMENTED = os.path.join('video_frames/augmented')

## Read gesture classes defined in file CLASSES.txt
actions = np.array(open("../CLASSES.txt").read().splitlines())
## Number of videos recorded per gesture
no_sequences = 10
## Number of frames for each video
sequence_length = 30

# For each action create folder in PATH location if folder does not alredy exists
for action in actions:
    try:
        os.makedirs(os.path.join(PATH, action))
    except:
        pass

# Starting MediaPipe hands solution with predefined parameters
with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    for act in actions:
        print("Currently extracting: ", act)
        # Get every single sequence that was augmented
        for sequence in os.listdir(PATH_VIDEO_AUGMENTED + "/" + act):
            # Create folders for each sequence
            try:
                os.makedirs(os.path.join(PATH, act, sequence))
            except:
                pass

            # Extract hand landmarks from each frame
            for frame_num in range(len(os.listdir(PATH_VIDEO_AUGMENTED + "/" + act + "/" + sequence))):
                ## Read a frame from augmented video
                image = cv2.imread(PATH_VIDEO_AUGMENTED + "/" + act + "/" + sequence + "/" + act + "_" + str(frame_num) + ".jpg")
                ## Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Check if at least a hand is currently detected
                if results.multi_hand_landmarks:
                    # Draw hand landmarks
                    for idx, hand_handedness in enumerate(results.multi_handedness):
                        # Get all hand landmarks and handedness of detected hands
                        # ZIP: takes iterables (can be zero or more), aggregates them in a tuple, and returns it.
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                              results.multi_handedness):
                            ## Get hand landmarks by calling landmarks_transformer
                            h_landmarks = lt.process_landmarks(hand_landmarks, image)
                else:
                    h_landmarks = np.zeros(63)

                # Wait until ESC key is pressed to close program
                if cv2.waitKey(10) & 0xFF == 27:
                    break

                ## Save numpy array containg landmarks to PATH location
                npy_path = os.path.join(PATH, act, str(sequence), str(frame_num))
                np.save(npy_path, np.array(h_landmarks))
                # print(str(frame_num), ": ", h_landmarks)
