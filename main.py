## \mainpage Dynamic hand gesture recognition using RNN with LSTM
# This is the official documentation of the dynamic hand gesture recognition system based on Google's MediaPipe
# framework and Recurrent Neural Network with Long Short Term Memory project which can be seen at the following
# GitHub repo: https://github.com/Tobic0/dynamic-hand-gesture-recognition \n\n
# This documentation contains a brief overview of the main code pages used for executing the application:
# - video_recorder
# - video_augmentation
# - extract_from_video
# - keypoint_classification
# - main \n\n
# In particular the working flow is the following: first you need to record the hand gestures using the video_recorder
# where you can select which gesture to record or using external tools, it is important to record many videos as
# possible as, the more the better; afterwards using video_augmentation we increment the number of videos by
# augmenting the original one by changing randomly the brightness and by adding a random rotation. At this point all
# the input data is ready for running the extract_from_video script which uses MediaPipe hands solution for detecting
# and generating hand landmarks which are then stored in an appropriate folder after appropriate transformations are
# made by the landmarks_transformer. Afterwards we can train our
# recurrent neural network by running the keypoint_classification script which at the end will save the model
# together with the weights that been computed. At the end it is only necessary to run the main script for a real
# time application. Specifically the program uses the MediaPipe framework to detect hands and if it detects one then the
# landmarks are extracted and processed using the landmarks_transformer, and using the neural network previously
# trained it predicts which hand gesture was performed.

## @package main
# Documentation for the real time application code. \n
# This is the main program which is run for real time application. It tries to open the webcam or camera which is
# used to detect if there are hands using Google's MediaPipe framework and if hands are detected the RNN model is
# called to predict the hand gesture that was performed.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import mediapipe as mp
import landmarks_transformer as lt
import time
# import rospy
# from std_msgs.msg import String

# Mediapipe setup variables for indicating type of ML solution
## MediaPipe solution drawing utils
mp_drawing = mp.solutions.drawing_utils
## MediaPipe drawing styles for detected hands
mp_drawing_styles = mp.solutions.drawing_styles
## Define which MediaPipe solution to use
mp_hands = mp.solutions.hands

## Read gesture classes defined in file CLASSES.txt
actions = np.array(open("CLASSES.txt").read().splitlines())
## Path where tflite model is saved
TF_LITE_PATH = 'gesture.tflite'
## Number of frames for each video
sequence_length = 30


## Main function for real time application
def main():
    # Start rospy node
    # rospy.init_node('hand_gesture_recognizer')
    # pub = rospy.Publisher('/hand_gesture', String, queue_size=10)
    # Load keras classification model in order to predict gesture
    model = keras.models.load_model('gesture.h5')

    # Load tflite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=TF_LITE_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # List containing last 30 frames (with or without hands)
    sequence = []
    # List containing last 5 gestures that were performed
    sentence = []
    # Threshold value used to determine whether to use or not the predicted class
    threshold = 0.85
    # Value used for determine after how many landmarks to predict the next gesture
    wait_for_landmarks = 0

    # Open webcam or camera
    cap = cv2.VideoCapture(0)
    # Start MediaPipe hand solution
    with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5,
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
                    # Increment by 1 each time a new landmark has been added to sequence
                    wait_for_landmarks += 1

                    # Check if sequence length is 30, as classification is trained on 30 frames
                    if len(sequence) == 30:
                        # Count how many zeros arrays are contained in sequence
                        count = 0
                        for s in sequence:
                            if np.count_nonzero(s) == 0:
                                count += 1
                        # If there are more than 10 zero arrays don't predict, as the prediction might not be accurate
                        if (count < 10) & (wait_for_landmarks > 20):
                            # res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            res = interpreter.get_tensor(output_details[0]['index'])
                            # Reset to 0 when prediction gets called
                            wait_for_landmarks = 0

                            if (np.argmax(res) > threshold) & (actions[np.argmax(res)] != "general"):
                                # Check if sentence has at least one action predicted, if the current action is not
                                # the same as the last action, append it, otherwise just append
                                if len(sentence) > 0:
                                    # pub.publish(actions[np.argmax(res)])
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                                # Print prediction for each class for current sequence of landmarks
                                # print("threshold: ", res, " -> ", actions[np.argmax(res)])

                            # Reduce the sentence length always to 4
                            if len(sentence) > 4:
                                sentence = sentence[-4:]

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
            # Show image
            cv2.imshow('Dynamic hand gesture recognition', image)
            # Wait until ESC key os pressed to close program
            if cv2.waitKey(10) & 0xFF == 27:
                break
            print("FPS: ", 1.0 / (time.time() - start_time))
    cap.release()


if __name__ == "__main__":
    main()
