tensorboard --logdir=test
# Hand-Gesture-Recognition
Hand gesture recognition project based on Google's mediapipe library for hand and finger tracking, and the application of a feedforward neural network for classifying hand gestures.

## Index
1. [Introduction](#Introduction)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Basic working flow](#basic-working-flow)
5. [Training](#Training)
6. [References](#References)

## Introduction
The main goal of this project is to be able to classify different hand gestures that can later be used to control an ur5 robotic arm, for example interrupt the arm movement if a stop gesture is recognized. 
The recognition process is achieved through Google's mediapipe hands solution for hand keypoint detection and a feedforward neural network or multilayer perceptron for classifying the gestures.

## Requirements
* matplotlib 3.5.1
* mediapipe  0.8.9.1
* opencv 4.5.5.64
* scikit-learn 1.0.2
* seaborn 0.11.2
* tensorflow 2.8.0

## Usage
To execute the code you need to have a webcam connected and just need to run the following command:
```
python3 main.py
```

## Basic working flow
The basic working idea is that mediapipe generates 21 3D landmarks on current detected hands in a webcam and extracts each landmarks x and y value (each one going from 0.0 to 1.0):

| 0 | 1 | 2 | 3 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [0.81, 0.82] | [0.71, 0.76] | [0.63, 0.65] | [0.59, 0.55] | .... | [0.91, 0.40] | [0.92, 0.34] | [0.93, 0.27] |

<p align="center">
  <img src="https://google.github.io/mediapipe/images/mobile/hand_landmarks.png" title="Mediapipe's hand landmarks" style="width:80%;">
</p>

The x and y values are then multiplied, respectively, by the frame width and frame height in order to get pixel coordinates for each keypoint.

| 0 | 1 | 2 | 3 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [521.86, 394.18] | [454.98, 366.42] | [406.18, 312.48] | [376.36, 262.41] | .... | [581.03, 193.41] | [588.94, 161.57] | [593.04, 129.48] |

After that the landmarks are transformed to relative position with respect to the wrist keypoint (keypoint 0), so that the hand landmarks are not relative to the current position of the hand in the captured frame.

| 0 | 1 | 2 | 3 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [0.0, 0.0] | [-66.87, -27.76] | [-115.68, 81.69] | [-145.50, -131.76] | .... | [59.17, 200.77] | [67.08, -232.61] | [71.18, 264.70] |

Then convert the list into a one-dimensional list.

| One dimensional list |
| ------------------------------- |
| 0.0, 0.0, -66.87, -27.76, -115.68, 81.69, -145.50, -131.76, .... , 59.17, 200.77, 67.08, -232.61, 71.18, 264.70 |

Get max & min values of landmarks list in order to apply min-max normalization method.

| Normalzied one dimensional list |
| ------------------------------- |
| 0.0, 0.0, -0.20, -0.08, -0.35, 0.25, -0.44, -0.39, ...., 0.18, -0.61, 0.20, -0.70, 0.21, -0.80 |

At this point, based on the weights gained during the training phase, it prints the current detected class.

## Training

