import numpy as np


##
# Get landmarks in order to transform and normalize them. Get image in order to get image width and height for correct
# transformation
def process_landmarks(landmarks, image):
    h, w, _ = image.shape
    h_landmarks = []

    # Get x, y and z coordinates of each landmark
    for i, landmark in enumerate(landmarks.landmark):
        h_landmarks.append(np.array([landmark.x, landmark.y, landmark.z]))

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
    return h_landmarks

