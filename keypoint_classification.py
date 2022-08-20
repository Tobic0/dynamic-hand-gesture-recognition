## @package keypoint_classification
# Documentation for the RNN structure and training. \n
# This code is used to define the recurrent neural network and to train it on the input data extracted
# with extract_from_video
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
import numpy as np
import os

## Define path where to save the gesture keypoints
PATH = os.path.join('video/gestures_data')
## Define path where to save tensorflow lite version of the model
TF_LITE_PATH = 'gesture.tflite'

## Read gesture classes defined in file CLASSES.txt
actions = np.array(open("CLASSES.txt").read().splitlines())
## Number of videos recorded per gesture
no_sequences = 10
## Number of frames for each video
sequence_length = 30

## Dictionary containing labels
label_map = {label: num for num, label in enumerate(actions)}

## Declaration of sequences and labels lists
sequences, labels = [], []

# For each action inside the actions list, get the number of sequences and loop through each of them in order to
for action in actions:
    for sequence in os.listdir(PATH+"/"+action):
        ## Vector that contains the collection of 30 frames which define a gesture
        window = []
        for frame in range(len(os.listdir(PATH + "/" + action + "/" + sequence))):
            res = np.load(os.path.join(PATH, action, sequence, '{}.npy'.format(frame)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


## Extraction of all vectors containing the data of each hand
X = np.array(sequences)
## List containing all the labels associated to all input vectors
y = to_categorical(labels).astype(int)

## Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

## Definition of the log folder for tensorboard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

## Definition of type of Keras model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(sequence_length, 63)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
print(history.history.keys())

# Save model weights
model.save('gesture.h5')

# Convert model to tensorflow lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable tensorflow ops
converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open(TF_LITE_PATH, 'wb').write(tflite_quantized_model)

