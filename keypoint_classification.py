from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
import numpy as np
import os


PATH = os.path.join('video/gestures_data')

actions = np.array(['general', 'hello', 'pick-up', 'stop', 'okay', 'continue', 'next', 'previous', 'hold'])
# Number of videos recorded per gesture
no_sequences = 10
# Number of frames for each video
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
print(label_map)

'''
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(PATH, action, str(sequence), '{}.npy'.format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
print(np.array(sequences).shape)'''

#
# Trying to change the fact that you don't need to use fixed number of sequences for each action
#

test_sequences, test_labels = [], []
# For each action inside the actions list, get the number of sequences and loop through each of them in order to
for action in actions:
    for sequence in range(len(os.listdir(PATH+"/"+action))):
        test_window = []
        # print(action, "->", sequence, "->", len(os.listdir(PATH + "/" + action + "/" + str(sequence))))
        for frame in range(len(os.listdir(PATH + "/" + action + "/" + str(sequence)))):
            res = np.load(os.path.join(PATH, action, str(sequence), '{}.npy'.format(frame)))
            test_window.append(res)
        test_sequences.append(test_window)
        test_labels.append(label_map[action])
print("test: ", np.array(test_sequences).shape)


X = np.array(test_sequences)
y = to_categorical(test_labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(sequence_length, 63)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=550, callbacks=[tb_callback])
print(history.history.keys())

# Save model weights
model.save('gesture.h5')
