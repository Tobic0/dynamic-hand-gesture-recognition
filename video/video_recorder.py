import os
import cv2
import numpy as np


PATH = os.path.join('video_frames/raw')

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
            cv2.putText(_image, str(a)+") Record new key-points for action " + actions[a - 1], (5, 20 + (20 * a)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    elif 1 <= op <= actions.size:
        cv2.putText(_image, "Current action: " + actions[op - 1], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)
        cv2.putText(_image, "Current action: " + actions[op - 1], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(_image, "Press 0 to go back", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)
        cv2.putText(_image, "Press 0 to go back", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)


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

while cap.isOpened():
    key = cv2.waitKey(10)

    # Wait until ESC key is pressed to close program
    if key == 27:
        break
    opt = select_options(key, opt)

    success, image = cap.read()
    image = cv2.flip(image, 1)

    if recording:
        for sequence in range(no_sequences):
            # Create folder for sequence number
            try:
                os.makedirs(os.path.join(PATH, actions[opt - 1],
                                         str(len(os.listdir(PATH + "/" + actions[opt - 1])))))
            except:
                pass
            for frame_num in range(sequence_length + 1):
                success, image = cap.read()
                image = cv2.flip(image, 1)

                if frame_num == 0:
                    cv2.putText(image, 'Starting collection', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions[opt - 1],
                                                                                         sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(actions[opt - 1],
                                                                                         sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(PATH + "/" + actions[opt - 1] + "/" +
                                str(len(os.listdir(PATH + "/" + actions[opt - 1])) - 1) + "/" +
                                actions[opt - 1] + "_" + str(frame_num - 1) + ".jpg", image)

                cv2.imshow('Dynamic hand gesture recognition', image)
                if frame_num == 0:
                    cv2.waitKey(2000)

                # Wait until ESC key is pressed to close program
                if cv2.waitKey(10) & 0xFF == 27:
                    break

                # Add a little wait for getting better movement images
                cv2.waitKey(25)

        recording = False
        print("Finished recording!")

    draw_options(image, opt)

    cv2.putText(image, "'ESC' to exit", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, "'ESC' to exit", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Dynamic hand gesture recognition', image)

cap.release()
cv2.destroyAllWindows()
