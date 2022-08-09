import os
import numpy as np
import random
from PIL import Image, ImageEnhance


PATH = os.path.join('video_frames/augmented')
PATH_VIDEO_RAW = os.path.join('video_frames/raw')

# Read gesture classes defined in file CLASSES.txt
actions = np.array(open("../CLASSES.txt").read().splitlines())
# Number of videos recorded per gesture
no_sequences = 10
# Number of frames for each video
sequence_length = 30
# Number of augmentation to apply
no_augmentation = 4


# Generate random rotation
def random_rotate(min_rotation=-15, max_rotation=15, step_rotation=5):
    return random.randrange(min_rotation, max_rotation, step_rotation)


def random_brightness(min_brightness=1, max_brightness=180, step_brightness=10):
    return (random.randrange(min_brightness, max_brightness, step_brightness) / 100) + 0.1


for action in actions:
    try:
        os.makedirs(os.path.join(PATH, action))
    except:
        pass


def augment_video():
    # Get all actions
    for act in os.listdir(PATH_VIDEO_RAW):
        # Get every single sequence that was registered
        for sequence in os.listdir(PATH_VIDEO_RAW + "/" + act):
            # Create folders for original sequences
            try:
                os.makedirs(os.path.join(PATH, act, sequence))
            except:
                pass
            # Augment 6 times the same sequence of frame numbers
            for aug in range(no_augmentation):
                # create folders for augmented sequences from original sequences
                try:
                    os.makedirs(os.path.join(PATH, act, sequence + "_" + str(aug)))
                except:
                    pass

                rdm_brightness = random_brightness()
                rdm_rotation = random_rotate(step_rotation=3)
                # Apply augmentation to all frames of the same sequence
                for frame_num in os.listdir(PATH_VIDEO_RAW + "/" + act + "/" + sequence):
                    print(frame_num)
                    original_frame = Image.open(PATH_VIDEO_RAW + "/" + act + "/" + sequence + "/" + str(frame_num))
                    augmented_frame = original_frame.copy()
                    augmented_frame = ImageEnhance.Brightness(augmented_frame).enhance(rdm_brightness).rotate(rdm_rotation)

                    if aug == 0:
                        original_frame.save(PATH + "/" + act + "/" + sequence + "/" + str(frame_num))

                    augmented_frame.save(PATH + "/" + act + "/" + sequence + "_" + str(aug) + "/" + str(frame_num))

                    '''
                    original_frame = cv2.imread(PATH_VIDEO_RAW + "/" + act + "/" + sequence + "/" + str(frame_num))
                    augmented_frame = increase_brightness(original_frame, value=rdm_brightness).rotate(rdm_rotation)

                    # If first iteration of augmentation add also the original images to new folder
                    if aug == 0:
                        cv2.imwrite(PATH + "/" + act + "/" + sequence + "/" + str(frame_num), original_frame)
                    # augmented_frame = frame_num.add_mask().rotate(random_rotate(10, 80, 10)).fx(mp.vfx.colorx, random_brightness(100, 140, 1))
                    cv2.imwrite(PATH + "/" + act + "/" + sequence + "_" + str(aug) + "/" + str(frame_num), augmented_frame)
'''


if __name__ == "__main__":
    augment_video()
