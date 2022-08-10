## @package video_augmentation
# Documentation for video augmentation application. \n
# This code is used for generating new modified videos from an original video (video augmentation) that were
# recorder with the video_recorder or with external tools and put inside the PATH_VIDEO_RAW folder. \n
# In particular this program generates no_augmentation of videos by modifying the brightness and applying random
# rotation to copies of the original frames collection.
import os
import numpy as np
import random
from PIL import Image, ImageEnhance

## Path of folder location to save augmented videos
PATH = os.path.join('video_frames/augmented')
## Path of folder containing the raw videos recorder with video_recorder
PATH_VIDEO_RAW = os.path.join('video_frames/raw')

## Read gesture classes defined in file CLASSES.txt
actions = np.array(open("../CLASSES.txt").read().splitlines())
## Number of videos recorded per gesture
no_sequences = 10
## Number of frames for each video
sequence_length = 30
## Number of augmentation to apply to each video
no_augmentation = 4


## Function for generating random rotation, by default, between -15 and 15 degrees with a step rotation of 5 degree
# - min_rotation: integer for specifying the starting position
# - max_rotation: integer for specifying the ending position
# - step_rotation: optional integer for indicating the increment \n\n
# returns a random value between min_rotation and max_rotation
def random_rotate(min_rotation=-15, max_rotation=15, step_rotation=5):
    return random.randrange(min_rotation, max_rotation, step_rotation)


## Function for generating random brightness values, by default, between 1 and 180 with a step of 10
# - min_brightness: integer for specifying the starting position
# - max_brightness: integer for specifying the ending position
# - step_brightness: optional integer for indicating the increment \n\n
# returns a random value between min_brightness and max_brightness divided by 100 in order to get a float value between
# 0 and 1
def random_brightness(min_brightness=1, max_brightness=180, step_brightness=10):
    return (random.randrange(min_brightness, max_brightness, step_brightness) / 100) + 0.1


# For each action defined in actions try to create folders if they not exist already
for action in actions:
    try:
        os.makedirs(os.path.join(PATH, action))
    except:
        pass


## Main function of video_augmentation which iterates over all prerecorded videos found in PATH_VIDEO_RAW,
# extract each frame and applies augmentation by generating copies of the original frames but with random
# brightness and rotation applied
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
                    # Get original frame from PATH_VIDEO_RAW
                    original_frame = Image.open(PATH_VIDEO_RAW + "/" + act + "/" + sequence + "/" + str(frame_num))
                    # Make a copy from original frame
                    augmented_frame = original_frame.copy()
                    # Give random brightness and add random rotation to copied frame
                    augmented_frame = ImageEnhance.Brightness(augmented_frame).enhance(rdm_brightness).rotate(rdm_rotation)

                    # If first iteration o augmentation add also the original frame to PATH
                    if aug == 0:
                        original_frame.save(PATH + "/" + act + "/" + sequence + "/" + str(frame_num))

                    # Save augmented frame to PATH
                    augmented_frame.save(PATH + "/" + act + "/" + sequence + "_" + str(aug) + "/" + str(frame_num))


if __name__ == "__main__":
    augment_video()
