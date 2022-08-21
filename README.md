tensorboard --logdir=test

Raw data output generated by MediaPipe's hand landmark model

| 0 | 1 | 2 | .... | 18 | 19 | 20 |
| - | - | - | ---- | -- | -- | -- |
| [0.6, 0.6, 0.0] | [0.68, 0.56, 0.009] | [0.74, 0.49, 0.007] | .... | [0.53, 0.38, 0.01] | [0.54, 0.41, 0.02] | [0.54, 0.44, 0.02] |

The x and y values are then multiplied, respectively, by the frame width and frame height in order to get pixel coordinates for each keypoint.

| 0 | 1 | 2 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [386.9, 292, 0.0] | [438, 272, 0.009] | [473.7, 239, 0.007] | .... | [342.7, 183.3, 0.01] | [347, 198.9, 0.02] | [350, 211.8, 0.02] |

After that the landmark coordinates are transformed to relative position with respect to the wrist keypoint (keypoint 0), so that the hand landmarks are not relative to the current position of the hand in the captured frame.

| 0 | 1 | 2 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [0.0, 0.0, 0.0] | [51.1, -20, 0.009] | [86.8, -53, 0.007] | .... | [-44.2, -108.7, 0.01] | [-39.8, -93.1, 0.02] | [-36, -80.2, 0.02] |

Then convert the list into a one-dimensional list.

| One dimensional list |
| ------------------------------- |
| 0.0, 0.0, 0.0, 51.1, -20, 0.009, 86.8, -53, 0.007, .... , -44.2, -108.7, 0.01, -39.8, -93.1, 0.02, -36, -80.2, 0.02 |

Get max & min values of landmarks list in order to apply min-max normalization method.

| Normalzied one dimensional list |
| ------------------------------- |
| 0.0, 0.0, 0.0, 0.36, -0.1, 0.0, 0.61, -0.3, 0.0, .... , -0.3, -0.7, 0.0, -0.2, -0.6, 0.0001, -0.2, -0.5, 0.0001|
