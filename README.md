tensorboard --logdir=test

| 0 | 1 | 2 | 3 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [0.81, 0.82] | [0.71, 0.76] | [0.63, 0.65] | [0.59, 0.55] | .... | [0.91, 0.40] | [0.92, 0.34] | [0.93, 0.27] |

The x and y values are then multiplied, respectively, by the frame width and frame height in order to get pixel coordinates for each keypoint.

| 0 | 1 | 2 | 3 | .... | 18 | 19 | 20 |
| - | - | - | - | ---- | -- | -- | -- |
| [521.86, 394.18] | [454.98, 366.42] | [406.18, 312.48] | [376.36, 262.41] | .... | [581.03, 193.41] | [588.94, 161.57] | [593.04, 129.48] |

After that the landmark coordinates are transformed to relative position with respect to the wrist keypoint (keypoint 0), so that the hand landmarks are not relative to the current position of the hand in the captured frame.

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
