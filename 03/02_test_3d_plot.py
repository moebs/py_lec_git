# conda install matplotlib
# pip install mediapipe
# envs = py36_mediapipe

import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes(projection='3d')

ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))

plt.show()