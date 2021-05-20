# conda install matplotlib
# pip install mediapipe
# envs = py36_mediapipe


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 2*3.141592, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()


