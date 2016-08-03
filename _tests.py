import os
import sys

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1000)

# av = [0.5, 1, 4]
# cv = [0.1, 0.5, 0.7]
ac = [(5, 0.5), (20, 0.5), (20, 0.2)]

ys = []
for a, c in ac:#zip(av, cv):
    y = (1. / (1 + (np.exp(-a * (x - c)))))
    ys.append(y)

colors = 'rgbcmyk'
plt.figure()
plt.hold(True)
plt.grid()
for i, y in enumerate(ys):
    plt.plot(x, y, colors[i] + '-', linewidth=5)
plt.legend(('a=5, c=0.5', 'a=20, c=0.5', 'a=20, c=0.2'), loc=4)
plt.show()