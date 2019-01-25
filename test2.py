
import numpy as np
import math

rot = -45 * math.pi/180
R = np.array([
    [math.cos(rot), 0, math.sin(rot)],
    [0, 1, 0],
    [-math.sin(rot), 0, math.cos(rot)]
])

v = np.array([0, 10, 1])

out = np.dot(v, R)

print(out)
