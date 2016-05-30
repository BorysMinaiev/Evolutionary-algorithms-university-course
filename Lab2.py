# Created by Borys Minaiev

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random

generations = 0
n = 100
fr = 0
to = pi
dim = 2

class Point:
    z = 0.0

def f_one(x, i):
    m = 10
    return -sin(x) * (sin((i * x * x) / pi) ** (2 * m))

@np.vectorize
def f(x, y):
    res = f_one(x, 1) + f_one(y, 2)
    return res

def calc(point):
    for i in range(len(point.coords)):
        point.z += f_one(point.coords[i], i + 1)

def gen_f(delta):
    x = y = np.arange(fr, to, delta)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    return X, Y, Z

if dim == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = gen_f(0.01)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

pts = []
for i in range(n):
    point = Point()
    point.coords = []
    for j in range(dim):
        point.coords.append(random.uniform(fr, to))
    calc(point)
    pts.append(point)

# for gen in range(generations):


if dim == 2:
    for point in pts:
        ax.scatter(point.coords[0], point.coords[1], point.z, c='r', marker='o')

    plt.show()