# Created by Borys Minaiev

import matplotlib.pyplot as plt
import numpy as np
from math import *

class Point:
    x_int = 0
    x = 0
    y = 0
    prob = 0

def f(x):
    return (exp(-x) - exp(x)) * cos(x) / (exp(x) + exp(-x))
vf = np.vectorize(f)

fr, to = -5.0, 5.0
eps = 0.001
generations = 20
bits = 1
while eps * (1 << bits) < to - fr:
    bits = bits + 1

print bits
n = 30

def get_double(integer):
    return fr + (to - fr) / (1 << bits) * integer

def calc_fitness(points):
    min_y = min([point.y for point in points])
    sum = 0
    for point in points:
        sum += point.y - min_y
    for point in points:
        point.prob = (point.y - min_y) / sum

def random_point(points):
    p = np.random.rand()
    cur = 0
    for point in points:
        cur += point.prob
        if cur >= p:
            return point
    assert 0

def update_point(point):
    point.x = get_double(point.x_int)
    point.y = f(point.x)

def crossingover(p1, p2):
    k = np.random.randint(0, bits)
    nval = 0
    for i in range(bits):
        if i < k:
            nval |= (p1.x_int & (1 << i))
        else:
            nval |= (p2.x_int & (1 << i))
    res = Point()
    res.x_int = nval
    update_point(res)
    return res

def change_point(p):
    k = np.random.randint(0, bits)
    res = Point()
    res.x_int = p.x_int ^ (1 << k)
    update_point(res)
    return res

pts = []
for i in range(n):
    point = Point()
    point.x_int = np.random.randint(0, 1 << bits)
    update_point(point)
    pts.append(point)

t = np.arange(fr, to, 0.01)
plt.figure(figsize=(15, 10))

for gen in range(generations):
    plt.subplot(4, 5, gen + 1)
    plt.title("generation {}".format(gen + 1))
    plt.plot(t, vf(t))
    plt.plot([point.x for point in pts], [point.y for point in pts], 'r*')

    calc_fitness(pts)

    next_pts = pts[:]
    for it in range(n):
        p1 = random_point(pts)
        p2 = random_point(pts)
        next_pts.append(crossingover(p1, p2))
    for it in range(n):
        next_pts.append(change_point(random_point(pts)))
    calc_fitness(next_pts)
    pts = []
    for it in range(n):
        pts.append(random_point(next_pts))

plt.show()