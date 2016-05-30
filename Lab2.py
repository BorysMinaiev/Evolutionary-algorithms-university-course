# Created by Borys Minaiev

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random

# vals for bigger dimensions:
# max_generations = 1000
# n = 2000

max_generations = 2
generations_wait = 100
eps = 1e-6
n = 100
fr = 0
to = pi
dim = 2

# answers:
# dim = 5 -> -4.687
# dim = 10 -> -9.66

class Point:
    z = 0.0
    prob = 0.0
    prob_sum = 0.0

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

def calc_fitness(points):
    max_z = max([point.z for point in points])
    sum = 0
    for point in points:
        sum += max_z - point.z + 1
    prob_sum = 0.0
    for point in points:
        point.prob = (max_z - point.z + 1) / sum
        prob_sum += point.prob
        point.prob_sum = prob_sum

def random_point(points):
    p = np.random.rand()
    l, r = 0, len(points)
    while r - l > 1:
        mid = (l + r) // 2
        if points[mid].prob_sum <= p:
            l = mid
        else:
            r = mid
    return points[l]

def crossingover(p1, p2):
    res = Point()
    res.coords = []
    w = random.uniform(0, 1)
    for v1, v2 in zip(p1.coords, p2.coords):
        val = v1 * w + v2 * (1 - w)
        res.coords.append(val)
    calc(res)
    return res

def mutation(point):
    res = Point()
    res.coords = point.coords[:]
    k = random.randint(0, dim - 1)
    val = res.coords[k] + exp(random.uniform(-5, 5))
    if val < fr:
        val = fr
    if val > to:
        val = to
    res.coords[k] = val
    calc(res)
    return res

def cmp_fun(a, b):
    if a.z < b.z:
        return 1
    return -1

def get_best(pts, n):
    pts = sorted(pts, cmp = cmp_fun)
    next = pts[-n:]
    return next

last_best = 0
best_cnt = 0
for gen in range(max_generations):
    best = min([point.z for point in pts])
    if best < last_best - eps:
        last_best = best
        best_cnt = 1
    else:
        best_cnt += 1
    if best_cnt >= generations_wait:
        break
    tot_sum = sum([point.z for point in pts])
    print "after %d iterations best f = %.6f, av = %.6f" % (gen, best, tot_sum / n)
    calc_fitness(pts)
    next_pts = pts[:]

    for i in range(n):
        p1, p2 = random_point(pts), random_point(pts)
        next_pts.append(crossingover(p1, p2))

    for i in range(n):
        next_pts.append(mutation(random_point(pts)))
    pts = get_best(next_pts, n)


if dim == 2:
    for point in pts:
        ax.scatter(point.coords[0], point.coords[1], point.z, c='r', marker='o')

    plt.show()