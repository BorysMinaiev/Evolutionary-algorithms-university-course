# Created by Borys Minaiev

# you need download igraph from http://igraph.org/
from igraph import *
import random
import numpy as np

# set generations = 0 to see optimal solution
size = 1000
generations = 50

f = open('bayg29.tsp', 'r')
n = 29

dist = []
for i in range(n):
    cur = [0 for j in range(n)]
    dist.append(cur)

i = 0
for line in f:
    j = i
    for val in line.split():
        j += 1
        dist[i][j] = dist[j][i] = float(val)
    i += 1

def convert(pos):
    a = [i for i in range(n)]
    res = []
    for i in range(n):
        res.append(a[pos[i]])
        del a[pos[i]]
    return res

def gen_random():
    res = []
    for i in range(n):
        res.append(random.randint(0, n - i - 1))
    return res

f_coords = open('bayg29.pos', 'r')

def get_len(sol):
    len = 0.0
    for i in range(n):
        len += dist[sol[i]][sol[(i + 1) % n]]
    return len

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

def cmp_fun(a, b):
    if a.z < b.z:
        return 1
    return -1

def get_best(pts, n):
    pts = sorted(pts, cmp = cmp_fun)
    next = pts[-n:]
    return next

class Point:
    z = 0.0
    prob = 0.0
    prob_sum = 0.0

def crossingover(p1, p2):
    k = random.randint(0, n - 1)
    res = Point()
    res.value = []
    for i in range(n):
        if i < k:
            res.value.append(p1.value[i])
        else:
            res.value.append(p2.value[i])
    res.z = get_len(convert(res.value))
    return res

def mutation(p):
    k = random.randint(0, n - 1)
    res = Point()
    res.value = p.value[:]
    res.value[k] = random.randint(0, n - 1 - k)
    res.z = get_len(convert(res.value))
    return res

coords = []
for line in f_coords:
    tmp = line.split()
    coords.append((float(tmp[1]), float(tmp[2])))

f_sol = open('bayg29.sol', 'r')
sol = []
for line in f_sol:
    sol.append(int(line) - 1)
optimal = get_len(sol)


pts = []
for i in range(size):
    point = Point()
    point.value = gen_random()
    point.z = get_len(convert(point.value))
    pts.append(point)

for gen in range(generations):
    best_len = min([point.z for point in pts])
    sum_len = sum([point.z for point in pts])
    for point in pts:
        if point.z == best_len:
            sol = convert(point.value)
    print "after %d generations best length is %d, av = %.3f" % (gen, best_len, float(sum_len) / size)
    calc_fitness(pts)
    next = pts[:]

    for i in range(size):
        p1, p2 = random_point(pts), random_point(pts)
        next.append(crossingover(p1, p2))

    for i in range(size):
        next.append(mutation(random_point(pts)))

    pts = get_best(next, size)


g = Graph()
g.add_vertices(n)
for i in range(n):
    g.add_edge(sol[i], sol[(i + 1) % n])
print "found length = %.0f" % get_len(sol)
print "optimal path length is %.0f" % optimal
plot(g, vertex_label = [(i+1) for i in range(n)], layout = coords, bbox = (900, 900), margin = 50, vertex_size = 30, vertex_color = "yellow")