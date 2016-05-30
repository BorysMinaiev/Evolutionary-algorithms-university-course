# Created by Borys Minaiev

# you need download igraph from http://igraph.org/
from igraph import *

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

f_coords = open('bayg29.pos', 'r')

class Point:
    x = 0.0
    y = 0.0
    id = -1

pts = []
coords = []
for line in f_coords:
    point = Point()
    tmp = line.split()
    point.x = tmp[1]
    point.y = tmp[2]
    point.id = tmp[0]
    pts.append(point)
    coords.append((float(tmp[1]), float(tmp[2])))

f_sol = open('bayg29.sol', 'r')
sol = []
for line in f_sol:
    sol.append(int(line) - 1)

g = Graph()
g.add_vertices(n)
len = 0.0
for i in range(n):
    len += dist[sol[i]][sol[(i + 1) % n]]
    g.add_edge(sol[i], sol[(i + 1) % n])
print "total length = %.1f" % len
plot(g, vertex_label = [(i+1) for i in range(n)], layout = coords, bbox = (900, 900), margin = 50, vertex_size = 30, vertex_color = "yellow")