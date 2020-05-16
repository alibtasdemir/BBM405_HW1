#    Copyright 2019 Atikur Rahman Chitholian
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from heapq import heappop, heappush
from math import inf
import numpy as np
import time


class Graph:
    def __init__(self, directed=True):
        self.edges = {}
        self.directed = directed

    def add_edge(self, node1, node2, cost = 1, __reversed=False):
        try: neighbors = self.edges[node1]
        except KeyError: neighbors = {}
        neighbors[node2] = cost
        self.edges[node1] = neighbors
        if not self.directed and not __reversed: self.add_edge(node2, node1, cost, True)

    def neighbors(self, node):
        try: return self.edges[node]
        except KeyError: return []

    def cost(self, node1, node2):
        try: return self.edges[node1][node2]
        except: return inf

    def uniform_cost_search(self, start, goal):
        found, fringe, visited, came_from, cost_so_far = False, [(0, start)], set([start]), {start: None}, {start: 0}
        print('{:11s} | {}'.format('Expand Node', 'Fringe'))
        print('--------------------')
        print('{:11s} | {}'.format('-', str((0, start))))
        while not found and len(fringe):
            _, current = heappop(fringe)
            print('{:11s}'.format(current), end=' | ')
            if current == goal: found = True; break
            for node in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, node)
                if node not in visited or cost_so_far[node] > new_cost:
                    visited.add(node); came_from[node] = current; cost_so_far[node] = new_cost
                    heappush(fringe, (new_cost, node))
            print(', '.join([str(n) for n in fringe]))
        if found: print(); return came_from, cost_so_far[goal]
        else: print('No path from {} to {}'.format(start, goal)); return None, inf

    @staticmethod
    def print_path(came_from, goal):
        parent = came_from[goal]
        if parent:
            Graph.print_path(came_from, parent)
        else: print(goal, end='');return
        print(' =>', goal, end='')


    def __str__(self):
        return str(self.edges)


def read_mtx(file):
    book = {}
    with open("USAir97.net", 'r') as f:
        rawdict = f.readlines()
    for el in rawdict:
        proc = el.split(',')
        id = int(proc[0])
        name = proc[1].strip().strip('\"')
        book[id] = name

    with open(file, 'r') as f:
        raw = f.readlines()
    raw = raw[23:]
    v, e = int(raw[0].split()[0]), int(raw[0].split()[2])
    raw = raw[1:]
    graph = list(map(str.split, raw))
    for i, line in enumerate(graph):
        s, d, w = int(line[0]), int(line[1]), float(line[2])    # SOURCE, DEST, WEIGHT
        s = book[s]
        d = book[d]
        graph[i] = [s, d, w]

    return book, graph


if __name__ == '__main__':
    book, raw_graph = read_mtx("inf-USAir97.mtx")
    graph = Graph(directed=False)
    for edge in raw_graph:
        graph.add_edge(edge[0], edge[1], cost=edge[2])
    start, goal = ["John F Kennedy Intl", 'Deadhorse']
    start_time = time.time_ns()
    traced_path, cost = graph.uniform_cost_search(start, goal)
    print("--- %s nanoseconds ---" % (time.time_ns() - start_time))
    if (traced_path): print('Path:', end=' '); Graph.print_path(traced_path, goal); print('\nCost:', cost)