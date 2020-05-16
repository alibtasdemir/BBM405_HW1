from collections import deque
import time

class Graph:
    def __init__(self, directed=True):
        self.edges = {}
        self.directed = directed

    def add_edge(self, node1, node2, __reversed=False):
        try: neighbors = self.edges[node1]
        except KeyError: neighbors = set()
        neighbors.add(node2)
        self.edges[node1] = neighbors
        if not self.directed and not __reversed: self.add_edge(node2, node1, True)

    def neighbors(self, node):
        try:
            return self.edges[node]
        except KeyError:
            return []

    def depth_first_search(self, start, goal):
        found, fringe, visited, came_from = False, deque([start]), set([start]), {start: None}
        print('{:11s} | {}'.format('Expand Node', 'Fringe'))
        print('--------------------')
        print('{:11s} | {}'.format('-', start))
        while not found and len(fringe):
            current = fringe.pop()
            print('{:11s}'.format(current), end=' | ')
            if current == goal: found = True; break
            for node in self.neighbors(current):
                if node not in visited: visited.add(node); fringe.append(node); came_from[node] = current
            print(', '.join(fringe))
        if found: print(); return came_from
        else: print('No path from {} to {}'.format(start, goal))


    def breadth_first_search(self, start, goal):
        found, fringe, visited, came_from = False, deque([start]), set([start]), {start: None}
        print('{:11s} | {}'.format('Expand Node', 'Fringe'))
        print('--------------------')
        print('{:11s} | {}'.format('-', start))
        while not found and len(fringe):
            current = fringe.pop()
            print('{:11s}'.format(current), end=' | ')
            if current == goal: found = True; break
            for node in self.neighbors(current):
                if node not in visited: visited.add(node); fringe.appendleft(node); came_from[node] = current
            print(', '.join(fringe))
        if found:
            print(); return came_from
        else:
            print('No path from {} to {}'.format(start, goal))

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


def process(graph, algo='DFS', out=False):

    print(algo)
    start, goal = ["John F Kennedy Intl", 'Deadhorse']
    start_time = time.time_ns()

    if algo == 'DFS':
        traced_path = graph.depth_first_search(start, goal)
    else:
        traced_path = graph.breadth_first_search(start, goal)
    print("--- %s nanoseconds ---" % (time.time_ns() - start_time))
    if out:
        if traced_path:
            print('Path:', end=' ')
            Graph.print_path(traced_path, goal)
            print()


if __name__ == '__main__':

    book, raw_graph = read_mtx("inf-USAir97.mtx")
    graph = Graph(directed=False)
    for edge in raw_graph:
        graph.add_edge(edge[0], edge[1])

    process(graph, algo='DFS', out=True)
