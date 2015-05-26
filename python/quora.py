#! /usr/bin/env python

import sys

class Node:
    def __init__(self, weight, neighbors):
        self.e = dict(zip(neighbors, [weight for _ in neighbors]))
        self.messages = []
        self.nn = len(neighbors)
    def recv(self, other):
        self.messages.append(4) # FIXME
    def update(self):
        pass

def f(N, T, A):
    nodes = [Node(T[n-1], adj) for n, adj in A.items()]
    for _ in range(N): # iterations to convergence
        for node in nodes:
            pass
    return min(nodes, key=lambda node: 5).nn # TODO: placeholder

if __name__ == '__main__':
    N = int(sys.stdin.readline().strip('\n'))
    T = list(map(int, sys.stdin.readline().strip('\n').split()))
    adj = dict(zip(range(1, N+1), [set() for _ in range(N)]))
    for line in sys.stdin:
        u, v = map(int, line.split())
        adj[u].add(v)
        adj[v].add(u)
    print(f(N, T, adj))
