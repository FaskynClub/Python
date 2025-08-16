from collections import defaultdict, deque, OrderedDict
import math
import itertools
import random
# 2) Dijkstra (grafo dirigido, pesos no negativos)
def dijkstra(graph, src):
    import heapq
    dist = {v: math.inf for v in graph}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist