from collections import defaultdict, deque, OrderedDict
import math
import itertools
import random
# 3) N-Queens (conteo soluciones)
def n_queens_count(n):
    cols = set(); d1 = set(); d2 = set()
    count = 0
    def backtrack(r=0):
        nonlocal count
        if r==n:
            count += 1; return
        for c in range(n):
            if c in cols or (r-c) in d1 or (r+c) in d2:
                continue
            cols.add(c); d1.add(r-c); d2.add(r+c)
            backtrack(r+1)
            cols.remove(c); d1.remove(r-c); d2.remove(r+c)
    backtrack()
    return count