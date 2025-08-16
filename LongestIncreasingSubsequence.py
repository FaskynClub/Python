from collections import defaultdict, deque, OrderedDict
import math
import itertools
import random
# 1) Longest Increasing Subsequence (LIS) con reconstrucción O(n log n)
def lis(seq):
    import bisect
    tails = []
    prev = [-1]*len(seq)
    idx = []
    for i, x in enumerate(seq):
        j = bisect.bisect_left(tails, x)
        if j == len(tails):
            tails.append(x); idx.append(i)
        else:
            tails[j] = x; idx[j] = i
        prev[i] = idx[j-1] if j>0 else -1
    # reconstrucción
    k = idx[-1]
    res = []
    while k!=-1:
        res.append(seq[k]); k = prev[k]
    return list(reversed(res))
