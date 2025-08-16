from collections import defaultdict, deque, OrderedDict
import math
import itertools
import random
# 4) Sudoku solver (backtracking)
def solve_sudoku(board):
    # board: lista de listas 9x9 con 0 como vacío
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empties = []
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v==0:
                empties.append((r,c))
            else:
                rows[r].add(v); cols[c].add(v); boxes[(r//3)*3+(c//3)].add(v)
    def bt(i=0):
        if i==len(empties): return True
        r,c = empties[i]
        b = (r//3)*3+(c//3)
        for v in range(1,10):
            if v not in rows[r] and v not in cols[c] and v not in boxes[b]:
                board[r][c]=v; rows[r].add(v); cols[c].add(v); boxes[b].add(v)
                if bt(i+1): return True
                board[r][c]=0; rows[r].remove(v); cols[c].remove(v); boxes[b].remove(v)
        return False
    bt()
    return board