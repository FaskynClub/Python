from collections import defaultdict, deque, OrderedDict
import math
import itertools
import random
# 5) 0/1 Knapsack (programación dinámica)
def knapsack(weights, values, W):
    n = len(weights)
    dp = [0]*(W+1)
    for i in range(n):
        w, val = weights[i], values[i]
        for cap in range(W, w-1, -1):
            dp[cap] = max(dp[cap], dp[cap-w]+val)
    return dp[W]

# 6) Edit distance (Levenshtein)
def edit_distance(a,b):
    n,m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

# 7) KMP (búsqueda de patrón)
def kmp(text, pattern):
    # construir lps
    lps = [0]*len(pattern)
    k = 0
    for i in range(1,len(pattern)):
        while k>0 and pattern[k]!=pattern[i]:
            k = lps[k-1]
        if pattern[k]==pattern[i]:
            k += 1
        lps[i]=k
    # búsqueda
    res=[]; j=0
    for i,ch in enumerate(text):
        while j>0 and pattern[j]!=ch:
            j = lps[j-1]
        if pattern[j]==ch:
            j += 1
            if j==len(pattern):
                res.append(i-j+1); j = lps[j-1]
    return res

# 8) Orden topológico (Kahn)
def topo_sort(edges, n):
    g = [[] for _ in range(n)]
    indeg = [0]*n
    for u,v in edges:
        g[u].append(v); indeg[v]+=1
    q = deque([i for i in range(n) if indeg[i]==0])
    order=[]
    while q:
        u=q.popleft(); order.append(u)
        for v in g[u]:
            indeg[v]-=1
            if indeg[v]==0: q.append(v)
    return order if len(order)==n else None

# 9) Máxima subarray (Kadane)
def max_subarray(arr):
    best = cur = arr[0]
    for x in arr[1:]:
        cur = max(x, cur+x)
        best = max(best, cur)
    return best

# 10) Criba de Eratóstenes
def sieve(n):
    is_prime = [True]*(n+1)
    is_prime[0]=is_prime[1]=False
    p=2
    while p*p<=n:
        if is_prime[p]:
            for k in range(p*p, n+1, p):
                is_prime[k]=False
        p+=1
    return [i for i in range(n+1) if is_prime[i]]

# 11) Convex Hull (Graham scan)
def convex_hull(points):
    points = sorted(points)
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

# 12) Trie con autocompletado
class Trie:
    def __init__(self):
        self.trie = {}
    def insert(self, word):
        t = self.trie
        for ch in word:
            t = t.setdefault(ch, {})
        t['$']=True
    def _dfs(self, node, pref, out):
        if '$' in node: out.append(pref)
        for ch, nxt in node.items():
            if ch=='$': continue
            self._dfs(nxt, pref+ch, out)
    def autocomplete(self, prefix):
        t = self.trie
        for ch in prefix:
            if ch not in t: return []
            t = t[ch]
        out=[]; self._dfs(t, prefix, out)
        return out

# 13) LRU Cache (OrderedDict)
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.od = OrderedDict()
    def get(self, k):
        if k not in self.od: return -1
        v = self.od.pop(k)
        self.od[k]=v
        return v
    def put(self, k, v):
        if k in self.od: self.od.pop(k)
        elif len(self.od) >= self.cap:
            self.od.popitem(last=False)
        self.od[k]=v

# 14) Union-Find (Disjoint Set Union)
class DSU:
    def __init__(self, n):
        self.p=list(range(n)); self.r=[0]*n; self.c=n
    def find(self, x):
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self, a,b):
        ra,rb = self.find(a), self.find(b)
        if ra==rb: return False
        if self.r[ra]<self.r[rb]: ra,rb=rb,ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1
        self.c-=1; return True

# 15) Mediana de dos arrays ordenados (O(log n))
def median_two_sorted(a,b):
    if len(a)>len(b): a,b=b,a
    n,m=len(a),len(b)
    total=n+m; half=total//2
    lo,hi=0,n
    while lo<=hi:
        i=(lo+hi)//2
        j=half-i
        aL=a[i-1] if i>0 else -math.inf
        aR=a[i] if i<n else math.inf
        bL=b[j-1] if j>0 else -math.inf
        bR=b[j] if j<m else math.inf
        if aL<=bR and bL<=aR:
            if total%2: return min(aR,bR)
            return (max(aL,bL)+min(aR,bR))/2
        elif aL>bR: hi=i-1
        else: lo=i+1

# 16) Quickselect (k-ésimo menor)
def quickselect(arr, k):
    import random
    def select(l, r, k):
        if l==r: return arr[l]
        pivot = arr[random.randint(l,r)]
        i, j = l, r
        while i<=j:
            while arr[i]<pivot: i+=1
            while arr[j]>pivot: j-=1
            if i<=j:
                arr[i],arr[j]=arr[j],arr[i]
                i+=1; j-=1
        if k<=j: return select(l,j,k)
        if k>=i: return select(i,r,k)
        return arr[k]
    return select(0, len(arr)-1, k)

# 17) Rabin-Karp (hash rolling)
def rabin_karp(text, pattern):
    n,m=len(text),len(pattern)
    if m==0 or m>n: return []
    base=257; mod=10**9+7
    powm=pow(base,m-1,mod)
    h=0; hp=0
    for i in range(m):
        h=(h*base+ord(text[i]))%mod
        hp=(hp*base+ord(pattern[i]))%mod
    res=[]
    for i in range(n-m+1):
        if h==hp and text[i:i+m]==pattern: res.append(i)
        if i<n-m:
            h=( (h - ord(text[i])*powm)*base + ord(text[i+m]) )%mod
    return res

# 18) Generar paréntesis balanceados (backtracking)
def gen_parentheses(n):
    res=[]
    def bt(s='', open=0, close=0):
        if len(s)==2*n: res.append(s); return
        if open<n: bt(s+'(', open+1, close)
        if close<open: bt(s+')', open, close+1)
    bt()
    return res

# 19) Árbol binario de búsqueda (insert y altura)
class BST:
    class Node:
        def __init__(self, v): self.v=v; self.l=None; self.r=None
    def __init__(self): self.root=None
    def insert(self, v):
        def rec(node, v):
            if not node: return BST.Node(v)
            if v<node.v: node.l=rec(node.l,v)
            elif v>node.v: node.r=rec(node.r,v)
            return node
        self.root=rec(self.root,v)
    def height(self):
        def h(n):
            return -1 if not n else 1+max(h(n.l), h(n.r))
        return h(self.root)
    def inorder(self):
        res=[]
        def dfs(n):
            if not n: return
            dfs(n.l); res.append(n.v); dfs(n.r)
        dfs(self.root)
        return res

# 20) BFS en laberinto (camino más corto)
def shortest_path_maze(grid, start, goal):
    R,C=len(grid),len(grid[0])
    q=deque([(*start,0)])
    seen={start}
    parents={start:None}
    while q:
        r,c,d=q.popleft()
        if (r,c)==goal:
            # reconstrucción
            path=[]; cur=goal
            while cur is not None:
                path.append(cur); cur=parents[cur]
            path.reverse()
            return d, path
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc]==0 and (nr,nc) not in seen:
                seen.add((nr,nc)); parents[(nr,nc)]=(r,c); q.append((nr,nc,d+1))
    return None

if __name__ == '__main__':
    # P1
    seq1 = [10,9,2,5,3,7,101,18]
    print('P1 - LIS ->', lis(seq1))
    # P2
    graph2 = {
        'A':[('B',4),('C',2)],
        'B':[('C',5),('D',10)],
        'C':[('E',3)],
        'D':[('F',11)],
        'E':[('D',4)],
        'F':[]
    }
    print('P2 - Dijkstra desde A ->', dijkstra(graph2,'A'))
    # P3
    print('P3 - N-Queens N=8 ->', n_queens_count(8), 'soluciones')
    # P4
    sudoku = [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9],
    ]
    solved = solve_sudoku([row[:] for row in sudoku])
    print('P4 - Sudoku (primeras 3 filas) ->', solved[:3])
    # P5
    print('P5 - Knapsack W=5 ->', knapsack([2,3,4,5], [3,4,5,8], 5))
    # P6
    print("P6 - Edit distance('gato','pato') ->", edit_distance('gato','pato'))
    # P7
    print("P7 - KMP ->", kmp('ababcabcabababd','ababd'))
    # P8
    edges8 = [(0,1),(0,2),(1,3),(2,3)]
    print('P8 - Topo sort ->', topo_sort(edges8,4))
    # P9
    print('P9 - Máxima subarray ->', max_subarray([-2,1,-3,4,-1,2,1,-5,4]))
    # P10
    primes = sieve(100)
    print('P10 - Primos hasta 100 (primeros 10) ->', primes[:10], 'total=', len(primes))
    # P11
    points11 = [(0,0),(1,1),(1,0),(0,1),(0.5,0.5),(-1,-1),(2,2),(2,0)]
    print('P11 - Convex hull ->', convex_hull(points11))
    # P12
    trie = Trie()
    for w in ['python','pycharm','pyramid','java']: trie.insert(w)
    print("P12 - Autocomplete 'py' ->", trie.autocomplete('py'))
    # P13
    lru = LRUCache(2)
    lru.put(1,1); lru.put(2,2); _=lru.get(1); lru.put(3,3)
    # ahora 2 debe haberse expulsado
    print('P13 - LRU get(2) ->', lru.get(2), '; get(1) ->', lru.get(1))
    # P14
    dsu = DSU(5)
    dsu.union(0,1); dsu.union(1,2); dsu.union(3,4)
    print('P14 - Componentes ->', dsu.c, '; find(2)==find(0)?', dsu.find(2)==dsu.find(0))
    # P15
    print('P15 - Mediana ->', median_two_sorted([1,3],[2,4]))
    # P16
    print('P16 - Quickselect k=3 ->', quickselect([7,10,4,3,20,15],3))
    # P17
    print("P17 - Rabin-Karp ->", rabin_karp('aaaaa','aa'))
    # P18
    print('P18 - Paréntesis n=3 ->', gen_parentheses(3))
    # P19
    bst = BST()
    for v in [5,3,7,2,4,6,8]: bst.insert(v)
    print('P19 - BST inorder ->', bst.inorder(), '; altura ->', bst.height())
    # P20
    maze = [
        [0,1,0,0,0],
        [0,1,0,1,0],
        [0,0,0,1,0],
        [1,1,0,0,0],
        [0,0,0,1,0],
    ]
    dist, path = shortest_path_maze(maze, (0,0), (4,4))
    print('P20 - BFS dist ->', dist, '; path ->', path)