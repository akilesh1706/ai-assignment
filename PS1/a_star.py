# a_star.py
# Usage:
#   python a_star.py --runs 50 --rows 40 --cols 40 --obs 0.2 --seed 1234 --out a_star_metrics.csv

import argparse
import csv
import math
import random
import time
from collections import deque, defaultdict
import heapq
import numpy as np
import matplotlib.pyplot as plt

# --- Heuristics ---
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

HEURISTICS = {
    0: ("Manhattan", manhattan),
    1: ("Euclidean", euclidean),
    2: ("Chebyshev", chebyshev),
}

# --- Grid class ---
class Grid:
    def __init__(self, rows, cols, obstacle_prob, seed=None):
        self.rows = rows
        self.cols = cols
        self.obstacle_prob = obstacle_prob
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.grid = [[0] * cols for _ in range(rows)]
        self._generate()

    def _generate(self):
        rnd = random.Random(self.seed)
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = 1 if rnd.random() < self.obstacle_prob else 0

    def is_free(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def random_free_cell(self, rnd):
        for _ in range(5000):
            r = rnd.randrange(self.rows)
            c = rnd.randrange(self.cols)
            if self.is_free(r, c): return (r, c)
        return (-1, -1)

# --- A* ---
def a_star_search(grid: Grid, start, goal, heuristic_func):
    t0 = time.perf_counter()
    R, C = grid.rows, grid.cols
    sr, sc = start
    gr, gc = goal

    open_heap = []
    gscore = [[math.inf] * C for _ in range(R)]
    parent = [[None] * C for _ in range(R)]
    visited = [[False] * C for _ in range(R)]

    gscore[sr][sc] = 0.0
    heapq.heappush(open_heap, (heuristic_func(start, goal), 0.0, sr, sc))
    nodes_expanded = 0

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        if visited[r][c]: continue
        visited[r][c] = True
        nodes_expanded += 1

        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)
            while cur:
                path.append(cur)
                cur = parent[cur[0]][cur[1]]
            path.reverse()
            t1 = time.perf_counter()
            return {
                "found": True,
                "path_cost": gscore[r][c],
                "nodes_expanded": nodes_expanded,
                "time_ms": (t1 - t0) * 1000,
                "path": path
            }

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not grid.is_free(nr, nc): continue
            ng = gscore[r][c] + 1
            if ng < gscore[nr][nc]:
                gscore[nr][nc] = ng
                parent[nr][nc] = (r, c)
                heapq.heappush(open_heap, (ng + heuristic_func((nr, nc), goal), ng, nr, nc))

    t1 = time.perf_counter()
    return {"found": False, "path_cost": 0.0, "nodes_expanded": nodes_expanded,
            "time_ms": (t1 - t0) * 1000, "path": []}

# Save ASCII files
def save_grid_ascii(grid: Grid, start, goal):
    with open("last_grid.txt", "w") as f:
        f.write("# Grid with Obstacles (S=start, G=goal)\n")
        for r in range(grid.rows):
            row = ""
            for c in range(grid.cols):
                if (r, c) == start: row += "S"
                elif (r, c) == goal: row += "G"
                else: row += "#" if grid.grid[r][c] == 1 else "."
            f.write(row + "\n")

def save_path_ascii(path, grid: Grid, name):
    arr = []
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            row.append("#" if grid.grid[r][c] == 1 else ".")
        arr.append(row)

    for i, (r, c) in enumerate(path):
        arr[r][c] = "S" if i == 0 else "G" if i == len(path)-1 else "*"

    fname = f"last_path_{name}.txt"
    with open(fname, "w") as f:
        f.write(f"# A* Path using {name}\n")
        for row in arr:
            f.write("".join(row) + "\n")

# Experiment Runner
def run_experiments(runs, rows, cols, obs_prob, seed, out_csv):
    rnd = random.Random(seed)
    fieldnames = ["run","heuristic_name","found","path_cost","nodes_expanded","time_ms"]
    with open(out_csv,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(runs):
            G = Grid(rows, cols, obs_prob, rnd.randrange(1<<30))
            s = G.random_free_cell(rnd)
            g = G.random_free_cell(rnd)

            if i == 0:
                save_grid_ascii(G, s, g)

            for hidx, (hname, hfunc) in HEURISTICS.items():
                res = a_star_search(G, s, g, hfunc)
                if i == 0 and res["found"]:
                    save_path_ascii(res["path"], G, hname)

                writer.writerow({
                    "run": i,
                    "heuristic_name": hname,
                    "found": int(res["found"]),
                    "path_cost": res["path_cost"],
                    "nodes_expanded": res["nodes_expanded"],
                    "time_ms": res["time_ms"]
                })

    print("Done. Metrics saved & ASCII visualization written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--obs", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--out", type=str, default="a_star_metrics.csv")
    a = parser.parse_args()
    run_experiments(a.runs, a.rows, a.cols, a.obs, a.seed, a.out)
