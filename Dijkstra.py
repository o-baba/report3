import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Coord = Tuple[int, int]  # (row, col)


@dataclass
class SearchResult:
    path: List[Coord]
    cost: float
    visited: int
    time_ms: float
    order: List[Coord]


def neighbors(pos: Coord, grid: np.ndarray, allow_diag: bool) -> List[Tuple[Coord, float]]:
    r, c = pos
    h, w = grid.shape

    steps: List[Tuple[int, int, float]] = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
    ]
    if allow_diag:
        d = math.sqrt(2.0)
        steps += [(-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d)]

    out: List[Tuple[Coord, float]] = []
    for dr, dc, cost in steps:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 0:
            out.append(((rr, cc), cost))
    return out


def reconstruct(came: Dict[Coord, Optional[Coord]], start: Coord, goal: Coord) -> List[Coord]:
    if goal not in came:
        return []
    cur: Optional[Coord] = goal
    path: List[Coord] = []
    while cur is not None:
        path.append(cur)
        if cur == start:
            break
        cur = came.get(cur)
    path.reverse()
    if not path or path[0] != start:
        return []
    return path


def dijkstra(grid: np.ndarray, start: Coord, goal: Coord, diag: bool) -> SearchResult:
    t0 = time.perf_counter()
    dist: Dict[Coord, float] = {start: 0.0}
    came: Dict[Coord, Optional[Coord]] = {start: None}
    pq: List[Tuple[float, Coord]] = []
    heappush(pq, (0.0, start))
    order: List[Coord] = []

    while pq:
        cur_cost, cur = heappop(pq)
        if cur_cost != dist.get(cur, float("inf")):
            continue
        order.append(cur)
        if cur == goal:
            break
        for nxt, sc in neighbors(cur, grid, diag):
            nc = cur_cost + sc
            if nc < dist.get(nxt, float("inf")):
                dist[nxt] = nc
                came[nxt] = cur
                heappush(pq, (nc, nxt))

    t1 = time.perf_counter()
    path = reconstruct(came, start, goal)
    return SearchResult(path, dist.get(goal, float("inf")), len(order), (t1 - t0) * 1000.0, order)


def heuristic(a: Coord, b: Coord, diag: bool) -> float:
    (r1, c1), (r2, c2) = a, b
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return math.sqrt(dr * dr + dc * dc) if diag else float(dr + dc)


def astar(grid: np.ndarray, start: Coord, goal: Coord, diag: bool) -> SearchResult:
    t0 = time.perf_counter()
    g: Dict[Coord, float] = {start: 0.0}
    came: Dict[Coord, Optional[Coord]] = {start: None}
    pq: List[Tuple[float, Coord]] = []
    heappush(pq, (heuristic(start, goal, diag), start))
    order: List[Coord] = []

    while pq:
        fcur, cur = heappop(pq)
        if (g.get(cur, float("inf")) + heuristic(cur, goal, diag)) != fcur:
            continue
        order.append(cur)
        if cur == goal:
            break
        gcur = g[cur]
        for nxt, sc in neighbors(cur, grid, diag):
            ng = gcur + sc
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                heappush(pq, (ng + heuristic(nxt, goal, diag), nxt))

    t1 = time.perf_counter()
    path = reconstruct(came, start, goal)
    return SearchResult(path, g.get(goal, float("inf")), len(order), (t1 - t0) * 1000.0, order)


def generate_grid(h: int, w: int, obstacle_prob: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) < obstacle_prob).astype(np.int8)  # 1=obstacle, 0=free


def pick_start_goal(grid: np.ndarray, seed: int) -> Tuple[Coord, Coord]:
    rng = np.random.default_rng(seed + 12345)
    free = np.argwhere(grid == 0)
    if free.shape[0] < 2:
        raise RuntimeError("Not enough free cells; reduce obstacle probability.")
    s = free[rng.integers(0, free.shape[0])]
    g = free[rng.integers(0, free.shape[0])]
    while (g == s).all():
        g = free[rng.integers(0, free.shape[0])]
    return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))


def ensure_free(grid: np.ndarray, pos: Coord) -> None:
    grid[pos[0], pos[1]] = 0


def find_pair_with_cost(grid: np.ndarray, target_cost: int, seed: int, max_tries: int = 8000) -> Tuple[Coord, Coord]:
    """Try to find (start, goal) such that Dijkstra (4-neigh) cost equals target_cost.
    This makes Figure 1 likely match the screenshot style. If not found, fall back.
    """
    rng = np.random.default_rng(seed)
    free = np.argwhere(grid == 0)
    if free.shape[0] < 2:
        raise RuntimeError("Not enough free cells.")
    for _ in range(max_tries):
        s = free[rng.integers(0, free.shape[0])]
        g = free[rng.integers(0, free.shape[0])]
        if (g == s).all():
            continue
        start = (int(s[0]), int(s[1]))
        goal = (int(g[0]), int(g[1]))
        rd = dijkstra(grid, start, goal, diag=False)
        if rd.path and abs(rd.cost - float(target_cost)) < 1e-9:
            return start, goal
    # fallback
    return pick_start_goal(grid, seed)


def plot_fig1(grid: np.ndarray, start: Coord, goal: Coord, rd: SearchResult, ra: SearchResult, out_path: str) -> None:
    img = 1 - grid  # free=1, obstacle=0
    plt.figure(figsize=(10, 6))
    plt.imshow(img, interpolation="nearest")

    if rd.path:
        pr = [p[0] for p in rd.path]
        pc = [p[1] for p in rd.path]
        plt.plot(pc, pr, linewidth=2, label="Dijkstra path")

    if ra.path:
        pr = [p[0] for p in ra.path]
        pc = [p[1] for p in ra.path]
        plt.plot(pc, pr, linewidth=2, linestyle="--", label="A* path")

    plt.scatter([start[1]], [start[0]], marker="o", s=80, label="Start")
    plt.scatter([goal[1]], [goal[0]], marker="x", s=80, label="Goal")

    title = (
        f"Grid: {grid.shape[0]}x{grid.shape[1]}, obstacles={float(grid.mean()):.2f}\n"
        f"Dijkstra: cost={rd.cost:.3f}, visited={rd.visited}, time={rd.time_ms:.2f} ms\n"
        f"A*: cost={ra.cost:.3f}, visited={ra.visited}, time={ra.time_ms:.2f} ms"
    )
    plt.title(title)
    plt.axis("off")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def sweep_one_grid(h: int, w: int, p_list: List[float], seed_base: int, diag: bool) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(p_list):
        grid = generate_grid(h, w, p, seed_base + i)
        start, goal = pick_start_goal(grid, seed_base + i)
        ensure_free(grid, start)
        ensure_free(grid, goal)

        rd = dijkstra(grid, start, goal, diag=diag)

        # Ensure reachable by resampling start/goal on the same fixed grid
        tries = 0
        while not rd.path and tries < 120:
            tries += 1
            start, goal = pick_start_goal(grid, seed_base + i + tries)
            ensure_free(grid, start)
            ensure_free(grid, goal)
            rd = dijkstra(grid, start, goal, diag=diag)

        ra = astar(grid, start, goal, diag=diag)

        rows.append({
            "grid": f"{h}x{w}",
            "p": p,
            "diag": diag,
            "seed": seed_base + i,
            "start": str(start),
            "goal": str(goal),
            "dij_cost": rd.cost,
            "dij_steps": max(0, len(rd.path) - 1),
            "dij_visited": rd.visited,
            "dij_time_ms": rd.time_ms,
            "astar_cost": ra.cost,
            "astar_steps": max(0, len(ra.path) - 1),
            "astar_visited": ra.visited,
            "astar_time_ms": ra.time_ms,
        })
    return pd.DataFrame(rows)


def main() -> None:
    # -------------------------
    # Figure 1: example grid + paths
    # -------------------------
    h1, w1 = 40, 60
    obstacle_prob = 0.28
    seed1 = 0
    grid1 = generate_grid(h1, w1, obstacle_prob, seed1)

    # Try to match the screenshot-ish cost (27) on 4-neighborhood
    start1, goal1 = find_pair_with_cost(grid1, target_cost=27, seed=1, max_tries=12000)
    ensure_free(grid1, start1)
    ensure_free(grid1, goal1)

    rd1 = dijkstra(grid1, start1, goal1, diag=False)

    # If still unreachable (rare), resample a few times
    tries = 0
    while not rd1.path and tries < 200:
        tries += 1
        start1, goal1 = pick_start_goal(grid1, seed1 + tries)
        ensure_free(grid1, start1)
        ensure_free(grid1, goal1)
        rd1 = dijkstra(grid1, start1, goal1, diag=False)

    ra1 = astar(grid1, start1, goal1, diag=False)
    plot_fig1(grid1, start1, goal1, rd1, ra1, "fig1_grid_paths_example.png")

    # -------------------------
    # Figure 2 & 3: 4-neighborhood sweep (40x60)
    # -------------------------
    p_list = [0.20, 0.25, 0.30, 0.35, 0.40]
    df_4 = sweep_one_grid(40, 60, p_list, seed_base=100, diag=False)
    df_4.to_csv("sweep_4neigh_40x60.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_4["p"], df_4["dij_visited"], marker="o", linewidth=2, label="Dijkstra visited")
    plt.plot(df_4["p"], df_4["astar_visited"], marker="s", linewidth=2, label="A* visited")
    plt.xlabel("Obstacle probability p")
    plt.ylabel("Visited nodes")
    plt.title("Visited nodes vs obstacle probability (grid 40x60, 4-neighborhood)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_visited_vs_p_4neigh.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_4["p"], df_4["dij_time_ms"], marker="o", linewidth=2, label="Dijkstra time (ms)")
    plt.plot(df_4["p"], df_4["astar_time_ms"], marker="s", linewidth=2, label="A* time (ms)")
    plt.xlabel("Obstacle probability p")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs obstacle probability (grid 40x60, 4-neighborhood)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3_runtime_vs_p_4neigh.png", dpi=250)
    plt.close()

    # -------------------------
    # Figure 4~6: 4-neighborhood vs 8-neighborhood sweep (40x60)
    # -------------------------
    df4 = sweep_one_grid(40, 60, p_list, seed_base=200, diag=False)
    df8 = sweep_one_grid(40, 60, p_list, seed_base=200, diag=True)
    df4v8 = pd.concat([df4, df8], ignore_index=True)
    df4v8.to_csv("sweep_4v8_40x60.csv", index=False)

    # visited vs p (4v8, Dijkstra + A*)
    plt.figure(figsize=(9, 5))
    d4 = df4v8[df4v8["diag"] == False]
    d8 = df4v8[df4v8["diag"] == True]

    plt.plot(d4["p"], d4["dij_visited"], marker="o", linewidth=2, label="Dijkstra (4-neigh) visited")
    plt.plot(d8["p"], d8["dij_visited"], marker="o", linewidth=2, linestyle="--", label="Dijkstra (8-neigh) visited")
    plt.plot(d4["p"], d4["astar_visited"], marker="s", linewidth=2, label="A* (4-neigh) visited")
    plt.plot(d8["p"], d8["astar_visited"], marker="s", linewidth=2, linestyle="--", label="A* (8-neigh) visited")

    plt.xlabel("Obstacle probability p")
    plt.ylabel("Visited nodes")
    plt.title("Visited nodes vs p (grid 40x60): 4-neighborhood vs 8-neighborhood")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig4_visited_vs_p_4v8.png", dpi=250)
    plt.close()

    # runtime vs p (4v8)
    plt.figure(figsize=(9, 5))
    plt.plot(d4["p"], d4["dij_time_ms"], marker="o", linewidth=2, label="Dijkstra (4-neigh) time")
    plt.plot(d8["p"], d8["dij_time_ms"], marker="o", linewidth=2, linestyle="--", label="Dijkstra (8-neigh) time")
    plt.plot(d4["p"], d4["astar_time_ms"], marker="s", linewidth=2, label="A* (4-neigh) time")
    plt.plot(d8["p"], d8["astar_time_ms"], marker="s", linewidth=2, linestyle="--", label="A* (8-neigh) time")
    plt.xlabel("Obstacle probability p")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs p (grid 40x60): 4-neighborhood vs 8-neighborhood")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig5_runtime_vs_p_4v8.png", dpi=250)
    plt.close()

    # steps and cost vs p (use Dijkstra only, 4 vs 8)
    plt.figure(figsize=(9, 5))
    plt.plot(d4["p"], d4["dij_steps"], marker="o", linewidth=2, label="Dijkstra steps (4)")
    plt.plot(d8["p"], d8["dij_steps"], marker="o", linewidth=2, linestyle="--", label="Dijkstra steps (8)")
    plt.plot(d4["p"], d4["dij_cost"], marker="s", linewidth=2, label="Dijkstra cost (4)")
    plt.plot(d8["p"], d8["dij_cost"], marker="s", linewidth=2, linestyle="--", label="Dijkstra cost (8)")
    plt.xlabel("Obstacle probability p")
    plt.ylabel("Steps / Cost")
    plt.title("Steps and Cost vs p (grid 40x60): 4-neighborhood vs 8-neighborhood")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("fig6_steps_cost_vs_p_4v8.png", dpi=250)
    plt.close()

    # -------------------------
    # Figure 7: visited ratio bar across conditions (like screenshot)
    # -------------------------
    # Conditions (12 bars):
    #  - 40x60 diag=False, p=0.20..0.40 (5)
    #  - 60x90 diag=False, p=0.20..0.40 (5)
    #  - 40x60 diag=True,  p=0.30 (1)
    #  - 60x90 diag=True,  p=0.35 (1)
    conds: List[Tuple[int, int, float, bool, int]] = []
    for p in p_list:
        conds.append((40, 60, p, False, 3000))
    for p in p_list:
        conds.append((60, 90, p, False, 4000))
    conds.append((40, 60, 0.30, True,  5000))
    conds.append((60, 90, 0.35, True,  6000))

    rows = []
    for (h, w, p, diag, seed_base) in conds:
        df = sweep_one_grid(h, w, [p], seed_base=seed_base, diag=diag)
        r = df.iloc[0]
        ratio = float(r["astar_visited"]) / float(r["dij_visited"]) if float(r["dij_visited"]) > 0 else float("nan")
        rows.append({
            "grid": f"{h}x{w}",
            "p": p,
            "diag": diag,
            "dij_visited": float(r["dij_visited"]),
            "astar_visited": float(r["astar_visited"]),
            "visited_ratio": ratio,
        })

    df_ratio = pd.DataFrame(rows)
    df_ratio.to_csv("visited_ratio_conditions.csv", index=False)

    x_labels = [f'{r["grid"]}, p={r["p"]:.2f}, diag={r["diag"]}' for _, r in df_ratio.iterrows()]
    y = df_ratio["visited_ratio"].to_numpy()

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(y)), y)
    plt.xticks(np.arange(len(y)), x_labels, rotation=90)
    plt.ylabel("Visited ratio (A* / Dijkstra)")
    plt.title("Search efficiency comparison: visited ratio across conditions")
    plt.tight_layout()
    plt.savefig("fig7_visited_ratio_bar.png", dpi=250)
    plt.close()

    print("Generated figures:")
    print("  fig1_grid_paths_example.png")
    print("  fig2_visited_vs_p_4neigh.png")
    print("  fig3_runtime_vs_p_4neigh.png")
    print("  fig4_visited_vs_p_4v8.png")
    print("  fig5_runtime_vs_p_4v8.png")
    print("  fig6_steps_cost_vs_p_4v8.png")
    print("  fig7_visited_ratio_bar.png")
    print("Generated CSV logs:")
    print("  sweep_4neigh_40x60.csv, sweep_4v8_40x60.csv, visited_ratio_conditions.csv")


if __name__ == "__main__":
    main()
