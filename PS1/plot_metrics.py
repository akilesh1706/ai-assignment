# plot_metrics.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Helper: render an ASCII grid file (last_grid.txt / last_path_*.txt)
def render_and_print_ascii(file_path, save_png=True):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]
        if not lines:
            return
        header = lines[0] if lines[0].startswith("#") else ""
        grid_lines = lines[1:] if header else lines
        print(f"\n=== {os.path.basename(file_path)} ===")
        if header:
            print(header)
        for ln in grid_lines:
            print(ln)

        # convert ASCII → array
        h = len(grid_lines)
        w = len(grid_lines[0]) if h > 0 else 0
        arr = np.zeros((h, w), dtype=int)
        mapping = {'.': 0, '#': 1, 'S': 2, 'G': 3, '*': 4}
        for r, ln in enumerate(grid_lines):
            for c, ch in enumerate(ln):
                arr[r, c] = mapping.get(ch, 0)

        if save_png and h > 0 and w > 0:
            cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            plt.figure(figsize=(7,7))
            plt.imshow(arr, cmap=cmap, norm=norm, origin='upper', interpolation='none')

            # ✅ Add visible borders around each grid cell
            plt.grid(which='major', color='gray', linewidth=0.5)
            plt.xticks(np.arange(-0.5, w, 1))
            plt.yticks(np.arange(-0.5, h, 1))
            plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.gca().set_aspect('equal', adjustable='box')

            plt.title(os.path.basename(file_path))
            outpng = os.path.splitext(file_path)[0] + ".png"
            plt.savefig(outpng, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved image: {outpng}")

    except Exception as e:
        print(f"Could not read/plot {file_path}: {e}")

# =============== Load Metrics CSV ===============
csv_path = "a_star_metrics.csv"

if not os.path.exists(csv_path):
    print(f"CSV not found: {csv_path}. Run a_star.py first.")
else:
    df = pd.read_csv(csv_path)
    df["path_cost"] = pd.to_numeric(df.get("path_cost", pd.Series()), errors="coerce")
    df["time_ms"] = pd.to_numeric(df.get("time_ms", pd.Series()), errors="coerce")

    grouped = df.groupby("heuristic_name")[["path_cost","nodes_expanded","time_ms"]].mean()
    print("\nAverage Metrics:\n", grouped)

    # Show grid + each path overlay
    if os.path.exists("last_grid.txt"):
        render_and_print_ascii("last_grid.txt")
    else:
        print("\n(last_grid.txt not found — enable saving grid in a_star.py)")

    for pfile in sorted(glob.glob("last_path_*.txt")):
        render_and_print_ascii(pfile)

    # ---- Plot Metrics ----
    plt.figure()
    grouped["time_ms"].plot(kind="bar")
    plt.title("Average Computation Time (ms)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    grouped["nodes_expanded"].plot(kind="bar")
    plt.title("Average Nodes Expanded")
    plt.tight_layout()
    plt.show()

    plt.figure()
    grouped["path_cost"].plot(kind="bar")
    plt.title("Average Path Cost")
    plt.tight_layout()
    plt.show()
