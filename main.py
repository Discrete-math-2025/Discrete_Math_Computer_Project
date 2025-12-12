import argparse
import csv
import matplotlib.pyplot as plt

<<<<<<< HEAD
from algorithms.dijkstra import dijkstra
from algorithms.duan_algo import run_duan
from data_generator import generate_graph
=======
>>>>>>> d2d8bc5289061d53885780ae34acc68f45b3060e

def plot_results_from_csv(csv_path, save_path=None, show=False):
    sizes = []
    times_dijkstra = []
    times_duan = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sizes.append(int(row["size"]))
                times_dijkstra.append(float(row["time_dijkstra"]))
                times_duan.append(float(row["time_duan"]))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_dijkstra, marker="o", label="Dijkstra", linewidth=2)
    plt.plot(sizes, times_duan, marker="s", label="Duan", linestyle="--", linewidth=2)

    plt.title("Algorithm Performance Comparison")
    plt.xlabel("Graph Size (N)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()

<<<<<<< HEAD
    print(f"Generating Graph (N={args.n}, Avg Degree={args.density})...")
    graph = generate_graph(args.n, args.density)
    source, target = 0, args.n - 1
=======
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
>>>>>>> d2d8bc5289061d53885780ae34acc68f45b3060e

    if show:
        plt.show()

<<<<<<< HEAD
    nx_path, nx_dist, nx_dur = run_networkx(graph, source, target)
    print(f"{'NetworkX (Ref)':<20} | {nx_dur:<10.4f} | {nx_dist:<10} | ✅ Verified")

    start = time.time()
    d_path, d_dist = dijkstra(graph, source, target)
    d_dur = (time.time() - start) * 1000

    d_status = "✅ Match" if d_dist == nx_dist else f"❌ Fail ({d_dist})"
    print(f"{'Dijkstra (Custom)':<20} | {d_dur:<10.4f} | {d_dist:<10} | {d_status}")

    start = time.time()
    duan_path, duan_dist = run_duan(graph, source, target)
    duan_dur = (time.time() - start) * 1000

    duan_status = "✅ Match" if duan_dist == nx_dist else f"❌ Fail ({duan_dist})"
    print(f"{'Duan et al. (2025)':<20} | {duan_dur:<10.4f} | {duan_dist:<10} | {duan_status}")
    print("-" * 60)

    if d_dist != duan_dist:
        print("Algorithms found different distances!")
        print(f"Dijkstra Path Len: {len(d_path)}")
        print(f"Duan Path Len:     {len(duan_path)}")
=======

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark and Visualization Interface")
    parser.add_argument("--mode", choices=["benchmark", "visualize"], required=True)
    parser.add_argument("--csv", type=str, help="Path to CSV file with results")
    parser.add_argument("--save", type=str, help="Output path for saving the plot")
    parser.add_argument("--show", action="store_true", help="Display the plot on screen")
    return parser

>>>>>>> d2d8bc5289061d53885780ae34acc68f45b3060e

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "benchmark":
        from benchmark import run_benchmark_suite
        run_benchmark_suite()

    elif args.mode == "visualize":
        if not args.csv:
            print("Error: --csv path is required in visualize mode.")
        else:
            plot_results_from_csv(
                csv_path=args.csv,
                save_path=args.save,
                show=args.show
            )
