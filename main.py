import argparse
import csv
import matplotlib.pyplot as plt


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

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark and Visualization Interface")
    parser.add_argument("--mode", choices=["benchmark", "visualize"], required=True)
    parser.add_argument("--csv", type=str, help="Path to CSV file with results")
    parser.add_argument("--save", type=str, help="Output path for saving the plot")
    parser.add_argument("--show", action="store_true", help="Display the plot on screen")
    return parser


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
