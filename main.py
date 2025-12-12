import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_results_from_csv(csv_path, save_path=None, show=False):
    """
    Читає CSV з колонками:
        size, time_dijkstra, time_duan
    Будує графік порівняння.
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f" Помилка читання CSV: {e}")
        return

    required = {"size", "time_dijkstra", "time_duan"}
    if not required.issubset(df.columns):
        print(" Помилка: CSV має містити колонки size, time_dijkstra, time_duan")
        return

    sizes = df["size"]
    times_dijkstra = df["time_dijkstra"]
    times_duan = df["time_duan"]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_dijkstra, marker='o', label='Dijkstra', linewidth=2)
    plt.plot(sizes, times_duan, marker='s', label='Duan et al.', linestyle='--', linewidth=2)

    plt.title("Порівняння алгоритмів Dijkstra та Duan")
    plt.xlabel("Розмір графа (N)")
    plt.ylabel("Час виконання (мс)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Графік збережено у {save_path}")

    if show:
        plt.show()

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark + Visualization для SSSP алгоритмів"
    )

    parser.add_argument(
        "--mode",
        choices=["benchmark", "visualize"],
        required=True,
        help="Режим роботи: benchmark або visualize"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="Шлях до CSV файлу з результатами (для visualize)"
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Куди зберегти графік (PNG)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Показати графік на екрані"
    )

    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # Режим 1: Benchmark
    if args.mode == "benchmark":
        print("Запуск benchmark...")
        from benchmark import run_benchmark_suite
        run_benchmark_suite()

    # Режим 2: Visualization
    elif args.mode == "visualize":
        if not args.csv:
            print("Помилка: потрібно вказати --csv шлях до файлу")
        else:
            print(f"Будуємо графік з {args.csv} ...")
            plot_results_from_csv(
                csv_path=args.csv,
                save_path=args.save,
                show=args.show
            )



