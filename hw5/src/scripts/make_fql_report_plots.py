import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("plots") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_csv(csv_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    success_rates: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(float(row["step"])))
            success_rates.append(float(row["eval/success_rate"]))
    return steps, success_rates


def ensure_plot_dirs(base_plot_dir: Path) -> None:
    for question in ("q1", "q2", "q3"):
        (base_plot_dir / question).mkdir(parents=True, exist_ok=True)


def save_q3_best_agents_plot(exp_dir: Path, output_dir: Path) -> Path | None:
    selected_runs = [
        ("q3-antsoccer-alpha=10", "antsoccer"),
        ("q3-cube-alpha=300", "cube-single"),
    ]

    plot_runs: list[tuple[str, list[int], list[float]]] = []
    for run_name, task_label in selected_runs:
        eval_csv = exp_dir / run_name / "eval.csv"
        if not eval_csv.exists():
            continue
        steps, success_rates = load_eval_csv(eval_csv)
        if not steps:
            continue
        plot_runs.append((f"{task_label} ({run_name})", steps, success_rates))

    if not plot_runs:
        return None

    plt.figure(figsize=(8, 5))
    for label, steps, success_rates in plot_runs:
        plt.plot(steps, success_rates, marker="o", linewidth=2, label=label)

    plt.xlabel("Training steps")
    plt.ylabel("Success rate")
    plt.title("Q3: Best FQL Agents by Task")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "best_fql_tasks.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path("exp"))
    parser.add_argument("--plot_dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    ensure_plot_dirs(args.plot_dir)

    plot_path = save_q3_best_agents_plot(args.exp_dir / "q3", args.plot_dir / "q3")
    if plot_path is not None:
        print(plot_path)
    else:
        print("No Q3 plot was generated. Check that the selected exp/q3 runs contain eval.csv files.")


if __name__ == "__main__":
    main()
