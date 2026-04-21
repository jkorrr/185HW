import argparse
import csv
import os
import re
from collections import defaultdict
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


def parse_alpha(run_name: str) -> float | None:
    patterns = [
        r"alpha=([0-9]+(?:\.[0-9]+)?)",
        r"_a([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, run_name)
        if match:
            return float(match.group(1))
    return None


def task_label_from_run_name(run_name: str) -> str:
    lowered = run_name.lower()
    if "cube" in lowered:
        return "cube-single"
    if "antsoccer" in lowered:
        return "antsoccer"
    if "antmaze" in lowered:
        return "antmaze"
    return re.sub(r"[-_]?alpha=.*$", "", run_name)


def discover_runs(exp_group_dir: Path) -> list[dict]:
    runs: list[dict] = []
    if not exp_group_dir.exists():
        return runs

    for run_dir in sorted(p for p in exp_group_dir.iterdir() if p.is_dir()):
        eval_csv = run_dir / "eval.csv"
        if not eval_csv.exists():
            continue

        steps, success_rates = load_eval_csv(eval_csv)
        if not steps:
            continue

        runs.append(
            {
                "name": run_dir.name,
                "path": run_dir,
                "steps": steps,
                "success_rates": success_rates,
                "alpha": parse_alpha(run_dir.name),
                "task": task_label_from_run_name(run_dir.name),
                "final_success": success_rates[-1],
                "best_success": max(success_rates),
            }
        )
    return runs


def ensure_plot_dirs(base_plot_dir: Path) -> None:
    for question in ("q1", "q2", "q3"):
        (base_plot_dir / question).mkdir(parents=True, exist_ok=True)


def save_q2_best_agents_plot(runs: list[dict], output_dir: Path) -> Path | None:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_task[run["task"]].append(run)

    best_runs: list[dict] = []
    for task_runs in by_task.values():
        best_run = max(task_runs, key=lambda run: (run["best_success"], run["final_success"]))
        best_runs.append(best_run)

    if not best_runs:
        return None

    plt.figure(figsize=(8, 5))
    for run in sorted(best_runs, key=lambda item: item["task"]):
        label = f"{run['task']} ({run['name']})"
        plt.plot(run["steps"], run["success_rates"], marker="o", linewidth=2, label=label)

    plt.xlabel("Training steps")
    plt.ylabel("Success rate")
    plt.title("Q2: Best IQL Agents by Task")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "best_iql_tasks.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_q2_alpha_sweep_plot(runs: list[dict], output_dir: Path) -> Path | None:
    cube_runs = [run for run in runs if run["task"] == "cube-single" and run["alpha"] is not None]
    if not cube_runs:
        return None

    cube_runs.sort(key=lambda run: (run["alpha"], run["name"]))

    plt.figure(figsize=(8, 5))
    for run in cube_runs:
        alpha = int(run["alpha"]) if run["alpha"].is_integer() else run["alpha"]
        label = f"alpha={alpha}"
        plt.plot(run["steps"], run["success_rates"], marker="o", linewidth=2, label=label)

    plt.xlabel("Training steps")
    plt.ylabel("Success rate")
    plt.title("Q2: IQL Alpha Sweep on cube-single")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "cube_alpha_sweep.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path("exp"))
    parser.add_argument("--plot_dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    ensure_plot_dirs(args.plot_dir)

    q2_runs = discover_runs(args.exp_dir / "q2")
    q2_output_dir = args.plot_dir / "q2"

    best_agents_plot = save_q2_best_agents_plot(q2_runs, q2_output_dir)
    alpha_sweep_plot = save_q2_alpha_sweep_plot(q2_runs, q2_output_dir)
    if best_agents_plot is not None:
        print(best_agents_plot)
    if alpha_sweep_plot is not None:
        print(alpha_sweep_plot)

    if best_agents_plot is None and alpha_sweep_plot is None:
        print("No Q2 plots were generated. Check that exp/q2 contains eval.csv files.")


if __name__ == "__main__":
    main()
