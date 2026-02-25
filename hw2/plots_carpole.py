import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = "exp"
X_COL = "Train_EnvstepsSoFar"
Y_COL = "Eval_AverageReturn"

def load_run_csvs(run_dirs):
    runs = []
    for d in sorted(run_dirs):
        csv_path = os.path.join(d, "log.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if X_COL not in df.columns or Y_COL not in df.columns:
            raise ValueError(
                f"{csv_path} missing required columns. "
                f"Need '{X_COL}' and '{Y_COL}'. Found: {list(df.columns)}"
            )

        df = df[[X_COL, Y_COL]].dropna()
        name = os.path.basename(d)
        runs.append((name, df))
    return runs

def plot_runs(runs, title, outfile):
    if not runs:
        print(f"[warn] no runs to plot for: {title}")
        return

    plt.figure()
    for name, df in runs:
        plt.plot(df[X_COL].to_numpy(), df[Y_COL].to_numpy(), label=name)

    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Average Return")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")

def main():
    small_dirs = glob.glob(os.path.join(EXP_DIR, "CartPole-v0_cartpole*_sd*"))
    small_dirs = [d for d in small_dirs if "_lb_" not in os.path.basename(d)]

    large_dirs = glob.glob(os.path.join(EXP_DIR, "CartPole-v0_cartpole_lb*_sd*"))

    small_runs = load_run_csvs(small_dirs)
    large_runs = load_run_csvs(large_dirs)

    plot_runs(
        small_runs,
        "CartPole Small Batch (no lb): Eval Avg Return vs Env Steps",
        "cartpole_small_batch.png",
    )
    plot_runs(
        large_runs,
        "CartPole Large Batch (lb): Eval Avg Return vs Env Steps",
        "cartpole_large_batch.png",
    )

if __name__ == "__main__":
    main()