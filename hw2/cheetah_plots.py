import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = "exp"
X_COL = "Train_EnvstepsSoFar"
LOSS_COL = "Baseline Loss"
RET_COL = "Eval_AverageReturn"

def find_latest_run():
    runs = sorted(glob.glob(os.path.join(EXP_DIR, "HalfCheetah-v4_cheetah_baseline_sd1_*")))
    if not runs:
        raise FileNotFoundError(f"No runs found matching {EXP_DIR}/HalfCheetah-v4_cheetah_baseline_sd1_*")
    return runs[-1]

def load_log(run_dir):
    csv_path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    needed = {X_COL, LOSS_COL, RET_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns {missing}. Found: {list(df.columns)}")
    return df[[X_COL, LOSS_COL, RET_COL]].dropna(), csv_path

def plot_curve(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print("Saved:", outpath)

def main():
    run_dir = find_latest_run()
    df, csv_path = load_log(run_dir)
    print("Using:", run_dir)
    print("Log:", csv_path)

    x = df[X_COL].to_numpy()

    plot_curve(
        x,
        df[LOSS_COL].to_numpy(),
        title=f"HalfCheetah Baseline Loss vs Env Steps\n({os.path.basename(run_dir)})",
        xlabel="Environment Steps",
        ylabel="Baseline Loss",
        outpath="cheetah_baseline_loss.png",
    )

    plot_curve(
        x,
        df[RET_COL].to_numpy(),
        title=f"HalfCheetah Eval Return vs Env Steps\n({os.path.basename(run_dir)})",
        xlabel="Environment Steps",
        ylabel="Eval Average Return",
        outpath="cheetah_eval_return.png",
    )

if __name__ == "__main__":
    main()