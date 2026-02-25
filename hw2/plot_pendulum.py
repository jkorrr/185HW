import glob, os
import pandas as pd
import matplotlib.pyplot as plt

X = "Train_EnvstepsSoFar"
Y = "Eval_AverageReturn"

baseline_log = glob.glob("exp/InvertedPendulum-v4_pendulum_baseline_sd1_*/log.csv")[0]
tuned_log    = glob.glob("exp/InvertedPendulum-v4_pendulum_tune_b_sd1_*/log.csv")[0]

df_base = pd.read_csv(baseline_log)[[X, Y]].dropna()
df_tune = pd.read_csv(tuned_log)[[X, Y]].dropna()

plt.figure()
plt.plot(df_base[X], df_base[Y], label="Baseline (default)")
plt.plot(df_tune[X], df_tune[Y], label="Tuned (tune_b)")

plt.xlabel("Environment Steps")
plt.ylabel("Eval Average Return")
plt.title("InvertedPendulum-v4: Baseline vs Tuned")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pendulum_baseline_vs_tune_b.png", dpi=200)
print("Saved pendulum_baseline_vs_tune_b.png")