import pandas as pd
import matplotlib.pyplot as plt

# fixed-temperature run (3.4)
df_fixed = pd.read_csv("exp/HalfCheetah-v4_sac_sd1_20260303_054821/log.csv")
eval_fixed = df_fixed[df_fixed["Eval_AverageReturn"].notna()]

# auto-tuned run (3.5)
df_auto = pd.read_csv("exp/HalfCheetah-v4_sac_autotune_sd1_20260303_061828/log.csv")
eval_auto = df_auto[df_auto["Eval_AverageReturn"].notna()]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) Eval return over training (both runs)
axes[0].plot(eval_fixed["step"], eval_fixed["Eval_AverageReturn"], label="Fixed temperature", marker="o")
axes[0].plot(eval_auto["step"], eval_auto["Eval_AverageReturn"], label="Auto-tuned temperature", marker="o")
axes[0].set_xlabel("Environment steps")
axes[0].set_ylabel("Eval Average Return")
axes[0].set_title("HalfCheetah SAC: Eval Return")
axes[0].legend()

# (2) Temperature (alpha) over training for auto-tuned run
# Use learned alpha if present; fall back to "temperature" column if that's what you logged
if "alpha" in df_auto.columns and df_auto["alpha"].notna().any():
    axes[1].plot(df_auto["step"], df_auto["alpha"], marker="o")
    axes[1].set_ylabel("Alpha (learned)")
else:
    axes[1].plot(df_auto["step"], df_auto["temperature"], marker="o")
    axes[1].set_ylabel("Temperature (alpha)")

axes[1].set_xlabel("Environment steps")
axes[1].set_title("Auto-tuned α over Training")

plt.tight_layout()
plt.savefig("halfcheetah_sac_fixed_vs_autotune.png", dpi=200)
plt.show()