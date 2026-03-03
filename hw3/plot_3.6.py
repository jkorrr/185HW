# import pandas as pd
# import matplotlib.pyplot as plt

# # ----- EDIT THESE TWO PATHS -----
# singleq_path = "exp/Hopper-v4_sac_singleq_sd1_20260303_064127/log.csv"
# doubleq_path = "exp/Hopper-v4_sac_clipq_sd1_20260303_064213/log.csv"
# # --------------------------------

# df_single = pd.read_csv(singleq_path)
# df_double = pd.read_csv(doubleq_path)

# # eval rows only (Eval_AverageReturn only filled at eval time)
# eval_single = df_single[df_single["Eval_AverageReturn"].notna()]
# eval_double = df_double[df_double["Eval_AverageReturn"].notna()]

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # (1) Eval return
# axes[0].plot(eval_single["step"], eval_single["Eval_AverageReturn"], marker="o", label="Single-Q (mean backup)")
# axes[0].plot(eval_double["step"], eval_double["Eval_AverageReturn"], marker="o", label="Clipped Double-Q (min backup)")
# axes[0].set_xlabel("Environment steps")
# axes[0].set_ylabel("Eval Average Return")
# axes[0].set_title("Hopper-v4: Eval Return")
# axes[0].legend()

# # (2) Q values (logged during training; just plot raw series)
# axes[1].plot(df_single["step"], df_single["q_values"], label="Single-Q q_values")
# axes[1].plot(df_double["step"], df_double["q_values"], label="Double-Q q_values")
# axes[1].set_xlabel("Environment steps")
# axes[1].set_ylabel("Mean Q value (logged)")
# axes[1].set_title("Hopper-v4: Q Values")
# axes[1].legend()

# plt.tight_layout()
# plt.savefig("hopper_singleq_vs_doubleq_eval_and_qvalues.png", dpi=200)
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

singleq_path = "exp/Hopper-v4_sac_singleq_sd1_20260303_064127/log.csv"
doubleq_path = "exp/Hopper-v4_sac_clipq_sd1_20260303_064213/log.csv"

df_s = pd.read_csv(singleq_path)
df_d = pd.read_csv(doubleq_path)

print("Single-Q non-NaN q_values:", df_s["q_values"].notna().sum())
print("Double-Q non-NaN q_values:", df_d["q_values"].notna().sum())

eval_s = df_s[df_s["Eval_AverageReturn"].notna()]
eval_d = df_d[df_d["Eval_AverageReturn"].notna()]

q_s = df_s[df_s["q_values"].notna()]
q_d = df_d[df_d["q_values"].notna()]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Eval return
axes[0].plot(eval_s["step"], eval_s["Eval_AverageReturn"], marker="o", label="Single-Q")
axes[0].plot(eval_d["step"], eval_d["Eval_AverageReturn"], marker="o", label="Clipped Double-Q")
axes[0].set_xlabel("Environment steps")
axes[0].set_ylabel("Eval Average Return")
axes[0].set_title("Hopper-v4: Eval Return")
axes[0].legend()

# Q values (only if they exist)
if len(q_s) == 0 and len(q_d) == 0:
    axes[1].text(0.5, 0.5, "q_values are all NaN in log.csv\n(fix logging in run_sac.py)",
                 ha="center", va="center", transform=axes[1].transAxes)
else:
    if len(q_s) > 0:
        axes[1].plot(q_s["step"], q_s["q_values"], label="Single-Q q_values")
    if len(q_d) > 0:
        axes[1].plot(q_d["step"], q_d["q_values"], label="Double-Q q_values")
    axes[1].legend()

axes[1].set_xlabel("Environment steps")
axes[1].set_ylabel("Mean Q value (logged)")
axes[1].set_title("Hopper-v4: Q Values")

plt.tight_layout()
plt.savefig("hopper_single_vs_clipq_eval_and_qvalues.png", dpi=200)
plt.show()