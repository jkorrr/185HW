import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("exp/CartPole-v1_dqn_sd1_20260302_225851/log.csv")
# df = pd.read_csv("exp/LunarLander-v2_dqn_sd1_20260302_233143/log.csv")
df = pd.read_csv("exp/HalfCheetah-v4_sac_sd1_20260303_054821/log.csv")
eval_df = df[df["Eval_AverageReturn"].notna()]

plt.plot(eval_df["step"], eval_df["Eval_AverageReturn"], marker="o")
plt.xlabel("Environment steps")
plt.ylabel("Eval Average Return")
plt.title("SAC Half Cheetah")

# plt.savefig("lunar_eval_return.png", dpi=200)
plt.savefig("halfcheetahsac3.4.png", dpi = 200)
plt.show()