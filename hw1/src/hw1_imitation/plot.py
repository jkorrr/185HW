import pandas as pd
import matplotlib.pyplot as plt

MSE_POLICY_LOG_PATH = "exp/seed_42_20260205_153318/log.csv"
FLOW_MATCHING_POLICY_LOG_PATH = "exp/seed_42_20260205_155920/log.csv"

df = pd.read_csv(FLOW_MATCHING_POLICY_LOG_PATH)

loss_df = df.dropna(subset=["train/loss"])
reward_df = df.dropna(subset=["eval/mean_reward"])

plt.figure()
plt.plot(loss_df["step"], loss_df["train/loss"])
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Flow Matching Policy Training Loss")
plt.grid(True)
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(reward_df["step"], reward_df["eval/mean_reward"])
plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.title("Flow Matching Policy Evaluation Reward")
plt.grid(True)
plt.savefig("reward_curve.png")

plt.show()
