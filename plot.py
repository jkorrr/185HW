import pandas as pd
import matplotlib.pyplot as plt

log_path = "exp/seed_42_2026xxxx/log.csv"  
df = pd.read_csv(log_path)

# Loss curve
plt.figure()
plt.plot(df["step"], df["train/loss"])
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("MSE Policy Training Loss")
plt.grid(True)
plt.savefig("loss_curve.png")

# Reward curve
plt.figure()
plt.plot(df["step"], df["eval/mean_reward"])
plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.title("MSE Policy Evaluation Reward")
plt.grid(True)
plt.savefig("reward_curve.png")

plt.show()
