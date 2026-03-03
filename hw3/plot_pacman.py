import csv
import math
import matplotlib.pyplot as plt

LOG = "exp/MsPacman_dqn_sd1_20260303_013131/log.csv"  # change if needed

def parse_float(x):
    if x is None:
        return None
    x = x.strip()
    if x == "" or x.lower() == "nan":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return v if math.isfinite(v) else None

train_steps, train_returns = [], []
eval_steps, eval_returns = [], []

with open(LOG, newline="") as f:
    for row in csv.DictReader(f):
        s = parse_float(row.get("step", ""))

        tr = parse_float(row.get("Train_EpisodeReturn", ""))
        if s is not None and tr is not None:
            train_steps.append(s)
            train_returns.append(tr)

        er = parse_float(row.get("Eval_AverageReturn", ""))
        if s is not None and er is not None:
            eval_steps.append(s)
            eval_returns.append(er)

plt.plot(train_steps, train_returns, label="Train return")
plt.plot(eval_steps, eval_returns, label="Eval return")

plt.xlabel("Environment steps")
plt.ylabel("Return")
plt.title("MsPacman-v0 DQN: Train vs Eval Return")
plt.legend()
plt.tight_layout()
plt.savefig("mspacman_train_vs_eval.png", dpi=200)
plt.show()