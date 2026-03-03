# plot_2_6_lr.py
import csv
import math
import matplotlib.pyplot as plt

RUNS = [
    ("baseline", "exp/LunarLander-v2_dqn_sd1_20260302_233143/log.csv"),
    ("lr=1e-4",   "exp/LunarLander-v2_dqn-1e-4_sd1_20260303_021153/log.csv"),
    ("lr=3e-4",   "exp/LunarLander-v2_dqn-3e-4_sd1_20260303_021155/log.csv"),
    ("lr=3e-3",   "exp/LunarLander-v2_dqn-3e-3_sd1_20260303_021156/log.csv"),
]

def parse_float(x: str):
    """Return float if x is a valid finite number, else None."""
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

for label, path in RUNS:
    steps, rets = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s = parse_float(row.get("step", ""))
            r = parse_float(row.get("Eval_AverageReturn", ""))
            if s is None or r is None:
                continue
            steps.append(s)
            rets.append(r)

    plt.plot(steps, rets, label=label)

plt.xlabel("Environment steps")
plt.ylabel("Eval Average Return")
plt.title("LunarLander-v2 DQN: Learning Rate Sweep")
plt.legend()
plt.tight_layout()
plt.savefig("lunarlander_lr_sweep.png", dpi=200)
plt.show()