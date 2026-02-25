import glob, os
import pandas as pd
import matplotlib.pyplot as plt

X = "Train_EnvstepsSoFar"
Y = "Eval_AverageReturn"

runs = sorted(glob.glob("exp/LunarLander-v2_lunar_lander_lambda*_sd1_*/log.csv"))

plt.figure()
for p in runs:
    df = pd.read_csv(p)
    name = os.path.basename(os.path.dirname(p)) 
    lam = name.split("lambda", 1)[1].split("_", 1)[0]
    df = df[[X, Y]].dropna()
    plt.plot(df[X].to_numpy(), df[Y].to_numpy(), label=f"λ={lam}")

plt.xlabel("Env steps")
plt.ylabel("Eval avg return")
plt.title("LunarLander-v2: GAE λ comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lunarlander_lambdas.png", dpi=200)
print("Saved lunarlander_lambdas.png")