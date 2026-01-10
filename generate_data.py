import numpy as np
import pandas as pd

np.random.seed(42)

n = 1500  # exceeds Cultus requirement
time = np.arange(n)

target = 50 + 10 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 2, n)
feature_1 = target + np.random.normal(0, 1, n)
feature_2 = 20 + 5 * np.cos(2 * np.pi * time / 168)
feature_3 = np.random.uniform(0.3, 0.9, n)

df = pd.DataFrame({
    "target": target,
    "feature_1": feature_1,
    "feature_2": feature_2,
    "feature_3": feature_3
})

df.to_csv("data.csv", index=False)
print("Generated data.csv with", n, "rows")
