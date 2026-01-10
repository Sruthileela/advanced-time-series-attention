import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 1500
time = np.arange(n_rows)

# Generate seasonal + trend data
target = (
    0.05 * time +
    10 * np.sin(2 * np.pi * time / 24) +
    5 * np.sin(2 * np.pi * time / 168) +
    np.random.normal(0, 1, n_rows)
)

feature_1 = target + np.random.normal(0, 0.5, n_rows)
feature_2 = np.cos(2 * np.pi * time / 24) + np.random.normal(0, 0.2, n_rows)
feature_3 = np.random.normal(0, 1, n_rows)

df = pd.DataFrame({
    "target": target,
    "feature_1": feature_1,
    "feature_2": feature_2,
    "feature_3": feature_3
})

df.to_csv("data.csv", index=False)
print("data.csv generated with", len(df), "rows")
