import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500
t = np.arange(n)

target = (
    0.03 * t +
    8 * np.sin(2 * np.pi * t / 24) +
    4 * np.sin(2 * np.pi * t / 168) +
    np.random.normal(0, 1, n)
)

feature_1 = target + np.random.normal(0, 0.3, n)
feature_2 = np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.2, n)
feature_3 = np.sin(2 * np.pi * t / 168) + np.random.normal(0, 0.2, n)
feature_4 = np.random.normal(0, 1, n)

df = pd.DataFrame({
    "target": target,
    "feature_1": feature_1,
    "feature_2": feature_2,
    "feature_3": feature_3,
    "feature_4": feature_4
})

df.to_csv("data.csv", index=False)
