import numpy as np
import pandas as pd

CSV_PATH = r"outputs/metrics.csv"

df = pd.read_csv("outputs/metrics.csv")
df["inlier_ratio"] = df["inliers_ransac"] / df["matches_total"].replace(0, np.nan)

print("Frames:", len(df))
print("Inlier ratio stats:\n", df["inlier_ratio"].describe())

bad = df[(df["matches_total"] < 150) | (df["inlier_ratio"] < 0.4)]
print("\nBad frames (low matches or low inlier ratio):")
print(bad[["frame","matches_total","inliers_ransac","inlier_ratio","epi_mean_desc","epi_mean_geom_in"]].head(30))

print("\nTop 10 epi_mean_desc spikes:")
print(df.sort_values("epi_mean_desc", ascending=False)[
    ["frame","matches_total","inliers_ransac","inlier_ratio","epi_med_desc","epi_mean_desc"]
].head(10))