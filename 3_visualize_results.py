import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns

# =========================
# CONFIG
# =========================
INPUT_CSV = "Final_Predictions.csv"
OUTPUT_DIR = "final_visualizations"
TOP_K_WORST = 20
TOP_K_BEST = 20
DPI = 180

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

required_cols = [
    "True_X", "True_Y", "True_Z",
    "Pred_X", "Pred_Y", "Pred_Z"
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Residuals
df["Err_X"] = df["Pred_X"] - df["True_X"]
df["Err_Y"] = df["Pred_Y"] - df["True_Y"]
df["Err_Z"] = df["Pred_Z"] - df["True_Z"]

df["AbsErr_X"] = np.abs(df["Err_X"])
df["AbsErr_Y"] = np.abs(df["Err_Y"])
df["AbsErr_Z"] = np.abs(df["Err_Z"])

# Euclidean error recompute just in case
df["Euclidean_Error"] = np.sqrt(
    df["Err_X"]**2 + df["Err_Y"]**2 + df["Err_Z"]**2
)

# =========================
# METRICS
# =========================


def mae(x):
    return np.mean(np.abs(x))


def rmse(x):
    return np.sqrt(np.mean(x**2))


def bias(x):
    return np.mean(x)


def std(x):
    return np.std(x)


summary = {
    "Num_Samples": len(df),

    "Euclidean_Mean": df["Euclidean_Error"].mean(),
    "Euclidean_Median": df["Euclidean_Error"].median(),
    "Euclidean_Std": df["Euclidean_Error"].std(),
    "Euclidean_P90": np.percentile(df["Euclidean_Error"], 90),
    "Euclidean_P95": np.percentile(df["Euclidean_Error"], 95),
    "Euclidean_P99": np.percentile(df["Euclidean_Error"], 99),
    "Euclidean_Max": df["Euclidean_Error"].max(),

    "X_MAE": mae(df["Err_X"]),
    "X_RMSE": rmse(df["Err_X"]),
    "X_Bias": bias(df["Err_X"]),
    "X_STD": std(df["Err_X"]),

    "Y_MAE": mae(df["Err_Y"]),
    "Y_RMSE": rmse(df["Err_Y"]),
    "Y_Bias": bias(df["Err_Y"]),
    "Y_STD": std(df["Err_Y"]),

    "Z_MAE": mae(df["Err_Z"]),
    "Z_RMSE": rmse(df["Err_Z"]),
    "Z_Bias": bias(df["Err_Z"]),
    "Z_STD": std(df["Err_Z"]),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("Saved metrics summary.")

# =========================
# HELPER FUNCTIONS
# =========================


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close()


# =========================
# 1) EUCLIDEAN ERROR DISTRIBUTION
# =========================
plt.figure()
sns.histplot(df["Euclidean_Error"], bins=40, kde=True, color="royalblue")
plt.axvline(df["Euclidean_Error"].mean(),
            color="red", linestyle="--", label="Mean")
plt.axvline(df["Euclidean_Error"].median(),
            color="green", linestyle="--", label="Median")
plt.title("Euclidean Error Distribution")
plt.xlabel("Error (mm)")
plt.ylabel("Count")
plt.legend()
save_fig("01_euclidean_error_hist.png")

plt.figure()
sns.boxplot(x=df["Euclidean_Error"], color="orange")
plt.title("Euclidean Error Boxplot")
plt.xlabel("Error (mm)")
save_fig("02_euclidean_error_boxplot.png")

plt.figure()
sns.violinplot(x=df["Euclidean_Error"], color="skyblue")
plt.title("Euclidean Error Violin Plot")
plt.xlabel("Error (mm)")
save_fig("03_euclidean_error_violin.png")

# CDF
sorted_err = np.sort(df["Euclidean_Error"].values)
cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
plt.figure()
plt.plot(sorted_err, cdf, color="purple")
plt.title("CDF of Euclidean Error")
plt.xlabel("Error (mm)")
plt.ylabel("Cumulative Probability")
save_fig("04_euclidean_error_cdf.png")

# Percentile plot
percentiles = np.arange(0, 101)
perc_values = np.percentile(df["Euclidean_Error"], percentiles)
plt.figure()
plt.plot(percentiles, perc_values, color="darkcyan")
plt.title("Euclidean Error Percentile Curve")
plt.xlabel("Percentile")
plt.ylabel("Error (mm)")
save_fig("05_euclidean_error_percentiles.png")

# =========================
# 2) DIMENSION-WISE ANALYSIS
# =========================
dims = [
    ("X", "True_X", "Pred_X", "Err_X", "AbsErr_X"),
    ("Y", "True_Y", "Pred_Y", "Err_Y", "AbsErr_Y"),
    ("Z", "True_Z", "Pred_Z", "Err_Z", "AbsErr_Z"),
]

for dim_name, true_col, pred_col, err_col, abs_err_col in dims:
    # Signed error histogram
    plt.figure()
    sns.histplot(df[err_col], bins=40, kde=True, color="teal")
    plt.axvline(df[err_col].mean(), color="red", linestyle="--", label="Bias")
    plt.title(f"{dim_name} Signed Error Distribution")
    plt.xlabel("Signed Error (mm)")
    plt.ylabel("Count")
    plt.legend()
    save_fig(f"10_{dim_name}_signed_error_hist.png")

    # Absolute error histogram
    plt.figure()
    sns.histplot(df[abs_err_col], bins=40, kde=True, color="coral")
    plt.title(f"{dim_name} Absolute Error Distribution")
    plt.xlabel("Absolute Error (mm)")
    plt.ylabel("Count")
    save_fig(f"11_{dim_name}_abs_error_hist.png")

    # Boxplot
    plt.figure()
    sns.boxplot(x=df[err_col], color="lightgreen")
    plt.title(f"{dim_name} Signed Error Boxplot")
    plt.xlabel("Signed Error (mm)")
    save_fig(f"12_{dim_name}_signed_error_box.png")

    # Scatter True vs Pred
    plt.figure()
    plt.scatter(df[true_col], df[pred_col], alpha=0.6, s=20)
    mn = min(df[true_col].min(), df[pred_col].min())
    mx = max(df[true_col].max(), df[pred_col].max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
    plt.title(f"{dim_name}: True vs Predicted")
    plt.xlabel(f"True {dim_name} (mm)")
    plt.ylabel(f"Predicted {dim_name} (mm)")
    plt.legend()
    save_fig(f"13_{dim_name}_true_vs_pred.png")

    # Residual plot
    plt.figure()
    plt.scatter(df[true_col], df[err_col], alpha=0.6, s=20, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{dim_name}: Residual Plot")
    plt.xlabel(f"True {dim_name} (mm)")
    plt.ylabel(f"Residual = Pred - True (mm)")
    save_fig(f"14_{dim_name}_residual_plot.png")

    # Bland-Altman style
    mean_vals = (df[true_col] + df[pred_col]) / 2
    diffs = df[err_col]
    md = np.mean(diffs)
    sd = np.std(diffs)
    plt.figure()
    plt.scatter(mean_vals, diffs, alpha=0.6, s=20, color="brown")
    plt.axhline(md, color='blue', linestyle='--', label='Mean Diff')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--', label='+1.96 SD')
    plt.axhline(md - 1.96 * sd, color='green',
                linestyle='--', label='-1.96 SD')
    plt.title(f"{dim_name}: Bland-Altman Plot")
    plt.xlabel(f"Mean of True and Pred {dim_name} (mm)")
    plt.ylabel("Difference (Pred - True) (mm)")
    plt.legend()
    save_fig(f"15_{dim_name}_bland_altman.png")

# =========================
# 3) 3D VISUALIZATION
# =========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["True_X"], df["True_Y"], df["True_Z"],
           c="blue", s=18, alpha=0.5, label="True")
ax.scatter(df["Pred_X"], df["Pred_Y"], df["Pred_Z"],
           c="red", s=18, alpha=0.5, label="Pred")
ax.set_title("3D Scatter: True vs Predicted")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend()
save_fig("20_3d_true_vs_pred.png")

# 3D error vectors
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["True_X"], df["True_Y"], df["True_Z"], c="blue", s=10, alpha=0.4)

for i in range(len(df)):
    ax.plot(
        [df.loc[i, "True_X"], df.loc[i, "Pred_X"]],
        [df.loc[i, "True_Y"], df.loc[i, "Pred_Y"]],
        [df.loc[i, "True_Z"], df.loc[i, "Pred_Z"]],
        color="red", alpha=0.15
    )

ax.set_title("3D Error Vectors (True -> Pred)")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
save_fig("21_3d_error_vectors.png")

# 3D failure map
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(
    df["True_X"], df["True_Y"], df["True_Z"],
    c=df["Euclidean_Error"], cmap="turbo", s=24, alpha=0.85
)
fig.colorbar(p, ax=ax, shrink=0.7, label="Euclidean Error (mm)")
ax.set_title("3D Failure Map (Color = Error)")
ax.set_xlabel("True X (mm)")
ax.set_ylabel("True Y (mm)")
ax.set_zlabel("True Z (mm)")
save_fig("22_3d_failure_map.png")

# =========================
# 4) 2D PROJECTIONS
# =========================
projections = [
    ("X", "Y", "True_X", "True_Y", "Pred_X", "Pred_Y"),
    ("X", "Z", "True_X", "True_Z", "Pred_X", "Pred_Z"),
    ("Y", "Z", "True_Y", "True_Z", "Pred_Y", "Pred_Z"),
]

for a, b, ta, tb, pa, pb in projections:
    plt.figure(figsize=(8, 7))
    plt.scatter(df[ta], df[tb], c="blue", alpha=0.4, s=20, label="True")
    plt.scatter(df[pa], df[pb], c="red", alpha=0.4, s=20, label="Pred")
    plt.title(f"{a}{b} Projection: True vs Pred")
    plt.xlabel(f"{a} (mm)")
    plt.ylabel(f"{b} (mm)")
    plt.legend()
    save_fig(f"30_projection_{a}{b}.png")

# =========================
# 5) CORRELATION / HEATMAP
# =========================
corr_cols = [
    "True_X", "True_Y", "True_Z",
    "Pred_X", "Pred_Y", "Pred_Z",
    "Err_X", "Err_Y", "Err_Z",
    "Euclidean_Error"
]
corr = df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
save_fig("40_correlation_heatmap.png")

# =========================
# 6) ERROR VS TRUE / PRED
# =========================
for dim_name, true_col, pred_col, err_col, abs_err_col in dims:
    plt.figure()
    plt.scatter(df[true_col], df[abs_err_col],
                alpha=0.6, s=20, color="darkorange")
    plt.title(f"{dim_name}: Absolute Error vs True Value")
    plt.xlabel(f"True {dim_name} (mm)")
    plt.ylabel("Absolute Error (mm)")
    save_fig(f"50_{dim_name}_abs_error_vs_true.png")

    plt.figure()
    plt.scatter(df[pred_col], df[abs_err_col],
                alpha=0.6, s=20, color="darkgreen")
    plt.title(f"{dim_name}: Absolute Error vs Predicted Value")
    plt.xlabel(f"Predicted {dim_name} (mm)")
    plt.ylabel("Absolute Error (mm)")
    save_fig(f"51_{dim_name}_abs_error_vs_pred.png")

# =========================
# 7) TOP BEST / WORST SAMPLES
# =========================
df_sorted = df.sort_values("Euclidean_Error", ascending=False)
worst_df = df_sorted.head(TOP_K_WORST).copy()
best_df = df.sort_values(
    "Euclidean_Error", ascending=True).head(TOP_K_BEST).copy()

worst_df.to_csv(os.path.join(OUTPUT_DIR, "top_worst_samples.csv"), index=False)
best_df.to_csv(os.path.join(OUTPUT_DIR, "top_best_samples.csv"), index=False)

# Bar chart worst samples
plt.figure(figsize=(12, 6))
plt.bar(range(len(worst_df)), worst_df["Euclidean_Error"], color="crimson")
plt.title(f"Top {TOP_K_WORST} Worst Samples by Euclidean Error")
plt.xlabel("Rank")
plt.ylabel("Error (mm)")
save_fig("60_top_worst_samples_bar.png")

# Bar chart best samples
plt.figure(figsize=(12, 6))
plt.bar(range(len(best_df)), best_df["Euclidean_Error"], color="seagreen")
plt.title(f"Top {TOP_K_BEST} Best Samples by Euclidean Error")
plt.xlabel("Rank")
plt.ylabel("Error (mm)")
save_fig("61_top_best_samples_bar.png")

# Compare error components in worst cases
plt.figure(figsize=(13, 6))
x = np.arange(len(worst_df))
w = 0.25
plt.bar(x - w, worst_df["AbsErr_X"], width=w, label="AbsErr_X")
plt.bar(x,     worst_df["AbsErr_Y"], width=w, label="AbsErr_Y")
plt.bar(x + w, worst_df["AbsErr_Z"], width=w, label="AbsErr_Z")
plt.title("Worst Samples: Error Components")
plt.xlabel("Worst Sample Rank")
plt.ylabel("Absolute Error (mm)")
plt.legend()
save_fig("62_worst_samples_error_components.png")

# =========================
# 8) PAIRPLOT-LIKE SMALL MULTI-SCATTER
# =========================
sns.pairplot(
    df[["True_X", "True_Y", "True_Z", "Pred_X", "Pred_Y", "Pred_Z", "Euclidean_Error"]].sample(
        min(500, len(df)), random_state=42
    ),
    corner=True,
    diag_kind="hist"
)
plt.suptitle("Pairplot of True/Pred/Error (Sampled)", y=1.02)
plt.savefig(os.path.join(OUTPUT_DIR, "70_pairplot_sampled.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()

# =========================
# 9) TEXT REPORT
# =========================
report_lines = []
report_lines.append("FINAL VISUAL ANALYSIS REPORT")
report_lines.append("=" * 50)
report_lines.append(f"Number of samples: {len(df)}")
report_lines.append("")
report_lines.append("Euclidean Error:")
report_lines.append(f"  Mean   : {df['Euclidean_Error'].mean():.4f} mm")
report_lines.append(f"  Median : {df['Euclidean_Error'].median():.4f} mm")
report_lines.append(f"  Std    : {df['Euclidean_Error'].std():.4f} mm")
report_lines.append(
    f"  P90    : {np.percentile(df['Euclidean_Error'], 90):.4f} mm")
report_lines.append(
    f"  P95    : {np.percentile(df['Euclidean_Error'], 95):.4f} mm")
report_lines.append(
    f"  P99    : {np.percentile(df['Euclidean_Error'], 99):.4f} mm")
report_lines.append(f"  Max    : {df['Euclidean_Error'].max():.4f} mm")
report_lines.append("")

for dim_name, true_col, pred_col, err_col, abs_err_col in dims:
    report_lines.append(f"{dim_name} Dimension:")
    report_lines.append(f"  MAE  : {mae(df[err_col]):.4f} mm")
    report_lines.append(f"  RMSE : {rmse(df[err_col]):.4f} mm")
    report_lines.append(f"  Bias : {bias(df[err_col]):.4f} mm")
    report_lines.append(f"  Std  : {std(df[err_col]):.4f} mm")
    report_lines.append("")

with open(os.path.join(OUTPUT_DIR, "final_visual_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"All visualizations saved to: {OUTPUT_DIR}")
print("Done.")
