from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs")
PRED_FILE = OUT_DIR / "test_predictions.csv"


def metric_row(group_name, sub):
    y = sub["age"].to_numpy()
    p = sub["prediction"].to_numpy()
    err = p - y

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "group": group_name,
        "N": len(sub),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Mean_Error": np.mean(err),
        "Median_Absolute_Error": np.median(np.abs(err)),
        "Within_5_years_pct": 100 * np.mean(np.abs(err) <= 5),
    }


def main():
    df = pd.read_csv(PRED_FILE)

    overall = pd.DataFrame([metric_row("Overall", df)])
    by_sex = pd.DataFrame(
        [metric_row(name, sub) for name, sub in df.groupby("sex")]
    )
    by_site = pd.DataFrame(
        [metric_row(name, sub) for name, sub in df.groupby("site")]
    )

    bins = [0, 29, 39, 49, 59, 69, 79, 120]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    df["age_group"] = pd.cut(
        df["age"], bins=bins, labels=labels, include_lowest=True
    )
    by_age = pd.DataFrame(
        [metric_row(str(name), sub) for name, sub in df.groupby("age_group", observed=True)]
    )

    overall.to_csv(OUT_DIR / "overall_metrics.csv", index=False)
    by_sex.to_csv(OUT_DIR / "fairness_by_sex.csv", index=False)
    by_site.to_csv(OUT_DIR / "fairness_by_site.csv", index=False)
    by_age.to_csv(OUT_DIR / "error_by_age_group.csv", index=False)

    # Absolute error by sex
    groups = []
    labels_for_plot = []
    for name, sub in df.groupby("sex"):
        groups.append(sub["absolute_error"].to_numpy())
        labels_for_plot.append(name)

    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, showfliers=False)
    plt.xticks(range(1, len(labels_for_plot) + 1), labels_for_plot)
    plt.ylabel("Absolute error (years)")
    plt.title("Prediction Error by Sex")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "error_by_sex.png", dpi=200)
    plt.close()

    # MAE by sex and site
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(by_sex["group"], by_sex["MAE"])
    axes[0].set_title("MAE by Sex")
    axes[0].set_ylabel("MAE (years)")

    axes[1].bar(by_site["group"], by_site["MAE"])
    axes[1].set_title("MAE by Site")
    axes[1].set_ylabel("MAE (years)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mae_by_group.png", dpi=200)
    plt.close()

    print("\nOVERALL")
    print(overall.to_string(index=False))

    print("\nBY SEX")
    print(by_sex.to_string(index=False))

    print("\nBY SITE")
    print(by_site.to_string(index=False))

    print("\nBY AGE GROUP")
    print(by_age.to_string(index=False))

    print("\nSaved fairness tables and plots.")


if __name__ == "__main__":
    main()
