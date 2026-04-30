from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, pearsonr, spearmanr

# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def prepare_statistical_data(data_path: str | Path) -> pd.DataFrame:

    stat_df = pd.read_csv(data_path)

    # Clean city values.
    stat_df["city"] = stat_df["city"].astype(str).str.lower().str.strip()
    stat_df["city"] = stat_df["city"].replace(
        {
            "tehr@n": "tehran",
            "thr": "tehran",
            "thran": "tehran",
            "tehran ": "tehran",
            "nan": np.nan,
        }
    )

    # Convert time into datetime format and extract hour.
    stat_df["time"] = pd.to_datetime(stat_df["time"], errors="coerce")
    stat_df["hour"] = stat_df["time"].dt.hour

    # Make sure amount is numeric.
    stat_df["amount"] = pd.to_numeric(stat_df["amount"], errors="coerce")

    # Keep only rows needed for this hypothesis test.
    stat_df = stat_df.dropna(subset=["amount", "city", "hour"])

    return stat_df


# -----------------------------------------------------------------------------
# Descriptive statistics
# -----------------------------------------------------------------------------


def get_city_summary(stat_df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for transaction amount grouped by city."""
    city_summary = (
        stat_df.groupby("city")["amount"]
        .agg(count="count", mean="mean", median="median", std="std")
        .sort_values("mean", ascending=False)
    )
    return city_summary


def get_hour_summary(stat_df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for transaction amount grouped by hour."""
    hour_summary = (
        stat_df.groupby("hour")["amount"]
        .agg(count="count", mean="mean", median="median", std="std")
        .sort_index()
    )
    return hour_summary


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------


def run_anova_by_group(
    stat_df: pd.DataFrame,
    group_column: str,
    value_column: str = "amount",
) -> Tuple[float, float]:

    groups = [
        group[value_column].dropna()
        for _, group in stat_df.groupby(group_column)
        if len(group[value_column].dropna()) > 1
    ]

    if len(groups) < 2:
        raise ValueError(
            f"ANOVA requires at least two valid groups for {group_column}."
        )

    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def run_hour_amount_correlations(stat_df: pd.DataFrame) -> dict[str, float]:
    """
    Run Pearson and Spearman correlation checks between hour and amount.

    These are supporting checks for the time portion of the statistical analysis.
    """
    pearson_r, pearson_p = pearsonr(stat_df["hour"], stat_df["amount"])
    spearman_r, spearman_p = spearmanr(stat_df["hour"], stat_df["amount"])

    return {
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p_value": spearman_p,
    }


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def save_city_boxplot_iqr_filtered(
    stat_df: pd.DataFrame,
    figures_dir: str | Path,
    filename: str = "amount_by_city_boxplot_iqr_filtered.png",
) -> Path:

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    q1 = stat_df["amount"].quantile(0.25)
    q3 = stat_df["amount"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    visual_df = stat_df[
        (stat_df["amount"] >= lower_bound) & (stat_df["amount"] <= upper_bound)
    ]

    print("IQR-filtered city boxplot")
    print("Original rows:", len(stat_df))
    print("Rows used in visualization:", len(visual_df))
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=visual_df, x="city", y="amount")
    plt.title("Transaction Amount by City with Extreme Outliers Removed")
    plt.xlabel("City")
    plt.ylabel("Transaction Amount")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = figures_dir / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def save_city_boxplot_log_scale(
    stat_df: pd.DataFrame,
    figures_dir: str | Path,
    filename: str = "amount_by_city_boxplot_log_scale.png",
) -> Path:
    """Save a city boxplot using a log-scaled y-axis."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=stat_df, x="city", y="amount")
    plt.yscale("log")
    plt.title("Transaction Amount by City (Log Scale)")
    plt.xlabel("City")
    plt.ylabel("Transaction Amount (Log Scale)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = figures_dir / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def save_average_amount_by_hour(
    stat_df: pd.DataFrame,
    figures_dir: str | Path,
    filename: str = "average_amount_by_hour.png",
) -> Path:
    """Save a line plot of average transaction amount by hour."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    hourly_avg = stat_df.groupby("hour")["amount"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly_avg, x="hour", y="amount", marker="o")
    plt.title("Average Transaction Amount by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Transaction Amount")
    plt.xticks(range(0, 24))
    plt.tight_layout()

    output_path = figures_dir / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def save_median_amount_by_hour(
    stat_df: pd.DataFrame,
    figures_dir: str | Path,
    filename: str = "median_amount_by_hour.png",
) -> Path:
    """Save a line plot of median transaction amount by hour."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    hourly_median = stat_df.groupby("hour")["amount"].median().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly_median, x="hour", y="amount", marker="o")
    plt.title("Median Transaction Amount by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Median Transaction Amount")
    plt.xticks(range(0, 24))
    plt.tight_layout()

    output_path = figures_dir / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def save_city_hour_heatmap(
    stat_df: pd.DataFrame,
    figures_dir: str | Path,
    filename: str = "city_hour_amount_heatmap.png",
) -> Path:
    """Save a heatmap of average transaction amount by city and hour."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    city_hour_pivot = stat_df.pivot_table(
        values="amount",
        index="city",
        columns="hour",
        aggfunc="mean",
    )

    plt.figure(figsize=(14, 7))
    sns.heatmap(city_hour_pivot)
    plt.title("Average Transaction Amount by City and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("City")
    plt.tight_layout()

    output_path = figures_dir / filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def run_statistical_analysis(
    data_path: str | Path = "../data/trx-10k.csv",
    figures_dir: str | Path = "../figures",
) -> None:
    stat_df = prepare_statistical_data(data_path)

    print("Rows available for statistical testing:", len(stat_df))
    print("Cities included:")
    print(stat_df["city"].value_counts())
    print()

    print("City summary:")
    city_summary = get_city_summary(stat_df)
    print(city_summary)
    print()

    city_f_stat, city_p_value = run_anova_by_group(stat_df, "city")
    print("City ANOVA F-statistic:", city_f_stat)
    print("City ANOVA p-value:", city_p_value)
    print()

    print("Hour summary:")
    hour_summary = get_hour_summary(stat_df)
    print(hour_summary)
    print()

    hour_f_stat, hour_p_value = run_anova_by_group(stat_df, "hour")
    print("Hour ANOVA F-statistic:", hour_f_stat)
    print("Hour ANOVA p-value:", hour_p_value)
    print()

    correlations = run_hour_amount_correlations(stat_df)
    print("Supporting correlation checks:")
    print("Pearson r:", correlations["pearson_r"])
    print("Pearson p-value:", correlations["pearson_p_value"])
    print("Spearman rho:", correlations["spearman_rho"])
    print("Spearman p-value:", correlations["spearman_p_value"])
    print()

    print("Saving figures...")
    saved_figures = [
        save_city_boxplot_iqr_filtered(stat_df, figures_dir),
        save_city_boxplot_log_scale(stat_df, figures_dir),
        save_average_amount_by_hour(stat_df, figures_dir),
        save_median_amount_by_hour(stat_df, figures_dir),
        save_city_hour_heatmap(stat_df, figures_dir),
    ]

    for figure_path in saved_figures:
        print("Saved:", figure_path)


if __name__ == "__main__":
    # The default paths assume this file is stored in the src/ folder.
    run_statistical_analysis()
