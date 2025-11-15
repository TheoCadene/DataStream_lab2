import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Regression CSVs
CAPY_REG_PATH = os.environ.get("CAPY_REG", "capymoa_regression.csv")
RIVER_REG_PATH = os.environ.get("RIVER_REG", "regression.csv")

# Multiclass CSVs
CAPY_MC_PATH = os.environ.get("CAPY_MC", "capymoa_multiclass_classification.csv")
RIVER_MC_PATH = os.environ.get("RIVER_MC", "multiclass_classification.csv")

# Binary CSVs
CAPY_BIN_PATH = os.environ.get("CAPY_BIN", "capymoa_binary_classification.csv")
RIVER_BIN_PATH = os.environ.get("RIVER_BIN", "binary_classification.csv")

# Anomaly detection CSVs
CAPY_AD_PATH = os.environ.get("CAPY_AD", "capymoa_anomaly_detection.csv")
RIVER_AD_PATH = os.environ.get("RIVER_AD", "anomaly_detection.csv")

# Output folders
OUT_REG = Path("./resultats/regression/")
OUT_MC  = Path("./resultats/multiclass/")
OUT_BIN = Path("./resultats/binary/")
OUT_AD  = Path("./resultats/anomaly/")
for p in (OUT_REG, OUT_MC, OUT_BIN, OUT_AD):
    p.mkdir(parents=True, exist_ok=True)


def _norm_dataset_col(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    return df

def _ensure_columns(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

def _barplot_and_save(pivot: pd.DataFrame, title: str, ylab: str, out_path: Path):
    cols = [c for c in ["CapyMOA", "River"] if c in pivot.columns]
    if cols:
        pivot = pivot[cols]

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylab)
    ax.legend(title="Library")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(out_path)


def _best_per_dataset(df: pd.DataFrame, metric: str, maximize: bool) -> pd.DataFrame:
    """Return (dataset, library, model, metric) best rows per dataset."""
    sub = df.copy()
    agg = sub.groupby(["dataset", "library", "model"], as_index=False)[metric].mean()
    agg["rank"] = agg.groupby("dataset")[metric].rank(ascending=not maximize, method="min")
    best = agg[agg["rank"] == 1].sort_values("dataset")
    return best[["dataset", "library", "model", metric]]


# =========================
# Regression
# =========================
def prepare_long_df_regression(capy_df: pd.DataFrame, river_df: pd.DataFrame) -> pd.DataFrame:
    c = _norm_dataset_col(capy_df.copy())
    r = _norm_dataset_col(river_df.copy())

    c["library"] = "CapyMOA"
    r["library"] = "River"

    common_cols = ["step", "track", "model", "dataset",
                   "MAE", "RMSE", "Memory in Mb", "Time in s"]

    for df in (c, r):
        _ensure_columns(df, common_cols)

    df_all = pd.concat([
        c[["library"] + common_cols],
        r[["library"] + common_cols]
    ], ignore_index=True)

    for col in ["MAE", "RMSE", "Memory in Mb", "Time in s"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    if "R2" in river_df.columns:
        r2 = river_df[["model", "dataset", "R2"]].copy()
        r2["dataset"] = r2["dataset"].astype(str).str.lower()
        df_all = df_all.merge(r2, on=["model", "dataset"], how="left")

    return df_all


def plot_metric_regression(df_all: pd.DataFrame, metric: str, dataset: str, out_dir: Path = OUT_REG):
    dataset_key = dataset.lower()
    sub = df_all[df_all["dataset"] == dataset_key].copy()
    if sub.empty:
        print(f" Aucun résultat (regression) pour le dataset {dataset}")
        return

    if metric not in sub.columns:
        print(f" La métrique {metric} n'existe pas (regression).")
        return

    agg = sub.groupby(["library", "model"], as_index=False)[metric].mean()
    pivot = agg.pivot(index="model", columns="library", values=metric)
    out_path = out_dir / f"{dataset_key}_{metric.replace(' ', '_')}.png"
    _barplot_and_save(pivot, f"{metric} — {dataset}", metric, out_path)


def run_all_regression():
    if not (Path(CAPY_REG_PATH).exists() and Path(RIVER_REG_PATH).exists()):
        print(" CSV regression manquants, partie regression sautée.")
        return
    capy = pd.read_csv(CAPY_REG_PATH)
    river = pd.read_csv(RIVER_REG_PATH)
    df_all = prepare_long_df_regression(capy, river)

    metrics = [m for m in ["RMSE", "MAE", "Time in s", "Memory in Mb", "R2"] if m in df_all.columns]
    datasets = sorted(df_all["dataset"].dropna().unique())

    # plots
    for ds in datasets:
        print(f"\n=== Regression | Dataset : {ds} ===")
        for m in metrics:
            plot_metric_regression(df_all, m, ds)

    # best-per-dataset CSVs
    for m in metrics:
        maximize = (m == "R2")
        best = _best_per_dataset(df_all[["dataset", "library", "model", m]].dropna(subset=[m]), m, maximize)
        out_csv = OUT_REG / f"best_by_dataset_{m.replace(' ', '_')}.csv"
        best.to_csv(out_csv, index=False)
        print(out_csv)


# =========================
# Multiclass
# =========================
def prepare_long_df_multiclass(capy_df: pd.DataFrame, river_df: pd.DataFrame) -> pd.DataFrame:
    c = _norm_dataset_col(capy_df.copy())
    r = _norm_dataset_col(river_df.copy())

    c["library"] = "CapyMOA"
    r["library"] = "River"

    common_cols = ["step", "track", "model", "dataset",
                   "Accuracy", "F1", "MicroF1", "MacroF1",
                   "Memory in Mb", "Time in s"]

    for df in (c, r):
        _ensure_columns(df, common_cols)

    df_all = pd.concat([
        c[["library"] + common_cols],
        r[["library"] + common_cols]
    ], ignore_index=True)

    for col in ["Accuracy", "F1", "MicroF1", "MacroF1", "Memory in Mb", "Time in s"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    return df_all


def plot_metric_multiclass(df_all: pd.DataFrame, metric: str, dataset: str, out_dir: Path = OUT_MC):
    aliases = {
        "acc": "Accuracy",
        "accuracy": "Accuracy",
        "f1": "F1" if "F1" in df_all.columns else ("MacroF1" if "MacroF1" in df_all.columns else "MicroF1"),
        "macro_f1": "MacroF1",
        "macrof1": "MacroF1",
        "micro_f1": "MicroF1",
        "microf1": "MicroF1",
        "time": "Time in s",
        "runtime": "Time in s",
        "memory": "Memory in Mb",
    }
    met = aliases.get(metric.lower(), metric)

    dataset_key = dataset.lower()
    sub = df_all[df_all["dataset"] == dataset_key].copy()
    if sub.empty:
        print(f"Aucun résultat (multiclass) pour le dataset {dataset}")
        return

    if met not in sub.columns:
        print(f" La métrique {metric} n'existe pas (multiclass).")
        return

    agg = sub.groupby(["library", "model"], as_index=False)[met].mean()
    pivot = agg.pivot(index="model", columns="library", values=met)
    out_path = out_dir / f"{dataset_key}_{met.replace(' ', '_')}.png"
    _barplot_and_save(pivot, f"{met} — {dataset}", met, out_path)


def run_all_multiclass():
    if not (Path(CAPY_MC_PATH).exists() and Path(RIVER_MC_PATH).exists()):
        print(" CSV multiclass manquants, partie multiclass sautée.")
        return

    capy = pd.read_csv(CAPY_MC_PATH)
    river = pd.read_csv(RIVER_MC_PATH)
    df_all = prepare_long_df_multiclass(capy, river)

    metrics = [m for m in ["Accuracy", "F1", "MacroF1", "MicroF1", "Time in s", "Memory in Mb"] if m in df_all.columns]
    datasets = sorted(df_all["dataset"].dropna().unique())

    for ds in datasets:
        print(f"\n=== Multiclass | Dataset : {ds} ===")
        for m in metrics:
            plot_metric_multiclass(df_all, m, ds)

    # best-per-dataset CSVs
    for m in metrics:
        maximize = m not in ["Time in s", "Memory in Mb"]
        best = _best_per_dataset(df_all[["dataset", "library", "model", m]].dropna(subset=[m]), m, maximize)
        out_csv = OUT_MC / f"best_by_dataset_{m.replace(' ', '_')}.csv"
        best.to_csv(out_csv, index=False)
        print(out_csv)


# =========================
# Binary
# =========================
def prepare_long_df_binary(capy_df: pd.DataFrame, river_df: pd.DataFrame) -> pd.DataFrame:
    c = _norm_dataset_col(capy_df.copy())
    r = _norm_dataset_col(river_df.copy())

    c["library"] = "CapyMOA"
    r["library"] = "River"

    common_cols = ["step", "track", "model", "dataset",
                   "Accuracy", "F1", "MacroF1", "MicroF1",
                   "Memory in Mb", "Time in s"]

    for df in (c, r):
        _ensure_columns(df, common_cols)

    df_all = pd.concat([
        c[["library"] + common_cols],
        r[["library"] + common_cols]
    ], ignore_index=True)

    for col in ["Accuracy", "F1", "MacroF1", "MicroF1", "Memory in Mb", "Time in s"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    return df_all


def plot_metric_binary(df_all: pd.DataFrame, metric: str, dataset: str, out_dir: Path = OUT_BIN):
    aliases = {
        "acc": "Accuracy",
        "accuracy": "Accuracy",
        "f1": "F1",
        "macro_f1": "MacroF1",
        "macrof1": "MacroF1",
        "micro_f1": "MicroF1",
        "microf1": "MicroF1",
        "time": "Time in s",
        "runtime": "Time in s",
        "memory": "Memory in Mb",
    }
    met = aliases.get(metric.lower(), metric)

    dataset_key = dataset.lower()
    sub = df_all[df_all["dataset"] == dataset_key].copy()
    if sub.empty:
        print(f" Aucun résultat (binary) pour le dataset {dataset}")
        return

    if met not in sub.columns:
        print(f" La métrique {metric} n'existe pas (binary).")
        return

    agg = sub.groupby(["library", "model"], as_index=False)[met].mean()
    pivot = agg.pivot(index="model", columns="library", values=met)
    out_path = out_dir / f"{dataset_key}_{met.replace(' ', '_')}.png"
    _barplot_and_save(pivot, f"{met} — {dataset}", met, out_path)


def run_all_binary():
    if not (Path(CAPY_BIN_PATH).exists() and Path(RIVER_BIN_PATH).exists()):
        print(" CSV binary manquants, partie binary sautée.")
        return

    capy = pd.read_csv(CAPY_BIN_PATH)
    river = pd.read_csv(RIVER_BIN_PATH)
    df_all = prepare_long_df_binary(capy, river)

    metrics = [m for m in ["Accuracy", "F1", "MacroF1", "MicroF1", "Time in s", "Memory in Mb"] if m in df_all.columns]
    datasets = sorted(df_all["dataset"].dropna().unique())

    for ds in datasets:
        print(f"\n=== Binary | Dataset : {ds} ===")
        for m in metrics:
            plot_metric_binary(df_all, m, ds)

    # best-per-dataset CSVs
    for m in metrics:
        maximize = m not in ["Time in s", "Memory in Mb"]
        best = _best_per_dataset(df_all[["dataset", "library", "model", m]].dropna(subset=[m]), m, maximize)
        out_csv = OUT_BIN / f"best_by_dataset_{m.replace(' ', '_')}.csv"
        best.to_csv(out_csv, index=False)
        print(out_csv)


# =========================
# Anomaly detection
# =========================
def prepare_long_df_anomaly(capy_df: pd.DataFrame, river_df: pd.DataFrame) -> pd.DataFrame:
    c = _norm_dataset_col(capy_df.copy())
    r = _norm_dataset_col(river_df.copy())

    c["library"] = "CapyMOA"
    r["library"] = "River"

    common_cols = ["step", "track", "model", "dataset",
                   "Accuracy", "F1", "MacroF1", "MicroF1",
                   "Memory in Mb", "Time in s"]

    for df in (c, r):
        _ensure_columns(df, common_cols)

    df_all = pd.concat([
        c[["library"] + common_cols],
        r[["library"] + common_cols]
    ], ignore_index=True)

    for col in ["Accuracy", "F1", "MacroF1", "MicroF1", "Memory in Mb", "Time in s"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    return df_all


def plot_metric_anomaly(df_all: pd.DataFrame, metric: str, dataset: str, out_dir: Path = OUT_AD):
    aliases = {
        "acc": "Accuracy",
        "accuracy": "Accuracy",
        "f1": "F1",
        "macro_f1": "MacroF1",
        "macrof1": "MacroF1",
        "micro_f1": "MicroF1",
        "microf1": "MicroF1",
        "time": "Time in s",
        "runtime": "Time in s",
        "memory": "Memory in Mb",
    }
    met = aliases.get(metric.lower(), metric)

    dataset_key = dataset.lower()
    sub = df_all[df_all["dataset"] == dataset_key].copy()
    if sub.empty:
        print(f"⚠ Aucun résultat (anomaly) pour le dataset {dataset}")
        return

    if met not in sub.columns:
        print(f"⚠ La métrique {metric} n'existe pas (anomaly).")
        return

    agg = sub.groupby(["library", "model"], as_index=False)[met].mean()
    pivot = agg.pivot(index="model", columns="library", values=met)
    out_path = out_dir / f"{dataset_key}_{met.replace(' ', '_')}.png"
    _barplot_and_save(pivot, f"{met} — {dataset}", met, out_path)


def run_all_anomaly():
    if not (Path(CAPY_AD_PATH).exists() and Path(RIVER_AD_PATH).exists()):
        print("⚠ CSV anomaly manquants, partie anomaly sautée.")
        return

    capy = pd.read_csv(CAPY_AD_PATH)
    river = pd.read_csv(RIVER_AD_PATH)
    df_all = prepare_long_df_anomaly(capy, river)

    metrics = [m for m in ["Accuracy", "F1", "MacroF1", "MicroF1", "Time in s", "Memory in Mb"] if m in df_all.columns]
    datasets = sorted(df_all["dataset"].dropna().unique())

    for ds in datasets:
        print(f"\n=== Anomaly | Dataset : {ds} ===")
        for m in metrics:
            plot_metric_anomaly(df_all, m, ds)

    for m in metrics:
        maximize = m not in ["Time in s", "Memory in Mb"]
        best = _best_per_dataset(df_all[["dataset", "library", "model", m]].dropna(subset=[m]), m, maximize)
        out_csv = OUT_AD / f"best_by_dataset_{m.replace(' ', '_')}.csv"
        best.to_csv(out_csv, index=False)
        print(out_csv)


# =========================
# Main
# =========================
def main(args: Optional[List[str]] = None):
    """
    Usage:
      python benchmark_both.py            # lance toutes les parties présentes
      python benchmark_both.py reg       # uniquement regression
      python benchmark_both.py mc        # uniquement multiclass
      python benchmark_both.py bin       # uniquement binary
      python benchmark_both.py ad        # uniquement anomaly detection

    Personnalisez les chemins via variables d'env :
      CAPY_REG, RIVER_REG, CAPY_MC, RIVER_MC, CAPY_BIN, RIVER_BIN, CAPY_AD, RIVER_AD
    """
    if args is None:
        args = sys.argv[1:]

    mode = args[0].lower() if args else "all"
    if mode in ("reg", "regression"):
        run_all_regression()
    elif mode in ("mc", "multiclass"):
        run_all_multiclass()
    elif mode in ("bin", "binary"):
        run_all_binary()
    elif mode in ("ad", "anomaly", "anomaly_detection"):
        run_all_anomaly()
    else:
        run_all_regression()
        run_all_multiclass()
        run_all_binary()
        run_all_anomaly()


if __name__ == "__main__":
    main()
