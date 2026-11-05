"""Render report-ready model comparison figures from the website's verified JSON.

Run only after: .\\.venv\\Scripts\\python.exe src/export_website_data.py

All values come from docs/data/results.json. No ML fitting, EOS calculations,
synthetic measurements, or statistical significance claims are made.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/data/results.json"
OUTPUT = ROOT / "results/figures"
COLORS = {"Ridge": "#087f74", "Random_Forest": "#628da5", "Extra_Trees": "#db8b4e"}
LABELS = {"Ridge": "Ridge", "Random_Forest": "Random Forest", "Extra_Trees": "Extra Trees"}
FEATURES = ["Original_5", "Enhanced_12", "Richer_28", "Richer_28_Morgan128"]
N_FEATURES = [5, 12, 28, 156]
TARGETS = {"m": ("Segment number m", "MAE (dimensionless)"),
           "sigma": ("Segment diameter sigma", "MAE (angstrom)"),
           "epsilon_k": ("Dispersion energy epsilon/k", "MAE (K)")}


def finish(fig, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", OUTPUT / f"{name}.png")


def format_axes(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)


def cv_plots(data: dict) -> None:
    all_rows = data["experiments"]["cv_models"]
    for target, (target_label, unit) in TARGETS.items():
        fig, ax = plt.subplots(figsize=(9, 5.1))
        for model in COLORS:
            xs, ys, std = [], [], []
            for label, n in zip(FEATURES, N_FEATURES):
                matches = [row for row in all_rows
                           if row["target"] == target and row["model"] == model
                           and row["features"] == label]
                if matches:
                    xs.append(n)
                    ys.append(matches[0]["cv_mae_mean"])
                    std.append(matches[0]["cv_mae_std"])
            if xs:
                ax.errorbar(xs, ys, yerr=std, color=COLORS[model], marker="o",
                            markersize=6, capsize=4, linewidth=2.3,
                            label=LABELS[model])
        ax.set_xscale("log")
        ax.set_xticks(N_FEATURES, [str(x) for x in N_FEATURES])
        ax.set_xlim(4, 190)
        ax.set_ylim(bottom=0)
        format_axes(ax, "Number of molecular features", unit)
        ax.set_title(f"Model evolution: {target_label}", loc="left", pad=18,
                     fontsize=15, fontweight="bold")
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.text(.125, .015, "Three-fold CV on the 155-molecule training pool. Error bars are fold SD, not confidence intervals.",
                 color="#657984", fontsize=8)
        finish(fig, f"cv_mae_vs_features_{target}")


def collision_plot(data: dict) -> None:
    rows = data["experiments"]["collisions"]
    assert [r["features"] for r in rows] == FEATURES, "Unexpected feature order"
    nums = [r["colliding_molecules"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.barh(range(4), nums, color=["#d78951", "#86b7a5", "#86b7a5", "#087f74"], height=.55)
    ax.set_yticks(range(4), ["5 basic", "12 enhanced", "28 rich", "156 incl. Morgan128"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(nums) * 1.2 + 1)
    for i, value in enumerate(nums):
        ax.text(value + max(nums) * .015, i, str(value), va="center", fontweight="bold", fontsize=11)
    format_axes(ax, "Training molecules sharing an identical feature vector", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=.2)
    ax.set_title("Feature collisions fall as molecular representation improves", loc="left",
                 fontsize=14, fontweight="bold", pad=18)
    fig.text(.125,.01,"Computed on the 155 training-pool molecules only; this is not prediction accuracy.",
             fontsize=8,color="#657984")
    finish(fig, "descriptor_collisions")


def downstream_plots(data: dict) -> None:
    for kind, field, title in [
        ("pressure", "macro_mean_molecule_ape_percent", "Saturation-pressure error"),
        ("density", "mean_density_ape_percent", "Saturated-liquid-density error"),
    ]:
        label = "development_vle" if kind == "pressure" else "development_density"
        rows = data["development"].get(label) or []
        names = [("Ridge_5", "Ridge · 5", "#087f74"),
                 ("Ridge_12", "Ridge · 12", "#628da5"),
                 ("ExtraTrees_156", "Extra Trees · 156", "#db8b4e")]
        values = []
        for key, name, color in names:
            matched = [r for r in rows if r["model"] == key]
            if not matched or not isinstance(matched[0].get(field), (int, float)):
                print(f"Skipped {kind}: missing development results")
                return
            values.append(matched[0][field])
        fig, ax = plt.subplots(figsize=(8.6, 4.5))
        bars = ax.bar([name for _, name, _ in names], values,
                      color=[color for _, _, color in names], width=.57)
        ax.set_ylim(0, max(values) * 1.24)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=6,
                     fontsize=10, fontweight="bold")
        format_axes(ax, "Model / molecular representation", "Mean absolute percentage error (%)")
        ax.set_title(f"Development-validation {title.lower()}", loc="left", pad=18,
                     fontsize=15, fontweight="bold")
        fig.text(.125,.01,"39 development-validation molecules, four temperatures each. Reference: published-parameter PC-SAFT.",
                 fontsize=8,color="#657984")
        finish(fig, f"development_{kind}_model_error")


def overfitting_plot(data: dict) -> None:
    rows = data["experiments"].get("overfitting", [])
    names = [("Ridge_5", "Ridge · 5"), ("Ridge_12", "Ridge · 12"),
             ("ExtraTrees_156", "Extra Trees · 156")]
    for target, (title, unit) in TARGETS.items():
        fitted = {r["model"]: r for r in rows if r["target"] == target}
        if set(fitted) != {name for name, _ in names}:
            print("Skipped overfitting plot for", target, ": incomplete data")
            continue
        x = np.arange(3)
        tr = [fitted[k]["train_mae"] for k, _ in names]
        val = [fitted[k]["validation_mae"] for k, _ in names]
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ax.bar(x-.19, tr, width=.37, color="#82bfb2", label="Development training (116)")
        ax.bar(x+.19, val, width=.37, color="#e6a16c", label="Development validation (39)")
        ax.set_xticks(x, [label for _, label in names])
        format_axes(ax, "", unit)
        ax.set_title(f"Training–validation gap: {title}", loc="left", pad=18,
                     fontsize=15, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        fig.text(.125,.01,"The final held-out 39 molecules are not included in this chart.",
                 fontsize=8,color="#657984")
        finish(fig, f"development_train_validation_{target}")


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing {INPUT}. Run src/export_website_data.py first.")
    data=json.loads(INPUT.read_text(encoding="utf-8"))
    if len(data["curves"]) != 156 or data["metadata"]["test_molecules"] != 39:
        raise ValueError("Unexpected held-out data size; do not render unverifiable figures")
    if not data.get("experiments") or len(data["experiments"].get("cv_models", [])) < 30:
        raise ValueError("Website JSON is missing verified ML experiment comparisons")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv_plots(data)
    collision_plot(data)
    downstream_plots(data)
    overfitting_plot(data)
    print("Finished. Figures use saved experimental results only.")


if __name__ == "__main__":
    main()
