import math
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def safe_name(name: str) -> str:
    """Make filesystem-safe version of a string (letters, digits, underscore)."""
    return re.sub(r'[^A-Za-z0-9]+', '_', name.strip())

def plot_judge_vs_debater(
        task_name: str,
        judge_model: str,
        debater_model: str,
        means: List[float],
        errors: List[float],
        n_values: Tuple[int, int],
        p_value_str: str,
        out_dir: str = "./plots"
) -> str:
    """
    Plot Judge vs Debater results with journal-ready styling and save to file.
    Returns the path of the saved plot.
    """
    conditions = ["Judge Baseline", "Debater Baseline", "Before Training", "After SFT+DPO"]
    groups = ["Judge", "Debater", "Debater", "Debater"]

    df = pd.DataFrame({
        "Condition": conditions,
        "Mean": means,
        "Error": errors,
        "Group": groups
    })
    # ---------------- Style ----------------
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.1)
    debater_palette = ["#c6dbef", "#9ecae1", "#6baed6"]
    colors = ["#bdbdbd"] + debater_palette

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(7,5))
    bars = ax.bar(df["Condition"], df["Mean"], yerr=df["Error"],
                  capsize=5, color=colors, width=0.7,
                  error_kw=dict(lw=1.5))
    ax.axvline(0.5, color="lightgray", linestyle="--", lw=1)

    # ---------------- Annotations ----------------
    padding_above = 0.005
    padding_below = 0.01
    for bar, mean, err in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + err + padding_above,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9, color="black")
    ax.text(2, means[2] - errors[2] - padding_below, f"n = {n_values[0]}",
            ha="center", va="top", fontsize=9, color="black")
    ax.text(3, means[3] - errors[3] - padding_below, f"n = {n_values[1]}",
            ha="center", va="top", fontsize=9, color="black")

    p_value_lower = p_value_str.lower()
    if "n.s." in p_value_lower or ">" in p_value_lower:
        p_color = "gray"
    elif "decrease" in p_value_lower:
        p_color = "darkred"
    elif "increase" in p_value_lower:
        p_color = "darkgreen"
    elif "significant" in p_value_lower:
        p_color = "darkgreen" if means[3] > means[2] else "darkred"
    else:
        p_color = "black"
    ax.text(2.5, max(means)+0.045, p_value_str, ha="center",
            fontsize=10, fontweight="bold", color=p_color)

    # ---------------- Labels & Styling ----------------
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ymin = min(m - e for m, e in zip(means, errors))
    ymax = max(m + e for m, e in zip(means, errors))
    ymin = max(0.0, ymin - 0.02)
    ymax = ymax + 0.05
    ymin = math.floor(ymin / 0.05) * 0.05
    ymax = math.ceil(ymax / 0.05) * 0.05
    ax.set_ylim(ymin, ymax)

    ax.set_title(
        f"$\\bf{{{task_name}}}$\nJudge ({judge_model}) vs Debater ({debater_model})",
        pad=15, loc="center", ha="center"
    )
    ax.set_xticklabels(df["Condition"], rotation=0, ha="center", fontsize=9)

    sns.despine(ax=ax, top=True, right=True)
    ax.grid(False)
    plt.tight_layout()

    # ---------------- Save to File ----------------
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{safe_name(task_name)}__Judge-{safe_name(judge_model)}__Debater-{safe_name(debater_model)}.png"
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=500)
    plt.close(fig)   # close figure so it doesn’t display or hog memory

    return filepath
#
plot_judge_vs_debater(
    task_name="QuALITY",
    judge_model="GPT‑4.1 SFTed",
    debater_model="LLaMA-3-8B-262k",
    means=[0.634, 0.623, 0.761, 0.790],
    errors=[0.010, 0.010, (0.778-0.744)/2, (0.806-0.774)/2],
    n_values=(2502, 2502),
    p_value_str="p = .0153 (significant)"
)

plot_judge_vs_debater(
    task_name="QuALITY",
    judge_model="GPT-4.1 Nano SFTed",
    debater_model="GPT o4-mini",
    means=[0.575, 0.894, 0.565, 0.569],
    errors=[0.010, 0.006, (0.585-0.544)/2, (0.590-0.548)/2],
    n_values=(2200, 2200),
    p_value_str="p > .05 (n.s.)"
)

plot_judge_vs_debater(
    task_name="QuALITY",
    judge_model="GPT-4.1 Nano SFTed",
    debater_model="LLaMA-3-8B-262k",
    means=[0.575, 0.623, 0.571, 0.537],
    errors=[0.010, 0.010, (0.592-0.549)/2, (0.559-0.515)/2],
    n_values=(2000, 2005),
    p_value_str="p = .035 (decrease)"
)

plot_judge_vs_debater(
    task_name="Lojban",
    judge_model="GPT-4.1 Nano SFTed",
    debater_model="GPT o4-mini",
    means=[0.585, 0.824, 0.501, 0.490],
    errors=[0.012, 0.009, (0.525-0.477)/2, (0.514-0.465)/2],
    n_values=(1630, 1630),
    p_value_str="p > .05 (n.s.)"
)

plot_judge_vs_debater(
    task_name="Lojban",
    judge_model="GPT-4.1 Nano SFTed",
    debater_model="LLaMA-3-8B-262k",
    means=[0.585, 0.586, 0.525, 0.572],
    errors=[0.012, 0.012, (0.549-0.500)/2, (0.601-0.543)/2],
    n_values=(1630, 1131),
    p_value_str="p = .0143 (significant)"
)
