from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "reports" / "experiment_b" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 160,
})


def grouped_bars(filename, labels, series, title, ylabel="Score", rotate_labels=0):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, labels, rotation=rotate_labels, ha="right" if rotate_labels else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(4, len(series)))
    ymax = max(max(values) for values in series.values())
    ax.set_ylim(0, ymax * 1.24)
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


grouped_bars(
    "configuration_ablation.png",
    ["Zero-shot A", "B1", "B2", "B3"],
    {
        "mAP": [0.0116, 0.0223, 0.0661, 0.0551],
        "AP50": [0.0336, 0.0689, 0.1757, 0.1672],
        "AP75": [0.0049, 0.0089, 0.0279, 0.0139],
    },
    "Validation ablation across fine-tuning configurations",
)

grouped_bars(
    "b2_epoch_ablation.png",
    ["Epoch 1", "Epoch 2", "Epoch 3"],
    {
        "mAP": [0.0661, 0.0635, 0.0581],
        "AP50": [0.1757, 0.1824, 0.1571],
        "F1": [0.2158, 0.2310, 0.2133],
    },
    "B2 checkpoint-selection ablation",
)

grouped_bars(
    "scale_ablation.png",
    ["Native\n800/1333", "Global\n1024/1706", "Tiled\n384, 20%"],
    {
        "mAP": [0.0661, 0.0552, 0.1283],
        "AP50": [0.1757, 0.1450, 0.3640],
        "F1": [0.2158, 0.2275, 0.6152],
    },
    "Validation scale-adaptation ablation",
)

grouped_bars(
    "final_test_comparison.png",
    ["mAP", "AP50", "AP75", "Precision", "Recall", "F1", "Animal AP50", "Person AP50"],
    {
        "Experiment A": [0.0549, 0.1824, 0.0197, 0.3340, 0.3522, 0.3429, 0.3049, 0.0600],
        "Final B2 tiled": [0.1144, 0.3096, 0.0666, 0.3434, 0.5668, 0.4277, 0.5681, 0.0511],
    },
    "Final TestReal comparison",
    rotate_labels=20,
)

print(f"Wrote README figures to {OUT}")
