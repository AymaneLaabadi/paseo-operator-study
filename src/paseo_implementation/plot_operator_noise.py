"""Plot 1: operator bias and Jacobian alignment under noise shifts (data from step 5).

(a) RMS bias of H_psi vs the noise-averaged target, and the noise floor, against sigma.
(b) Cosine between the learned and reference Jacobian-vector products.
Error bars: standard error across images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"
R, LAM = 0.02, 0.005
TRAIN_EDGE = 0.04            # top of the training range
EMBED_EDGE = 0.1             # sigma where the timestep t = sigma / 0.1 * 999 passes 999

# Reference categorical palette (slots 1-4), plus neutral inks.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

CONDITIONS = [  # (experiment, sigma_given / sigma, label, colour, marker, linestyle)
    ("A", 1.0, "Gaussian, true $\\sigma$ given", BLUE, "o", "-"),
    ("B", 0.5, "Gaussian, $0.5\\sigma$ given", ORANGE, "v", "-"),
    ("B", 2.0, "Gaussian, $2\\sigma$ given", AQUA, "^", "-"),
    ("C", 1.0, "Poisson–Gaussian, $\\sigma$ given", YELLOW, "s", "--"),
]


def select(df, experiment, factor):
    g = df[df.experiment == experiment]
    return g[(g.sigma_given / g.sigma).round(2) == factor]


def bias_rms(g):
    """sqrt(mean B^2) per sigma, with its standard error (delta method)."""
    s = g.groupby("sigma").bias2
    mean, se = s.mean(), s.std() / s.count() ** 0.5
    return mean ** 0.5, se / (2 * mean ** 0.5)


def mean_se(g, column):
    s = g.groupby("sigma")[column]
    return s.mean(), s.std() / s.count() ** 0.5


def style(ax, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.02, 0.04, 0.08, 0.16, 0.32], ["0.02", "0.04", "0.08", "0.16", "0.32"])
    ax.minorticks_off()
    ax.set_xlabel("measurement noise $\\sigma$ (code units, images in [−1, 1])", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    for x, text in [(TRAIN_EDGE, "training\nrange ends"), (EMBED_EDGE, "timestep\npasses 999")]:
        ax.axvline(x, color=GRAY, linewidth=1, linestyle=":")
        ax.text(x * 1.04, 0.98, text, transform=ax.get_xaxis_transform(), va="top",
                fontsize=8, color=MUTED)


def main():
    df = pd.read_csv(RESULTS / "step5_operator_noise.csv")
    fig, (ax_bias, ax_cos) = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) bias and noise floor
    for experiment, factor, label, colour, marker, ls in CONDITIONS:
        mean, se = bias_rms(select(df, experiment, factor))
        ax_bias.errorbar(mean.index, mean, yerr=se, color=colour, marker=marker, markersize=6,
                         linestyle=ls, linewidth=2, capsize=2, label=label)
    floor = select(df, "A", 1.0).groupby("sigma")["var"].mean() ** 0.5
    ax_bias.plot(floor.index, floor, color=GRAY, linewidth=2, linestyle="-.", label="noise floor (target std)")
    ax_bias.text(0.36, floor.iloc[-1], "noise floor", va="center", fontsize=9, color=INK)
    mean, _ = bias_rms(select(df, "A", 1.0))
    ax_bias.text(0.36, mean.iloc[-1], "bias", va="center", fontsize=9, color=INK)
    for y, text in [(R, "solver's $r$ = 0.02"), ((R**2 + LAM) ** 0.5, "effective $\\sqrt{r^2+\\lambda}$ = 0.073")]:
        ax_bias.axhline(y, color=GRAY, linewidth=1, linestyle="--")
        ax_bias.text(0.17, y, text, va="bottom", fontsize=8, color=MUTED)
    ax_bias.set_ylim(0, 0.7)
    style(ax_bias, "per-element RMS (latent units)")
    ax_bias.set_title("(a) Operator bias rises beyond $\\sigma\\approx0.1$, not at the training edge",
                      loc="left", fontsize=11, color=INK)

    # (b) Jacobian direction
    for experiment, factor, label, colour, marker, ls in CONDITIONS[:3]:
        g = select(df, experiment, factor)
        mean, se = mean_se(g, "cos")
        ax_cos.errorbar(mean.index, mean, yerr=se, color=colour, marker=marker, markersize=6,
                        linestyle=ls, linewidth=2, capsize=2, label=label)
    mean, se = mean_se(select(df, "A", 1.0), "cos_single")
    ax_cos.errorbar(mean.index, mean, yerr=se, color=BLUE, marker="o", markersize=6, markerfacecolor="white",
                    linestyle=":", linewidth=2, capsize=2, label="true $\\sigma$, single-draw reference")
    ax_cos.text(0.36, 0.68, "averaged\nreference", va="center", fontsize=9, color=INK)
    ax_cos.text(0.36, 0.50, "single-draw\nreference", va="center", fontsize=9, color=INK)
    ax_cos.set_ylim(0, 1)
    style(ax_cos, "cosine(learned JVP, reference JVP)")
    ax_cos.set_title("(b) Jacobian direction is only partly right, and flat in $\\sigma$",
                     loc="left", fontsize=11, color=INK)

    for ax in (ax_bias, ax_cos):
        ax.set_xlim(0.017, 0.62)
        ax.legend(loc="lower left" if ax is ax_cos else "upper left", bbox_to_anchor=(0, 0.0) if ax is ax_cos
                  else (0.0, 0.86), fontsize=8, frameon=False, labelcolor=INK)

    fig.suptitle("SILO's SR×8 operator $H_\\psi$ under noise shifts (26 FFHQ images, K = 16 draws, "
                 "Poisson–Gaussian has no Jacobian test)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    out = RESULTS / "plot1_operator_noise.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
