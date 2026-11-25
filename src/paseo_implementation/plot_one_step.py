"""Plot 2: does one PASEO update help or hurt, with the fixed r vs the measured operator error?

y = gain over the same update with the measurement ignored (prior only), per image, averaged.
Bold: main stand-in prior (eta = 0.3, beta = 1), error bars = s.e. over images.
Thin: robustness settings (other eta / beta).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paseo_implementation.one_step_experiment import SETTINGS, gains
from paseo_implementation.plot_operator_noise import INK, MUTED, style

RESULTS = Path(__file__).resolve().parents[2] / "results"
COLOURS = {"fixed": "#eb6834", "measured": "#2a78d6"}        # reference palette slots 2 and 1
LABELS = {"fixed": "fixed $r$ = 0.02 (paper)", "measured": "measured error $r=\\sqrt{B^2+V}$"}
THIN_STYLES = ["--", ":", "-."]


def main():
    g = gains(pd.read_csv(RESULTS / "step6_one_step.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [("psnr_gain", "PSNR gain over prior-only update (dB)\n0 = ignoring the measurement",
               "(a) Fixed $r$ loses its benefit as $\\sigma$ grows; measured error never hurts"),
              ("err_reduction", "latent error reduction over prior-only update\n0 = ignoring the measurement",
               "(b) Same in latent space, $\\|z-z_{true}\\|/\\|z_{true}\\|$")]

    for ax, (column, ylabel, title) in zip(axes, panels):
        ax.axhline(0, color=INK, linewidth=1)
        for r_name, colour in COLOURS.items():
            for k, (eta, beta) in enumerate(SETTINGS):
                s = g[(g.r_name == r_name) & (g.eta == eta) & (g.beta == beta)].groupby("sigma")[column]
                mean, se = s.mean(), s.std() / s.count() ** 0.5
                if k == 0:
                    ax.errorbar(mean.index, mean, yerr=se, color=colour, marker="o", markersize=6,
                                linewidth=2.5, capsize=2, label=LABELS[r_name] + "  ($\\eta$=0.3, $\\beta$=1)")
                else:
                    ax.plot(mean.index, mean, color=colour, linewidth=1, alpha=0.8,
                            linestyle=THIN_STYLES[k - 1],
                            label=f"other prior: $\\eta$={eta}, $\\beta$={beta}" if r_name == "fixed" else None)
        style(ax, ylabel)
        ax.set_xlim(0.017, 0.5)
        ax.set_title(title, loc="left", fontsize=11, color=INK)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(LABELS["fixed"] + "  ($\\eta$=0.3, $\\beta$=1)"),
             labels.index(LABELS["measured"] + "  ($\\eta$=0.3, $\\beta$=1)")]
    order += [i for i in range(len(labels)) if i not in order]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3,
               fontsize=8, frameon=False, labelcolor=INK,
               title="thin lines: other stand-in priors (same line styles for both colours)", title_fontsize=8)
    fig.suptitle("One PASEO measurement update on real FFHQ latents (P=4, C=2, $\\rho$=0.15, $\\lambda$=0.005; "
                 "26 images; measured $r$ is in-sample)", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = RESULTS / "plot2_one_step.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
