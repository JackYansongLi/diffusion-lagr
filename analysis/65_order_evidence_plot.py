# -*- coding: utf-8 -*-
"""65_order_evidence_plot.py --- C02 plot stage: evidence on data ordering

Panel (a): the scatter of the batch-mean series for three reads of the same data --
           our single-GPU read (256 consecutive trajectories per batch), the four-window
           read of a four-GPU run, and a randomly shuffled read that gives the pure-noise
           floor of the same statistic.
Panel (b): the lag-1 autocorrelation of the same three series, which is what makes the
           gradient noise correlated in time rather than white.

Both panels are quantities measured on the training data itself (see
RESULTS_ANALYSIS.md section 7.19); the figure exists to make the argument of the
discussion document checkable rather than asserted.

Style follows the project figure set: English labels, Agg backend, dpi 150,
bbox_inches="tight".

Input : results/order_evidence.npz        (from 64_order_evidence_data.py)
Output: results/figs/C02_fig_order_evidence.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs")
os.makedirs(FIGS, exist_ok=True)

z = np.load(os.path.join(RESULTS, "order_evidence.npz"))
labels = ["shuffled\n(noise floor)", "four-window read\n(4-GPU style)", "consecutive read\n(our training)"]
keys = ["wn", "ob", "wb"]
cols = ["0.65", "tab:orange", "tab:blue"]
x = np.arange(3)

std = [float(z["std_%s" % k]) for k in keys]
rho = [float(z["rho1_%s" % k]) for k in keys]
struct = [float(z.get("structural_%s" % k, np.nan)) for k in keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))

b1 = ax1.bar(x, std, color=cols, width=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("std of the batch-mean series")
ax1.set_title("Batch-to-batch fluctuation of the training stream", fontsize=9.5)
ax1.grid(alpha=0.2, axis="y", lw=0.4)
for xi, v in zip(x, std):
    ax1.text(xi, v * 1.04, "%.3e" % v, ha="center", fontsize=8)
ax1.set_ylim(0, max(std) * 1.22)
ax1.annotate("", xy=(2, std[2]), xytext=(2, std[1]),
             arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
ax1.text(2.08, 0.5 * (std[1] + std[2]), "%.2f$\\times$" % (std[2] / std[1]), fontsize=8.5, va="center")

b2 = ax2.bar(x, rho, color=cols, width=0.6)
ax2.axhline(0.0, color="0.4", lw=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel(r"lag-1 autocorrelation $\rho_1$ of the batch-mean series")
ax2.set_title("Temporal correlation of the gradient noise", fontsize=9.5)
ax2.grid(alpha=0.2, axis="y", lw=0.4)
for xi, v in zip(x, rho):
    ax2.text(xi, v + (0.03 if v >= 0 else -0.06), "%.3f" % v, ha="center", fontsize=8)
ax2.set_ylim(min(rho) - 0.12, max(rho) + 0.12)

fig.tight_layout()
out = os.path.join(FIGS, "C02_fig_order_evidence.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", out, flush=True)
print("std:", ["%.3e" % v for v in std], "rho1:", ["%.3f" % v for v in rho], flush=True)
