# -*- coding: utf-8 -*-
"""63_deficit_profile_plot.py --- C01 plot stage: where the retrained model loses flatness

Panel (a): F4(tau) on log-log axes for the DNS, the published DM-1c and our retrained
DM-1c, each with its 10-batch envelope (the scatter that the published error bars expose).
Panel (b): the same data as ratios to the published model, showing that the deficit of the
retrained model is confined to the smallest lags and closes monotonically with lag, while
the DNS-to-published gap of the model family is scale-independent in the inertial range.

Style follows the project figure set: English labels, Agg backend, dpi 150,
bbox_inches="tight", no smoothing, markers sparsified for readability.

Input : results/deficit_profile.npz      (from 62_deficit_profile_data.py)
Output: results/figs/C01_fig_deficit_profile.png
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

z = np.load(os.path.join(RESULTS, "deficit_profile.npz"))
tau = z["tau"]
STYLE = [("dns", "DNS (327,680 traj.)", "tab:green", "o"),
         ("official", "published DM-1c (358,400)", "0.25", "s"),
         ("ours", "retrained DM-1c (358,400)", "tab:blue", "^")]
EVERY = 48

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

for tag, lab, col, mk in STYLE:
    y, lo, hi = z["F4_%s" % tag], z["lo_%s" % tag], z["hi_%s" % tag]
    ax1.fill_between(tau, lo, hi, color=col, alpha=0.18, lw=0)
    ax1.plot(tau, y, "-", color=col, lw=1.3, label=lab)
    ax1.plot(tau[::EVERY], y[::EVERY], mk, color=col, ms=3.6, mfc="none", mew=1.0)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(0.09, 200)
ax1.set_ylim(1.0, 200.0)
ax1.set_xlabel(r"$\tau/\tau_\eta$")
ax1.set_ylabel(r"$F^{(4)}_\tau$")
ax1.set_title(r"Generalized flatness $F^{(4)}_\tau$: DNS, published DM-1c, retrained DM-1c", fontsize=9.5)
ax1.grid(alpha=0.22, which="both", lw=0.4)
ax1.legend(fontsize=8, loc="lower left", framealpha=0.9)

ax2.axhline(1.0, color="0.5", lw=1.0, ls=":")
for num, lab, col, mk in [("ours", "retrained / published", "tab:blue", "^"),
                          ("dns", "DNS / published", "tab:green", "o")]:
    r = z["ratio_%s" % num]
    ax2.plot(tau, r, "-", color=col, lw=1.4, label=lab)
    ax2.plot(tau[::EVERY], r[::EVERY], mk, color=col, ms=3.8, mfc="none", mew=1.0)

ax2.set_xscale("log")
ax2.set_xlim(0.09, 200)
ax2.set_ylim(0.65, 1.45)
ax2.set_xlabel(r"$\tau/\tau_\eta$")
ax2.set_ylabel(r"ratio of $F^{(4)}_\tau$")
ax2.set_title("Same data as ratios: the deficit closes with lag", fontsize=9.5)
ax2.grid(alpha=0.22, which="both", lw=0.4)
ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax2.annotate("", xy=(1.05, z["ratio_ours"][0]), xytext=(1.05, 1.0),
             arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=0.9))
ax2.text(1.15, 0.5 * (1.0 + z["ratio_ours"][0]), "%.1f%% at $0.1\\tau_\\eta$" % (100 * (z["ratio_ours"][0] - 1)),
         color="tab:blue", fontsize=8, va="center")
ax2.text(1.15, z["ratio_ours"][19] - 0.05, "%.1f%% at $2\\tau_\\eta$" % (100 * (z["ratio_ours"][19] - 1)),
         color="tab:blue", fontsize=8, va="center")

fig.tight_layout()
out = os.path.join(FIGS, "C01_fig_deficit_profile_mod_pub.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", out, flush=True)
