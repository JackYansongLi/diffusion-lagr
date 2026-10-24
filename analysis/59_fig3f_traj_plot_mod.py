#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""59_fig3f_traj_plot_mod.py -- revision of Fig. 3f: four panels stacked vertically in the layout of the
paper (D -> C -> B -> A).

Differences from the initial version (see RESULTS_ANALYSIS.md section 7.16)
--------------------------------------------
- initial version: one panel with four overlaid curves told apart by colour and line style, plus a title
and a legend
- this revision: **the layout of the original figure** -- four panels stacked vertically, D, C, B and A
from the top down,
  **sharing the x axis** (only the bottom panel shows the ticks 0/50/100/150/200 and the label tau/tau_eta),
  each panel holding **one thin black solid line**
  without markers, legend or title; the y range is [-4, 4] with ticks -4/0/4, and the upper left of each
  panel carries its letter
  (D/C/B/A) while the upper right carries the formula label V_i^(n)/sigma(V_i^(n))
  (specifications taken from the text layer of Fig. 3f on page 4 of the paper: the letters sit at y =
  295/354/411/471, the formula at x ~ 519-543,
   the y ticks -4/0/4 repeat in every panel, the x ticks 0/50/100/150/200 come with the label tau/tau_eta,
   and every drawing in the region uses colour 0,0,0)
- the data and conventions are exactly those of the initial version: the npz written by 58_fig3f_traj_data.py
  (sigma = the standard deviation over all samples of that stage)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ROW_ORDER = ["D", "C", "B", "A"]          # the paper: D -> C -> B -> A from the top down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(RESULTS, "fig3f_traj_dm1c_repro.npz"))
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "figs"))
    ap.add_argument("--out-suffix", default="_mod")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--fig-w", type=float, default=4.1)
    ap.add_argument("--fig-h", type=float, default=5.4)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=200.0)
    ap.add_argument("--ymin", type=float, default=-4.0)
    ap.add_argument("--ymax", type=float, default=4.0)
    ap.add_argument("--lw", type=float, default=0.6)
    ap.add_argument("--log", default=os.path.join(RESULTS, "log_59_fig3f_mod.txt"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    z = np.load(args.npz)
    V = z["V"]
    labels = [str(x) for x in z["labels"]]
    t = z["t"]
    i = args.traj

    fig, axes = plt.subplots(4, 1, figsize=(args.fig_w, args.fig_h), sharex=True,
                             gridspec_kw=dict(hspace=0.10))
    n_clip_total = 0
    for ax, tag in zip(axes, ROW_ORDER):
        k = labels.index(tag)
        y = V[k, i]
        ax.plot(t, y, color="k", lw=args.lw, solid_joinstyle="round")
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_yticks([args.ymin, 0.0, args.ymax])
        ax.set_yticklabels(["-4", "0", "4"], fontsize=8)
        ax.tick_params(axis="y", length=2.5, pad=2)
        ax.grid(alpha=0.20, lw=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # the paper puts the letter on the left of each panel (vertically centred) and the formula in its upper right
        ax.text(0.015, 0.50, tag, transform=ax.transAxes, fontsize=10, va="center", ha="left")
        ax.text(0.985, 0.90, r"$V_i^{(n)}/\sigma(V_i^{(n)})$", transform=ax.transAxes,
                fontsize=8.5, va="top", ha="right")
        n_clip = int(np.sum((y < args.ymin) | (y > args.ymax)))
        n_clip_total += n_clip
        say("     %s: this trajectory mean %+.4f std %.4f range [%.2f, %.2f] | points clipped on y %d"
            % (tag, y.mean(), y.std(), y.min(), y.max(), n_clip))
    axes[-1].set_xticks([0, 50, 100, 150, 200])
    axes[-1].set_xlabel(r"$\tau/\tau_\eta$", fontsize=10.5)
    # tight_layout warns "not compatible" with sharex subplots and can misplace the margins, so they are
    # set explicitly
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.10, hspace=0.10)
    out = os.path.join(args.out_dir, "A19_fig3f_traj_repro%s.png" % args.out_suffix)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    say("[59-mod] traj=%d | 4x1 layout top-down %s | y in [%.0f,%.0f] x in [%.0f,%.0f] | total clipped %d %s | -> %s"
        % (i, " ".join(ROW_ORDER), args.ymin, args.ymax, args.xmin, args.xmax,
           n_clip_total, "OK" if n_clip_total == 0 else "WARNING, element out of frame", os.path.basename(out)))
    log.close()


if __name__ == "__main__":
    main()
