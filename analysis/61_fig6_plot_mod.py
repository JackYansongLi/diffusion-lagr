#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""61_fig6_plot_mod.py -- revision of Fig. 6: the main panel only (the inset is gone), with our curve drawn
on top of the paper.

Differences from the initial version (61_fig6_plot.py / A20_fig6_cmp.png, now frozen)
------------------------------------------------------------
1. **the upper right inset is gone** (the A/B sampling runs were dropped, so the A/B/C curves of the inset
   are not drawn)
2. **our curve sits above that of the paper**: the drawing order and zorder were changed (ours 3, paper 2)
   and our line is slightly thicker
   (the initial version drew ours first, so the dashed curve of the paper sat on top)
3. upper-right was added to the legend candidates (free once the inset is gone); the best is still chosen
by the number of points inside the box
4. the layout checks remain: occluded points (and their fraction) and points clipped on y; the inset check
is gone with the inset

The data and conventions are exactly those of the initial version: `fig6_dm1c_repro.npz` (loss_step/loss)
and `MOESM9: Figure6.out` of the paper,
log-log with x = Iterations and y = <Lsimple_n>; see RESULTS_ANALYSIS.md section 7.17.
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
MOESM9 = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data",
                      "42256_2024_810_MOESM9_ESM.txt")


def paper_block(path, name):
    rows, on = [], False
    for line in open(path, errors="replace"):
        s = line.strip()
        if not on:
            if s.startswith("# " + name + ".out"):
                on = True
            continue
        if not s or s.startswith("#"):
            break
        v = s.split()
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            break
    return np.array(rows, dtype=float)


def cover(ax, box, xs, ys):
    x0, y0, x1, y1 = box
    m = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(RESULTS, "fig6_dm1c_repro.npz"))
    ap.add_argument("--moesm9", default=MOESM9)
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "figs"))
    ap.add_argument("--out-suffix", default="_mod")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--log", default=os.path.join(RESULTS, "log_61_fig6_mod.txt"))
    ap.add_argument("--xmin", type=float, default=1.0)
    ap.add_argument("--xmax", type=float, default=4.0e5)
    ap.add_argument("--ymin", type=float, default=0.02)
    ap.add_argument("--ymax", type=float, default=1.2)
    ap.add_argument("--lw-paper", type=float, default=1.0)
    ap.add_argument("--lw-ours", type=float, default=1.6)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    z = np.load(args.npz)
    step, loss = z["loss_step"], z["loss"]
    p_loss = paper_block(args.moesm9, "Figure6")
    say("[61-mod] ours %d points (last step %.5g) | paper %d points (last %.5g)"
        % (len(step), step.max(), len(p_loss), p_loss[:, 0].max()))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_xscale("log"); ax.set_yscale("log")
    # the paper is drawn first (zorder 2) and ours after (zorder 3), so ours lies on top
    ax.plot(p_loss[:, 0][1:], p_loss[:, 1][1:], "--", color="0.15", lw=args.lw_paper, alpha=0.9,
            zorder=2, label="DM-1c, Li et al. (334.5k steps)")
    ax.plot(step[step > 0], loss[step > 0], "-", color="#1f77b4", lw=args.lw_ours,
            zorder=3, label="DM-1c, this work (266k steps)")
    ax.set_xlim(args.xmin, args.xmax); ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel("Iterations"); ax.set_ylabel(r"$\langle L_n^{\mathrm{simple}}\rangle$")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.set_title("Fig.6: DM-1c training loss (reproduction)", fontsize=10)

    xs = np.concatenate([step[step > 0], p_loss[:, 0][1:]])
    ys = np.concatenate([loss[step > 0], p_loss[:, 1][1:]])
    rects = {"lower-left": [0.03, 0.03, 0.34, 0.22], "lower-right": [0.63, 0.03, 0.34, 0.22],
             "upper-left": [0.03, 0.62, 0.34, 0.33], "upper-right": [0.63, 0.62, 0.34, 0.33]}
    lgx, lgy = np.log10(args.xmin), np.log10(args.ymin)
    lw_, lh_ = np.log10(args.xmax) - lgx, np.log10(args.ymax) - lgy
    best = None
    for name, r in rects.items():
        n = cover(ax, [10 ** (lgx + r[0] * lw_), 10 ** (lgy + r[1] * lh_),
                       10 ** (lgx + (r[0] + r[2]) * lw_), 10 ** (lgy + (r[1] + r[3]) * lh_)], xs, ys)
        say("[61-mod] legend candidate %-12s, points inside %d" % (name, n))
        if best is None or n < best[1]:
            best = (name, n)
    frac = 100.0 * best[1] / max(len(xs), 1)
    say("[61-mod] legend placed: %s (%d/%d = %.3f%%) %s"
        % (best[0], best[1], len(xs), frac, "OK" if frac < 1.0 else "WARNING, too much overlap"))
    ax.legend(loc=best[0].replace("-", " "), fontsize=8, framealpha=0.92)

    nclip = int(((loss[step > 0] < args.ymin) | (loss[step > 0] > args.ymax)).sum()) + \
        int(((p_loss[1:, 1] < args.ymin) | (p_loss[1:, 1] > args.ymax)).sum())
    # overlap check: the distance between our curve and that of the paper inside their shared x range
    m = (p_loss[:, 0] >= step[step > 0].min()) & (p_loss[:, 0] <= step.max())
    xi = p_loss[m, 0]
    yi_ours = np.interp(xi, step[step > 0], loss[step > 0])
    say("[61-mod] check: y clipped %d points OK | %d points in the shared x range, median loss difference ours minus paper %+.4g (relative %+.2f%%) | our zorder 3 > paper 2, so ours lies on top"
        % (nclip, int(m.sum()), float(np.median(yi_ours - p_loss[m, 1])),
           100 * float(np.median(yi_ours - p_loss[m, 1])) / float(np.median(p_loss[m, 1]))))

    out = os.path.join(args.out_dir, "A20_fig6_cmp%s.png" % args.out_suffix)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    say("[61-mod] → %s" % os.path.basename(out))
    log.close()


if __name__ == "__main__":
    main()
