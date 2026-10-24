#!/usr/bin/env python3
"""27_fig4c_width_vs_N.py -- width of the Fig. 4c bars against sample size (the concluding figure):
ordered cut against shuffled cut.

Answers whether the DNS at 1/10 of the samples widens the way DM-3c does:
  DM-3c (generated samples, no index-block correlation): 0.0116 on the full set -> 0.0449 on the 1/10
  slice, i.e. **3.87x** (about sqrt(10): pure sampling noise, following the sqrt(N) law)
  DNS (training data, with index-block correlation): 0.1555 -> 0.1692, i.e. **1.09x**, almost unchanged,
  so the width is set by the block structure and barely depends on N

Data sources:
  full-set widths: the convention-C rows of `fig4c_calibers_<ds>.npz` (global S2 denominator, average
  first, labelled in Chinese in the npz) and the "G(" rows (extremes first)
  slice widths: w_seq / w_rand in `fig4c_split_<ds>.npz` (D1-O1 = global denominator, average first, the
  same definition as convention C)
  paper width: both npz hold paper_width; the shuffled estimate for the full set is extrapolated from the
  slice as 1/sqrt(N) (open points with a dashed guide on the figure)

Output: results/figs/A14_fig4c_width_vs_N.png
Usage: python3 27_fig4c_width_vs_N.py
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
TAG = {"dm3c": "DM-3c", "dns": "DNS"}
COLORS = {"dm3c": "#1f77b4", "dns": "#d62728"}


def collect(ds):
    zc = np.load(os.path.join(RESULTS, "fig4c_calibers_%s.npz" % ds))
    nm = [str(x) for x in zc["names"]]
    kC = [i for i, s in enumerate(nm) if s.startswith("猜测1(")][0]
    F = zc["F"]
    w_full_seq = float(np.median((zc["his"][kC] - zc["los"][kC]) / F))
    zs = np.load(os.path.join(RESULTS, "fig4c_split_%s.npz" % ds))
    nms = [str(x) for x in zs["names"]]
    k = nms.index("D1-O1")
    return dict(w_full=w_full_seq, w_slice_seq=float(zs["w_seq"][k]),
                w_slice_rand=float(np.median(zs["w_rand"][:, k])),
                N_full=zs["M"] * 10, N_slice=int(zs["M"]),
                paper=float(zs["paper_width"]))


def main():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for ds in ("dm3c", "dns"):
        d = collect(ds)
        c = COLORS[ds]
        Ns = [d["N_slice"], d["N_full"]]
        # ordered cut
        ax.plot(Ns, [d["w_slice_seq"], d["w_full"]], "o-", color=c, lw=1.8, ms=7,
                label="%s: sequential split" % TAG[ds])
        # shuffled: measured on the slice and extrapolated to the full set as 1/sqrt(N)
        w_ex = d["w_slice_rand"] / np.sqrt(d["N_full"] / d["N_slice"])
        ax.plot([d["N_slice"]], [d["w_slice_rand"]], "s", color=c, ms=7, mfc="white", mew=1.8,
                label="%s: shuffled x40 (measured)" % TAG[ds])
        ax.plot(Ns, [d["w_slice_rand"], w_ex], "s--", color=c, ms=7, mfc="white", mew=1.8, lw=1.2, alpha=0.85,
                label="%s: shuffled (1/sqrt(N) extrapolation)" % TAG[ds])
        ax.axhline(d["paper"], color=c, ls=":", lw=1.3, alpha=0.75,
                   label="%s: paper (MOESM7)" % TAG[ds])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of trajectories N", fontsize=10.5)
    ax.set_ylabel("relative full width (hi - lo)/F", fontsize=10.5)
    ax.set_title("Fig.4c error-bar width vs sample size\n"
                 "DM-3c: width ~ 1/sqrt(N) (pure sampling noise) · DNS: width set by index-block structure",
                 fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(top=0.75)          # leave room at the top for the legend (the data reach 0.165, the paper DNS value)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9, columnspacing=1.2)
    out = os.path.join(FIGS, "A14_fig4c_width_vs_N.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved %s" % out)
    for ds in ("dm3c", "dns"):
        d = collect(ds)
        print("  [%s] full (ordered) %.4f | slice (ordered) %.4f (x%.2f) | slice (shuffled) %.4f | paper %.4f"
              % (TAG[ds], d["w_full"], d["w_slice_seq"], d["w_slice_seq"] / d["w_full"],
                 d["w_slice_rand"], d["paper"]))


if __name__ == "__main__":
    main()
