#!/usr/bin/env python3
"""23_fig4c_mixed_mod.py -- reproduction of the fourth-order mixed-component flatness of Fig. 4c
(**revision**; the initial 23_fig4c_mixed.py is left untouched).

Changes against the initial version (2026-10-07):
  1. the upper and lower half-widths pass through `np.maximum(..., 0)`: the mixed flatness is a **nonlinear
     ratio**, so the pooled mean need not fall inside the batch min/max window
     (convention B leaves 65 of 1024 points outside it, and the initial version raised
     `ValueError: yerr must not contain negative values`),
     and on those points that half-width is set to 0 (no data are invented and the curve is unchanged).
  2. the number of such points is printed, which helps judge whether a convention is suitable.
  3. a new convention **G** (= the guess-1/G variant: global S2 denominator, extremes taken per component
     pair and then averaged over the pairs), `--err-def G`:
     its envelope is read from `24_fig4c_calibers_<ds>.npz`, which already holds the lo/hi of all seven
     conventions.
     Evidence (the analysis of 2026-10-07 plus measurements): A is 4-10x too narrow and B 3-9x too wide,
     while **G matches the width of the paper for the DNS (1.06x)** and gives
     0.34x for DM-3c, still the best of the conventions tried (A gives 0.10x); G is also exactly what the
     caption of Fig. 5 of the paper describes as
     "error bars computed solely over the three different velocity components".
  4. **the vertical range follows the paper, 8e-1..3e1** (the initial version used 1e0..3e1 and clipped the
     lower ends below 1: convention A reaches 0.974,
     B 0.895 and the band of the paper 0.949). `--ylim-lo/--ylim-hi` were added, defaulting to 0.8/30;
     convention B benefits from `--ylim-lo 0.7`. A check also reports whether any drawn element is clipped.

The docstring of the initial version follows.
----------------------
23_fig4c_mixed.py -- reproduction of the fourth-order mixed-component flatness of Fig. 4c (initial
version).

Fig. 4c of the paper plots the fourth-order **mixed-component** flatness on log-log axes,
    F(4,ij)_tau = <(dtauVi)^2 (dtauVj)^2> / [S(2)_tau]^2, averaged over ij = xy, xz, yz,
for the DNS and DM-3c only (as requested on 2026-10-06: two reproduction panels in total).
Each panel compares our reproduction with the original result of the paper (two-entry legend: this work /
paper).

Error bars (as in the caption of the paper: the dataset is cut into independent batches and, at each tau,
the min/max is taken):
  - convention B (the paper's, the default, validated on Figs. 4a/4b) = trajectory blocks x component pairs
    kept apart, i.e. 30 curves for the DNS and DM-3c
  - convention A (the earlier one, kept for reference) = 10 trajectory blocks with the three pairs averaged
    inside each block, i.e. 10 curves

Axes (measured from the vector layer of the Fig. 4c panel): x logarithmic with major ticks at 10^-1 and
10^2 plus margin, xlim(0.043, 200) as in Figs. 4a/4b;
y logarithmic, where the paper labels only 10^0 and 10^1 (the data span 0.95..17.4), so ylim(1e0, 3e1)
with yticks [1, 10].
Legend: F decreases with tau (17 -> 1), so the upper right is empty and the legend sits there, as in the
panel of the paper.

Data sources:
  ours : results/fig4c_mixed_<ds>.npz (written by 22_fig4c_mixed_batches.py)
  paper : the blocks of Data/Li_et_al_2024_source_data/42256_2024_810_MOESM7_ESM.txt
         `Figure4c_<DNS|DM-3c>.out`, columns = tau, mean, -, + (the error half-width)

Output: results/figs/A12_fig4c_<ds><suffix>.png
Usage: python3 23_fig4c_mixed_mod.py --dataset dns|dm3c [--err-def B|A|G] [--stats-tag _smoke] [--out-suffix _smoke]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data")
FIGS = os.path.join(RESULTS, "figs")
os.makedirs(FIGS, exist_ok=True)

LABEL = {"dns": "DNS", "dm3c": "DM-3c"}
BLOCK = {"dns": "DNS", "dm3c": "DM-3c"}
C_OURS, C_PAPER = "tab:blue", "0.35"
EVERY = 32


def parse_src(fn, blk):
    cur, rows = None, []
    for l in open(os.path.join(SRC, fn), errors="replace"):
        if l.startswith("#"):
            cur = l[1:].strip()
        elif l.strip() and cur and cur.split()[0] == blk:
            rows.append([float(v) for v in l.split()])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c"])
    ap.add_argument("--err-def", default="B", choices=["A", "B", "C", "G"],
                    help="bar convention: B = the paper's (30 component-pair batches); A = 10 trajectory blocks with the pairs averaged inside")
    ap.add_argument("--stats-tag", default="", help="read fig4c_mixed_<ds><tag>.npz (used for smoke tests)")
    ap.add_argument("--out-suffix", default="", help="output filename suffix")
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--ylim-lo", type=float, default=0.8, help="lower limit of the vertical axis (8e-1 in Fig. 4c of the paper)")
    ap.add_argument("--ylim-hi", type=float, default=30.0, help="upper limit of the vertical axis (3e1 in Fig. 4c of the paper)")
    args = ap.parse_args()
    ds = args.dataset
    sl = slice(None, None, args.every)

    z = np.load(os.path.join(RESULTS, "fig4c_mixed_%s%s.npz" % (ds, args.stats_tag)))
    tau = z["tau"]
    F = z["F_mean"]
    if args.err_def in ("G", "C"):
        # C = guess 1 (global S2 denominator, pairs averaged inside a block before the extremes, 10 curves)
        # G = global S2 denominator, extremes per pair, then averaged over pairs (30 curves)
        cz = np.load(os.path.join(RESULTS, "fig4c_calibers_%s.npz" % ds))
        pre = "G(" if args.err_def == "G" else "猜测1("
        k = [i for i, nm in enumerate(cz["names"]) if nm.startswith(pre)][0]
        lo, hi = cz["los"][k], cz["his"][k]
        P, n_err = None, (30 if args.err_def == "G" else 10)
    elif args.err_def == "B":
        P = z["Fb_pairs"].reshape(-1, len(tau))            # (10x3, nl): every batch and component pair
        lo, hi = z["B_lo"], z["B_hi"]
    else:
        P = z["Fb"]                                        # (10, nl): pairs already averaged inside the block
        lo, hi = z["A_lo"], z["A_hi"]
    n_err = P.shape[0] if P is not None else n_err

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    m = np.isfinite(F) & (F > 0)
    ln, = ax.plot(tau[m], F[m], ls="-", lw=1.4, color=C_OURS, zorder=3,
                  label="this work")
    yl = np.maximum(F[m] - lo[m], 0.0)
    yh = np.maximum(hi[m] - F[m], 0.0)
    n_clip = int(((F[m] - lo[m]) < 0).sum() + ((hi[m] - F[m]) < 0).sum())
    e = ax.errorbar(tau[m][sl], F[m][sl], yerr=[yl[sl], yh[sl]],
                    color=C_OURS, ls="none", marker="o", ms=3.6, mfc=C_OURS, mec=C_OURS,
                    capsize=2.5, capthick=1.0, elinewidth=1.0, zorder=4)

    r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4c_%s.out" % BLOCK[ds])
    n = min(len(tau), len(r))
    xr, mr, lo_r, hi_r = r[:n, 0], r[:n, 1], r[:n, 2], r[:n, 3]
    ln2, = ax.plot(xr, mr, ls="--", lw=1.3, color=C_PAPER, zorder=2, label="paper")
    ylo = np.maximum(mr - lo_r, mr * 0.02)
    ax.fill_between(xr, ylo, mr + hi_r, color=C_PAPER, alpha=0.25, lw=0, zorder=1)
    ax.plot(xr[sl], mr[sl], ls="none", marker="s", ms=3.2, mfc="none", mec=C_PAPER, zorder=2)
    h_paper = (ln2, Patch(facecolor=C_PAPER, alpha=0.35, edgecolor=C_PAPER, lw=0.8))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.043, 200)
    ax.set_ylim(args.ylim_lo, args.ylim_hi)
    ax.set_xticks([1e-1, 1e0, 1e1, 1e2])
    ax.set_yticks([1e0, 1e1])
    ax.set_xlabel("$\\tau/\\tau_\\eta$")
    ax.set_ylabel("$F_\\tau^{(4,ij)}$")
    ax.grid(alpha=0.22, which="both", lw=0.4)
    ax.set_title("Fig.4c reproduction: mixed flatness (%s)" % LABEL[ds], fontsize=10)
    # the curves decrease with tau, so the upper right is empty and the legend goes there, as in the paper
    ax.legend([(ln, e), h_paper], ["this work", "paper"], loc="upper right",
              fontsize=9, framealpha=0.9, borderpad=0.6)

    out = os.path.join(FIGS, "A12_fig4c_%s%s.png" % (ds, args.out_suffix))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved %s | bar convention %s (%d curves) | data %s | stride every=%d"
          % (out, args.err_def, n_err, os.path.basename(z.filename) if hasattr(z, "filename") else "npz",
             args.every), flush=True)

    # clipping check: do all drawn elements (both ends of our bars and of the paper band) fall inside the
    # vertical range
    lo_min, hi_max = float(min(lo[m].min(), ylo.min())), float(max(hi[m].max(), (mr + hi_r).max()))
    clip_us = int((lo[m] < args.ylim_lo).sum()) + int((hi[m] > args.ylim_hi).sum())
    clip_pa = int((ylo < args.ylim_lo).sum()) + int(((mr + hi_r) > args.ylim_hi).sum())
    print("  y axis %.2g..%.2g | lowest %.4f, highest %.4f => %s (ours outside %d, paper outside %d)"
          % (args.ylim_lo, args.ylim_hi, lo_min, hi_max,
             "fully visible" if (clip_us + clip_pa) == 0 else "**an element is clipped**", clip_us, clip_pa), flush=True)
    ratio = F[:n] / mr
    w_us = np.median((hi[:n] - lo[:n]) / F[:n])
    w_pa = np.median((hi_r + lo_r) / mr)
    print("  ours/paper ratio, median %.4f (%.3f..%.3f) | relative full width ours %.4f vs paper %.4f (ours/paper %.2f)"
          " | points outside (half-width set to 0) %d" % (np.median(ratio), ratio.min(), ratio.max(), w_us, w_pa, w_us / w_pa, n_clip), flush=True)
    for k in (0, 99, 499, 1023):
        print("    tau=%7.1f ours %8.4f [%.4f, %.4f] | paper %8.4f [%.4f, %.4f]"
              % (tau[k], F[k], lo[k], hi[k], mr[k], mr[k] - lo_r[k], mr[k] + hi_r[k]), flush=True)


if __name__ == "__main__":
    main()
