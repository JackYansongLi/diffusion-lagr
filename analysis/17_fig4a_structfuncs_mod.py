"""17_fig4a_structfuncs_mod.py -- structure functions S_tau(p) of Fig. 4a (**revision**; the initial
17_fig4a_structfuncs.py is left untouched).

Changes against the initial version (requested 2026-10-05; the layout follows a measurement of the
vector layer of Fig. 4a in the paper):
  1) legend: p = 2, 4, 6 in **three vertical rows**, **this work in the left column and paper in the right**
     (two side-by-side legends, each with a column title).
  2) horizontal axis: about 0.35 decade of margin at each end, so xlim(0.043, 200) (measured in the paper:
     left edge ~4.3e-2, right edge ~2e2);
     the top of the vertical axis gets margin as well, ylim(1e-5, 4e4), so the last data points are not
     hidden by the frame.
  3) thinned points: the curve itself still joins all 1024 points, but **markers and bars are drawn every
     32 points** (about 32 per curve,
     against the ~35 markers per curve measured in the paper).
  4) the legend carries the **bar style** (the legend of the paper shows line, marker and bars together): a
     tuple of (curve, errorbar container) plus
     HandlerTuple(ndivide=1) draws a single combined entry.
  6) the **convention** of the bars is switchable (`--err-def`): A = the earlier one (10 trajectory blocks
     with the components merged); B = **the convention of the paper**
     (captions of Figs. 4 and 5: 30 component batches for the DNS and DM-3c with the components kept
     apart, 10 blocks for DM-1c). Recomputed with B,
     the bars agree with the paper to 1.0-1.1 times (DNS) and 0.93-0.98 times (DM-3c) => **B is the correct
     convention**.
  5) the two sources are told apart consistently across figures (decided 2026-10-05):
       this work = solid line, filled circle and **error bars** (capped: capsize 2.5 / capthick 1.0 /
       elinewidth 1.0)
       paper     = dashed line, open square and **error band** (translucent fill of the same colour, full
       resolution; its lower edge is clipped at 2% of the mean on the log axis)
     Reason: under the convention of the paper both sides have nearly the same error size, so bars alone are
     hard to tell apart, hence bars against a band.

Layout measured from Fig. 4a of the paper (2024 NMI 6:393-403, vector layer of the Fig. 4 page):
  - the legend holds only three vertical rows, DNS / DM-1c / DM-3c (lower left, middle right), and p = 2, 4, 6
    are **text labels inside the data area at the right end of the curves**;
  - about 35 markers per curve; the bars are even smaller than the markers (caption: "Error bars may
    appear smaller than the data points.");
  - the frame spans x in [0.043, 200] (three decades plus ~0.35 decade of margin on each side) and
    y in [1e-5, ~3e4].

Output: results/figs/A10_fig4a_<ds>_mod.png
Usage: python 17_fig4a_structfuncs_mod.py --dataset dns|dm1c|dm3c
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
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

LABEL = {"dns": "DNS", "dm1c": "DM-1c", "dm3c": "DM-3c"}
BLOCK = {"dns": "DNS", "dm1c": "DM-1c", "dm3c": "DM-3c"}
CP = {2: "k", 4: "tab:blue", 6: "tab:red"}
PS = (2, 4, 6)
EVERY = 32                      # marker/bar stride (1024/32 = 32 per curve, against ~35 in the paper)


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
    ap.add_argument("--dataset", required=True, choices=["dns", "dm1c", "dm3c"])
    ap.add_argument("--out-suffix", default="_mod")
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--err-def", default="A", choices=["A", "B"],
                    help="bar convention: A = the earlier one (10 trajectory blocks, components merged); B = the paper's (components kept apart, 30 component batches for the DNS and DM-3c)")
    args = ap.parse_args()
    ds = args.dataset
    ev = args.every
    sl = slice(None, None, ev)                       # thinned sampling

    z = np.load(os.path.join(RESULTS, "structfuncs_%s_full.npz" % ds))
    lags = z["lags"]
    tau = 0.1 * lags
    ours = {p: z["S%d" % p][:len(lags)] for p in PS}

    fb = os.path.join(RESULTS, "fig4a_batches_%s.npz" % ds)
    have_err = os.path.exists(fb)
    suf = "" if args.err_def == "A" else "_cB"
    if have_err:
        zb = np.load(fb)
        if suf and ("S2_lo%s" % suf) not in zb.files:
            print("[warn] %s lacks the %s convention key (for single-component data both coincide) => falling back to A"
                  % (os.path.basename(fb), args.err_def), flush=True)
            suf = ""
        err = {p: (zb["S%d_lo%s" % (p, suf)], zb["S%d_hi%s" % (p, suf)]) for p in PS}
    else:
        print("[warn] %s is missing => our bars are omitted" % fb, flush=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    h_ours, h_paper = [], []
    for p in PS:
        c = CP[p]
        y = ours[p]
        m = y > 0
        x = tau[m]
        yy = y[m]
        # the curve itself: all points joined (no markers, kept out of the legend; the errorbar entry carries it)
        ln, = ax.plot(x, yy, ls="-", lw=1.2, color=c, zorder=3)
        # thinned markers and bars: this work = filled circle with capped bars
        if have_err:
            lo, hi = err[p]
            e = ax.errorbar(x[sl], yy[sl], yerr=[yy[sl] - lo[m][sl], hi[m][sl] - yy[sl]],
                            color=c, ls="none", marker="o", ms=3.4, mfc=c, mec=c,
                            capsize=2.5, capthick=1.0, elinewidth=1.0, zorder=4)
        else:
            e = ax.plot(x[sl], yy[sl], ls="none", marker="o", ms=3.4, color=c, zorder=4)
        h_ours.append((ln, e))

        r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4a_%s_p%d.out" % (BLOCK[ds], p))
        n = min(len(tau), len(r))
        xr, mr, lo_r, hi_r = r[:n, 0], r[:n, 1], r[:n, 2], r[:n, 3]
        ln2, = ax.plot(xr, mr, ls="--", lw=1.1, color=c, alpha=0.9, zorder=2)
        # paper = band (translucent, full resolution) with open-square markers (thinned)
        # on a log axis the lower edge must be clipped: at small tau the half-width can exceed the mean itself
    # (DNS p6 reaches a relative full width of 2.79)
        ylo = np.maximum(mr - lo_r, mr * 0.02)
        yhi = mr + hi_r
        band = ax.fill_between(xr, ylo, yhi, color=c, alpha=0.22, lw=0, zorder=1)
        ax.plot(xr[sl], mr[sl], ls="none", marker="s", ms=3.0, mfc="none", mec=c,
                alpha=0.9, zorder=2)
        # legend handle: the PolyCollection of fill_between cannot be drawn in a HandlerTuple, so a translucent
    # Patch stands in for it
        h_paper.append((ln2, Patch(facecolor=c, alpha=0.35, edgecolor=c, lw=0.8)))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.043, 200)                              # about 0.35 decade of margin on each side (as measured in the paper)
    ax.set_ylim(1e-5, 4e4)                               # margin at the top so the last points are not hidden
    ax.set_yticks(10.0 ** np.arange(-5, 5, 3))
    ax.set_xlabel("$\\tau/\\tau_\\eta$")
    ax.set_ylabel("$S_\\tau^{(p)}$")
    ax.grid(alpha=0.22, which="both", lw=0.4)
    ax.set_title("Fig.4a reproduction: Lagrangian structure functions (%s)" % LABEL[ds], fontsize=10)

    # two side-by-side legends: this work on the left (p = 2/4/6 vertical), paper on the right
    kw = dict(loc="upper left", fontsize=8, framealpha=0.9, handlelength=2.4, borderpad=0.5,
              handler_map={tuple: HandlerTuple(ndivide=1)})   # the tuple handle draws one entry with line, marker and bars
    lg1 = ax.legend(h_ours, ["$p$=%d" % p for p in PS],
                    title="this work", bbox_to_anchor=(0.015, 0.985), **kw)
    ax.add_artist(lg1)
    ax.legend(h_paper, ["$p$=%d" % p for p in PS],
              title="paper", bbox_to_anchor=(0.255, 0.985), **kw)

    out = os.path.join(FIGS, "A10_fig4a_%s%s.png" % (ds, args.out_suffix))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out, "| stride every=%d => %d markers per curve" % (ev, int(np.ceil(m.sum() / ev))), flush=True)

    for p in PS:
        r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4a_%s_p%d.out" % (BLOCK[ds], p))
        n = min(len(tau), len(r))
        ratio = ours[p][:n] / r[:n, 1]
        msg = "  p=%d: ours/paper ratio, median %.4f (%.3f..%.3f)" % (p, np.median(ratio), ratio.min(), ratio.max())
        if have_err:
            lo, hi = err[p]
            msg += (" | relative bar width ours %.4f vs paper %.4f"
                    % (np.median((hi[:n] - lo[:n]) / ours[p][:n]),
                       np.median((r[:n, 3] + r[:n, 2]) / r[:n, 1])))
        print(msg, flush=True)


if __name__ == "__main__":
    main()
