# -*- coding: utf-8 -*-
"""49_fig4a_cmp.py -- three-way comparison for Fig. 4a: the source curve of the paper, the released
DM-1c samples and our retrained DM-1c samples.

On top of 17_fig4a_structfuncs_mod.py (the script behind A10_fig4a_<ds>_mod.png) this adds **our
retrained DM-1c samples** as a third source, writes a new file and touches no existing figure.
Layout and statistics follow 17_mod item by item: p = 2/4/6 told apart by colour, markers and bars every
32 points, bars defaulting to convention B
(the caption of the paper: components kept apart), the same log-log view and legend style.

Visual encoding of the three sources (three legend columns):
  paper            : dashed line, open square, translucent band (full resolution)
  this work (released)  : solid line, filled circle, capped bars
  ours   (retrained)    : dash-dot line, open triangle, thin bars

Usage: python 49_fig4a_cmp.py [--official dm1c --ours dm1c_repro]
Output: results/figs/A10_fig4a_dm1c_cmp.png (override with --out-name)
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

BLOCK = {"dm1c": "DM-1c", "dm1c_repro": "DM-1c"}      # the same source block of the paper (both are DM-1c)
CP = {2: "k", 4: "tab:blue", 6: "tab:red"}
PS = (2, 4, 6)
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
    ap.add_argument("--official", default="dm1c")
    ap.add_argument("--ours", default="dm1c_repro")
    ap.add_argument("--err-def", default="B", choices=["A", "B"],
                    help="bar convention (default B = the caption of the paper; for single-component DM-1c A and B coincide)")
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--out-name", default="A10_fig4a_dm1c_cmp.png")
    args = ap.parse_args()
    sl = slice(None, None, args.every)
    suf = "" if args.err_def == "A" else "_cB"

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    h_off, h_ours, h_paper = [], [], []
    qtxt = {}
    for ds, kind in ((args.official, "official"), (args.ours, "ours")):
        z = np.load(os.path.join(RESULTS, "structfuncs_%s_full.npz" % ds))
        lags = z["lags"]
        tau = 0.1 * lags
        ours = {p: z["S%d" % p][:len(lags)] for p in PS}
        fb = os.path.join(RESULTS, "fig4a_batches_%s.npz" % ds)
        have_err = os.path.exists(fb)
        err = {}
        if have_err:
            zb = np.load(fb)
            s = suf
            if s and ("S2_lo%s" % s) not in zb.files:
                print("[warn] %s lacks the %s convention key => falling back to A" % (os.path.basename(fb), args.err_def))
                s = ""
            err = {p: (zb["S%d_lo%s" % (p, s)], zb["S%d_hi%s" % (p, s)]) for p in PS}
        for p in PS:
            c = CP[p]
            y = ours[p]
            m = y > 0
            x, yy = tau[m], y[m]
            if kind == "official":
                ln, = ax.plot(x, yy, ls="-", lw=1.2, color=c, zorder=3)
                if have_err:
                    lo, hi = err[p]
                    e = ax.errorbar(x[sl], yy[sl], yerr=[yy[sl] - lo[m][sl], hi[m][sl] - yy[sl]],
                                    color=c, ls="none", marker="o", ms=3.4, mfc=c, mec=c,
                                    capsize=2.5, capthick=1.0, elinewidth=1.0, zorder=4)
                else:
                    e = ax.plot(x[sl], yy[sl], ls="none", marker="o", ms=3.4, color=c, zorder=4)
                h_off.append((ln, e))
            else:
                ln, = ax.plot(x, yy, ls="-.", lw=1.3, color=c, zorder=5)
                if have_err:
                    lo, hi = err[p]
                    e = ax.errorbar(x[sl], yy[sl], yerr=[yy[sl] - lo[m][sl], hi[m][sl] - yy[m][sl]],
                                    color=c, ls="none", marker="^", ms=3.8, mfc="none", mec=c, mew=1.0,
                                    capsize=1.6, capthick=0.8, elinewidth=0.8, alpha=0.85, zorder=6)
                else:
                    e = ax.plot(x[sl], yy[sl], ls="none", marker="^", ms=3.8, mfc="none", mec=c, zorder=6)
                h_ours.append((ln, e))
        qtxt[ds] = {}
        for p in PS:
            r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4a_%s_p%d.out" % (BLOCK[ds], p))
            n = min(len(tau), len(r))
            ratio = ours[p][:n] / r[:n, 1]
            qtxt[ds][p] = (float(np.median(ratio)), float(ratio.min()), float(ratio.max()))
            note = ""
            if have_err:
                lo, hi = err[p]
                note = (" | relative bar width ours %.4f vs paper %.4f"
                        % (np.median((hi[:n] - lo[:n]) / ours[p][:n]),
                           np.median((r[:n, 3] + r[:n, 2]) / r[:n, 1])))
            print("  [%s] p=%d: ours/paper ratio, median %.4f (%.3f..%.3f)%s" % (ds, p, *qtxt[ds][p], note),
                  flush=True)

    # the source curve of the paper: drawn once per order
    for p in PS:
        c = CP[p]
        r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4a_%s_p%d.out" % (BLOCK[args.official], p))
        xr, mr, lo_r, hi_r = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        # z-order (2026-10-14 visual pass): the solid line below, the dashed and dash-dot above, so overlapping
    # sources remain identifiable through the gaps
        ln2, = ax.plot(xr, mr, ls="--", lw=1.1, color=c, alpha=0.9, zorder=4)
        ylo = np.maximum(mr - lo_r, mr * 0.02)
        ax.fill_between(xr, ylo, mr + hi_r, color=c, alpha=0.22, lw=0, zorder=1)
        ax.plot(xr[sl], mr[sl], ls="none", marker="s", ms=3.2, mfc="none", mec=c, alpha=0.95, zorder=4.2)
        h_paper.append((ln2, Patch(facecolor=c, alpha=0.35, edgecolor=c, lw=0.8)))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.043, 200)
    ax.set_ylim(1e-5, 4e4)
    ax.set_yticks(10.0 ** np.arange(-5, 5, 3))
    ax.set_xlabel("$\\tau/\\tau_\\eta$")
    ax.set_ylabel("$S_\\tau^{(p)}$")
    ax.grid(alpha=0.22, which="both", lw=0.4)
    ax.set_title("Fig.4a: structure functions — DM-1c paper samples (solid) vs our DM-1c "
                 "(dash-dot) vs paper source (dashed)", fontsize=9.5)

    kw = dict(loc="upper left", fontsize=8, framealpha=0.9, handlelength=2.4, borderpad=0.5,
              handler_map={tuple: HandlerTuple(ndivide=1)})
    lg1 = ax.legend(h_off, ["$p$=%d" % p for p in PS], title="paper samples", bbox_to_anchor=(0.005, 0.985), **kw)
    ax.add_artist(lg1)
    lg2 = ax.legend(h_ours, ["$p$=%d" % p for p in PS], title="ours", bbox_to_anchor=(0.215, 0.985), **kw)
    ax.add_artist(lg2)
    ax.legend(h_paper, ["$p$=%d" % p for p in PS], title="paper source", bbox_to_anchor=(0.40, 0.985), **kw)

    out = os.path.join(FIGS, args.out_name)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, "| bar convention %s | stride every=%d" % (args.err_def, args.every), flush=True)


if __name__ == "__main__":
    main()
