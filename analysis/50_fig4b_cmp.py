# -*- coding: utf-8 -*-
"""50_fig4b_cmp.py -- three-way comparison for Fig. 4b: the source curve of the paper, the released
DM-1c samples and our retrained DM-1c samples.

On top of 19_fig4b_flatness_mod.py (the script behind A11_fig4b_<ds>_mod.png) this adds the retrained
DM-1c result as a third source, writes a new file and touches no existing figure. Conventions and layout
follow 19_mod item by item:
p = 4/6/8 told apart by colour, markers every 32 points, bars defaulting to convention B (the caption of
the paper: components kept apart),
the log-log view xlim(0.043,200) / ylim(1e0,3e9), the legend in the free upper right corner, and two legend
columns becoming three.

Visual encoding of the three sources:
  paper source     : dashed line, open square, translucent band (full resolution)
  paper samples    : solid line, filled circle, capped bars
  ours (retrained) : dash-dot line, open triangle, thin bars

Usage: python 50_fig4b_cmp.py [--official dm1c --ours dm1c_repro]
Output: results/figs/A11_fig4b_dm1c_cmp.png
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

BLOCK = {"dm1c": "DM-1c", "dm1c_repro": "DM-1c"}
CP = {4: "k", 6: "tab:blue", 8: "tab:red"}
PS = (4, 6, 8)
IX = {2: 0, 4: 1, 6: 2, 8: 3}
EVERY = 32


def parse_src(fn, blk):
    cur, rows = None, []
    for l in open(os.path.join(SRC, fn), errors="replace"):
        if l.startswith("#"):
            cur = l[1:].strip()
        elif l.strip() and cur and cur.split()[0] == blk:
            rows.append([float(v) for v in l.split()])
    return np.array(rows)


def curves_for(ds, err_def, every):
    """→ tau, {p: F}, {p: (lo, hi)}, nb_eff"""
    z = np.load(os.path.join(RESULTS, "flatness_%s_full.npz" % ds))
    lags = z["lags"]
    tau = 0.1 * lags
    mean = z["mean"]                       # (nl, 4) = S2/S4/S6/S8
    Vbc = z["Sbc"]                         # (nb, nl, 4, C)
    nb, nl, _, C = Vbc.shape
    F, err = {}, {}
    for p in PS:
        F[p] = mean[:, IX[p]] / mean[:, IX[2]] ** (p / 2.0)
        if err_def == "A":
            S = Vbc.mean(axis=3)
            Fb = S[:, :, IX[p]] / S[:, :, IX[2]] ** (p / 2.0)
        else:
            S = Vbc.transpose(0, 3, 1, 2).reshape(nb * C, nl, 4)
            Fb = S[:, :, IX[p]] / S[:, :, IX[2]] ** (p / 2.0)
        err[p] = (Fb.min(axis=0), Fb.max(axis=0))
    return tau, F, err, (nb if err_def == "A" else nb * C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default="dm1c")
    ap.add_argument("--ours", default="dm1c_repro")
    ap.add_argument("--err-def", default="B", choices=["A", "B"])
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--out-name", default="A11_fig4b_dm1c_cmp.png")
    args = ap.parse_args()
    sl = slice(None, None, args.every)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    h_off, h_ours, h_paper = [], [], []
    for ds, kind in ((args.official, "official"), (args.ours, "ours")):
        tau, F, err, nb_eff = curves_for(ds, args.err_def, args.every)
        print("  [%s] bar convention %s (%d batches)" % (ds, args.err_def, nb_eff), flush=True)
        for p in PS:
            c = CP[p]
            y = F[p]
            m = np.isfinite(y) & (y > 0)
            lo, hi = err[p]
            if kind == "official":
                ln, = ax.plot(tau[m], y[m], ls="-", lw=1.2, color=c, zorder=3)
                e = ax.errorbar(tau[m][sl], y[m][sl], yerr=[y[m][sl] - lo[m][sl], hi[m][sl] - y[m][sl]],
                                color=c, ls="none", marker="o", ms=3.4, mfc=c, mec=c,
                                capsize=2.5, capthick=1.0, elinewidth=1.0, zorder=4)
                h_off.append((ln, e))
            else:
                ln, = ax.plot(tau[m], y[m], ls="-.", lw=1.3, color=c, zorder=5)
                e = ax.errorbar(tau[m][sl], y[m][sl], yerr=[y[m][sl] - lo[m][sl], hi[m][sl] - y[m][sl]],
                                color=c, ls="none", marker="^", ms=3.8, mfc="none", mec=c, mew=1.0,
                                capsize=1.6, capthick=0.8, elinewidth=0.8, alpha=0.85, zorder=6)
                h_ours.append((ln, e))
            r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4b_%s_p%d.out" % (BLOCK[ds], p))
            n = min(len(tau), len(r))
            ratio = y[:n] / r[:n, 1]
            print("      p=%d: ours/paper ratio, median %.4f (%.3f..%.3f) | relative full width ours %.4f vs paper %.4f"
                  % (p, np.median(ratio), ratio.min(), ratio.max(),
                     np.median((hi[:n] - lo[:n]) / y[:n]), np.median((r[:n, 3] + r[:n, 2]) / r[:n, 1])),
                  flush=True)

    # the source curve of the paper (once per order)
    for p in PS:
        c = CP[p]
        r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4b_%s_p%d.out" % (BLOCK[args.official], p))
        xr, mr, lo_r, hi_r = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        # z-order as in the Fig. 4a comparison: solid below, dashed and dash-dot above
        ln2, = ax.plot(xr, mr, ls="--", lw=1.1, color=c, alpha=0.9, zorder=4)
        ylo = np.maximum(mr - lo_r, mr * 0.02)
        ax.fill_between(xr, ylo, mr + hi_r, color=c, alpha=0.22, lw=0, zorder=1)
        ax.plot(xr[sl], mr[sl], ls="none", marker="s", ms=3.2, mfc="none", mec=c, alpha=0.95, zorder=4.2)
        h_paper.append((ln2, Patch(facecolor=c, alpha=0.35, edgecolor=c, lw=0.8)))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.043, 200)
    ax.set_ylim(1e0, 3e9)
    ax.set_yticks(10.0 ** np.arange(0, 10, 3))
    ax.set_xlabel("$\\tau/\\tau_\\eta$")
    ax.set_ylabel("$F_\\tau^{(p)}$")
    ax.grid(alpha=0.22, which="both", lw=0.4)
    ax.set_title("Fig.4b: generalized flatness — DM-1c paper samples (solid) vs our DM-1c "
                 "(dash-dot) vs paper source (dashed)", fontsize=9.5)

    # three legend columns: tighter font and handles with evenly spaced anchors, so the "ours" and
    # "paper source" blocks do not squeeze each other (2026-10-14 visual pass)
    kw = dict(loc="upper right", fontsize=7.5, framealpha=0.9, handlelength=2.0, borderpad=0.45,
              handler_map={tuple: HandlerTuple(ndivide=1)})
    lg1 = ax.legend(h_off, ["$p$=%d" % p for p in PS], title="paper samples",
                    bbox_to_anchor=(0.575, 0.985), **kw)
    ax.add_artist(lg1)
    lg2 = ax.legend(h_ours, ["$p$=%d" % p for p in PS], title="ours",
                    bbox_to_anchor=(0.79, 0.985), **kw)
    ax.add_artist(lg2)
    ax.legend(h_paper, ["$p$=%d" % p for p in PS], title="paper source",
              bbox_to_anchor=(1.0, 0.985), **kw)

    out = os.path.join(FIGS, args.out_name)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, "| bar convention %s | stride every=%d" % (args.err_def, args.every), flush=True)


if __name__ == "__main__":
    main()
