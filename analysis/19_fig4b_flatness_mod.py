"""19_fig4b_flatness_mod.py -- generalized flatness of Fig. 4b (**revision**; the initial
19_fig4b_flatness.py is left untouched).

Change against the initial version (2026-10-06): **both legends moved to the upper right corner**.
Reason: F_tau^(p) of Fig. 4b **decreases** with tau (the curves run from the upper left to the lower right),
so the upper left corner holds data and the initial legend covered the curves;
the upper right corner is free (at large tau the three orders sit near y ~ 3/15/100, far below the 1e9 top),
so moving the legends there hides nothing.

Everything else follows the initial version (the bars default to convention B = the 30 component batches of
the paper; band on the paper side, bars on ours; markers every 32 points).
Output: results/figs/A11_fig4b_<ds>_mod.png

The docstring of the initial version follows.
----------------------
One figure, one script: reproduction of the generalized flatness F_tau^(p) of Fig. 4b (initial version).

Fig. 4b of the paper plots the generalized flatness F_tau^(p) = S_tau^(p) / [S_tau^(2)]^(p/2) on log-log axes
for p = 4, 6, 8 and
compares the DNS, DM-1c and DM-3c. As established with Fig. 4a there is **one figure per dataset**, each
with the three orders p = 4, 6, 8 and
our reproduction against the original result of the paper, with error bars.

Error bars (as in the caption of the paper): the dataset is cut into independent batches and, at each tau,
the min/max of the batch F values is taken.
  - convention A (the earlier one) = 10 trajectory blocks with the three components merged
  - convention B (the paper's, captions of Figs. 4 and 5) = components kept apart, i.e. 30 component
    batches for the DNS and DM-3c and 10 for DM-1c
    (F is a ratio, so each batch has its own F = S_p^(batch) / [S_2^(batch)]^(p/2) before the min/max)

Layout (as finalised for Fig. 4a): ours = solid line, filled circle and capped bars (every 32nd point);
the paper = dashed line, open square and translucent band (full resolution); two legends (this work /
paper), each with p = 4/6/8 in three vertical rows.
Axes as in Fig. 4b of the paper: x logarithmic with major ticks at 1e-1 and 1e2 plus margin, so
xlim(0.043, 200) (as in Fig. 4a);
y logarithmic with major ticks at 1e0 and 1e9 (one every three decades) and margin at the top,
ylim(1e0, 3e9).

Data sources:
  ours : results/flatness_<ds>_full.npz (written by 18_flatness_stats.py: mean(nl,4) = S2/S4/S6/S8,
         Sbc(nb,nl,4,C) = the S values of every batch and component)
  paper : the blocks of Data/Li_et_al_2024_source_data/42256_2024_810_MOESM7_ESM.txt
         `Figure4b_<DNS|DM-1c|DM-3c>_p<4|6|8>.out`, columns = tau, F, -, + (the error half-width)

Output: results/figs/A11_fig4b_<ds><suffix>.png
Usage: python 19_fig4b_flatness_mod.py --dataset dns|dm1c|dm3c [--err-def A|B]
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
CP = {4: "k", 6: "tab:blue", 8: "tab:red"}
PS = (4, 6, 8)
IX = {2: 0, 4: 1, 6: 2, 8: 3}          # index on the orders axis of the npz
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
    ap.add_argument("--dataset", required=True, choices=["dns", "dm1c", "dm3c"])
    ap.add_argument("--err-def", default="B", choices=["A", "B"],
                    help="bar convention: A = 10 trajectory blocks with the components merged; B = the paper's (components kept apart, 30 component batches for the DNS and DM-3c)")
    ap.add_argument("--out-suffix", default="_mod")
    ap.add_argument("--every", type=int, default=EVERY)
    args = ap.parse_args()
    ds = args.dataset
    sl = slice(None, None, args.every)

    z = np.load(os.path.join(RESULTS, "flatness_%s_full.npz" % ds))
    lags = z["lags"]
    tau = 0.1 * lags
    mean = z["mean"]                                   # (nl, 4) = full-resolution mean of S2/S4/S6/S8
    Vbc = z["Sbc"]                                     # (nb, nl, 4, C)
    nb, nl, _, C = Vbc.shape

    ours, err = {}, {}
    for p in PS:
        ours[p] = mean[:, IX[p]] / mean[:, IX[2]] ** (p / 2.0)
        if args.err_def == "A":
            S = Vbc.mean(axis=3)                       # (nb, nl, 4) with the components merged
            F = S[:, :, IX[p]] / S[:, :, IX[2]] ** (p / 2.0)
        else:
            S = Vbc.transpose(0, 3, 1, 2).reshape(nb * C, nl, 4)   # transpose before reshaping
            F = S[:, :, IX[p]] / S[:, :, IX[2]] ** (p / 2.0)
        err[p] = (F.min(axis=0), F.max(axis=0))
    n_err = nb if args.err_def == "A" else nb * C

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    h_ours, h_paper = [], []
    for p in PS:
        c = CP[p]
        x = tau
        y = ours[p]
        m = np.isfinite(y) & (y > 0)
        lo, hi = err[p]
        ln, = ax.plot(x[m], y[m], ls="-", lw=1.2, color=c, zorder=3)
        e = ax.errorbar(x[m][sl], y[m][sl], yerr=[y[m][sl] - lo[m][sl], hi[m][sl] - y[m][sl]],
                        color=c, ls="none", marker="o", ms=3.4, mfc=c, mec=c,
                        capsize=2.5, capthick=1.0, elinewidth=1.0, zorder=4)
        h_ours.append((ln, e))

        r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4b_%s_p%d.out" % (BLOCK[ds], p))
        n = min(len(tau), len(r))
        xr, mr, lo_r, hi_r = r[:n, 0], r[:n, 1], r[:n, 2], r[:n, 3]
        ln2, = ax.plot(xr, mr, ls="--", lw=1.1, color=c, alpha=0.9, zorder=2)
        ylo = np.maximum(mr - lo_r, mr * 0.02)         # clip the lower edge on the log axis (at small tau the half-width can exceed the mean)
        band = ax.fill_between(xr, ylo, mr + hi_r, color=c, alpha=0.22, lw=0, zorder=1)
        ax.plot(xr[sl], mr[sl], ls="none", marker="s", ms=3.0, mfc="none", mec=c, alpha=0.9, zorder=2)
        h_paper.append((ln2, Patch(facecolor=c, alpha=0.35, edgecolor=c, lw=0.8)))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.043, 200)
    ax.set_ylim(1e0, 3e9)
    ax.set_yticks(10.0 ** np.arange(0, 10, 3))         # 1e0/1e3/1e6/1e9 (as in the paper)
    ax.set_xlabel("$\\tau/\\tau_\\eta$")
    ax.set_ylabel("$F_\\tau^{(p)}$")
    ax.grid(alpha=0.22, which="both", lw=0.4)
    ax.set_title("Fig.4b reproduction: generalized flatness (%s)" % LABEL[ds], fontsize=10)

    # both legends moved to the **upper right** (F decreases with tau, so the upper left holds data)
    kw = dict(loc="upper right", fontsize=8, framealpha=0.9, handlelength=2.4, borderpad=0.5,
              handler_map={tuple: HandlerTuple(ndivide=1)})
    lg1 = ax.legend(h_ours, ["$p$=%d" % p for p in PS], title="this work",
                    bbox_to_anchor=(0.745, 0.985), **kw)
    ax.add_artist(lg1)
    ax.legend(h_paper, ["$p$=%d" % p for p in PS], title="paper",
              bbox_to_anchor=(0.985, 0.985), **kw)

    out = os.path.join(FIGS, "A11_fig4b_%s%s.png" % (ds, args.out_suffix))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out, "| bar convention %s (%d batches) | stride every=%d" % (args.err_def, n_err, args.every),
          flush=True)

    for p in PS:
        r = parse_src("42256_2024_810_MOESM7_ESM.txt", "Figure4b_%s_p%d.out" % (BLOCK[ds], p))
        n = min(len(tau), len(r))
        ratio = ours[p][:n] / r[:n, 1]
        lo, hi = err[p]
        w_us = np.median((hi[:n] - lo[:n]) / ours[p][:n])
        w_pa = np.median((r[:n, 3] + r[:n, 2]) / r[:n, 1])
        print("  p=%d: ours/paper ratio, median %.4f (%.3f..%.3f) | relative full width of the bars ours %.4f vs paper %.4f (ratio %.2f)"
              % (p, np.median(ratio), ratio.min(), ratio.max(), w_us, w_pa, w_pa / w_us), flush=True)


if __name__ == "__main__":
    main()
