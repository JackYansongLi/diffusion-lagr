# -*- coding: utf-8 -*-
"""48_fig1a_fig2_cmp_mod.py -- three-way comparison for Figs. 1a and 2: the source curve of the paper,
the released DM-1c samples and our retrained DM-1c samples.

Background: A7_fig1a_dm1c_mod.png and A8_fig2_dm1c_mod.png show only the released DM-1c samples against
the source data of the paper.
This script adds **our retrained DM-1c samples** to the same figure, writes a new file and **overwrites
nothing**.

Conventions: the curves reuse the functions of 14_fig1a_pdf_mod.py (coarse_bins / trim_ends /
bridge_gaps / adaptive_curve) and their defaults, so the released and the retrained curve share every
parameter
(the same number of coarse bins, the same end rule, the same shift), and any difference comes from the
data alone.

Usage: python 48_fig1a_fig2_cmp_mod.py
Output:
  results/figs/A7_fig1a_dm1c_cmp.png (four lags: released, retrained and the paper, one shift)
  results/figs/A8_fig2_dm1c_cmp.png (acceleration PDF with the autocorrelation inset)
"""
import argparse
import importlib
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_m14 = importlib.import_module("14_fig1a_pdf_mod")
RESULTS, FIGS = _m14.RESULTS, _m14.FIGS
parse_src, coarse_bins, trim_ends, bridge_gaps = _m14.parse_src, _m14.coarse_bins, _m14.trim_ends, _m14.bridge_gaps

# released samples = samples_358400x2000x1_DM-1c of the paper; retrained = our own DM-1c
# style: released = solid blue (the DM-1c colour of the existing A7/A8 figures); retrained = dashed red, so
# they stay distinguishable when they overlap
STYLE = {"dm1c": dict(color="tab:blue", ls="-", lw=1.2),
         "dm1c_repro": dict(color="tab:red", ls="--", lw=1.4)}
LABELS = {"dm1c": "DM-1c (paper samples)", "dm1c_repro": "DM-1c (ours)"}


def series_list(official, ours):
    """Two (ds, style, label) entries; when ours and official share a name an orange dash-dot keeps them apart
    (a pipeline self-check)."""
    out = [(official, STYLE.get(official, dict(color="tab:blue", ls="-", lw=1.2)),
            LABELS.get(official, official))]
    if ours:
        dup = (ours == official)
        out.append((ours, dict(color="tab:orange", ls="-.", lw=1.4) if dup else
                    STYLE.get(ours, dict(color="tab:red", ls="--", lw=1.4)),
                    LABELS.get(ours, ours) + (" [dup test]" if dup else "")))
    return out
SRC_FIG1A = "42256_2024_810_MOESM4_ESM.txt"
SRC_FIG2 = "42256_2024_810_MOESM5_ESM.txt"


def build_curves(ds, args):
    """Return the figstats content of one dataset plus (xc, pdf, agg, wb) per lag, with the defaults of 08_mod."""
    z = np.load(os.path.join(RESULTS, "figstats_%s.npz" % ds))
    edges = z["inc_edges"]
    wb = float(edges[1] - edges[0])
    H = z["hist_inc"]
    out = []
    for k in range(len(z["pdf_lags"])):
        agg, xc, hc = coarse_bins(H[k], edges, nbins=args.nbins)
        n_fill = 0
        if args.bridge_gaps > 0:
            xc, hc, n_fill = bridge_gaps(xc, hc, args.bridge_gaps)
        if args.tail_min_count > 0:
            xc, hc = trim_ends(xc, hc, max(1, args.tail_min_count))
        pdf = np.where(hc > 0, hc / (H[k].sum() * agg * wb), np.nan)
        out.append((xc, pdf, agg, wb, n_fill))
    return z, edges, H, wb, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-min-count", type=int, default=1)
    ap.add_argument("--nbins", type=int, default=200)
    ap.add_argument("--bridge-gaps", type=int, default=0)
    ap.add_argument("--out-suffix", default="_cmp_mod")
    ap.add_argument("--official", default="dm1c", help="dataset name of the released DM-1c samples (default dm1c)")
    ap.add_argument("--ours", default="dm1c_repro", help="dataset name of the retrained DM-1c samples (default dm1c_repro)")
    args = ap.parse_args()
    SERIES = series_list(args.official, args.ours)

    loaded = {}
    for ds, st, lab in SERIES:
        loaded[ds] = build_curves(ds, args)
        z, edges, H, wb, curves = loaded[ds]
        print("[%s] lags=%s fine bin Delta=%.3f sigma, aggregation %dx" %
              (lab, list(z["pdf_lags"] // 10), wb, curves[0][2]))

    # ================= Fig.1a =================
    z0 = loaded["dm1c"][0]
    pdf_lags = z0["pdf_lags"]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    # source data of the paper (one curve, drawn once)
    ref_blocks = {}
    for tau in pdf_lags:
        try:
            ref_blocks[tau] = parse_src(SRC_FIG1A, "Figure1a_DM-1c_tau" + str(tau // 10))
        except (KeyError, IndexError):
            pass
    for ds, st, lab in SERIES:
        curves = loaded[ds][4]
        for k, tau in enumerate(pdf_lags):
            xc, pdf, agg, wb, _ = curves[k]
            sh = 10.0 ** (-1.3 * k)
            # one legend entry per family: the group holds all four lags (1, 2, 5, 100),
            # so labelling it with a single lag is wrong (user, 2026-10-22)
            ax.semilogy(xc, pdf * sh,
                        label=("%s, $\\tau/\\tau_\\eta$ = 1, 2, 5, 100" % lab) if k == 0 else None,
                        **st)
            if k == 0 and len(curves[k][0]):
                print("[Fig.1a] %-24s tau/tau_eta=%d: range +-%.2f sigma, %d points"
                      % (lab, tau // 10, float(np.nanmax(np.abs(xc))), int(np.sum(~np.isnan(pdf)))))
    for k, tau in enumerate(pdf_lags):
        if tau in ref_blocks:
            ref = ref_blocks[tau]
            mr = ref[:, 1] > 0
            ax.semilogy(ref[mr, 0], ref[mr, 1], ":", color="0.4", lw=1.3,
                        label="paper" if k == 0 else None)
    ax.set_xlim(-40, 40)
    ax.set_ylim(1e-13, 1e1)
    ax.set_xlabel("$\\delta_\\tau V_i / \\sigma(\\delta_\\tau V_i)$")
    ax.set_ylabel("PDF$(\\delta_\\tau V_i)$")
    ax.set_title("Fig.1a: paper source (dotted) vs DM-1c paper samples (blue solid)\n"
                 "vs our DM-1c (red dashed); PDFs shifted by $10^{-1.3k}$", fontsize=10)
    # the legend leaves the data area (2026-10-10: a legend must not cover a curve)
    h_, l_ = ax.get_legend_handles_labels()
    # Fixed canvas and explicit margins (no tight bbox): the DNS panel uses the same
    # numbers, so the two panels of Fig.1 come out pixel-identical in size (user request).
    fig.subplots_adjust(left=0.155, right=0.975, top=0.905, bottom=0.175)
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    p1 = os.path.join(FIGS, "A7_fig1a_dm1c%s.png" % args.out_suffix)
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("saved", p1)

if __name__ == "__main__":
    main()
