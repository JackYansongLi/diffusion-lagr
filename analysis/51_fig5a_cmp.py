#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""51_fig5a_cmp.py -- three-way comparison for Fig. 5a: the source curve of the paper, the released
DM-1c samples and our retrained DM-1c samples.

On top of 06_fig5a_local_slope_mod.py (which writes A6_fig5a_local_slope_mod_<ds>.png) this adds the
retrained DM-1c result as a third source, writes a new file and touches no existing figure. Conventions
follow 06_mod item by item:
zeta(p,tau) = [dlogS_p/dlogtau]/[dlogS_2/dlogtau] (a second-order central difference in log10), the 31 lags
of the paper
are indexed directly (lag = tau*10 is always an integer), the mean and the bars come from one source (the
average of the batch curves with a min/max envelope),
the bars follow the caption of the paper (30 component batches for the DNS and DM-3c, 10 for the
single-component DM-1c), and the axes span 1.0..2.2 vertically and 3e-2..2.5e2 horizontally.

Visual encoding of the three sources:
  paper source  : grey dashed line, open square, grey band
  paper samples : blue solid line, filled circle, capped bars
  ours (retrained) : red dash-dot line, filled triangle, thin bars

Usage: python 51_fig5a_cmp.py [--official dm1c --ours dm1c_repro]
Output: results/figs/A15_fig5a_dm1c_cmp.png
"""
import argparse
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
SRC = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data")
FIGS = os.path.join(RESULTS, "figs")
SRC_FN = os.path.join(SRC, "42256_2024_810_MOESM8_ESM.txt")
BLOCK = {"dm1c": "Figure5a_DM-1c.out (tau,mean,-,+)", "dm1c_repro": "Figure5a_DM-1c.out (tau,mean,-,+)"}
YLIM = (1.0, 2.2)
XLIM = (3e-2, 2.5e2)


def paper_block(ds):
    lines = open(SRC_FN, errors="replace").read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("# " + BLOCK[ds])][0]
    n = 0
    while i + 1 + n < len(lines) and lines[i + 1 + n].strip() and not lines[i + 1 + n].startswith("#"):
        n += 1
    pa = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(n)])
    return pa[:, 0], pa[:, 1], pa[:, 1] - pa[:, 2], pa[:, 1] + pa[:, 3]


def zeta4(S2, S4, lags):
    lg = np.log10(np.asarray(lags, dtype=float))
    d4 = np.gradient(np.log10(S4), lg, axis=-1)
    d2 = np.gradient(np.log10(S2), lg, axis=-1)
    return d4 / d2


def zeta_series(ds, L):
    """Return (tau_points, mean, lo, hi, n_curves) on the 31 lags of the paper, batching as in 06_mod."""
    z = np.load(os.path.join(RESULTS, "flatness_%s_full.npz" % ds))
    orders = [int(x) for x in z["orders"]]
    i2, i4 = orders.index(2), orders.index(4)
    lags = np.asarray(z["lags"])
    Sbc = z["Sbc"]                                   # (nb, nl, 4, C)
    nb, nl, _, nc = Sbc.shape
    zb = np.empty((nb, nc, nl))
    for b in range(nb):
        for c in range(nc):
            zb[b, c] = zeta4(Sbc[b, :, i2, c], Sbc[b, :, i4, c], lags)
    zb = zb.reshape(-1, nl)
    return zb.mean(axis=0)[L - 1], zb.min(axis=0)[L - 1], zb.max(axis=0)[L - 1], zb.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default="dm1c")
    ap.add_argument("--ours", default="dm1c_repro")
    ap.add_argument("--out-name", default="A15_fig5a_dm1c_cmp.png")
    args = ap.parse_args()

    tau, pm, plo, phi = paper_block(args.official)
    L = np.rint(tau * 10).astype(int)

    res = {}
    for ds in (args.official, args.ours):
        m, lo, hi, nc = zeta_series(ds, L)
        res[ds] = (m, lo, hi, nc)
        ratio = m / pm
        n_clip = int(((m < YLIM[0]) | (m > YLIM[1])).sum() + ((lo < YLIM[0]) | (hi > YLIM[1])).sum())
        print("[%s] batch curves %d | ratio over the 31 points, median %.4f (%.3f..%.3f) | absolute difference median %.4f max %.4f at tau=%.1f | elements outside the view %d"
              % (ds, nc, np.median(ratio), ratio.min(), ratio.max(),
                 np.median(np.abs(m - pm)), np.abs(m - pm).max(), tau[int(np.argmax(np.abs(m - pm)))], n_clip),
              flush=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    h_ref = ax.axhline(2.0, color="0.45", ls=":", lw=1.2, zorder=1)
    ln_p = ax.plot(tau, pm, "s--", ms=4.0, mfc="white", mew=0.9, color="0.25", lw=1.1, zorder=3)[0]
    bd_p = ax.fill_between(tau, plo, phi, color="0.35", alpha=0.22, zorder=2)
    m, lo, hi, nc = res[args.official]
    eb = ax.errorbar(tau, m, yerr=[m - lo, hi - m], fmt="o-", ms=3.4, lw=1.5,
                     color="#1f77b4", capsize=2.0, elinewidth=0.9, zorder=4)
    m2, lo2, hi2, _ = res[args.ours]
    eb2 = ax.errorbar(tau, m2, yerr=[m2 - lo2, hi2 - m2], fmt="^-.", ms=4.0, lw=1.4,
                      color="tab:red", capsize=1.6, elinewidth=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$\tau/\tau_\eta$", fontsize=11)
    ax.set_ylabel(r"$\zeta(4,\tau)$", fontsize=11)
    ax.set_title("Fig.5a: local slope — DM-1c paper samples (blue) vs our DM-1c (red) vs paper source",
                 fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend([(ln_p, bd_p), eb, eb2, h_ref],
              ["paper source: dashed line + min/max band",
               "paper samples: solid line + min/max bars (DM-1c)",
               "ours: dash-dot + min/max bars (DM-1c, re-trained)",
               r"$\zeta=2$ (non-intermittent)"],
              fontsize=8, loc="lower left", framealpha=0.9)
    out = os.path.join(FIGS, args.out_name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
