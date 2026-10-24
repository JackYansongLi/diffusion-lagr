#!/usr/bin/env python3
"""26_fig4c_variants.py -- point-by-point comparison of the variant-1/2 error bars of Fig. 4c with the
paper (one figure per variant).

Variant 1 (as proposed: the denominator becomes the per-component product S2i*S2j), **full data**, reads
`fig4c_calibers_<ds>.npz`
  → `A13_fig4c_var1_<ds>.png`
Variant 2 (as proposed: shuffle before cutting the 10 batches instead of cutting in order), **sliced data**
(= 1/10 of the full set, M = one batch),
   one ordered cut against the point-by-point median of 40 shuffled draws, read from
   `fig4c_split_<ds>.npz` -> `A13_fig4c_var2_<ds>.png`
   (every width on a slice grows by about sqrt(10), so variant 2 is compared only between ordered and
   shuffled cuts, not against the paper)

Four columns per figure (shared tau/tau_eta axis):
  [1] log-log F_tau^(4,ij): the paper (grey band) and our conventions (bars every 32 points, nudged
      slightly in tau so they can be told apart)
  [2] **relative full width (hi-lo)/F**: the decisive column, where width differences invisible on the log
      axis are obvious (the paper is the grey dashed line)
  [3] relative deviation of the lower bound, point by point (our lo / paper lo - 1)
  [4] the same for the upper bound (our hi / paper hi - 1)

Usage: python3 26_fig4c_variants.py --variant 1|2 [--datasets dm3c,dns]
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
FIGS = os.path.join(RESULTS, "figs")
TAG = {"dm3c": "DM-3c", "dns": "DNS"}
EVERY = 32
COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#17becf"]
LSTYLES = ["-", "--", "-.", ":"]


def series_var1(ds):
    z = np.load(os.path.join(RESULTS, "fig4c_calibers_%s.npz" % ds))
    nm = [str(x) for x in z["names"]]
    want = [("C: global S2avg, avg-first", "猜测1("),
            ("G: global S2avg, extrema-first", "G("),
            ("var1a: global S2i*S2j, avg-first", "变体1a(D3 全局逐分量分母,先平均)"),
            ("var1b: per-batch S2i*S2j, avg-first", "变体1b(D2 逐批逐分量分母,先平均)")]
    out = []
    for label, pre in want:
        k = [i for i, s in enumerate(nm) if s.startswith(pre)][0]
        out.append((label, z["los"][k], z["his"][k]))
    return z["tau"], z["F"], z["paper_lo"], z["paper_hi"], out


def series_var2(ds):
    z = np.load(os.path.join(RESULTS, "fig4c_split_%s.npz" % ds))
    nm = [str(x) for x in z["names"]]
    out = []
    for label, key, which in [("sequential, avg-first", "D1-O1", "seq"),
                              ("shuffled x40, avg-first", "D1-O1", "rand"),
                              ("sequential, extrema-first", "D1-O2", "seq"),
                              ("shuffled x40, extrema-first", "D1-O2", "rand")]:
        k = nm.index(key)
        lo = z["seq_los"][k] if which == "seq" else z["rand_los"][k]
        hi = z["seq_his"][k] if which == "seq" else z["rand_his"][k]
        out.append((label, lo, hi))
    return z["tau"], z["Fref"], z["paper_lo"], z["paper_hi"], out


def render(ds, variant, tau, F, plo, phi, series):
    fig, ax = plt.subplots(4, 1, figsize=(7.4, 10.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.1, 1.05, 1, 1], hspace=0.13))
    a0, aw, a1, a2 = ax
    a0.fill_between(tau, plo, phi, color="0.35", alpha=0.22, label="paper (MOESM7)")
    a0.plot(tau, F, "-", color="0.15", lw=1.6, label="this work (mean)", zorder=5)
    aw.plot(tau, (phi - plo) / F, "--", color="0.35", lw=1.4, label="paper")
    for i, (label, lo, hi) in enumerate(series):
        c = COLORS[i % len(COLORS)]
        ls = LSTYLES[i % 4]
        off = 1.0 + 0.012 * (i - 1.5)      # nudge in tau, otherwise overlapping envelopes cannot be counted
        a0.errorbar(tau[::EVERY] * off, F[::EVERY], yerr=[(F - lo)[::EVERY], (hi - F)[::EVERY]],
                    fmt="o", ms=3.0, lw=0.9, capsize=1.8, color=c, label=label, zorder=4 - i)
        aw.plot(tau, (hi - lo) / F, ls, color=c, lw=1.4, label=label)
        a1.plot(tau, lo / plo - 1.0, ls, color=c, lw=1.2, label=label)
        a2.plot(tau, hi / phi - 1.0, ls, color=c, lw=1.2, label=label)
    aw.set_ylabel("rel. full width\n(hi - lo)/F", fontsize=8)
    aw.set_yscale("log")
    aw.grid(alpha=0.25, which="both")
    for a, ttl in zip((a1, a2), ("lower bound dev.", "upper bound dev.")):
        a.axhline(0, color="0.5", lw=0.8, ls="--")
        a.set_ylabel(ttl, fontsize=8)
        a.grid(alpha=0.25)
        a.set_yscale("symlog", linthresh=0.02)
    for a in (a0, aw, a1, a2):
        a.tick_params(labelsize=8)
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_xlim(0.1, 102.4)
    a0.set_ylim(0.8, 30)
    a0.set_ylabel(r"$F_{\tau}^{(4,ij)}$", fontsize=9.5, labelpad=10)
    a0.grid(alpha=0.25, which="both")
    a0.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.92, handlelength=1.6)
    aw.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9, handlelength=1.8)
    sub = ("variant 1: per-component denominator S2i*S2j (full data)" if variant == 1
           else "variant 2: sequential vs shuffled split (slice = 1/10 data, M = one batch)")
    a0.set_title("%s — Fig.4c %s" % (TAG[ds], sub), fontsize=10)
    a2.set_xlabel(r"$\tau/\tau_\eta$", fontsize=11)
    out = os.path.join(FIGS, "A13_fig4c_var%d_%s.png" % (variant, ds))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved %s" % out, flush=True)
    print("   %-34s width %-8.4f (paper)" % ("paper", np.median((phi - plo) / F)), flush=True)
    for label, lo, hi in series:
        dlo = np.abs(lo / plo - 1)
        dhi = np.abs(hi / phi - 1)
        print("   %-34s width %-8.4f | lo deviation median %.4f max %.4f | hi deviation median %.4f max %.4f"
              % (label, np.median((hi - lo) / F), np.median(dlo), dlo.max(), np.median(dhi), dhi.max()),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, required=True, choices=[1, 2])
    ap.add_argument("--datasets", default="dm3c,dns")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        tau, F, plo, phi, series = series_var1(ds) if args.variant == 1 else series_var2(ds)
        render(ds, args.variant, tau, F, plo, phi, series)


if __name__ == "__main__":
    main()
