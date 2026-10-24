#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""55_fig3d_s2_plot_panel_mod.py -- comparison for Fig. 3d: five panels (DNS / A / B / C / D), two curves
each.

Conventions (see RESULTS_ANALYSIS.md section 7.14)
-----------------------------------
- the original txt of the paper: `Figure3d_{DNS,A,B,C,D}.out (tau,mean,-,+)` in MOESM6, 1024 points,
  columns = (tau, mean, -, +), where **-/+ are the error half-widths** (lower = mean-, upper = mean+)
- our result: DNS from `structfuncs_dns_full.npz['S2']` with the envelope `fig4a_batches_dns.npz` (**cB = 30
  batches**,
  matching the -/+ 0.112/0.103 of the paper for the DNS: ours are 0.116/0.118);
  A/B/C/D from `fig3d_s2_dm1c_repro.npz` (a 10-batch envelope, the same convention as the Fig. 4a chain)
- axes: log-log; x = tau/tau_eta = 0.1*lag in (0.09, 130) (the tau grid of the paper, 0.1..102.4, is fully
  covered); y in [1e-3, 1e1]
Every style comes from a command-line argument (a project convention).
Usage: python 55_fig3d_s2_plot_panel_mod.py
Output: results/figs/A17_fig3d_{DNS,A,B,C,D}_cmp.png
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

SRC_FN = os.path.join(SRC, "42256_2024_810_MOESM6_ESM.txt")
FIGS = os.path.join(RESULTS, "figs")
DESC = {"DNS": "DNS", "A": "stage A (n = 0, final sample)", "B": "stage B (n = 0.06N)",
        "C": "stage C (n = 0.27N)", "D": "stage D (n = 0.52N)"}


def paper_block(lab):
    name = "Figure3d_%s.out (tau,mean,-,+)" % lab
    lines = open(SRC_FN, errors="replace").read().splitlines()
    i = [k for k, l in enumerate(lines) if l.strip() == "# " + name][0]
    n = 0
    while i + 1 + n < len(lines) and lines[i + 1 + n].strip() and not lines[i + 1 + n].startswith("#"):
        n += 1
    return np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)])


def ours_curves(lab, s2npz):
    if lab == "DNS":
        z = np.load(os.path.join(RESULTS, "structfuncs_dns_full.npz"))
        b = np.load(os.path.join(RESULTS, "fig4a_batches_dns.npz"))
        lag = z["lags"]
        return lag, z["S2"][:len(lag)], b["S2_lo_cB"][:len(lag)], b["S2_hi_cB"][:len(lag)], "30-batch envelope (cB)"
    z = np.load(s2npz)
    lag = z["lags"]
    return (lag, z["%s_S2b" % lab], z["%s_S2_lo" % lab], z["%s_S2_hi" % lab], "10-batch envelope")


PANEL = {"DNS": "a", "A": "b", "B": "c", "C": "d", "D": "e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-npz", default=os.path.join(RESULTS, "fig3d_s2_dm1c_repro.npz"))
    ap.add_argument("--out-suffix", default="_cmp_panel")
    ap.add_argument("--panel-labels", default="1", choices=["0", "1"],
                    help="1 = prefix each panel title with (a)..(e), matching the caption")
    ap.add_argument("--out-dir", default=FIGS)
    ap.add_argument("--fig-w", type=float, default=7.0)
    ap.add_argument("--fig-h", type=float, default=4.8)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--xmin", type=float, default=0.09)
    ap.add_argument("--xmax", type=float, default=130.0)
    ap.add_argument("--ymin", type=float, default=1e-3)
    ap.add_argument("--ymax", type=float, default=1e1)
    ap.add_argument("--marker-every", type=int, default=30)
    ap.add_argument("--ours-color", default="#1f77b4")
    ap.add_argument("--paper-color", default="0.25")
    ap.add_argument("--legend", default="below", choices=["below", "lower-right", "upper-left"])
    ap.add_argument("--title-prefix", default="Fig.3d: second-order structure function "
                                              "$S_\\tau^{(2)}$")
    ap.add_argument("--log", default=os.path.join(RESULTS, "log_55_fig3d.txt"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    say("[55] S2 npz=%s" % os.path.basename(args.s2_npz))
    stats = []
    for lab in ("DNS", "A", "B", "C", "D"):
        ref = paper_block(lab)
        xr, mr, lo_r, hi_r = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
        lag, y, lo, hi, env = ours_curves(lab, args.s2_npz)
        tau = 0.1 * lag
        ratio = y / mr
        rel = np.median((hi - lo) / y)
        rel_p = np.median((hi_r + lo_r) / mr)

        fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))
        # z-order (fixed in a visual pass): solid below, dashed and dash-dot above, so nearly coincident curves
        # remain identifiable through the gaps
        ax.fill_between(xr, np.maximum(mr - lo_r, mr * 1e-3), mr + hi_r, color=args.paper_color,
                        alpha=0.20, lw=0, zorder=1)
        ax.fill_between(tau, lo, hi, color=args.ours_color, alpha=0.25, lw=0, zorder=2)
        ax.loglog(tau, y, ls="-", lw=1.6, color=args.ours_color,
                  marker="o", ms=3.2, markevery=1 + len(tau) // args.marker_every,
                  label="ours (%s, %s)" % ("DNS" if lab == "DNS" else "DM-1c re-trained stage %s" % lab, env),
                  zorder=3)
        ax.loglog(xr, mr, ls="--", lw=1.1, color=args.paper_color,
                  marker="s", ms=3.4, mfc="white", mew=1.0, markevery=1 + len(xr) // args.marker_every,
                  label="paper source (Figure3d_%s)" % lab, zorder=4)
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_xlabel(r"$\tau/\tau_\eta$", fontsize=11)
        ax.set_ylabel(r"$S_\tau^{(2)}$", fontsize=11)
        if args.panel_labels == "1":
            ax.set_title("(%s) %s\n%s" % (PANEL[lab], args.title_prefix, DESC[lab]), fontsize=9.5)
        else:
            ax.set_title("%s\n%s" % (args.title_prefix, DESC[lab]), fontsize=9.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        if args.legend == "below":
            ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=1, framealpha=0.95, borderaxespad=0.0)
        else:
            ax.legend(fontsize=8.5, loc=("lower right" if args.legend == "lower-right" else "upper left"),
                      framealpha=0.92)
        out = os.path.join(args.out_dir, "A17_fig3d_%s%s.png" % (lab, args.out_suffix))
        fig.tight_layout()
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        say("[55] %-4s ratio median %.4f (%.3f..%.3f) | S2(tau=1) ours %.4e paper %.4e | "
            "relative envelope width ours %.4f paper %.4f | -> %s"
            % (lab, np.median(ratio), ratio.min(), ratio.max(), y[9], mr[9], rel, rel_p,
               os.path.basename(out)))
        stats.append(dict(lab=lab, med=float(np.median(ratio)), rmin=float(ratio.min()),
                          rmax=float(ratio.max()), s2_tau1_ours=float(y[9]), s2_tau1_paper=float(mr[9]),
                          env_ours=float(rel), env_paper=float(rel_p)))
    np.savez(os.path.join(RESULTS, "fig3d_cmp_stats.npz"),
             **{("stats_%s" % s["lab"]): np.array([s[k] for k in
                  ("med", "rmin", "rmax", "s2_tau1_ours", "s2_tau1_paper", "env_ours", "env_paper")])
                for s in stats})
    say("[55] done: 5 figures -> %s" % args.out_dir)
    log.close()


if __name__ == "__main__":
    main()
