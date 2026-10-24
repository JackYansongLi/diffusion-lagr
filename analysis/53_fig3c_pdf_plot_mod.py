#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""53_fig3c_pdf_plot_mod.py -- comparison for Fig. 3c: five panels (DNS / A / B / C / D), two curves each.

Conventions (see RESULTS_ANALYSIS.md section 7.13):
  the original txt of the paper = Figure3c_{DNS,A,B,C,D}.out of MOESM6 (y is already PDF*sigma, x is
dtauVi/sigma)
  our result     = <key>_y_paper in fig3c_pdf_dm1c_repro.npz, written by script 52
                   (**aggregated onto the x grid of the paper**; the fine-bin version peaks 3.6% higher and
                   must not be used for the comparison)
Axes: x in [-30, 30] and y in [1e-7, 1e0] (the range of the Fig. 3c panels; the extremes of the five source
curves, +-29.6, all fall inside).

Every style comes from a command-line argument (a project convention).
Usage: python 53_fig3c_pdf_plot_mod.py
Output: results/figs/A16_fig3c_{DNS,A,B,C,D}_cmp.png
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

NPZ = os.path.join(RESULTS, "fig3c_pdf_dm1c_repro.npz")
SRC_FN = os.path.join(SRC, "42256_2024_810_MOESM6_ESM.txt")
FIGS = os.path.join(RESULTS, "figs")

# key -> (block name in the paper, dataset description in the panel title, label in the output name)
PANELS = [
    ("dns", "Figure3c_DNS.out", "DNS", "DNS"),
    ("A", "Figure3c_A.out", "stage A (n = 0, final sample)", "A"),
    ("B", "Figure3c_B.out", "stage B (n = 0.06N)", "B"),
    ("C", "Figure3c_C.out", "stage C (n = 0.27N)", "C"),
    ("D", "Figure3c_D.out", "stage D (n = 0.52N)", "D"),
]
STAGE_OF = {"A": 0, "B": 48, "C": 216, "D": 416}


def paper_block(name):
    lines = open(SRC_FN, errors="replace").read().splitlines()
    i = [k for k, l in enumerate(lines) if l.strip() == "# " + name][0]
    n = 0
    while i + 1 + n < len(lines) and lines[i + 1 + n].strip() and not lines[i + 1 + n].startswith("#"):
        n += 1
    return np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)])


PANEL = {"A": "a", "B": "b", "C": "c", "D": "d", "dns": "e", "DNS": "e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--out-suffix", default="_cmp")
    ap.add_argument("--out-dir", default=FIGS)
    ap.add_argument("--fig-w", type=float, default=7.0)
    ap.add_argument("--fig-h", type=float, default=4.8)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--xmax", type=float, default=30.0)
    ap.add_argument("--ymin", type=float, default=1e-7)
    ap.add_argument("--ymax", type=float, default=1.0)
    ap.add_argument("--marker-every", type=int, default=10)
    ap.add_argument("--ours-color", default="#1f77b4")
    ap.add_argument("--paper-color", default="0.25")
    ap.add_argument("--title-prefix", default="Fig.3c: increment PDF at $\\tau=\\tau_\\eta$")
    ap.add_argument("--panel-labels", default="1", choices=["0", "1"],
                    help="1 = prefix each panel title with (a)..(e), as the caption refers to them")
    ap.add_argument("--legend", default="below", choices=["below", "lower-center", "upper-right"],
                    help="below = centred under the axes (default, keeps clear of the curves)")
    ap.add_argument("--log", default=os.path.join(RESULTS, "log_53_fig3c_pdf.txt"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    stats = []
    z = np.load(args.npz)
    say("[53] data %s | lag=%d (tau=tau_eta) | fine bin %.3f sigma | stages=%s"
        % (os.path.basename(args.npz), int(z["lag"]), float(z["fine_bin"]),
           [int(v) for v in z["stages"]]))

    for key, blk, desc, tag in PANELS:
        ref = paper_block(blk)
        xo = z[key + "_x_paper"]
        yo = z[key + "_y_paper"]
        dx = float(np.median(np.diff(ref[:, 0])))
        L1 = float(np.nansum(np.abs(yo - ref[:, 1])) * dx)
        shell = lambda y, a, b: float(np.nansum(y[(np.abs(xo) >= a) & (np.abs(xo) < b)]) * dx)
        # the fine-bin peak is diagnostic only (the npz stores no edges, so it is rebuilt from fine_bin/x_max; the
        # comparison always uses the aggregated version)
        fb, xm = float(z["fine_bin"]), float(z["x_max"])
        edges = np.arange(-xm, xm + fb / 2, fb)
        fine = z[key + "_hist"]
        fine_peak = float((fine / (fine.sum() * fb)).max())

        fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))
        ax.semilogy(ref[:, 0], ref[:, 1], ls="--", lw=1.1, color=args.paper_color,
                    marker="s", ms=3.2, mfc="white", mew=0.9, markevery=args.marker_every,
                    label="paper source (%s)" % blk.replace(".out", ""), zorder=3)
        ax.semilogy(xo, yo, ls="-", lw=1.5, color=args.ours_color,
                    marker="o", ms=3.4, markevery=args.marker_every,
                    label="ours (%s)" % ("DNS" if key == "dns" else "DM-1c re-trained, stage %s, n=%d"
                                         % (tag, STAGE_OF[tag])), zorder=4)
        ax.set_xlim(-args.xmax, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_xlabel(r"$\delta_\tau V_i \,/\, \sigma(\delta_\tau V_i)$", fontsize=11)
        ax.set_ylabel(r"$\mathrm{PDF}(\delta_\tau V_i)\,\times\,\sigma(\delta_\tau V_i)$", fontsize=11)
        lbl = PANEL.get(key if "key" in dir() else tag, "")
        if args.panel_labels == "1" and lbl:
            ax.set_title("(%s) %s\n%s" % (lbl, args.title_prefix, desc), fontsize=9.5)
        else:
            ax.set_title("%s\n%s" % (args.title_prefix, desc), fontsize=9.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        if args.legend == "below":
            # the legend leaves the data area (a project convention): two columns centred below the axes, which
            # bbox_inches="tight" includes in the crop
            ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=2, framealpha=0.95, borderaxespad=0.0)
        elif args.legend == "lower-center":
            ax.legend(fontsize=8.5, loc="lower center", framealpha=0.92)
        else:
            ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
        out = os.path.join(args.out_dir, "A16_fig3c_%s%s.png" % (tag, args.out_suffix))
        fig.tight_layout()
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        say("[53] %-4s peak ours %.4f vs paper %.4f | L1 %.5f | mass 0-1 sigma %.5f/%.5f  1-2 sigma %.5f/%.5f  "
            "2-5 sigma %.5f/%.5f  5-10 sigma %.2e/%.2e | fine-bin peak %.4f | -> %s"
            % (tag, np.nanmax(yo), ref[:, 1].max(), L1,
               shell(yo, 0, 1), shell(ref[:, 1], 0, 1), shell(yo, 1, 2), shell(ref[:, 1], 1, 2),
               shell(yo, 2, 5), shell(ref[:, 1], 2, 5), shell(yo, 5, 10), shell(ref[:, 1], 5, 10),
               fine_peak, os.path.basename(out)))
        stats.append(dict(key=key, L1=L1, peak_ours=float(np.nanmax(yo)), peak_paper=float(ref[:, 1].max()),
                          m_0_1=shell(yo, 0, 1), m_1_2=shell(yo, 1, 2), m_2_5=shell(yo, 2, 5),
                          p_0_1=shell(ref[:, 1], 0, 1), p_1_2=shell(ref[:, 1], 1, 2), p_2_5=shell(ref[:, 1], 2, 5)))
    np.savez(os.path.join(RESULTS, "fig3c_cmp_stats.npz"),
             **{("stats_%s" % s["key"]): np.array([s[k] for k in ("L1", "peak_ours", "peak_paper",
                                                                  "m_0_1", "m_1_2", "m_2_5",
                                                                  "p_0_1", "p_1_2", "p_2_5")])
                for s in stats})
    say("[53] done: 5 figures -> %s" % args.out_dir)
    log.close()


if __name__ == "__main__":
    main()
