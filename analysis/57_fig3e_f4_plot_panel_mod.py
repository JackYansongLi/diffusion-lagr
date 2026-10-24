#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""57_fig3e_f4_plot_panel_mod.py -- comparison for Fig. 3e: five panels (DNS / A / B / C / D), two curves
each.

Conventions (see RESULTS_ANALYSIS.md section 7.15)
-----------------------------------
- the original txt of the paper: `Figure3e_{DNS,A,B,C,D}.out (tau,mean,-,+)` in MOESM6, 1024 points with
tau = 0.1*lag,
  so it aligns point by point with our lag grid; columns = (tau, mean, -, +), where **-/+ are the error
  half-widths** (lower = mean-, upper = mean+)
- our curve: F_tau^(4) = S4(tau)/S2(tau)^2, computed **per unit** before any statistic is taken (each unit
  uses the S2 of its own curve as the denominator)
  * DNS: the 30 units of `fig3e_f4_dns.npz` (10 trajectory blocks x 3 components, components kept apart),
    the DNS convention of section 7.14
  * A/B/C/D: the 10 trajectory blocks of `fig3e_f4_dm1c_repro.npz`, the DM convention of section 7.14
  the mean and the envelope come from **the same set of unit values** (mean = average over the units,
  envelope = their min/max), so the two are consistent
- axes: log-log; x = tau/tau_eta = 0.1*lag in (0.09, 130) (as in Fig. 3d); y in [2e0, 1e2] (as requested,
  and since F stays at or above 2.99 along all five curves of the paper nothing is clipped; the script
  still prints its clipping check)
- legend: by default the upper right inside the axes (F decreases with tau, so that corner is free); the
  script proves the choice **numerically** (the display coordinates of the legend box are converted back to
  data coordinates and the curve points inside the box are counted) for every candidate corner, and
  --legend-scan prints the occluded count of all of them
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(DATA, "Li_et_al_2024_source_data", "42256_2024_810_MOESM6_ESM.txt")

DESC = {
    "DNS": "DNS",
    "A": "stage A (n = 0, final sample)",
    "B": "stage B (n = 0.06N)",
    "C": "stage C (n = 0.27N)",
    "D": "stage D (n = 0.52N, noisiest)",
}


def paper_block(lab):
    tag = "# Figure3e_%s.out" % lab
    lines = open(SRC, errors="replace").read().splitlines()
    idx = [k for k, l in enumerate(lines) if l.strip().startswith(tag)]
    if len(idx) != 1:
        raise RuntimeError("Figure3e_%s.out matches %d blocks (exactly 1 required)" % (lab, len(idx)))
    s = idx[0] + 1
    e = s
    while e < len(lines) and not lines[e].startswith("#"):
        e += 1
    rows = [[float(x) for x in lines[k].replace(",", " ").split()] for k in range(s, e) if lines[k].strip()]
    return np.array(rows)


def ours_curves(lab):
    """Return (lag, mean curve, lower bound, upper bound, convention note)."""
    if lab == "DNS":
        z = np.load(os.path.join(RESULTS, "fig3e_f4_dns.npz"))
        S2, S4 = z["S2u"], z["S4u"]
        F = S4 / S2 ** 2
        return z["lags"], F.mean(axis=0), F.min(axis=0), F.max(axis=0), "%d-unit envelope (chunk x comp)" % F.shape[0]
    z = np.load(os.path.join(RESULTS, "fig3e_f4_dm1c_repro.npz"))
    order = [str(o) for o in z["order"]]
    F = np.array([z["S4b"][order.index("%s_%d" % (lab, b))] / z["S2b"][order.index("%s_%d" % (lab, b))] ** 2
                  for b in range(10)])
    return z["lags"], F.mean(axis=0), F.min(axis=0), F.max(axis=0), "10-batch envelope"


def legend_box_data(ax, fig):
    """Data-coordinate extent of the current legend box plus the number of curve points inside it."""
    fig.canvas.draw()
    bb = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x0, x1, y0, y1


PANEL = {"DNS": "a", "A": "b", "B": "c", "C": "d", "D": "e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "figs"))
    ap.add_argument("--out-suffix", default="_cmp_panel")
    ap.add_argument("--panel-labels", default="1", choices=["0", "1"],
                    help="1 = prefix each panel title with (a)..(e), matching the caption")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--fig-w", type=float, default=7.0)
    ap.add_argument("--fig-h", type=float, default=4.6)
    ap.add_argument("--xmin", type=float, default=0.09)
    ap.add_argument("--xmax", type=float, default=130.0)
    ap.add_argument("--ymin", type=float, default=2e0)
    ap.add_argument("--ymax", type=float, default=1e2)
    ap.add_argument("--marker-every", type=int, default=30)
    ap.add_argument("--ours-color", default="#1f77b4")
    ap.add_argument("--paper-color", default="0.25")
    ap.add_argument("--legend", default="upper-right",
                    choices=["below", "lower-right", "lower-left", "upper-right", "upper-left"],
                    help="default upper-right: F decreases with tau, so the upper right is free (within a figure family the corner follows the curves)")
    ap.add_argument("--legend-scan", action="store_true", help="print the occluded count of every candidate corner and exit")
    ap.add_argument("--title-prefix", default="Fig.3e: flatness $F_\\tau^{(4)}$")
    ap.add_argument("--log", default=os.path.join(RESULTS, "log_57_fig3e.txt"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    stats = []
    for lab in ("DNS", "A", "B", "C", "D"):
        ref = paper_block(lab)
        xr, mr, lo_r, hi_r = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
        lag, y, lo, hi, env = ours_curves(lab)
        tau = 0.1 * lag
        ratio = y / mr

        fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))
        # z-order: solid below, dashed and dash-dot above, so nearly coincident curves remain identifiable
        # through the gaps (as in the Fig. 3c/3d family)
        ax.fill_between(xr, np.maximum(mr - lo_r, mr * 1e-9), mr + hi_r, color=args.paper_color,
                        alpha=0.20, lw=0, zorder=1)
        ax.fill_between(tau, lo, hi, color=args.ours_color, alpha=0.25, lw=0, zorder=2)
        ax.loglog(tau, y, ls="-", lw=1.6, color=args.ours_color,
                  marker="o", ms=3.2, markevery=1 + len(tau) // args.marker_every,
                  label="ours (%s, %s)" % ("DNS" if lab == "DNS" else "DM-1c re-trained stage %s" % lab, env),
                  zorder=3)
        ax.loglog(xr, mr, ls="--", lw=1.1, color=args.paper_color,
                  marker="s", ms=3.4, mfc="white", mew=1.0, markevery=1 + len(xr) // args.marker_every,
                  label="paper source (Figure3e_%s)" % lab, zorder=4)
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_xlabel(r"$\tau/\tau_\eta$", fontsize=11)
        ax.set_ylabel(r"$F_\tau^{(4)}$", fontsize=11)
        if args.panel_labels == "1":
            ax.set_title("(%s) %s\n%s" % (PANEL[lab], args.title_prefix, DESC[lab]), fontsize=9.5)
        else:
            ax.set_title("%s\n%s" % (args.title_prefix, DESC[lab]), fontsize=9.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)

        cands = ["upper-right", "upper-left", "lower-right", "lower-left", "below"]
        cover = {}
        for cand in cands:                      # prove it numerically: measure the occlusion of every corner
            if cand == "below":
                leg = ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                                ncol=1, framealpha=0.95, borderaxespad=0.0)
            else:
                loc = cand.replace("-", " ")
                leg = ax.legend(fontsize=8.5, loc=loc, framealpha=0.92, borderaxespad=0.6)
            fig.tight_layout(rect=(0, 0.27, 1, 1) if cand == "below" else None)
            bx0, bx1, by0, by1 = legend_box_data(ax, fig)
            # the check covers **every layer**: the main lines and the upper and lower edges of both envelopes
            # (a band edge running out of the frame was a real defect)
            n_in = int(np.sum((tau >= bx0) & (tau <= bx1) & (((y >= by0) & (y <= by1))
                                                            | ((hi >= by0) & (hi <= by1))
                                                            | ((lo >= by0) & (lo <= by1))))
                       + np.sum((xr >= bx0) & (xr <= bx1) & (((mr >= by0) & (mr <= by1))
                                                            | ((mr - lo_r >= by0) & (mr - lo_r <= by1))
                                                            | ((mr + hi_r >= by0) & (mr + hi_r <= by1)))))
            cover[cand] = (n_in, bx0, bx1, by0, by1)
            leg.remove()
        if args.legend_scan:
            say("[57] %-4s occluded counts per candidate corner: %s" % (lab, " | ".join("%s=%d" % (k, v[0]) for k, v in cover.items())))
            plt.close(fig)
            continue

        pick = args.legend
        if cover[pick][0] > 0:                  # the chosen corner covers data, so fall back to the first clear one
            zero = [k for k in cands if cover[k][0] == 0]
            if zero:
                say("[57] WARNING: %s covers %d points in that corner => switching to %s" % (lab, cover[pick][0], zero[0]))
                pick = zero[0]
        if pick == "below":
            ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                      ncol=1, framealpha=0.95, borderaxespad=0.0)
        else:
            ax.legend(fontsize=8.5, loc=pick.replace("-", " "), framealpha=0.92, borderaxespad=0.6)
        fig.tight_layout(rect=(0, 0.27, 1, 1) if pick == "below" else None)
        bx0, bx1, by0, by1 = legend_box_data(ax, fig)
        n_in = int(np.sum((tau >= bx0) & (tau <= bx1) & (((y >= by0) & (y <= by1))
                                                        | ((hi >= by0) & (hi <= by1))
                                                        | ((lo >= by0) & (lo <= by1))))
                   + np.sum((xr >= bx0) & (xr <= bx1) & (((mr >= by0) & (mr <= by1))
                                                        | ((mr - lo_r >= by0) & (mr - lo_r <= by1))
                                                        | ((mr + hi_r >= by0) & (mr + hi_r <= by1)))))
        # clipping check: any layer (curves, envelope edges) out of the frame is reported
        n_clip = int(np.sum(y < args.ymin) + np.sum(y > args.ymax) + np.sum(lo < args.ymin) + np.sum(hi > args.ymax)
                     + np.sum(mr < args.ymin) + np.sum(mr > args.ymax)
                     + np.sum(mr - lo_r < args.ymin) + np.sum(mr + hi_r > args.ymax))
        out = os.path.join(args.out_dir, "A18_fig3e_%s%s.png" % (lab, args.out_suffix))
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        say("[57] %-4s ratio median %.4f (%.3f..%.3f) | F(tau=1) ours %.4f paper %.4f | relative envelope width ours %.4f paper %.4f | "
            "→ %s" % (lab, np.median(ratio), ratio.min(), ratio.max(), y[9], mr[9],
                      np.median((hi - lo) / y), np.median((lo_r + hi_r) / mr), os.path.basename(out)))
        say("      legend(%s) box x in [%.3g, %.3g], y in [%.3g, %.3g] | points inside %d %s | points clipped on y %d %s"
            % (pick, bx0, bx1, by0, by1, n_in, "OK, no overlap" if n_in == 0 else "WARNING, overlapping",
               n_clip, "OK" if n_clip == 0 else "WARNING, element out of frame"))
        stats.append(dict(lab=lab, med=float(np.median(ratio)), rmin=float(ratio.min()), rmax=float(ratio.max()),
                          F1=float(y[9]), F1p=float(mr[9]), n_clip=n_clip, n_in=n_in, legend=pick))
    if not args.legend_scan:
        np.savez(os.path.join(RESULTS, "fig3e_cmp_stats.npz"),
                 lab=np.array([s["lab"] for s in stats]),
                 med=np.array([s["med"] for s in stats]), rmin=np.array([s["rmin"] for s in stats]),
                 rmax=np.array([s["rmax"] for s in stats]), F1=np.array([s["F1"] for s in stats]),
                 F1p=np.array([s["F1p"] for s in stats]))
        say("[57] done: 5 figures -> %s" % args.out_dir)
    log.close()


if __name__ == "__main__":
    main()
