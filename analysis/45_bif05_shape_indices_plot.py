"""45_bif05_shape_indices_plot.py -- FIGURE stage for the shape indices I_i(t)   (figure B03).

Reads the npz from 45_bif05_shape_indices_data.py.  Everything is a command-line choice, so style changes
never re-run the 5-minute sweep.

  --calibre both|plain|filtered   unconditional curves and/or the B05-selected ones
  --gauss lit|mc|none            Gaussian reference lines: literature 0.748/0.222/0.03 (Pumir/Biferale,
                                 quoted in VA08 Fig.13b) or this run's own Monte-Carlo values
  --band                         spatial-block band (min..max of the 1024 block means)
  --events                       mark the time of the <I2> minimum (maximum distortion)
  --xscale log|linear  --yscale log|linear  --xmax --ymin --ymax --t3 --markevery --dpi --out

Usage: 45_bif05_shape_indices_plot.py [--calibre both] [--gauss lit] [--band] [--events]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl

LIT = np.array([0.748, 0.222, 0.03])
COL = ["tab:blue", "tab:orange", "tab:green"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(tl.RESULTS, "bif05_shape_indices.npz"))
    ap.add_argument("--out", default="B03_bif05_shape_indices.png")
    ap.add_argument("--calibre", choices=["both", "plain", "filtered"], default="plain")
    ap.add_argument("--gauss", choices=["lit", "mc", "none"], default="lit")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--xscale", choices=["log", "linear"], default="log")
    ap.add_argument("--yscale", choices=["log", "linear"], default="log")
    ap.add_argument("--xmin", type=float, default=1e-2)   # VA08 Fig.13(b) axes: x 1e-2..1e3, y 1e-3..1e0
    ap.add_argument("--xmax", type=float, default=1e3)
    ap.add_argument("--ymin", type=float, default=1e-3)
    ap.add_argument("--ymax", type=float, default=1.5)   # above the paper's top tick 1e0 (frame headroom)
    ap.add_argument("--legend", default="lower right")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--figsize", default="6.8,4.8")
    ap.add_argument("--markevery", type=int, default=120)
    ap.add_argument("--title", default="Tetrad shape indices -- TURB-Lagr tetrads (unconditional)")
    a = ap.parse_args()

    d = np.load(a.inp)
    t_tau = d["t_tau"]
    mean, lo, hi = d["mean"], d["lo"], d["hi"]
    sel, ok = d["sel_mean"], d["sel_ok"]
    gref = d["gaussian_ref"] if a.gauss == "mc" else LIT
    eta = float(d["eta"])

    plt = tl.setup_style()
    fig, ax = plt.subplots(figsize=tuple(float(x) for x in a.figsize.split(",")))
    lw = 1.7
    if a.calibre in ("both", "plain"):
        for i in range(3):
            if a.band:
                ax.fill_between(t_tau, lo[:, i], hi[:, i], color=COL[i], alpha=0.15, lw=0)
            ax.plot(t_tau, mean[:, i], color=COL[i], lw=lw, marker="o", markevery=a.markevery, ms=4.5,
                    label=rf"$\langle I_{i+1}\rangle$")
    if a.calibre in ("both", "filtered"):
        for i in range(3):
            y = np.where(ok, sel[:, i], np.nan)
            ax.plot(t_tau, y, color=COL[i], lw=lw, ls="--",
                    label=rf"$\langle I_{i+1}\rangle$ (B05-selected)")
    ghost = None
    if a.gauss != "none":
        from matplotlib.lines import Line2D
        for i in range(3):
            ax.axhline(gref[i], color=COL[i], lw=1.0, ls=":")
        ghost = Line2D([], [], color="0.35", lw=1.0, ls=":",
                       label="isotropic Gaussian " + "/".join(f"{g:.3f}" for g in gref))
    # t = 0 itself cannot be drawn on a log x-axis (and VA08's x range starts at 1e-2), so mark the
    # first plotted sample and quote the exact t -> 0 values in the annotation.
    ax.plot([t_tau[1]], [mean[1, 0]], "kv", ms=7)
    ax.annotate(rf"$I(0)=(4/9,4/9,1/9)$" + "\n" + rf"$=(0.444,0.444,0.111)$" + "\n"
                + rf"at $t\to0$ (drawn from $t=0.10\,\tau_\eta$)",
                xy=(t_tau[1], mean[1, 0]), xytext=(0.04, 0.70), textcoords="axes fraction", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
    if a.events:
        tv = float(d["t_min_I2"]) / tl.TAU_ETA          # x axis is t / tau_eta, not physical time
        ax.axvline(tv, color="0.35", lw=1.0, ls="--")
        ax.annotate(rf"$\langle I_2\rangle$ min = {mean[:,1].min():.4f}" + "\n" + rf"at $t={tv:.1f}\,\tau_\eta$",
                    xy=(tv, mean[:, 1].min()), xytext=(0.52, 0.30), textcoords="axes fraction",
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
    ax.set_xscale(a.xscale); ax.set_yscale(a.yscale)
    ax.set_xlim(left=t_tau[0] if a.xscale == "linear" else a.xmin, right=a.xmax)
    ax.set_ylim(a.ymin, a.ymax)
    ax.set_xlabel(r"$t\,/\,\tau_\eta$")
    ax.set_ylabel(r"$I_i = g_i / r^2$")
    ax.set_title(a.title + rf"   ($\eta$ = {eta})")
    h, lb = ax.get_legend_handles_labels()
    if ghost is not None:
        h, lb = h + [ghost], lb + [ghost.get_label()]
    ax.legend(h, lb, loc=a.legend, fontsize=8.5, ncol=2)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    fig.tight_layout()
    tl.savefig(fig, a.out, dpi=a.dpi)
    print(f"source npz: {a.inp}  nq={int(d['nq'])}  calibre={a.calibre}  gauss={a.gauss}")
    print(f"  I(0) = {np.array2string(mean[0], precision=4)}   late <I> (t=1999) = "
          f"{np.array2string(mean[-1], precision=4)}")
    if ok.any():
        print(f"  selected: pass fraction {100*d['pass_frac'][-1]:.1f}% at the last time; "
              f"<I2>_sel(t=1999) = {sel[-1,1]:.4f}")


if __name__ == "__main__":
    main()
