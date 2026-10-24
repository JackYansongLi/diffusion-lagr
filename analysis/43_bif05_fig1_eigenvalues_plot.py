"""43_bif05_fig1_eigenvalues_plot.py -- FIGURE stage for Biferale et al. (2005) Fig. 1 (main panel).

Reads the npz produced by 43_bif05_fig1_eigenvalues_data.py and draws the figure.  NO data processing
here: every style choice is a command-line option, so restyling costs seconds instead of re-running
the 5-minute sweep.

  y axis  : <g_i>/eta^2 (paper normalisation), lower limit from --ymin
  curves  : three eigenvalues, paper-style markers, optional spatial-block band (--band)
  guide   : single t^3 dimensional line, range from --t3

Usage:
  43_bif05_fig1_eigenvalues_plot.py [--in NPZ] [--out NAME] [--band] [--ymin 1e-2]
                                    [--dpi D] [--title T] [--t3 A:B] [--markevery N]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl

COLS = ["tab:red", "tab:blue", "tab:green"]
SYMS = ["o", "s", "d"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(tl.RESULTS, "bif05_fig1_eigenvalues.npz"))
    ap.add_argument("--out", default="B01_bif05_fig1_eigenvalues.png")
    ap.add_argument("--band", action="store_true", help="draw the spatial-block band (off: matches the paper)")
    ap.add_argument("--ymin", type=float, default=1e-2)
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--t3", default="60:1200", help="first:last time index for the t^3 guide line")
    ap.add_argument("--markevery", type=int, default=120)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--figsize", default="6.6,4.6")
    ap.add_argument("--title", default="Biferale et al. (2005) Fig. 1 analogue: TURB-Lagr tetrads")
    a = ap.parse_args()

    d = np.load(a.inp)
    t_tau = d["t_tau"]
    mean = d["mean"]
    eta2 = float(d["eta2"]) if "eta2" in d.files else 1.0
    i0, i1 = (int(x) for x in a.t3.split(":"))

    plt = tl.setup_style()
    fig, ax = plt.subplots(figsize=tuple(float(x) for x in a.figsize.split(",")))
    for i in range(3):
        if a.band and "lo" in d.files:
            ax.fill_between(t_tau, d["lo"][:, i] / eta2, d["hi"][:, i] / eta2,
                            color=COLS[i], alpha=0.15, lw=0)
        ax.plot(t_tau, mean[:, i] / eta2, color=COLS[i], lw=1.5, marker=SYMS[i],
                markevery=a.markevery, ms=4.5, label=f"$\\langle g_{i+1} \\rangle$")
    ref = mean[i0, 0] / eta2 / t_tau[i0] ** 3
    ax.plot(t_tau[i0:i1], ref * t_tau[i0:i1] ** 3, "k--", lw=1.0, label="$t^3$ (dimensional)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=a.ymin)
    if a.xmin:
        ax.set_xlim(left=a.xmin)
    if a.xmax:
        ax.set_xlim(right=a.xmax)
    ax.set_xlabel(r"$t\,/\,\tau_\eta$")
    ax.set_ylabel(r"$\langle g_i \rangle \,/\, \eta^2$" + f"   ($\\eta = {np.sqrt(eta2):.4f}$)")
    ax.set_title(a.title)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    fig.tight_layout()
    tl.savefig(fig, a.out, dpi=a.dpi)
    print(f"source npz: {a.inp}   (nq={int(d['nq'])}, source={str(d['source'])})")


if __name__ == "__main__":
    main()
