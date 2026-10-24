"""44_bif05_volume_plot.py -- FIGURE stage for the tetrad volume V(t)   (figure B02).

Reads the npz from 44_bif05_volume_data.py.  The data are stored in physical units, so the vertical
normalisation is chosen here (no re-run of the sweep):

  --norm V0      V/V0        (starts at 1; pure growth, no eta/eps dependence)
  --norm eta3    V/eta^3     (Kolmogorov normalisation; eta = 0.0042 from TURB-Lagr)
  --norm Lint3   V/(u_rms^3 T_L^3) = V/L_int^3   (van Aartrijk 2008 Fig.13a axis; log axis 1e-8..1e2)
  --norm raw     physical box units

A t^(9/2) Richardson guide line is drawn by default (the inertial-range expectation, equivalent to
r^2 ~ t^3); --no-guide switches it off.  --band adds the spatial-block band.

Usage:
  44_bif05_volume_plot.py [--in NPZ] [--out NAME] [--norm V0|eta3|Lint3|raw]
                          [--band] [--no-guide] [--guide A:B] [--dpi D] [--title T]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl

LABELS = {
    "V0": (r"$V\,/\,V_0$", None),
    "eta3": (r"$V\,/\,\eta^3$", None),
    "Lint3": (r"$V\,/\,(u_{rms}^3 T_L^3)$", None),
    "raw": (r"$V$  (box units)", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(tl.RESULTS, "bif05_volume.npz"))
    ap.add_argument("--out", default="B02_bif05_volume.png")
    ap.add_argument("--norm", choices=list(LABELS), default="V0")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--no-guide", action="store_true")
    ap.add_argument("--guide", default="60:1200")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--figsize", default="6.6,4.6")
    ap.add_argument("--title", default="Tetrahedron volume -- TURB-Lagr tetrads")
    a = ap.parse_args()

    d = np.load(a.inp)
    t_tau = d["t_tau"]
    V = d["mean"]
    V0 = float(d["V0"])
    u, e_li, e_b24, l_int, t_l, eta, dx = d["constants"][:7]
    scale = {"V0": V0, "eta3": eta ** 3, "Lint3": l_int ** 3, "raw": 1.0}[a.norm]
    y = V / scale
    ylabel = LABELS[a.norm][0]

    plt = tl.setup_style()
    fig, ax = plt.subplots(figsize=tuple(float(x) for x in a.figsize.split(",")))
    if a.band and "lo" in d.files:
        ax.fill_between(t_tau, d["lo"] / scale, d["hi"] / scale, color="tab:blue", alpha=0.15, lw=0)
    ax.plot(t_tau, y, color="tab:blue", lw=1.6, marker="o", markevery=120, ms=4.5, label="$\\langle V \\rangle$")
    if not a.no_guide:
        i0, i1 = (int(x) for x in a.guide.split(":"))
        ref = y[i0] / t_tau[i0] ** 4.5
        ax.plot(t_tau[i0:i1], ref * t_tau[i0:i1] ** 4.5, "k--", lw=1.0, label=r"$t^{9/2}$ (Richardson)")
    ax.set_xscale("log"); ax.set_yscale("log")
    if a.ymin:
        ax.set_ylim(bottom=a.ymin)
    if a.xmax:
        ax.set_xlim(right=a.xmax)
    ax.set_xlabel(r"$t\,/\,\tau_\eta$")
    ax.set_ylabel(ylabel)
    ax.set_title(a.title + (f"   ($V_0$ = {V0:.3e})" if a.norm == "V0" else ""))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    fig.tight_layout()
    tl.savefig(fig, a.out, dpi=a.dpi)
    print(f"source npz: {a.inp}  (norm={a.norm}, nq={int(d['nq'])})\n"
          f"constants: u_rms={u:.3f} eps_Li={e_li:.2f} eps_B24={e_b24:.2f} L_int={l_int:.3f} T_L={t_l:.3f} eta={eta:.4f}")


if __name__ == "__main__":
    main()
