"""70_fig1f_3dtraj_style.py -- Fig.1f-style 3D trajectory colouring, matching Li et al. (2024).

The published figure colours the trajectory by |a_i| with a blue-white-red scale and ticks
at 0.1, 0.5 and 1.0. Neither the caption nor the Methods state how |a_i| is normalised, nor
which component i is meant (the label suggests a single component). This script therefore
exposes both choices and keeps everything else identical:

  --comp mag : c = |a|   (vector magnitude, i.e. all three components)
  --comp x|y|z: c = |a_i| (one component, as the published label reads)
  --norm p99 : normalise by the 99th percentile, vmax 1.0 (tick 1.0 = that percentile)
  --norm rms : normalise by the rms,              vmax 1.2 (tick 1.0 = one rms)
  --ref self : reference computed on the same quantity as the colour (per-component)
  --ref mag  : reference always the magnitude's, so the three components share one scale

The reference is always the DNS trajectory, so both panels share one scale -- the published
colour bar is shared as well. Colormap, tick positions and the colour-bar label follow the
published panel; the layout follows A9_fig1f_3dtraj_mod (colour bar in its own axes strip).

usage: run.bat 70_fig1f_3dtraj_style.py [--traj-idx 137] [--norm p99|rms]
                                        [--comp mag|x|y|z] [--ref self|mag]
output: A9_fig1f_3dtraj_<norm>[_<comp>].png
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs")
os.makedirs(FIGS, exist_ok=True)

DTS = 2.25e-3          # storage interval in code units (= 0.1 τη)
ETA = 4.2e-3           # Kolmogorov length in code units


def denorm_npy(path, mn, mx, idx, k):
    a = np.load(path, mmap_mode="r")
    u = np.asarray(a[idx:idx + k], dtype=np.float32)
    mn = np.asarray(mn, dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, 1, -1)
    return (u + 1.0) * ((mx - mn) / 2.0) + mn


def denorm_h5(path, idx, k):
    with h5py.File(path, "r") as f:
        mn = f["min"][()]
        mx = f["max"][()]
        u = f["train"][idx:idx + k].astype(np.float32)
    mn = np.asarray(mn, dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, 1, -1)
    return (u + 1.0) * ((mx - mn) / 2.0) + mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-idx", type=int, default=137)
    ap.add_argument("--n-plot", type=int, default=2000)
    ap.add_argument("--norm", choices=["p99", "rms"], default="p99")
    ap.add_argument("--comp", choices=["mag", "x", "y", "z"], default="mag")
    ap.add_argument("--ref", choices=["self", "mag"], default="self")
    args = ap.parse_args()
    i0, K = args.traj_idx, args.n_plot
    comp = {"mag": None, "x": 0, "y": 1, "z": 2}[args.comp]

    with h5py.File(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), "r") as f:
        mn3 = f["min"][()]
        mx3 = f["max"][()]
    vdm = denorm_npy(os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy"),
                     mn3, mx3, i0, 1)[0, :K, :]
    vdns = denorm_h5(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), i0, 1)[0, :K, :]

    def accel_of(v, c):
        """|a| (c is None) or |a_i| for one component, one value per stored step."""
        dv = np.diff(v, axis=0) / DTS
        a = np.linalg.norm(dv, axis=1) if c is None else np.abs(dv[:, c])
        return np.concatenate(([0.0], a))

    # diagnostics: how different are the four candidate references?
    stats = {n: accel_of(vdns, c) for n, c in
             [("mag", None), ("x", 0), ("y", 1), ("z", 2)]}
    print("DNS reference candidates (code units):")
    for n, a in stats.items():
        print("  %-3s p99 = %8.3f   rms = %8.3f" %
              (n, np.quantile(a, 0.99), np.sqrt(np.mean(a ** 2))))

    trajs = []
    for v, name in [(vdns, "DNS"), (vdm, "DM-3c")]:
        pos = np.cumsum(v, axis=0) * DTS / ETA                  # positions in η
        trajs.append((pos, accel_of(v, comp), name))

    ref_src = stats[args.comp] if (args.ref == "self" or comp is None) else stats["mag"]
    if args.norm == "p99":
        ref, vmax = float(np.quantile(ref_src, 0.99)), 1.0
    else:
        ref, vmax = float(np.sqrt(np.mean(ref_src ** 2))), 1.2

    # 2026-10-22: the DNS and DM-3c panels are stacked (they were side by side), so the shape of the
    # trajectory can be compared from one viewpoint
    fig = plt.figure(figsize=(5.4, 7.0))
    # the panel label sat to the right of each 3D axis (text2D), which costs no vertical space and lets the
    # two panels sit closer
    boxes = ([0.010, 0.508, 0.700, 0.478], [0.010, 0.014, 0.700, 0.478])
    for k, (pos, acc, name) in enumerate(trajs):
        ax = fig.add_axes(boxes[k], projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=acc / ref, s=1.5,
                   cmap="coolwarm", vmin=0.0, vmax=vmax)
        # the label goes in the free space right of the axis (figure coordinates) so it does not collide with the
        # z tick labels
        fig.text(boxes[k][0] + boxes[k][2] + 0.012,
                 boxes[k][1] + 0.5 * boxes[k][3], name,
                 fontsize=11, ha="left", va="center")
        ax.set_xlabel("x/η", fontsize=10)
        ax.set_ylabel("y/η", fontsize=10)
        ax.set_zlabel("z/η", fontsize=10)   # 2026-10-22: the z axis carried no label
        ax.tick_params(labelsize=8.5)
        try:
            ax.set_box_aspect(None, zoom=0.86)
        except TypeError:
            pass

    # the colour bar is shortened and centred vertically (it used to span almost the whole column)
    cax = fig.add_axes([0.880, 0.330, 0.016, 0.340])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=[0.1, 0.5, 1.0])
    cb.set_label("$|a_i|$", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    # a narrower canvas would squeeze or clip a long suptitle; the panel titles already read DNS / DM-3c and
    # the caption states the convention
    fig.suptitle("")
    fig.text(0.012, 0.992, "(c) Fig.1f reproduction: 3D trajectories",
             ha="left", va="top", fontsize=13)
    suffix = args.norm if comp is None else "%s_%s" % (args.norm, args.comp)
    if args.ref == "mag" and comp is not None:
        suffix += "_refmag"
    p = os.path.join(FIGS, "A9_fig1f_3dtraj_%s_mod.png" % suffix)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p, "| component =", args.comp, "| reference =", round(ref, 3),
          "code units, vmax", vmax)


if __name__ == "__main__":
    main()
