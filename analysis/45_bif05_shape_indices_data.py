"""45_bif05_shape_indices_data.py -- DATA stage for the shape indices I_i(t)   (figure B03).

I_i = g_i / r^2  (i = 1,2,3; sum_i I_i = 1), computed PER TETRAD and then averaged -- never as a ratio of
means.  Two calibres are accumulated in the same pass:

  (1) unconditional      : all tetrads
  (2) B05-selected       : tetrads with eigenvalues inside the ranges quoted by Biferale et al. (2005)
                           5e2 <= g1/eta^2 <= 5e5,  5e1 <= g2/eta^2 <= 5e4,  5 <= g3/eta^2 <= 5e3
                           (their eta = 0.005; we convert with OUR eta = 0.0042, see TETRAD_DM.md section 3.0(e)).
                           B05 report the plateau of the SELECTED subset: <I2> = 0.135 +- 0.003,
                           <I3> = 0.011 +- 0.001 -- NOT universal constants.

Reference values for the plot stage: initial shape I(0) = (4/9, 4/9, 1/9) (right-corner tetrad, exact),
and the isotropic-Gaussian values <I1,I2,I3> = 0.748/0.222/0.03 (Pumir et al.; Biferale et al.; quoted
in van Aartrijk et al. 2008 Fig. 13(b)).  This script re-derives the Gaussian values by Monte Carlo
(both Gaussian positions and uniform-in-box positions) as an independent check of the geometry pipeline.

Usage: 45_bif05_shape_indices_data.py [--nq N] [--chunk C] [--tag NAME] [--source dns|integrated]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl

# B05 eigenvalue selection ranges, in units of eta^2 (converted with our eta)
G1_LO, G1_HI = 5e2, 5e5
G2_LO, G2_HI = 5e1, 5e4
G3_LO, G3_HI = 5.0, 5e3


def gaussian_reference(n=400000, seed=0):
    """MC values of <I_i> for 4 independent Gaussian points (as used by Pumir/Biferale) and for 4
    independent uniform points in the box, i.e. the two natural 'uncorrelated' references."""
    rng = np.random.default_rng(seed)
    out = {}
    for lbl, X in (("gaussian", rng.normal(size=(n, 4, 3))),
                   ("uniform_box", rng.uniform(0.0, tl.LBOX, size=(n, 4, 3)))):
        g, r2, Ii = tl.inertia(tl.jacobi_rho(X[:, None, :, :]))
        out[lbl] = Ii[:, 0].mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=tl.NQ)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--source", choices=["dns", "integrated"], default="dns")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""

    e2 = tl.ETA ** 2
    lim = np.array([[G1_LO * e2, G1_HI * e2], [G2_LO * e2, G2_HI * e2], [G3_LO * e2, G3_HI * e2]])

    nblk = 1024
    unc_sum = np.zeros((nblk, tl.NT, 3))
    sel_sum = np.zeros((nblk, tl.NT, 3))
    cnt = np.zeros(nblk)
    sel_cnt = np.zeros((nblk, tl.NT))
    pass_tot = np.zeros(tl.NT)
    nq_done = 0
    t0 = time.time()
    idx = tl.balanced(a.nq)
    print(f"selection ranges (box units): g1 [{lim[0,0]:.4g},{lim[0,1]:.4g}]  "
          f"g2 [{lim[1,0]:.4g},{lim[1,1]:.4g}]  g3 [{lim[2,0]:.4g},{lim[2,1]:.4g}]")
    for sl, rho, W in tl.iter_rhoW(idx, chunk=a.chunk, source=a.source):
        g, r2, Ii = tl.inertia(rho)                      # (n,T,3) each
        m = ((g[..., 0] >= lim[0, 0]) & (g[..., 0] <= lim[0, 1])
             & (g[..., 1] >= lim[1, 0]) & (g[..., 1] <= lim[1, 1])
             & (g[..., 2] >= lim[2, 0]) & (g[..., 2] <= lim[2, 1]))
        pass_tot += m.sum(0)
        blk = tl.pencil_column(4 * sl)
        for b in np.unique(blk):
            sb = blk == b
            unc_sum[b] += Ii[sb].sum(0)
            cnt[b] += sb.sum()
            sel_sum[b] += np.where(m[sb][..., None], Ii[sb], 0.0).sum(0)
            sel_cnt[b] += m[sb].sum(0)
        nq_done += Ii.shape[0]
        el = time.time() - t0
        print(f"  {nq_done:6d}/{a.nq} quartets  el {el/60:4.1f}min  eta {el/nq_done*(a.nq-nq_done)/60:4.1f}min", flush=True)

    used = cnt > 0
    blk_mean = unc_sum[used] / cnt[used][:, None, None]
    mean = blk_mean.mean(0)
    lo, hi = blk_mean.min(0), blk_mean.max(0)
    sd_blk = blk_mean.std(0)
    tot = cnt[used].sum()

    sc = sel_cnt[used].sum(0)                                  # (T,)
    sel_mean = sel_sum[used].sum(0) / np.maximum(sc, 1)[:, None]
    pass_frac = pass_tot / tot
    good = sc > 1000                                           # only report where the subset is meaningful
    t_phys = np.arange(tl.NT) * tl.DTS
    t_tau = t_phys / tl.TAU_ETA

    exact0 = np.array([4.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0])
    print(f"\nI(0) measured = {np.array2string(mean[0], precision=5)}   exact (4/9,4/9,1/9) = "
          f"{np.array2string(exact0, precision=5)}   sum = {mean[0].sum():.6f}")
    mon = mean[:, 1].argmin()
    print(f"<I2> minimum: {mean[mon, 1]:.5f} at t = {t_phys[mon]:.4f} = {t_tau[mon]:.1f} tau_eta")
    # Crossing diagnostics.  I1 starts BELOW its Gaussian value and rises through it, while I2 and I3
    # start ABOVE theirs, fall through, and only come back late in the run -- so report both a downward
    # and an upward crossing; a bare "mean >= target" test would just return t = 0 for I2 and I3.
    for j, tgt in ((0, 0.748), (1, 0.222), (2, 0.03)):
        y = mean[:, j]
        start = "above" if y[0] > tgt else "below"
        sgn = np.sign(y - tgt)
        ch = np.where(np.diff(sgn) != 0)[0]                     # sign changes: true crossings only
        s = f"  <I{j+1}> vs Gaussian {tgt} (starts {start}): "
        s += ", ".join(f"{'down' if sgn[k] > 0 else 'up'} at t = {t_tau[k]:.2f} tau_eta" for k in ch[:4])
        print(s if ch.size else s + f"no crossing in window (min {y.min():.5f}, max {y.max():.5f})")
    print("\ng3 lower threshold 5 eta^2 = %.4g; our g3(0) = %.4g (%.3f eta^2) => initial tetrads are all excluded"
          % (5 * e2, 0.25 * (tl.DX / 2) ** 2, 0.25 * (tl.DX / 2) ** 2 / e2))
    print("\n  t   t/tau    <I1>     <I2>     <I3>   |  sel <I1>  sel <I2>  sel <I3>  pass%  n_sel")
    for t in [0, 50, 200, 400, 630, 1000, 1500, 1999]:
        sb = sel_mean[t] if good[t] else np.array([np.nan] * 3)
        print(f"{t:5d} {t_tau[t]:7.1f}  {mean[t,0]:.5f}  {mean[t,1]:.5f}  {mean[t,2]:.5f}  |  "
              f"{sb[0]:.5f}   {sb[1]:.5f}   {sb[2]:.5f}   {100*pass_frac[t]:5.1f}  {int(sc[t])}")
    i35 = int(np.argmax(t_tau >= 35))
    row = np.where(good[i35:][:, None], sel_mean[i35:], np.nan)
    plat = np.nanmean(row, axis=0)
    print(f"\n  selected plateau, t >= 35 tau_eta (t index {i35}..1999): "
          f"<I1> = {plat[0]:.4f}  <I2> = {plat[1]:.4f}  <I3> = {plat[2]:.4f}"
          f"   [B05 reports <I2> = 0.135 +- 0.003, <I3> = 0.011 +- 0.001]")
    gref = gaussian_reference()
    print(f"\n  MC Gaussian reference <I_i>: {np.array2string(gref['gaussian'], precision=4)}"
          f"   (literature 0.748/0.222/0.03)")
    print(f"  MC uniform-in-box   <I_i>: {np.array2string(gref['uniform_box'], precision=4)}")

    os.makedirs(tl.RESULTS, exist_ok=True)
    out = os.path.join(tl.RESULTS, f"bif05_shape_indices{tag}.npz")
    np.savez_compressed(out, t_phys=t_phys, t_tau=t_tau, mean=mean, lo=lo, hi=hi, sd_blk=sd_blk,
                        sel_mean=sel_mean, sel_ok=good, sel_count=sc, pass_frac=pass_frac,
                        blk_mean=blk_mean, blk_labels=np.where(used)[0], blk_cnt=cnt[used],
                        exact0=exact0, gaussian_ref=gref["gaussian"], uniform_ref=gref["uniform_box"],
                        lim=lim, eta=tl.ETA, nq=nq_done, source=a.source, t_min_I2=float(t_phys[mon]))
    print("\ndata ->", out)
    print("figure: run 45_bif05_shape_indices_plot.py (style is free)")


if __name__ == "__main__":
    main()
