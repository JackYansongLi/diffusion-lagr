"""44_bif05_volume_data.py -- DATA stage for the tetrad VOLUME V(t)   (figure B02).

V = (1/3)|det(rho)| is the volume of the four-particle tetrahedron (Biferale et al. 2005 definition;
van Aartrijk et al. 2008 write V = |det r| -- the two differ by a constant factor 3, see
TETRAD_DM.md section 3.4(2)).  Internally cross-checked against V = (1/3) sqrt(g1 g2 g3), and the
initial value against the closed form V0 = h^3/6 = dx^3/48 (right-corner tetrahedron, legs h = dx/2).

Stored in PHYSICAL (box) units: every normalisation -- V/V0, V/eta^3, V/(u_rms^3 T_L^3) -- is applied
in the plot stage, so changing the axis never re-runs this 5-minute sweep.

Usage: 44_bif05_volume_data.py [--nq N] [--chunk C] [--tag NAME] [--source dns|integrated]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=tl.NQ)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--source", choices=["dns", "integrated"], default="dns")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""

    nblk = 1024
    sum_V = np.zeros((nblk, tl.NT))
    sum_V2 = np.zeros((nblk, tl.NT))
    cnt = np.zeros(nblk)
    chk_med, chk_p99, chk_bad, n_samp = [], [], 0, 0
    nq_done = 0
    t0 = time.time()
    idx = tl.balanced(a.nq)
    print(f"  sample: {idx.size} quartets, {len(np.unique(tl.pencil_column(4 * idx)))} distinct pencil columns")
    for sl, rho, W in tl.iter_rhoW(idx, chunk=a.chunk, source=a.source):
        # Vectorised exact 3x3 determinant (cofactor expansion): np.linalg.det() loops in Python over
        # small matrices and takes ~13 min for the full sweep, whereas the analytic form
        # V = (1/3)sqrt(g1 g2 g3) loses relative accuracy for near-degenerate rho (g3 at the float
        # noise floor => V ~ 0 there, harmless for the mean, but the identity check below must not be
        # judged by its max over such samples).
        r00, r01, r02 = rho[..., 0, 0], rho[..., 0, 1], rho[..., 0, 2]
        r10, r11, r12 = rho[..., 1, 0], rho[..., 1, 1], rho[..., 1, 2]
        r20, r21, r22 = rho[..., 2, 0], rho[..., 2, 1], rho[..., 2, 2]
        det = (r00 * (r11 * r22 - r12 * r21) - r01 * (r10 * r22 - r12 * r20)
               + r02 * (r10 * r21 - r11 * r20))
        g = tl.inertia(rho)[0]
        V = np.abs(det) / 3.0                                     # (n, T)
        chk = np.abs(np.abs(det) - np.sqrt(np.prod(g, axis=-1))) / np.maximum(3.0 * V, 1e-300)
        chk_med.append(float(np.median(chk))); chk_p99.append(float(np.percentile(chk, 99)))
        chk_bad += int((chk > 1e-8).sum()); n_samp += chk.size
        blk = tl.pencil_column(4 * sl)
        for i in range(V.shape[0]):
            sum_V[blk[i]] += V[i]
            sum_V2[blk[i]] += V[i] ** 2
            cnt[blk[i]] += 1
        nq_done += V.shape[0]
        el = time.time() - t0
        print(f"  {nq_done:6d}/{a.nq} quartets  el {el/60:4.1f}min  eta {el/nq_done*(a.nq-nq_done)/60:4.1f}min", flush=True)

    used = cnt > 0
    blk_mean = sum_V[used] / cnt[used][:, None]
    blk_var = sum_V2[used] / cnt[used][:, None] - blk_mean ** 2
    mean = blk_mean.mean(0)
    lo, hi = blk_mean.min(0), blk_mean.max(0)
    std = np.sqrt(np.maximum(blk_var, 0)).mean(0)
    sd_blk = blk_mean.std(0)                                     # spread of the block means
    t_phys = np.arange(tl.NT) * tl.DTS
    t_tau = t_phys / tl.TAU_ETA

    h = tl.DX / 2
    V0_cf = h ** 3 / 6.0
    V0 = mean[0]
    print(f"\nidentity check |det rho| = sqrt(g1 g2 g3): median {max(chk_med):.2e}, p99 {max(chk_p99):.2e}, "
          f"rel dev > 1e-8: {chk_bad}/{n_samp} (only near-degenerate rho with V ~ 0)")
    print(f"V(0): measured {V0:.6e}   closed form h^3/6 = {V0_cf:.6e}   ratio {V0/V0_cf:.4f}")
    print(f"constants: u_rms = {tl.U_RMS}, eps = {tl.EPS_LI} (Li; Biferale2024 gives {tl.EPS_B24}), "
          f"L_int = {tl.L_INT:.3f}, T_L = {tl.T_L:.3f}, L_int^3 = {tl.L_INT**3:.2f}")
    print(f"  V0/eta^3 = {V0/tl.ETA**3:.4f}   V0/L_int^3 = {V0/tl.L_INT**3:.3e}")
    print("  (VA08 Fig.13a starts at 1e-8 because their initial separations are ~(1/6)eta..15eta; ours is h=1.46eta,")
    print("   i.e. V0/L_int^3 = 1.9e-10, so that axis must be extended downwards to show our initial point)")
    print("\nlocal log-slope d ln<V>/d ln t (Richardson expectation 9/2):")
    for (w0, w1) in [(50, 200), (200, 600), (600, 1200), (1200, 1999)]:
        sl = (np.log(mean[w1]) - np.log(mean[w0])) / (np.log(t_tau[w1]) - np.log(t_tau[w0]))
        print(f"   t={w0:4d}-{w1:4d} (tau_eta {t_tau[w0]:6.1f}-{t_tau[w1]:6.1f}): slope = {sl:.2f}")

    os.makedirs(tl.RESULTS, exist_ok=True)
    out = os.path.join(tl.RESULTS, f"bif05_volume{tag}.npz")
    np.savez_compressed(out, t_phys=t_phys, t_tau=t_tau, mean=mean, lo=lo, hi=hi, std=std, sd_blk=sd_blk,
                        blk_mean=blk_mean, blk_labels=np.where(used)[0], blk_cnt=cnt[used], source=a.source,
                        V0=V0, V0_closed_form=V0_cf, nq=nq_done,
                        constants=np.array([tl.U_RMS, tl.EPS_LI, tl.EPS_B24, tl.L_INT, tl.T_L, tl.ETA, tl.DX]))
    print("data ->", out)
    print("figure: run 44_bif05_volume_plot.py (normalisation and style are free)")


if __name__ == "__main__":
    main()
