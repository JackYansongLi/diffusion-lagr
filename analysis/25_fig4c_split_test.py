#!/usr/bin/env python3
"""25_fig4c_split_test.py -- extended conventions for Fig. 4c: denominator variants x splitting scheme
(ordered against shuffled) x order of the extremes.

Answers two questions raised on 2026-10-07:
  1) does the bar width change if the denominator becomes the **per-component product** S2i*S2j
     instead of the global three-component average S2_avg?
  2) could the paper **shuffle with np.random.shuffle() before cutting the 10 blocks** (random
     subsampling)? An ordered cut would then suffer from the block correlation of "generated in one
     batch";

A key property makes 1/10 of the data sufficient here:
  the mixed statistic is **additive over trajectories**: within a block <di^2 dj^2>_B is the
  arithmetic mean of the per-trajectory time averages,
  since <·> is linear in both trajectory and time and the order is irrelevant. So for the M sliced
  trajectories one computes, once,
  qs[n,li,c] = sum_t dc^2 and qx[n,li,p] = sum_t di^2 dj^2, after which **any** split (ordered, any
  random permutation, bootstrap)
  reproduces the envelope width in seconds.

The slice size M is one block (DM-3c 35,840, DNS 32,768), so the width should be about sqrt(10) ~ 3.16
times that of the full set,
  which also tells us how small a sample the paper would need for DM-3c to recover that factor of 3.

Naming of the denominators:
  D0 = S2_avg of each block (the denominator of the early conventions A/B)
  D1 = the global S2_avg (the denominator of conventions C/G)
  D2 = S2i*S2j of each block (variant 1, per block)
  D3 = the global S2i*S2j (variant 1, global)
Order of the extremes: O1 = average over ij, then take the extremes (form C); O2 = extremes per pair,
then averaged over ij (form G).

Output: results/fig4c_split_<ds>.npz   Usage: python3 25_fig4c_split_test.py --dataset dm3c
"""
import argparse
import os

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
LAGS = np.arange(1, 1025)          # as in script 22: tau/tau_eta = 0.1*lag
PAIRS = ((0, 1), (0, 2), (1, 2))   # xy, xz, yz
TAG = {"dm3c": "DM-3c", "dns": "DNS"}
N_BATCH = 10
N_RAND = 40                        # number of random permutations (gives the width distribution)


def paper_band(ds):
    """The (mean, lo, hi) curves of Fig. 4c of the paper (1024 lags)."""
    fn = os.path.join(DATA, "Li_et_al_2024_source_data", "42256_2024_810_MOESM7_ESM.txt")
    lines = open(fn, errors="replace").read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("# Figure4c_%s.out" % TAG[ds])][0]
    pa = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(1024)])
    return pa[:, 1], pa[:, 1] - pa[:, 2], pa[:, 1] + pa[:, 3]


def paper_width(ds):
    """The median relative full width, (hi-lo)/mean, of Fig. 4c of the paper."""
    m, lo, hi = paper_band(ds)
    return float(np.median((hi - lo) / m))


def get_dataset(name):
    if name == "dns":
        return dict(kind="h5", path=os.path.join(DATA, "Lagr_u3c_diffusion.h5"), M=32768)
    return dict(kind="npy", path=os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy"), M=35840)


def _denorm(name):
    import h5py
    with h5py.File(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), "r") as f:
        return f["min"][()], f["max"][()]


def worker(task):
    """(i0, i1) -> qs(B,nl,3), qx(B,nl,3): the per-trajectory sum_t dc^2 and sum_t di^2 dj^2."""
    i0, i1, ds = task
    mn, mx = _denorm("dm3c")
    mn = np.asarray(mn, dtype=np.float32).reshape(1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, -1)
    if ds["kind"] == "h5":
        import h5py
        with h5py.File(ds["path"], "r") as f:
            u = f["train"][i0:i1].astype(np.float32)
    else:
        u = np.asarray(np.load(ds["path"], mmap_mode="r")[i0:i1], dtype=np.float32)
    v = (u + 1.0) * ((mx - mn) / 2.0) + mn
    del u
    B, K, C = v.shape
    qs = np.zeros((B, len(LAGS), C), dtype=np.float32)
    qx = np.zeros((B, len(LAGS), len(PAIRS)), dtype=np.float32)
    for li, tau in enumerate(LAGS):
        d = v[:, tau:, :] - v[:, :K - tau, :]
        m2 = d * d
        del d
        qs[:, li, :] = m2.sum(axis=1)
        for pi, (i, j) in enumerate(PAIRS):
            qx[:, li, pi] = (m2[:, :, i] * m2[:, :, j]).sum(axis=1)
        del m2
    del v
    return qs, qx


def envelopes(qs, qx, cnt, idx_b):
    """Given a list of block indices, return (lo, hi) for the four denominators x the two orders of the
    extremes."""
    out = {}
    S2g = qs.mean(0) / cnt[:, None]                       # (nl,3) the global per-component S2
    S2g_avg = S2g.mean(1)
    Xall = qx.mean(0) / cnt[:, None]
    Fref = Xall.mean(1) / S2g_avg ** 2
    for dname in ["D0", "D1", "D2", "D3"]:
        Fs = []                                            # Fs[k] = (nl, 3) the F of every block and component pair
        for k in range(len(idx_b)):
            Xk = qx[idx_b[k]].mean(0) / cnt[:, None]        # (nl,3)
            S2b = qs[idx_b[k]].mean(0) / cnt[:, None]       # (nl,3)
            if dname == "D0":
                den = np.repeat((S2b.mean(1) ** 2)[:, None], 3, axis=1)
            elif dname == "D1":
                den = np.repeat((S2g_avg ** 2)[:, None], 3, axis=1)
            elif dname == "D2":
                den = np.stack([S2b[:, i] * S2b[:, j] for i, j in PAIRS], axis=1)
            else:
                den = np.stack([S2g[:, i] * S2g[:, j] for i, j in PAIRS], axis=1)
            Fs.append(Xk / den)
        Fs = np.array(Fs)                                  # (nb, nl, 3)
        out[dname + "-O1"] = (Fs.mean(2).min(0), Fs.mean(2).max(0))
        out[dname + "-O2"] = (Fs.min(0).mean(1), Fs.max(0).mean(1))
    return out, Fref


def widths(env, Fref):
    return {k: float(np.median((v[1] - v[0]) / Fref)) for k, v in env.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--block", type=int, default=4096)
    args = ap.parse_args()
    ds = get_dataset(args.dataset)
    M = ds["M"]
    print("[%s] slice of M=%d trajectories (1/10 of the full set; expect about 3.16x the width) " % (TAG[args.dataset], M), flush=True)

    tasks = [(i, min(i + args.block, M), ds) for i in range(0, M, args.block)]
    if args.workers > 1:                                   # 6 x 98 MB blocks, safe under WSL
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            res = pool.map(worker, tasks)
    else:
        res = [worker(t) for t in tasks]
    qs = np.concatenate([r[0] for r in res], 0)
    qx = np.concatenate([r[1] for r in res], 0)
    del res
    cnt = (2000 - LAGS).astype(np.float64)                 # the number of sample pairs per lag, per trajectory
    pw = paper_width(args.dataset)
    print("  relative full width of the paper (median) %.4f" % pw, flush=True)

    # ordered cut
    idx0 = [np.arange(k * M // N_BATCH, (k + 1) * M // N_BATCH) for k in range(N_BATCH)]
    env0, Fref = envelopes(qs, qx, cnt, idx0)
    w0 = widths(env0, Fref)
    # shuffled cuts, repeated, to get a distribution
    rng = np.random.default_rng(0)
    rand_seeds = []
    rand_env = {k: ([], []) for k in env0}                # collect the envelope of every shuffled draw
    for r in range(N_RAND):
        perm = rng.permutation(M)
        idxr = [perm[k * M // N_BATCH:(k + 1) * M // N_BATCH] for k in range(N_BATCH)]
        e, _ = envelopes(qs, qx, cnt, idxr)
        rand_seeds.append(widths(e, Fref))
        for k in env0:
            rand_env[k][0].append(e[k][0])
            rand_env[k][1].append(e[k][1])
    med_env = {k: (np.median(rand_env[k][0], 0), np.median(rand_env[k][1], 0)) for k in env0}
    print("  %-8s %-12s %-12s %-8s %-10s" % ("conv.", "ordered width", "shuffled (med.)",
                                            "rand/ord", "ours/paper"), flush=True)
    for k in ["D1-O1", "D1-O2", "D3-O1", "D3-O2", "D0-O1", "D2-O1"]:
        r = np.array([d[k] for d in rand_seeds])
        print("  %-8s %-12.4f %-12.4f %-8.2f %-10.2f" % (k, w0[k], np.median(r), np.median(r) / w0[k], w0[k] / pw), flush=True)
    lo95 = {k: (np.percentile([d[k] for d in rand_seeds], 5), np.percentile([d[k] for d in rand_seeds], 95))
            for k in ["D1-O1", "D1-O2"]}
    for k, v in lo95.items():
        print("  [%s] shuffled 5-95%% interval: %.4f - %.4f" % (k, v[0], v[1]), flush=True)
    # invert: the effective sample size behind the width of the paper, with width ~ 1/sqrt(N)
    fac = w0["D1-O1"] / pw
    print("  => if the paper used the same convention on fewer %s samples: N_eff ~ M / %.2f^2 = %.0f (this slice holds %d)"
          "   (valid only for datasets **without index-block correlation**; the DNS width is set by the block structure, so it does not apply there)"
          % (TAG[args.dataset], fac, M / fac ** 2, M), flush=True)

    out = os.path.join(RESULTS, "fig4c_split_%s.npz" % args.dataset)
    nm = list(env0.keys())
    pm, plo, phi = paper_band(args.dataset)
    np.savez_compressed(out, tau=LAGS * 0.1, Fref=Fref, M=M, paper_width=pw,
                        names=np.array(nm),
                        seq_los=np.array([env0[k][0] for k in nm]),
                        seq_his=np.array([env0[k][1] for k in nm]),
                        rand_los=np.array([med_env[k][0] for k in nm]),
                        rand_his=np.array([med_env[k][1] for k in nm]),
                        paper_mean=pm, paper_lo=plo, paper_hi=phi,
                        w_seq=np.array([w0[k] for k in w0]),
                        w_rand=np.array([[d[k] for k in w0] for d in rand_seeds]))
    print("  wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
