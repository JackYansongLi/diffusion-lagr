"""Small original estimator study on real bundled DNS data, not a model improvement."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .data import small_hdf5
from .provenance import ROOT, manifest, sha256, write_json
from .statistics import acceleration, denormalize, local_exponents, trajectory_moments


def single_time_features(u):
    squares = u**2
    q = squares.mean(axis=1)
    cross = np.stack([np.mean(squares[:, :, i] * squares[:, :, j], axis=1)
                      for i, j in ((0, 1), (0, 2), (1, 2))], axis=1)
    return np.concatenate([q, cross], axis=1)


def single_time_from_features(features):
    mean = np.mean(features, axis=0)
    return np.array([mean[:3].mean(), np.mean(mean[3:] - mean[[0, 0, 1]] * mean[[1, 2, 2]])])


def bootstrap_single_time(velocity, acc, seed, replicates):
    vf, af = single_time_features(velocity), single_time_features(acc)
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 4))
    for b in range(replicates):
        ids = rng.integers(len(vf), size=len(vf))
        e, sv = single_time_from_features(vf[ids])
        a, sa = single_time_from_features(af[ids])
        values[b] = e, a, sv, sa
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("repro/results"))
    p.add_argument("--figures", type=Path, default=Path("repro/figures"))
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    started = time.monotonic()
    source = ROOT / "datasets/Lagr_u3c_diffusion-demo.h5"
    d, backend = small_hdf5(source)
    u = denormalize(d["train"], d["min"], d["max"])
    acc = acceleration(u, 1., "gradient")
    boot = bootstrap_single_time(u, acc, args.seed, args.bootstrap)
    e, sv = single_time_from_features(single_time_features(u))
    a, sa = single_time_from_features(single_time_features(acc))
    point = np.array([e, a, sv, sa])
    ci = np.quantile(boot, [.025, .975], axis=0)
    # All three scalar components stay within the same physical-trajectory unit.
    moments = trajectory_moments(u, [1, 10, 100])
    per_trace = moments.mean(axis=2)
    full = per_trace.mean(axis=0)
    rng = np.random.default_rng(args.seed)
    sizes = [16, 32, 64, 128, 256]
    subsamples = np.empty((len(sizes), 200, 3, 4))
    for k, n in enumerate(sizes):
        for b in range(200):
            ids = rng.choice(len(u), size=n, replace=False)
            subsamples[k, b] = per_trace[ids].mean(axis=0)
    relative = subsamples / full
    top3 = np.sort(per_trace, axis=0)[-3:].sum(axis=0) / per_trace.sum(axis=0)
    loo = (per_trace.sum(axis=0)[None, ...] - per_trace) / (len(u) - 1)
    max_loo = np.max(np.abs(loo / full - 1), axis=0)
    # Dense paper lag grid, but only on the bundled DNS subset, clearly labelled.
    dense_started = time.monotonic()
    lags = np.arange(1, 1025)
    dense = trajectory_moments(u, lags, orders=(2, 4)).mean(axis=(0, 2))
    local = local_exponents(dense, lags)[:, 1]
    log_s = np.log(dense)
    secant = (log_s[2:, 1] - log_s[:-2, 1]) / (log_s[2:, 0] - log_s[:-2, 0])
    dense_seconds = time.monotonic() - dense_started
    sparse_lags = np.unique(np.r_[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1024])
    sparse_local = local_exponents(dense[sparse_lags - 1], sparse_lags)[:, 1]
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    raw_path = args.results / "estimator_exploration_raw.npz"
    np.savez(raw_path, bootstrap_single_time=boot, subsample_structure=subsamples,
             sample_sizes=sizes, subsample_lags=[1, 10, 100], orders=[2, 4, 6, 8],
             dense_lags=lags, dense_structure=dense, dense_local_zeta4=local,
             secant_local_zeta4=secant, sparse_lags=sparse_lags, sparse_local_zeta4=sparse_local)
    summary = {
        "classification": "original_estimator_exploration_on_bundled_DNS_demo_NOT_model_improvement",
        "seed": args.seed, "physical_trajectories": len(u), "bootstrap_replicates": args.bootstrap,
        "single_time": {name:{"point":float(point[k]), "conditional_ci95":ci[:, k].tolist()}
                        for k,name in enumerate(("E", "A_sample_gradient", "Sigma_V", "Sigma_A_sample_gradient"))},
        "derivative": "np.gradient(u, 1.0, axis=1, edge_order=1), includes all 2000 times",
        "physical_time_conversion": {"dt":.00225, "A_factor":.00225**-2, "Sigma_A_factor":.00225**-4},
        "subsample_sizes": sizes, "subsample_replicates": 200, "subsample_lags": [1, 10, 100],
        "orders": [2, 4, 6, 8],
        "relative_subsample_q05_q50_q95": np.quantile(relative, [.05, .5, .95], axis=1).tolist(),
        "subsample_coefficient_of_variation": (subsamples.std(axis=1, ddof=1) / subsamples.mean(axis=1)).tolist(),
        "top_three_trajectory_fraction_of_total_moment": top3.tolist(),
        "max_leave_one_trajectory_out_relative_change": max_loo.tolist(),
        "dense_lag_grid": [1, 1024], "dense_grid_runtime_seconds":dense_seconds,
        "local_derivative_discretization_max_abs_difference_interior":float(np.max(np.abs(local[1:-1] - secant))),
        "sparse_grid_max_abs_difference_at_common_lags":float(np.max(np.abs(local[sparse_lags - 1] - sparse_local))),
        "runtime_seconds":time.monotonic() - started, "hdf5_backend":backend,
        "interpretation_limits":["Subset variation is conditional on these 256 traces, not convergence to a full-DNS truth.", "Physical traces share one DNS flow; exchangeability is an assumption, not demonstrated independence.", "Bootstrap intervals exclude between-training-seed and between-DNS uncertainty.", "No DM parameters changed or trained; this is an estimator study, not a claim of improved generation.", "Sparse grids can bias local exponents; dense integer grid used for the measured comparison."]}
    summary_path = args.results / "estimator_exploration.json"
    write_json(summary_path, summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), layout="constrained")
    q = np.quantile(relative[:, :, 1, :], [.05, .5, .95], axis=1)
    colors = ["#2563a6", "#df7d28", "#28967c", "#a44977"]
    for j, order in enumerate([2, 4, 6, 8]):
        axes[0].plot(sizes, q[1, :, j], "o-", color=colors[j], label=f"p={order}", ms=3)
        axes[0].fill_between(sizes, q[0, :, j], q[2, :, j], color=colors[j], alpha=.12)
    axes[0].axhline(1, color="0.6", lw=.8, ls=":")
    axes[0].set(xscale="log", xlabel="Physical trajectories sampled", ylabel="Moment / full-demo moment", title="Subset sensitivity at lag 10")
    axes[0].legend(frameon=False, fontsize=8)
    t = lags * .00225 / .021
    axes[1].semilogx(t, local, color=colors[0], label="Dense integer lags, log-grid derivative")
    axes[1].semilogx(t[1:-1], secant, color=colors[1], ls="--", label="Dense central-secant ratio")
    axes[1].semilogx(t[sparse_lags - 1], sparse_local, "o-", color=colors[3], ms=3, label="Sparse-grid derivative")
    axes[1].set(xlabel=r"$\tau/\tau_\eta$", ylabel=r"$\zeta(4,\tau)$", title="Local-slope estimator sensitivity")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Original analysis on 256 DNS demo traces; no model-performance claim", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(args.figures / f"demo_estimator_sensitivity.{suffix}", dpi=180)
    plt.close(fig)
    record = manifest(f"python -m repro.explore_estimators --seed {args.seed} --bootstrap {args.bootstrap}",
                      inputs=[source], seed=args.seed, classification=summary["classification"])
    outputs = [raw_path, summary_path] + list(args.figures.glob("demo_estimator_sensitivity.*"))
    record["outputs"] = [{"name":str(path), "sha256":sha256(path)} for path in outputs]
    write_json(args.results / "exploration_run_manifest.json", record)
    print(json.dumps({"single_time":summary["single_time"], "runtime_seconds":summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
