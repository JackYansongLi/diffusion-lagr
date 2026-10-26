"""Measure the bundled DNS demo; outputs are explicitly not paper reproduction."""
import argparse
from pathlib import Path
import time
import numpy as np

from .data import small_hdf5
from .provenance import ROOT, manifest, sha256, write_json
from .statistics import (acceleration, denormalize, flatness, mixed_trajectory_moments,
                         paper_batch_range, trajectory_bootstrap, trajectory_moments,
                         single_time_statistics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("repro/results"))
    parser.add_argument("--seed", type=int, default=230708529)
    args = parser.parse_args()
    started = time.monotonic()
    source3 = ROOT / "datasets/Lagr_u3c_diffusion-demo.h5"
    source1 = ROOT / "datasets/Lagr_u1c_diffusion-demo.h5"
    d3, backend = small_hdf5(source3)
    d1, _ = small_hdf5(source1)
    u = denormalize(d3["train"], d3["min"], d3["max"])
    u1 = denormalize(d1["train"], d1["min"], d1["max"])
    # The three scalar records refer to the same underlying physical trace.
    flattened = np.concatenate([u[..., i:i+1] for i in range(3)], axis=0)
    normalization_error = float(np.max(np.abs(flattened - u1)))
    if normalization_error > 1e-10:
        raise ValueError("one/three component demos do not identify the same raw velocities")
    lags = np.array([1, 2, 5, 10, 20, 50, 100, 500, 1000])
    moments = trajectory_moments(u, lags)
    mixed = mixed_trajectory_moments(u, lags)
    s = moments.mean(axis=(0, 2))
    envelope = paper_batch_range(moments, seed=args.seed)
    ci = trajectory_bootstrap(moments, seed=args.seed, replicates=400)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "demo_sufficient_statistics.npz"
    np.savez(raw, lags=lags, orders=[2, 4, 6, 8], trajectory_moments=moments,
             mixed_trajectory_moments=mixed, physical_trajectory_ids=np.arange(len(u)))
    summary = {
        "classification": "bundled_dns_demo_pipeline_validation_NOT_paper_reproduction",
        "shape": list(u.shape), "physical_trajectories": len(u),
        "dependent_scalar_component_records": len(u1),
        "normalization_max_abs_roundtrip_difference": normalization_error,
        "sampling_dt": .00225, "tau_eta_nominal": .021,
        "lag_samples": lags.tolist(),
        "lag_over_tau_eta_using_table_I": (lags * .00225 / .021).tolist(),
        "orders": [2, 4, 6, 8], "structure": s.tolist(),
        "flatness": flatness(s).tolist(),
        "mixed_flatness": (mixed.mean(axis=(0, 2)) / s[:, 0]**2).tolist(),
        "paper_style_minmax": {k:v.tolist() if isinstance(v, np.ndarray) else v
                               for k,v in envelope.items() if k != "curves"},
        "conditional_trajectory_bootstrap": {k:v.tolist() for k,v in ci.items()},
        "bootstrap_replicates": 400,
        "uncertainty_scope": "400 percentile bootstrap resamples of physical trajectories, keeping all times and components together. Conditional on exchangeable traces from this one DNS realization. No training-seed, between-DNS, systematic-discretization, or rare-tail convergence uncertainty.",
        "energy_component_mean_square": float(np.mean(u*u)),
        "component_mean_square": np.mean(u*u, axis=(0, 1)).tolist(),
        "acceleration_variance_by_convention": {
            f"{method}_dt_{dt}": float(np.mean(acceleration(u, dt, method)**2))
            for method in ("central", "forward", "gradient") for dt in (1., .00225)},
        "single_time_velocity": single_time_statistics(u),
        "single_time_acceleration_by_convention": {
            f"{method}_dt_{dt}": single_time_statistics(acceleration(u, dt, method))
            for method in ("central", "forward", "gradient") for dt in (1., .00225)},
        "acceleration_note": "Paper Table II A~1.7e-3 suggests per-sample differences; closeness of this small demo is only a clue, not proof of the authors' stencil or units.",
        "runtime_seconds": time.monotonic() - started,
        "hdf5_backend": backend,
        "limitations": ["256 physical traces, no independently generated DM samples", "Sparse lag grid cannot reproduce F5 local slopes", "Min-max envelopes are not 95% confidence intervals", "Extreme tails and eighth moments are not certified converged"]}
    summary_path = args.output / "demo_summary.json"
    write_json(summary_path, summary)
    record = manifest(f"python -m repro.validate_demo --output {args.output} --seed {args.seed}",
                      inputs=[source3, source1], seed=args.seed,
                      classification=summary["classification"])
    record["outputs"] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in (raw, summary_path)]
    write_json(args.output / "demo_run_manifest.json", record)
    print(f"Demo validation complete: {len(u)} physical traces; not paper reproduction")
    print(f"Normalization max error: {normalization_error:.3e}")
    print(f"Component-averaged mean square velocity: {summary['energy_component_mean_square']:.8g}")
    print(summary["acceleration_variance_by_convention"])


if __name__ == "__main__":
    main()
