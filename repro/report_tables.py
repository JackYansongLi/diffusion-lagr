"""Generate LaTeX tables directly from measured JSON; do not type results by hand."""
from pathlib import Path
import json
import numpy as np


def number(value):
    if value != 0 and (abs(value) < .01 or abs(value) >= 1e4):
        a, exponent = f"{value:.4e}".split("e")
        return rf"{a}\times10^{{{int(exponent)}}}"
    return f"{value:.5f}"


def main():
    root = Path(__file__).resolve().parents[1]
    results, output = root / "repro/results", root / "report"
    output.mkdir(exist_ok=True)
    demo = json.loads((results / "demo_summary.json").read_text())
    exp = json.loads((results / "estimator_exploration.json").read_text())
    schedule = json.loads((results / "schedule_metrics.json").read_text())["metrics"]
    rows = []
    refs = [3., 1.7e-3, -.41, 4.4e-5]
    for label, key, reference in zip([r"$E$", r"$A$ (sample index)", r"$\Sigma^V$", r"$\Sigma^A$ (sample index)"],
                                    ["E", "A_sample_gradient", "Sigma_V", "Sigma_A_sample_gradient"], refs):
        v = exp["single_time"][key]
        lo, hi = v["conditional_ci95"]
        rows.append(rf"{label} & ${number(v['point'])}$ & $[{number(lo)},\ {number(hi)}]$ & ${number(reference)}$ \\")
    (output / "single_time_table.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for method, label in [("central", "Central, interior only"), ("gradient", "Central + one-sided ends"), ("forward", "Forward difference")]:
        a = demo["acceleration_variance_by_convention"]
        rows.append(rf"{label} & ${number(a[method+'_dt_1.0'])}$ & ${number(a[method+'_dt_0.00225'])}$ \\")
    (output / "derivative_table.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for name, label in [("tanh6,1", "tanh6-1"), ("power4", "power4"), ("linear_paper", "Linear, paper"), ("linear_upstream", "Linear, upstream")]:
        v = schedule[name]
        rows.append(rf"{label} & {v['steps']} & ${number(v['beta_first'])}$ & ${number(v['beta_last'])}$ & ${number(v['alpha_bar_terminal_discrete'])}$ \\")
    (output / "schedule_table.tex").write_text("\n".join(rows) + "\n")
    rows = []
    cv = np.array(exp["subsample_coefficient_of_variation"])
    top = np.array(exp["top_three_trajectory_fraction_of_total_moment"])
    loo = np.array(exp["max_leave_one_trajectory_out_relative_change"])
    for i, order in enumerate([2, 4, 6, 8]):
        rows.append(rf"{order} & {100*cv[0,1,i]:.2f}\% & {100*cv[3,1,i]:.2f}\% & {100*top[0,i]:.2f}\% & {100*loo[0,i]:.2f}\% \\")
    (output / "sensitivity_table.tex").write_text("\n".join(rows) + "\n")
    macros = {
        "RoundTripError":number(demo["normalization_max_abs_roundtrip_difference"]),
        "DenseSeconds":f"{exp['dense_grid_runtime_seconds']:.2f}",
        "ExploreSeconds":f"{exp['runtime_seconds']:.2f}",
        "DenseDifference":number(exp["local_derivative_discretization_max_abs_difference_interior"]),
        "SparseDifference":number(exp["sparse_grid_max_abs_difference_at_common_lags"]),
        "SampleToPhysicalFactor":number(.00225**-2),
    }
    (output / "measured_macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k,v in macros.items()) + "\n")
    print("Generated measured LaTeX tables from JSON outputs")


if __name__ == "__main__":
    main()
