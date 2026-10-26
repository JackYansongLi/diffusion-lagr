"""Create analytic F3b schedule variants and a clearly labelled DNS-demo diagnostic."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .schedules import alpha_bar, betas, upstream_schedule_functions
from .provenance import ROOT, manifest, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("repro/figures"))
    p.add_argument("--results", type=Path, default=Path("repro/results"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#2563a6", "#df7d28", "#28967c", "#a44977"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    rows, metrics = [], {}
    reference = upstream_schedule_functions()
    for name, steps, label, color in zip(
            ["tanh6,1", "power4", "linear_paper", "linear_upstream"],
            [800, 800, 4000, 4000],
            ["tanh6-1, N=800", "power4, N=800", "Linear: paper endpoints, N=4000", "Linear: upstream code, N=4000"], colors):
        b = betas(name, steps)
        a = np.r_[1., np.cumprod(1 - b)]
        x = np.arange(steps + 1) / steps
        # Original Figure 3b displays noise variance, 1-alpha_bar, on a log axis.
        # n=0 has exactly zero noise and cannot be placed on the log curve.
        ax.semilogy(x[1:], 1 - a[1:], color=color, lw=1.8, label=label,
                    ls="--" if name == "linear_upstream" else "-")
        for i in range(steps + 1):
            rows.append([name, steps, i, x[i], a[i], 1 - a[i]])
        metrics[name] = {"steps": steps, "beta_first": float(b[0]), "beta_last": float(b[-1]),
                         "alpha_bar_terminal_discrete": float(a[-1])}
        if name in ("tanh6,1", "power4"):
            metrics[name]["analytic_discrete_max_abs_error_excluding_clipped_endpoint"] = float(np.max(np.abs(a[:-1] - alpha_bar(x[:-1], name))))
            metrics[name]["max_abs_beta_error_vs_upstream"] = float(np.max(np.abs(b - reference(name, steps))))
    for fraction, label in [(.06, "B"), (.27, "C"), (.52, "D")]:
        y = 1 - alpha_bar(fraction)
        ax.plot(fraction, y, "o", color=colors[0], ms=4)
        ax.annotate(label, (fraction, y), xytext=(3, 5), textcoords="offset points")
    ax.annotate("A: zero noise (outside log scale)", (0, 1e-12), xytext=(.035, 5e-12),
                textcoords="data", fontsize=8, arrowprops={"arrowstyle":"-", "color":"0.4"})
    ax.set(xlabel=r"Normalized forward diffusion step $n/N$", ylabel=r"Noise variance $1-\bar\alpha_n$",
           title="Figure 3b: noise schedules and code discrepancy", xlim=(0, 1), ylim=(1e-12, 1.5))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"figure3b_schedule_audit.{suffix}", dpi=180)
    plt.close(fig)
    csv_path = args.results / "schedule_curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "N", "n", "n_over_N", "alpha_bar_discrete", "noise_variance_discrete"])
        writer.writerows(rows)
    write_json(args.results / "schedule_metrics.json", {
        "classification": "analytic_panel_recalculation_with_unresolved_linear_variant", "metrics": metrics,
        "note": "No checkpoints or synthetic proxy data. Both paper Eq24 and upstream implementation are evaluated; paper/code linear endpoints differ by factor four at N=4000."})
    source = args.results / "demo_summary.json"
    demo = json.loads(source.read_text())
    x = np.array(demo["lag_over_tau_eta_using_table_I"])
    s, f = np.array(demo["structure"]), np.array(demo["flatness"])
    ci = np.array(demo["conditional_trajectory_bootstrap"]["flatness_ci95"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), layout="constrained")
    for j, order in enumerate([2, 4, 6, 8]):
        axes[0].loglog(x, s[:, j], "o-", ms=3, color=colors[j], label=f"p={order}")
        if order > 2:
            axes[1].loglog(x, f[:, j], "o-", ms=3, color=colors[j], label=f"p={order}")
            axes[1].fill_between(x, ci[0, :, j], ci[1, :, j], color=colors[j], alpha=.15)
    axes[0].set(ylabel=r"$S^{(p)}(\tau)$")
    axes[1].set(ylabel=r"$F^{(p)}(\tau)$")
    for ax in axes:
        ax.set_xlabel(r"$\tau/\tau_\eta$ (Table I units)")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Bundled DNS demo: 256 trajectories; NOT paper reproduction", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"demo_diagnostics.{suffix}", dpi=180)
    plt.close(fig)
    outputs = list(args.output.glob("*.pdf")) + list(args.output.glob("*.png")) + [csv_path, args.results / "schedule_metrics.json"]
    record = manifest(f"python -m repro.plot_audit --output {args.output} --results {args.results}",
                      inputs=[source, ROOT / "guided_diffusion/gaussian_diffusion.py"],
                      classification="analytic_schedule_audit_and_demo_diagnostic")
    record["outputs"] = [{"name": str(p), "sha256": sha256(p)} for p in outputs]
    write_json(args.results / "plot_run_manifest.json", record)
    print("Recomputed schedule curves; saved separately labelled demo diagnostics")


if __name__ == "__main__":
    main()
