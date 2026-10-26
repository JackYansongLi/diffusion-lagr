"""Analytic schedules, with paper and upstream linear variants kept distinct."""
import ast
import math
from pathlib import Path
import numpy as np


def alpha_bar(t, schedule="tanh6,1"):
    t = np.asarray(t, dtype=np.float64)
    if np.any(t < 0) or np.any(t > 1):
        raise ValueError("normalized diffusion time must be in [0,1]")
    if schedule == "tanh6,1":
        return (-np.tanh(7 * t - 6) + np.tanh(1)) / (np.tanh(6) + np.tanh(1))
    if schedule == "power4":
        return 1 - t**4
    raise ValueError(schedule)


def betas(schedule, steps):
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("need at least two diffusion steps")
    if schedule in ("linear_upstream", "linear_paper"):
        scale = 1000 / steps if schedule == "linear_upstream" else 1
        result = np.linspace(scale * 1e-4, scale * .02, steps)
    else:
        a = alpha_bar(np.arange(steps + 1) / steps, schedule)
        result = np.minimum(1 - a[1:] / a[:-1], .999)
    if not np.all((result > 0) & (result <= 1)):
        raise ValueError("invalid diffusion beta; increase steps")
    return result


def upstream_schedule_functions():
    """Load only the two audited NumPy schedule functions, without torch/MPI.

    Source is the locally checked-out, user-authorized upstream repository; no
    network code is downloaded or executed. This permits direct numerical
    comparisons rather than tests that only compare our code with itself.
    """
    path = Path(__file__).resolve().parents[1] / "guided_diffusion/gaussian_diffusion.py"
    tree = ast.parse(path.read_text())
    wanted = {"get_named_beta_schedule", "betas_for_alpha_bar"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if len(nodes) != 2:
        raise RuntimeError("upstream schedule source changed")
    scope = {"np": np, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["get_named_beta_schedule"]
