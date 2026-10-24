"""vec_regen.py -- re-run a plotting script and additionally write a vector PDF twin.

Why: the discussion document must embed vector figures (user annotation C10). The plotting
scripts are deliverables and are NOT edited (project rule: initial versions stay untouched).
Instead this driver patches matplotlib's savefig so that every .png a script writes is
followed by an identical .pdf next to it. On WSL it also translates the two hard-coded
Windows roots in memory; on Windows the paths are already correct and are left alone.

usage: python vec_regen.py <script.py> [script args...]
"""
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42      # embed TrueType, not Type-3 outlines
matplotlib.rcParams["ps.fonttype"] = 42
# matplotlib's default is 'line': gridlines sit above patches, so the grey grid is drawn on
# top of histogram bars. True puts the grid at zorder 0.5, below bars *and* curves.
matplotlib.rcParams["axes.axisbelow"] = True

import numpy as np


def _grid_below(fig):
    """Force gridlines under every artist (belt and braces: styles/scripts may override)."""
    for ax in fig.axes:
        try:
            ax.set_axisbelow(True)
        except Exception:
            pass
        for gl in list(ax.get_xgridlines()) + list(ax.get_ygridlines()):
            try:
                gl.set_zorder(0.5)
            except Exception:
                pass


def _flatten_alpha(fig, bg=(1.0, 1.0, 1.0)):
    """Replace translucent fills/lines by the equivalent opaque colour on a white page.

    PostScript has no alpha channel, so matplotlib's EPS backend composites translucent
    artists as solid colours -- an error envelope then appears as a filled block. Blending
    c' = a*c + (1-a)*bg and setting a = 1 reproduces the on-screen appearance exactly while
    staying vector.
    """
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba_array

    bg = np.asarray(bg, dtype=float)
    for art in fig.findobj():
        try:
            a_art = art.get_alpha()
        except Exception:
            a_art = None
        for getter, setter in (("get_facecolor", "set_facecolor"),
                               ("get_edgecolor", "set_edgecolor"),
                               ("get_color", "set_color")):
            if not hasattr(art, getter) or not hasattr(art, setter):
                continue
            try:
                col = getattr(art, getter)()
                if col is None:
                    continue
                rgba = to_rgba_array(col)
                if rgba.size == 0:
                    continue
                # matplotlib applies an artist-level alpha *instead of* the colour alpha,
                # so the effective value is one or the other -- multiplying double-counts it
                # and washes the band out to near-white.
                a = (np.full(len(rgba), float(a_art)) if a_art is not None
                     else rgba[:, 3])
                if np.all(a >= 0.999):
                    continue
                rgb = rgba[:, :3] * a[:, None] + (1.0 - a[:, None]) * bg[None, :]
                if isinstance(art, Collection) and len(rgb) > 1:
                    getattr(art, setter)(rgb)
                else:
                    getattr(art, setter)(tuple(rgb[0]))
                if a_art is not None:
                    art.set_alpha(1.0)
            except Exception:
                continue


_orig = matplotlib.figure.Figure.savefig
_written = []


def savefig_both(self, fname, *a, **kw):
    _grid_below(self)
    out = _orig(self, fname, *a, **kw)
    p = str(fname)
    if p.lower().endswith(".png"):
        pdf = p[:-4] + ".pdf"
        try:
            _orig(self, pdf, *a, **kw)
            _written.append(os.path.basename(pdf))
            print("[vec] ->", os.path.basename(pdf), flush=True)
            eps = p[:-4] + ".eps"
            # Figures with per-point colours (3D trajectories) or colormap artists must not
            # be flattened: collapsing a multi-colour Line3D to one blended colour destroys
            # the colour coding, and a colourbar's colour set is not a fill to composite.
            skip = {s.strip() for s in os.environ.get("VEC_NO_FLATTEN", "").split(",") if s.strip()}
            base = os.path.basename(p)[:-4]
            # prefix-aware: "A9_fig1f_3dtraj" must also protect "A9_fig1f_3dtraj_p99"
            if not any(base == k or base.startswith(k + "_") for k in skip):
                _flatten_alpha(self)      # EPS has no alpha: pre-composite over white
            _orig(self, eps, *a, **kw)
            print("[vec] ->", os.path.basename(eps), flush=True)
        except Exception as e:
            print("[vec] FAILED", pdf, repr(e), flush=True)
    return out


matplotlib.figure.Figure.savefig = savefig_both
plt.savefig = lambda *a, **kw: savefig_both(plt.gcf(), *a, **kw)

script = sys.argv[1]
extra = sys.argv[2:]
src = open(script, encoding="utf-8").read()

# The scripts read their inputs through LTS_ROOT, so no path rewriting is needed here.

sys.argv = [script] + extra
g = {"__name__": "__main__", "__file__": os.path.abspath(script), "__builtins__": __builtins__}
os.chdir(os.path.dirname(os.path.abspath(script)))
exec(compile(src, script, "exec"), g)
print("[vec] done", os.path.basename(script), "pdfs:", len(_written), flush=True)
