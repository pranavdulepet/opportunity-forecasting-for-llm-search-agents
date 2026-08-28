"""Render the paper's search-and-forecasting pipeline overview."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

from opportunity_forecasting import REPO_ROOT


FIG_W_IN = 456.48 / 72
FIG_H_IN = 216.71 / 72

LM_DIR = REPO_ROOT / "opportunity_forecasting" / "figures" / "fonts"
REGULAR_FONT = LM_DIR / 'lmsans10-regular.otf'
BOLD_FONT = LM_DIR / 'lmsans10-bold.otf'
for fp in (REGULAR_FONT, BOLD_FONT):
    if not fp.is_file():
        raise FileNotFoundError(fp)
    fontManager.addfont(str(fp))
FONT_REG = FontProperties(fname=str(REGULAR_FONT))
FONT_BOLD = FontProperties(fname=str(BOLD_FONT))
FONT_FAMILY = 'Latin Modern Sans'

mpl.rcParams.update({
    'font.family': FONT_FAMILY,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

BLUE = '#1f6fe5'
BLUE_DARK = '#0e5bd8'
BLUE_EDGE = '#8fbaf2'
BLUE_BG = '#f3f8ff'
RED = '#e32620'
RED_EDGE = '#ff8c8c'
RED_BG = '#fff4f4'
GREEN = '#178a3c'
GREEN_LIGHT = '#e8f7ec'
ORANGE = '#ef7600'
ORANGE_BG = '#fff7e8'
GRAY = '#7c858b'
GRAY_LIGHT = '#eef1f3'
DARK = '#222222'
WHITE = '#ffffff'
PDF_METADATA = {'CreationDate': None, 'ModDate': None}


def box(ax, x, y, w, h, fc, ec, lw=1.15, r=1.0, z=2, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.02,rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha
    )
    ax.add_patch(patch)
    return patch


def txt(ax, x, y, s, size=7.5, color=DARK, bold=False, ha='center', va='center', z=10, **kw):
    ax.text(x, y, s, fontsize=size, color=color,
            fontproperties=FONT_BOLD if bold else FONT_REG,
            ha=ha, va=va, zorder=z, **kw)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.25, ms=10, rad=0, z=7):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
        linewidth=lw, color=color, connectionstyle=f'arc3,rad={rad}',
        shrinkA=0, shrinkB=0, joinstyle='miter', zorder=z
    )
    ax.add_patch(patch)
    return patch


def polyline_arrow(ax, pts, color=GRAY, lw=1.15, ms=10, z=7):

    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=z))
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw, ms=ms, z=z)


def draw_search_icon(ax, cx, cy, scale=1.0, color=BLUE):
    ax.add_patch(Circle((cx, cy), 1.35*scale, facecolor='none', edgecolor=color, linewidth=2.1*scale, zorder=9))
    ax.add_line(Line2D([cx+0.95*scale, cx+1.85*scale], [cy-0.95*scale, cy-1.85*scale], color=color, linewidth=2.1*scale, zorder=9))


def draw_target_icon(ax, cx, cy, scale=1.0, color=RED):
    for r in [1.55, 0.95, 0.40]:
        ax.add_patch(Circle((cx, cy), r*scale, facecolor='none', edgecolor=color, linewidth=1.35*scale, zorder=9))
    ax.add_line(Line2D([cx-2.0*scale, cx-1.45*scale], [cy, cy], color=color, linewidth=1.2*scale, zorder=9))
    ax.add_line(Line2D([cx+1.45*scale, cx+2.0*scale], [cy, cy], color=color, linewidth=1.2*scale, zorder=9))
    ax.add_line(Line2D([cx, cx], [cy-2.0*scale, cy-1.45*scale], color=color, linewidth=1.2*scale, zorder=9))
    ax.add_line(Line2D([cx, cx], [cy+1.45*scale, cy+2.0*scale], color=color, linewidth=1.2*scale, zorder=9))


def draw_icon_circle(ax, cx, cy, kind):
    ax.add_patch(Circle((cx, cy), 1.15, facecolor='none', edgecolor=ORANGE, linewidth=1.0, zorder=9))
    if kind == 'expand':
        arrow(ax, cx-0.45, cy-0.45, cx+0.55, cy+0.55, color=ORANGE, lw=1.0, ms=6.5, z=10)
        ax.add_line(Line2D([cx-0.45, cx+0.35], [cy-0.45, cy-0.45], color=ORANGE, linewidth=1.0, zorder=10))
    elif kind == 'shield':
        verts = [(cx, cy+0.75), (cx+0.65, cy+0.35), (cx+0.52, cy-0.55), (cx, cy-0.85), (cx-0.52, cy-0.55), (cx-0.65, cy+0.35)]
        ax.add_patch(Polygon(verts, closed=True, facecolor='none', edgecolor=ORANGE, linewidth=1.0, zorder=10))
    elif kind == 'risk':

        xs = [cx-0.65, cx-0.25, cx+0.10, cx+0.55]
        ys = [cy-0.35, cy+0.05, cy-0.05, cy+0.45]
        ax.add_line(Line2D(xs, ys, color=ORANGE, linewidth=1.0, zorder=10))
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), 0.10, facecolor=ORANGE, edgecolor='none', zorder=11))


def beta_pdf(x, a, b):
    return math.gamma(a+b) / (math.gamma(a)*math.gamma(b)) * x**(a-1) * (1-x)**(b-1)


def draw_hurdle_beta(ax, x, y, w, h):
    base = y + 0.20*h
    ax.add_line(Line2D([x, x+w], [base, base], color=GRAY, linewidth=0.8, zorder=9))
    txt(ax, x, base-1.15, '0', size=5.7, color=DARK)
    txt(ax, x+w, base-1.15, '1', size=5.7, color=DARK)

    zero_atom_h = 0.46*h
    ax.add_line(Line2D([x, x], [base, base+zero_atom_h], color=RED, linewidth=1.8, zorder=10))
    ax.add_patch(Circle((x, base+zero_atom_h), 0.50, facecolor=RED, edgecolor=RED, zorder=11))
    txt(ax, x-1.6, base+zero_atom_h+1.35, 'mass\nat 0', size=4.9, color=RED, bold=True, ha='left', linespacing=0.9)

    one_atom_h = 0.30*h
    ax.add_line(Line2D([x+w, x+w], [base, base+one_atom_h], color=RED, linewidth=1.8, zorder=10))
    ax.add_patch(Circle((x+w, base+one_atom_h), 0.50, facecolor=RED, edgecolor=RED, zorder=11))
    txt(ax, x+w, base+one_atom_h+1.45, 'mass\nat 1', size=4.9, color=RED, bold=True, ha='right', linespacing=0.9)

    x_start = x + 0.75
    tail_w = w - 1.25
    vals = [0.01 + i * 0.975 / 180 for i in range(181)]
    dens = [beta_pdf(v, 2.0, 5.6) for v in vals]
    m = max(dens)
    pts = [(x_start + v*tail_w, base + (d/m)*0.56*h) for v, d in zip(vals, dens)]
    ax.fill([pts[0][0]] + [p[0] for p in pts] + [pts[-1][0]],
            [base] + [p[1] for p in pts] + [base], color=GREEN, alpha=0.11, zorder=8)
    ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts], color=GREEN, linewidth=1.75, zorder=10))
    txt(ax, x+0.50*w, base+0.68*h, 'interior Beta', size=5.4, color=GREEN, bold=True)


def render(output_dir: Path):
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(WHITE)


    box(ax, 2, 27.0, 70.0, 20.8, fc=BLUE_BG, ec=BLUE_EDGE, lw=0.9, r=1.1, z=1, alpha=0.92)
    box(ax, 2, 3.1, 96.0, 24.0, fc=RED_BG, ec='#ffd0d0', lw=0.8, r=1.1, z=1, alpha=0.88)


    box(ax, 0.8, 46.7, 14.6, 2.7, fc=BLUE_DARK, ec=BLUE_DARK, lw=0, r=0.45, z=5)
    txt(ax, 8.1, 48.05, 'Search model', size=7.4, color=WHITE, bold=True)
    box(ax, 0.8, 25.0, 17.0, 2.65, fc=RED, ec=RED, lw=0, r=0.45, z=5)
    txt(ax, 9.3, 26.3, 'Forecasting model', size=7.1, color=WHITE, bold=True)


    box(ax, 4.8, 34.0, 15.7, 10.0, fc=WHITE, ec=BLUE_DARK, lw=1.25, r=0.8, z=3)
    draw_search_icon(ax, 12.65, 40.0, scale=0.78, color=BLUE_DARK)
    txt(ax, 12.65, 36.9, 'Search model', size=7.0, color=DARK, bold=True)


    arrow(ax, 20.5, 39.0, 26.8, 39.0, color=BLUE_DARK, lw=1.2, ms=10)


    box(ax, 27.3, 32.3, 39.2, 12.3, fc=WHITE, ec=BLUE_EDGE, lw=1.0, r=0.9, z=3)
    txt(ax, 46.9, 42.5, 'Rollout + MC continuations', size=7.4, color=BLUE_DARK, bold=True)

    x0, y0 = 29.7, 38.2
    ax.add_line(Line2D([x0, 44.6], [y0, y0], color=BLUE_DARK, linewidth=1.15, zorder=7))
    for i, xx in enumerate([31.6, 35.1, 38.6, 42.1]):
        ax.add_patch(Circle((xx, y0), 0.44, facecolor=BLUE_DARK, edgecolor=BLUE_DARK, zorder=9))
    decision_x = 45.3
    ax.add_patch(Circle((decision_x, y0), 0.86, facecolor=ORANGE, edgecolor=ORANGE, zorder=10))
    txt(ax, decision_x, 35.7, 'decision\npoint', size=5.0, color=ORANGE, bold=True, linespacing=0.92)

    branch_ends = [
        (58.6, 42.0, GREEN, 'gain'),
        (58.6, 39.5, GRAY, 'no gain'),
        (58.6, 36.9, GREEN, 'gain'),
        (58.6, 34.3, GRAY, 'no gain'),
    ]
    for ex, ey, col, lab in branch_ends:
        ax.add_line(Line2D([decision_x, ex], [y0, ey], color=col, linewidth=1.1, zorder=7))
        ax.add_patch(Circle((ex, ey), 0.63, facecolor=col, edgecolor=col, zorder=9))
        txt(ax, ex+1.9, ey, lab, size=5.8, color=col if col != GRAY else DARK, ha='left')


    split_x, split_y, split_w, split_h = 35.6, 24.7, 15.2, 5.3
    arrow(ax, 43.2, 32.3, 43.2, split_y + split_h + 0.05, color=GREEN, lw=1.1, ms=9)
    box(ax, split_x, split_y, split_w, split_h, fc=GREEN_LIGHT, ec=GREEN, lw=1.2, r=0.7, z=4)
    txt(ax, 43.2, split_y + split_h/2, 'Goal-disjoint\nsplits', size=6.7, color=GREEN, bold=True, linespacing=0.9)


    box(ax, 3.2, 7.7, 13.5, 9.4, fc=WHITE, ec=RED, lw=1.2, r=0.8, z=3)
    draw_target_icon(ax, 9.95, 13.45, scale=0.64, color=RED)
    txt(ax, 9.95, 9.95, 'Forecasting\nmodel', size=6.2, color=RED, bold=True, linespacing=0.9)

    arrow(ax, 16.7, 12.5, 19.9, 12.5, color=RED, lw=1.2, ms=9)
    box(ax, 20.4, 8.1, 11.3, 8.6, fc=WHITE, ec=RED, lw=1.05, r=0.65, z=3)
    txt(ax, 26.05, 12.4, 'LLM\nbackbone', size=6.5, color=RED, bold=True, linespacing=0.9)

    arrow(ax, 31.7, 12.5, 36.1, 12.5, color=RED, lw=1.2, ms=9)
    box(ax, 36.6, 8.1, 13.4, 8.6, fc=WHITE, ec=RED, lw=1.05, r=0.65, z=3)
    txt(ax, 43.3, 12.4, 'Hurdle-Beta\nhead', size=6.4, color=RED, bold=True, linespacing=0.9)


    polyline_arrow(ax, [(43.2, split_y), (43.2, 20.5), (25.2, 20.5), (25.2, 17.2)], color=GREEN, lw=1.1, ms=9, z=8)

    arrow(ax, 50.0, 12.5, 54.2, 12.5, color=RED, lw=1.2, ms=9)


    box(ax, 54.6, 6.0, 22.4, 14.2, fc=WHITE, ec=RED_EDGE, lw=1.05, r=0.75, z=3)
    txt(ax, 65.8, 18.25, 'Forecast', size=7.4, color=RED, bold=True)
    draw_hurdle_beta(ax, 57.3, 8.0, 16.7, 8.2)


    arrow(ax, 77.0, 12.6, 80.4, 12.6, color=RED, lw=1.2, ms=9)
    box(ax, 81.0, 8.6, 18.1, 13.8, fc=ORANGE_BG, ec=ORANGE, lw=1.2, r=0.75, z=4)
    txt(ax, 90.05, 20.5, 'Downstream\nuse', size=7.2, color=ORANGE, bold=True, linespacing=0.9)

    rows = [
        ('expand', 'Budgeted expansion'),
        ('shield', 'Risk-neutral stop'),
        ('risk', 'Risk-adjusted stop'),
    ]
    row_y = [17.4, 14.6, 11.8]
    for (kind, label), yy in zip(rows, row_y):
        draw_icon_circle(ax, 84.1, yy, kind)
        txt(ax, 86.4, yy, label, size=5.8, color=DARK, ha='left')


    box(ax, 83.6, 26.0, 13.7, 3.2, fc=GREEN, ec=GREEN, lw=0, r=0.45, z=5)
    txt(ax, 90.45, 27.6, 'Commit best', size=6.8, color=WHITE, bold=True)
    arrow(ax, 90.45, 22.4, 90.45, 25.9, color=GREEN, lw=1.2, ms=9)

    box(ax, 82.3, 3.9, 16.6, 3.1, fc=BLUE_DARK, ec=BLUE_DARK, lw=0, r=0.45, z=5)
    txt(ax, 90.6, 5.45, 'Continue / expand', size=6.7, color=WHITE, bold=True)
    arrow(ax, 90.45, 8.6, 90.45, 7.0, color=BLUE_DARK, lw=1.2, ms=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = output_dir / 'pipeline_overview.pdf'
    png_out = output_dir / 'pipeline_overview.png'
    fig.savefig(png_out, dpi=600)
    fig.savefig(pdf_out, metadata=PDF_METADATA)
    plt.close(fig)
    print(pdf_out)
    print(png_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "figures" / "overview"
    )
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == '__main__':
    main()
