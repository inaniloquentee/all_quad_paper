from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .domain import SDFDomain
from .mesh import Mesh


def plot_mesh(mesh: Mesh, domain: SDFDomain, path: str | Path, samples: int = 400) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(mesh.vertices)

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {-1: "#1f2933", 1: "#7a8793"}
    widths = {-1: 0.45, 1: 0.35}
    for quad, region in zip(mesh.quads, mesh.regions):
        q = pts[list(quad) + [quad[0]]]
        ax.plot(
            q[:, 0],
            q[:, 1],
            color=colors.get(region, "#1f2933"),
            linewidth=widths.get(region, 0.4),
        )

    xmin, ymin, xmax, ymax = domain.bounds
    xs = np.linspace(xmin, xmax, samples)
    ys = np.linspace(ymin, ymax, samples)
    xx, yy = np.meshgrid(xs, ys)
    zz = domain.sdf(np.stack([xx, yy], axis=-1))
    ax.contour(xx, yy, zz, levels=[0.0], colors=["#d62828"], linewidths=1.2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    inside = sum(1 for region in mesh.regions if region < 0)
    outside = sum(1 for region in mesh.regions if region > 0)
    ax.set_title(f"{domain.name}: {inside} inside + {outside} outside quads")
    fig.tight_layout()
    fig.savefig(target, dpi=220)
    plt.close(fig)
