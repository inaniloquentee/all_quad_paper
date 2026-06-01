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
    for quad in mesh.quads:
        q = pts[list(quad) + [quad[0]]]
        ax.plot(q[:, 0], q[:, 1], color="#1f2933", linewidth=0.45)

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
    ax.set_title(f"{domain.name}: {len(mesh.quads)} quads")
    fig.tight_layout()
    fig.savefig(target, dpi=220)
    plt.close(fig)
