from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .domain import SDFDomain


@dataclass(frozen=True, order=True)
class Cell:
    level: int
    i: int
    j: int


class Quadtree:
    def __init__(self, origin: Tuple[float, float], size: float, leaves: Iterable[Cell]):
        self.origin = origin
        self.size = float(size)
        self.leaves: Set[Cell] = set(leaves)

    @classmethod
    def build(
        cls,
        domain: SDFDomain,
        min_depth: int = 2,
        max_depth: int = 6,
        boundary_band: float = 0.55,
    ) -> "Quadtree":
        xmin, ymin, xmax, ymax = domain.bounds
        span = max(xmax - xmin, ymax - ymin)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        origin = (cx - 0.5 * span, cy - 0.5 * span)

        leaves: Set[Cell] = set()
        stack = [Cell(0, 0, 0)]
        tree = cls(origin, span, [])
        while stack:
            cell = stack.pop()
            if cell.level < min_depth or (
                cell.level < max_depth and tree._needs_refinement(domain, cell, boundary_band)
            ):
                stack.extend(children(cell))
            else:
                leaves.add(cell)

        tree.leaves = leaves
        tree.balance(max_depth=max_depth)
        return tree

    def _needs_refinement(self, domain: SDFDomain, cell: Cell, boundary_band: float) -> bool:
        x0, y0, x1, y1 = self.bounds(cell)
        s = x1 - x0
        pts = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
                [0.5 * (x0 + x1), y0],
                [x1, 0.5 * (y0 + y1)],
                [0.5 * (x0 + x1), y1],
                [x0, 0.5 * (y0 + y1)],
            ]
        )
        vals = np.asarray(domain.sdf(pts), dtype=float)
        has_inside = np.any(vals <= 0.0)
        has_outside = np.any(vals >= 0.0)
        if has_inside and has_outside:
            return True
        return bool(np.min(np.abs(vals)) < boundary_band * s)

    def balance(self, max_depth: int) -> None:
        """Strongly balance the tree: corner-adjacent cells differ by at most 1."""

        while True:
            cells = sorted(self.leaves)
            refine: Set[Cell] = set()
            for idx, a in enumerate(cells):
                ab = self.bounds(a)
                for b in cells[idx + 1 :]:
                    if abs(a.level - b.level) <= 1:
                        continue
                    if not boxes_touch(ab, self.bounds(b)):
                        continue
                    coarse = a if a.level < b.level else b
                    if coarse.level < max_depth:
                        refine.add(coarse)
            if not refine:
                return
            self.leaves.difference_update(refine)
            for cell in refine:
                self.leaves.update(children(cell))

    def bounds(self, cell: Cell) -> Tuple[float, float, float, float]:
        h = self.size / (2**cell.level)
        x0 = self.origin[0] + cell.i * h
        y0 = self.origin[1] + cell.j * h
        return (x0, y0, x0 + h, y0 + h)

    def corners(self, cell: Cell) -> List[Tuple[float, float]]:
        x0, y0, x1, y1 = self.bounds(cell)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def collect_corners(self) -> Set[Tuple[float, float]]:
        pts: Set[Tuple[float, float]] = set()
        for cell in self.leaves:
            pts.update(round_point(p) for p in self.corners(cell))
        return pts

    def incident_min_sizes(self) -> Dict[Tuple[float, float], float]:
        sizes: Dict[Tuple[float, float], float] = {}
        for cell in self.leaves:
            x0, _y0, x1, _y1 = self.bounds(cell)
            s = x1 - x0
            for p in self.corners(cell):
                key = round_point(p)
                sizes[key] = min(sizes.get(key, s), s)
        return sizes


def children(cell: Cell) -> Sequence[Cell]:
    level = cell.level + 1
    return (
        Cell(level, 2 * cell.i, 2 * cell.j),
        Cell(level, 2 * cell.i + 1, 2 * cell.j),
        Cell(level, 2 * cell.i, 2 * cell.j + 1),
        Cell(level, 2 * cell.i + 1, 2 * cell.j + 1),
    )


def round_point(point: Tuple[float, float], digits: int = 12) -> Tuple[float, float]:
    return (round(float(point[0]), digits), round(float(point[1]), digits))


def boxes_touch(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    eps = 1.0e-12
    return not (
        a[2] < b[0] - eps
        or b[2] < a[0] - eps
        or a[3] < b[1] - eps
        or b[3] < a[1] - eps
    )
