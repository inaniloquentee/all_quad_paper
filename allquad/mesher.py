from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .domain import SDFDomain
from .mesh import Mesh, clean_polygon
from .quadtree import Cell, Quadtree, round_point


Point = Tuple[float, float]


@dataclass
class AllQuadMesher:
    domain: SDFDomain
    min_depth: int = 2
    max_depth: int = 6
    clearance_ratio: float = 0.25
    repelling: str = "normal"
    boundary_band: float = 0.55
    tol: float = 1.0e-10
    include_exterior: bool = True

    def generate(self) -> Mesh:
        tree = Quadtree.build(
            self.domain,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            boundary_band=self.boundary_band,
        )
        grid_points = tree.collect_corners()
        incident_sizes = tree.incident_min_sizes()
        moved_points = {
            p: self.domain.repel(
                np.array(p, dtype=float),
                self.clearance_ratio * incident_sizes[p],
                mode=self.repelling,
            )
            for p in grid_points
        }
        x_index, y_index = build_side_indices(grid_points)
        self._perturb_internal_hanging_points(tree, moved_points, x_index, y_index, incident_sizes)

        mesh = Mesh(tol=self.tol)
        for cell in sorted(tree.leaves):
            poly = self._cell_polygon(tree, cell, moved_points, x_index, y_index)
            if len(poly) < 3:
                continue
            pieces = self._split_polygon(poly)
            for clipped, region in pieces:
                if len(clipped) < 3:
                    continue
                # Applying midpoint subdivision consistently on both sides of every
                # shared segment is a generic closure of the paper's local 2-ref
                # templates. It keeps the implementation compact while producing a
                # conforming all-quad mesh at adaptive transitions.
                mesh.add_midpoint_subdivision(clipped, region=region)
        return mesh

    def _perturb_internal_hanging_points(
        self,
        tree: Quadtree,
        moved_points: Dict[Point, np.ndarray],
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
        incident_sizes: Dict[Point, float],
    ) -> None:
        """Avoid 180-degree midpoint-subdivision angles at transition nodes."""

        shifts: Dict[Point, np.ndarray] = {}
        for cell in tree.leaves:
            x0, y0, x1, y1 = tree.bounds(cell)
            center = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
            side_points = (
                points_on_horizontal(y_index, y0, x0, x1, reverse=False)[1:-1],
                points_on_vertical(x_index, x1, y0, y1, reverse=False)[1:-1],
                points_on_horizontal(y_index, y1, x0, x1, reverse=False)[1:-1],
                points_on_vertical(x_index, x0, y0, y1, reverse=False)[1:-1],
            )
            for side in side_points:
                for point in side:
                    key = round_point(point)
                    direction = center - np.asarray(key)
                    length = np.linalg.norm(direction)
                    if length > self.tol:
                        shifts[key] = shifts.get(key, np.zeros(2)) + direction / length

        for point, direction in shifts.items():
            length = np.linalg.norm(direction)
            if length <= self.tol:
                continue
            delta = 0.08 * incident_sizes[point] * direction / length
            candidate = moved_points[point] + delta
            original_sign = float(self.domain.sdf(moved_points[point]))
            candidate_sign = float(self.domain.sdf(candidate))
            if original_sign == 0.0 or original_sign * candidate_sign >= 0.0:
                moved_points[point] = candidate

    def _cell_polygon(
        self,
        tree: Quadtree,
        cell: Cell,
        moved_points: Dict[Point, np.ndarray],
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
    ) -> np.ndarray:
        x0, y0, x1, y1 = tree.bounds(cell)
        original: List[Point] = []
        original.extend(points_on_horizontal(y_index, y0, x0, x1, reverse=False))
        original.extend(points_on_vertical(x_index, x1, y0, y1, reverse=False)[1:])
        original.extend(points_on_horizontal(y_index, y1, x0, x1, reverse=True)[1:])
        original.extend(points_on_vertical(x_index, x0, y0, y1, reverse=True)[1:])
        return clean_polygon([moved_points[round_point(p)] for p in original], self.tol)

    def _split_polygon(self, polygon: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        pieces: List[Tuple[np.ndarray, int]] = []
        inside = self._clip_polygon(polygon, keep_inside=True)
        if len(inside) >= 3:
            pieces.append((inside, -1))
        if self.include_exterior:
            outside = self._clip_polygon(polygon, keep_inside=False)
            if len(outside) >= 3:
                pieces.append((outside, 1))
        return pieces

    def _clip_polygon(self, polygon: np.ndarray, keep_inside: bool) -> np.ndarray:
        vals = np.asarray(self.domain.sdf(polygon), dtype=float)
        inside = vals <= 0.0
        keep = inside if keep_inside else ~inside
        has_keep = bool(np.any(keep))
        has_other = bool(np.any(~keep))
        if has_keep and not has_other:
            return clean_polygon(polygon, self.tol)
        if not has_keep:
            return np.empty((0, 2), dtype=float)

        clipped: List[np.ndarray] = []
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            da = float(vals[i])
            db = float(vals[(i + 1) % n])
            a_keep = (da <= 0.0) if keep_inside else (da > 0.0)
            b_keep = (db <= 0.0) if keep_inside else (db > 0.0)
            if a_keep:
                clipped.append(a)
            if a_keep != b_keep:
                clipped.append(self.domain.segment_intersection(a, b, da, db))
        return clean_polygon(clipped, self.tol)


def build_side_indices(points: Iterable[Point]) -> Tuple[Dict[float, List[Point]], Dict[float, List[Point]]]:
    x_index: Dict[float, List[Point]] = {}
    y_index: Dict[float, List[Point]] = {}
    for p in points:
        key = round_point(p)
        x_index.setdefault(key[0], []).append(key)
        y_index.setdefault(key[1], []).append(key)
    for values in x_index.values():
        values.sort(key=lambda p: p[1])
    for values in y_index.values():
        values.sort(key=lambda p: p[0])
    return x_index, y_index


def points_on_horizontal(
    y_index: Dict[float, List[Point]],
    y: float,
    x0: float,
    x1: float,
    reverse: bool,
) -> List[Point]:
    yy = round(float(y), 12)
    lo = min(x0, x1) - 1.0e-12
    hi = max(x0, x1) + 1.0e-12
    pts = [p for p in y_index.get(yy, []) if lo <= p[0] <= hi]
    if reverse:
        pts.reverse()
    return pts


def points_on_vertical(
    x_index: Dict[float, List[Point]],
    x: float,
    y0: float,
    y1: float,
    reverse: bool,
) -> List[Point]:
    xx = round(float(x), 12)
    lo = min(y0, y1) - 1.0e-12
    hi = max(y0, y1) + 1.0e-12
    pts = [p for p in x_index.get(xx, []) if lo <= p[1] <= hi]
    if reverse:
        pts.reverse()
    return pts
