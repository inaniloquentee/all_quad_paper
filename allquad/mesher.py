from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .domain import SDFDomain
from .mesh import Mesh, clean_polygon, polygon_centroid
from .quadtree import Cell, Quadtree, round_point


Point = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


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
    adaptive: bool = False

    def generate(self) -> Mesh:
        min_depth = self.min_depth if self.adaptive else self.max_depth
        tree = Quadtree.build(
            self.domain,
            min_depth=min_depth,
            max_depth=self.max_depth,
            boundary_band=self.boundary_band,
        )
        base_points = tree.collect_corners()
        grid_points = base_points
        if self.adaptive:
            grid_points = self._augment_points_for_2ref(tree, base_points)
        else:
            incident_sizes = tree.incident_min_sizes()
        base_x_index, base_y_index = build_side_indices(base_points)
        base_incident_sizes = tree.incident_min_sizes()
        moved_base_points = {
            p: self.domain.repel(
                np.array(p, dtype=float),
                self.clearance_ratio * base_incident_sizes[p],
                mode=self.repelling,
            )
            for p in base_points
        }
        moved_points = {
            p: (
                moved_base_points[p]
                if p in moved_base_points
                else self._interpolate_inserted_point(p, moved_base_points, base_x_index, base_y_index)
            )
            for p in grid_points
        }
        moved_keys = {mesh_key(point, self.tol): key for key, point in moved_points.items()}
        x_index, y_index = build_side_indices(grid_points)

        mesh = Mesh(tol=self.tol)
        for cell in sorted(tree.leaves):
            active_cut_x_index = base_x_index if self.adaptive else x_index
            active_cut_y_index = base_y_index if self.adaptive else y_index
            poly = self._cell_polygon(tree, cell, moved_points, active_cut_x_index, active_cut_y_index)
            if len(poly) < 3:
                continue
            vals = np.asarray(self.domain.sdf(poly), dtype=float)
            if not (np.any(vals <= 0.0) and np.any(vals > 0.0)):
                region = -1 if float(np.mean(vals)) <= 0.0 else 1
                if region < 0 or self.include_exterior:
                    self._add_empty_cell(mesh, tree, cell, moved_points, x_index, y_index, region)
                continue
            pieces = self._split_polygon(poly)
            for clipped, region in pieces:
                if len(clipped) < 3:
                    continue
                if self.adaptive:
                    self._add_midpoint_subdivision(mesh, clipped, moved_points, moved_keys, region)
                else:
                    mesh.add_midpoint_subdivision(clipped, region=region)
        return mesh

    def _augment_points_for_2ref(self, tree: Quadtree, points: set[Point]) -> set[Point]:
        augmented = set(points)
        base_x_index, base_y_index = build_side_indices(points)
        cut_cells = {cell for cell in tree.leaves if self._cell_may_be_cut(tree, cell)}

        for cell in cut_cells:
            boundary = self._cell_boundary_points(tree, cell, base_x_index, base_y_index)
            for a, b in zip(boundary, boundary[1:] + boundary[:1]):
                pa = np.array(a, dtype=float)
                pb = np.array(b, dtype=float)
                da = float(self.domain.sdf(pa))
                db = float(self.domain.sdf(pb))
                if da * db > 0.0:
                    augmented.add(round_point(tuple(0.5 * (pa + pb))))

        protected_sides = self._cut_neighbor_sides(tree, cut_cells)
        for _iteration in range(64):
            x_index, y_index = build_side_indices(augmented)
            additions: set[Point] = set()
            for cell in tree.leaves:
                if cell in cut_cells:
                    continue
                additions.update(
                    self._structured_closure_points(
                        tree,
                        cell,
                        x_index,
                        y_index,
                        protected_sides.get(cell, set()),
                    )
                )
            additions.difference_update(augmented)
            if not additions:
                return augmented
            augmented.update(additions)
        return augmented

    def _structured_closure_points(
        self,
        tree: Quadtree,
        cell: Cell,
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
        protected_sides: set[int],
    ) -> List[Point]:
        x0, y0, x1, y1 = tree.bounds(cell)
        h = x1 - x0
        bottom = points_on_horizontal(y_index, y0, x0, x1, reverse=False)
        right = points_on_vertical(x_index, x1, y0, y1, reverse=False)
        top = points_on_horizontal(y_index, y1, x0, x1, reverse=False)
        left = points_on_vertical(x_index, x0, y0, y1, reverse=False)
        u_values = {round((p[0] - x0) / h, 12) for p in bottom + top}
        v_values = {round((p[1] - y0) / h, 12) for p in left + right}

        additions: List[Point] = []
        if 0 not in protected_sides:
            additions.extend(round_point((x0 + u * h, y0)) for u in u_values)
        if 2 not in protected_sides:
            additions.extend(round_point((x0 + u * h, y1)) for u in u_values)
        if 3 not in protected_sides:
            additions.extend(round_point((x0, y0 + v * h)) for v in v_values)
        if 1 not in protected_sides:
            additions.extend(round_point((x1, y0 + v * h)) for v in v_values)
        return additions

    def _cut_neighbor_sides(
        self,
        tree: Quadtree,
        cut_cells: set[Cell],
    ) -> Dict[Cell, set[int]]:
        cut_bounds = [tree.bounds(cell) for cell in cut_cells]
        protected: Dict[Cell, set[int]] = {}
        for cell in tree.leaves:
            if cell in cut_cells:
                continue
            bounds = tree.bounds(cell)
            for cut_bounds_item in cut_bounds:
                side = shared_side(bounds, cut_bounds_item)
                if side is not None:
                    protected.setdefault(cell, set()).add(side)
        return protected

    def _cell_may_be_cut(self, tree: Quadtree, cell: Cell) -> bool:
        x0, y0, x1, y1 = tree.bounds(cell)
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
        vals = np.asarray(self.domain.sdf(pts), dtype=float)
        return bool(np.any(vals <= 0.0) and np.any(vals > 0.0))

    def _interpolate_inserted_point(
        self,
        point: Point,
        moved_base_points: Dict[Point, np.ndarray],
        base_x_index: Dict[float, List[Point]],
        base_y_index: Dict[float, List[Point]],
    ) -> np.ndarray:
        x, y = point
        row = base_y_index.get(round(float(y), 12), [])
        left = [p for p in row if p[0] < x]
        right = [p for p in row if p[0] > x]
        if left and right:
            a = max(left, key=lambda p: p[0])
            b = min(right, key=lambda p: p[0])
            t = (x - a[0]) / (b[0] - a[0])
            return lerp(moved_base_points[a], moved_base_points[b], t)

        col = base_x_index.get(round(float(x), 12), [])
        lower = [p for p in col if p[1] < y]
        upper = [p for p in col if p[1] > y]
        if lower and upper:
            a = max(lower, key=lambda p: p[1])
            b = min(upper, key=lambda p: p[1])
            t = (y - a[1]) / (b[1] - a[1])
            return lerp(moved_base_points[a], moved_base_points[b], t)

        return np.array(point, dtype=float)

    def _add_empty_cell(
        self,
        mesh: Mesh,
        tree: Quadtree,
        cell: Cell,
        moved_points: Dict[Point, np.ndarray],
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
        region: int,
    ) -> None:
        x0, y0, x1, y1 = tree.bounds(cell)
        corners = [moved_points[round_point(p)] for p in tree.corners(cell)]
        side_nodes = [
            points_on_horizontal(y_index, y0, x0, x1, reverse=False)[1:-1],
            points_on_vertical(x_index, x1, y0, y1, reverse=False)[1:-1],
            points_on_horizontal(y_index, y1, x0, x1, reverse=False)[1:-1],
            points_on_vertical(x_index, x0, y0, y1, reverse=False)[1:-1],
        ]
        if not self.adaptive:
            mesh.add_midpoint_subdivision(
                self._cell_polygon(tree, cell, moved_points, x_index, y_index),
                region=region,
            )
            return
        if any(len(side) > 1 for side in side_nodes):
            self._add_structured_empty_cell(mesh, tree, cell, moved_points, x_index, y_index, region)
            return
        h = [moved_points[round_point(side[0])] if side else None for side in side_nodes]
        if not any(node is not None for node in h):
            mesh.add_quad(corners, region=region)
            return
        if not self._add_2ref_template(mesh, corners, h, region):
            self._add_structured_empty_cell(mesh, tree, cell, moved_points, x_index, y_index, region)

    def _add_structured_empty_cell(
        self,
        mesh: Mesh,
        tree: Quadtree,
        cell: Cell,
        moved_points: Dict[Point, np.ndarray],
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
        region: int,
    ) -> None:
        x0, y0, x1, y1 = tree.bounds(cell)
        h = x1 - x0
        bottom = points_on_horizontal(y_index, y0, x0, x1, reverse=False)
        right = points_on_vertical(x_index, x1, y0, y1, reverse=False)
        top = points_on_horizontal(y_index, y1, x0, x1, reverse=False)
        left = points_on_vertical(x_index, x0, y0, y1, reverse=False)
        u_values = sorted({round((p[0] - x0) / h, 12) for p in bottom + top})
        v_values = sorted({round((p[1] - y0) / h, 12) for p in left + right})

        points: Dict[Tuple[int, int], np.ndarray] = {}
        for iu, u in enumerate(u_values):
            for iv, v in enumerate(v_values):
                key = round_point((x0 + u * h, y0 + v * h))
                if key in moved_points:
                    points[(iu, iv)] = moved_points[key]
                    continue
                left_p = lerp(
                    moved_points[round_point((x0, y0))],
                    moved_points[round_point((x0, y1))],
                    v,
                )
                right_p = lerp(
                    moved_points[round_point((x1, y0))],
                    moved_points[round_point((x1, y1))],
                    v,
                )
                bottom_p = lerp(
                    moved_points[round_point((x0, y0))],
                    moved_points[round_point((x1, y0))],
                    u,
                )
                top_p = lerp(
                    moved_points[round_point((x0, y1))],
                    moved_points[round_point((x1, y1))],
                    u,
                )
                bilinear = (
                    (1.0 - u) * (1.0 - v) * moved_points[round_point((x0, y0))]
                    + u * (1.0 - v) * moved_points[round_point((x1, y0))]
                    + u * v * moved_points[round_point((x1, y1))]
                    + (1.0 - u) * v * moved_points[round_point((x0, y1))]
                )
                points[(iu, iv)] = (
                    (1.0 - u) * left_p
                    + u * right_p
                    + (1.0 - v) * bottom_p
                    + v * top_p
                    - bilinear
                )

        for iu in range(len(u_values) - 1):
            for iv in range(len(v_values) - 1):
                mesh.add_quad(
                    [
                        points[(iu, iv)],
                        points[(iu + 1, iv)],
                        points[(iu + 1, iv + 1)],
                        points[(iu, iv + 1)],
                    ],
                    region=region,
                )

    def _add_midpoint_subdivision(
        self,
        mesh: Mesh,
        polygon: np.ndarray,
        moved_points: Dict[Point, np.ndarray],
        moved_keys: Dict[Tuple[int, int], Point],
        region: int,
    ) -> int:
        poly = clean_polygon(polygon, self.tol)
        if len(poly) < 3:
            return 0

        center = polygon_centroid(poly)
        mids = []
        for i, vertex in enumerate(poly):
            nxt = poly[(i + 1) % len(poly)]
            midpoint = None
            a_key = moved_keys.get(mesh_key(vertex, self.tol))
            b_key = moved_keys.get(mesh_key(nxt, self.tol))
            if a_key is not None and b_key is not None:
                mid_key = round_point((0.5 * (a_key[0] + b_key[0]), 0.5 * (a_key[1] + b_key[1])))
                midpoint = moved_points.get(mid_key)
            mids.append(midpoint if midpoint is not None else 0.5 * (vertex + nxt))

        count = 0
        for i, vertex in enumerate(poly):
            prev_mid = mids[(i - 1) % len(poly)]
            next_mid = mids[i]
            if mesh.add_quad([vertex, next_mid, center, prev_mid], region=region):
                count += 1
        return count

    def _add_2ref_template(
        self,
        mesh: Mesh,
        corners: List[np.ndarray],
        side_nodes: List[np.ndarray | None],
        region: int,
    ) -> bool:
        c0, c1, c2, c3 = corners
        h0, h1, h2, h3 = side_nodes
        center = 0.25 * (c0 + c1 + c2 + c3)
        mask = sum((1 << i) for i, node in enumerate(side_nodes) if node is not None)

        def add(points: List[np.ndarray]) -> None:
            mesh.add_quad(points, region=region)

        if mask == 3 and h0 is not None and h1 is not None:
            add([c0, h0, center, c3])
            add([h0, c1, h1, center])
            add([center, h1, c2, c3])
        elif mask == 6 and h1 is not None and h2 is not None:
            add([c0, c1, h1, center])
            add([center, h1, c2, h2])
            add([c0, center, h2, c3])
        elif mask == 12 and h2 is not None and h3 is not None:
            add([c0, c1, center, h3])
            add([c1, c2, h2, center])
            add([center, h2, c3, h3])
        elif mask == 9 and h0 is not None and h3 is not None:
            add([c0, h0, center, h3])
            add([h0, c1, c2, center])
            add([h3, center, c2, c3])
        elif mask == 5 and h0 is not None and h2 is not None:
            add([c0, h0, h2, c3])
            add([h0, c1, c2, h2])
        elif mask == 10 and h1 is not None and h3 is not None:
            add([c0, c1, h1, h3])
            add([h3, h1, c2, c3])
        elif mask == 15 and all(node is not None for node in side_nodes):
            add([c0, h0, center, h3])  # type: ignore[arg-type]
            add([h0, c1, h1, center])  # type: ignore[arg-type]
            add([center, h1, c2, h2])  # type: ignore[arg-type]
            add([h3, center, h2, c3])  # type: ignore[arg-type]
        else:
            return False
        return True

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

    def _cell_boundary_points(
        self,
        tree: Quadtree,
        cell: Cell,
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
    ) -> List[Point]:
        x0, y0, x1, y1 = tree.bounds(cell)
        original: List[Point] = []
        original.extend(points_on_horizontal(y_index, y0, x0, x1, reverse=False))
        original.extend(points_on_vertical(x_index, x1, y0, y1, reverse=False)[1:])
        original.extend(points_on_horizontal(y_index, y1, x0, x1, reverse=True)[1:])
        original.extend(points_on_vertical(x_index, x0, y0, y1, reverse=True)[1:])
        if len(original) > 1 and original[0] == original[-1]:
            original.pop()
        return original

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


def shared_side(a: Bounds, b: Bounds) -> int | None:
    eps = 1.0e-12
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    y_overlap = min(ay1, by1) - max(ay0, by0)
    if abs(ay0 - by1) <= eps and x_overlap > eps:
        return 0
    if abs(ax1 - bx0) <= eps and y_overlap > eps:
        return 1
    if abs(ay1 - by0) <= eps and x_overlap > eps:
        return 2
    if abs(ax0 - bx1) <= eps and y_overlap > eps:
        return 3
    return None


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def mesh_key(point: np.ndarray, tol: float) -> Tuple[int, int]:
    p = np.asarray(point, dtype=float)
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))
