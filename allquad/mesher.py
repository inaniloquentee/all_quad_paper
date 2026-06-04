from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .domain import SDFDomain, cross2, segment_parameters
from .mesh import Mesh, clean_polygon, midpoint_subdivision_center, polygon_area, polygon_centroid
from .quadtree import Cell, Quadtree, children, round_point


Point = Tuple[float, float]
Bounds = Tuple[float, float, float, float]
SIDE_CORNERS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
)
CORNER_SIDES = (
    (3, 0),
    (0, 1),
    (1, 2),
    (2, 3),
)


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
    sparse_boundary: bool = True
    sparse_boundary_ratio: float = 1.0
    quality_relaxation: bool = False
    quality_relaxation_iters: int = 8
    last_effective_depth: int | None = field(default=None, init=False)
    last_sparse_attempts: List[Dict[str, float | bool]] = field(default_factory=list, init=False)

    def generate(self) -> Mesh:
        self.last_sparse_attempts = []
        if self._uses_sparse_polyline_depth():
            start_depth = self._effective_max_depth()
            last_mesh: Mesh | None = None
            last_error: Exception | None = None
            for depth in range(start_depth, self.max_depth + 1):
                try:
                    mesh = self._generate_at_depth(depth)
                except ValueError as exc:
                    last_error = exc
                    self.last_sparse_attempts.append(
                        {"ok": False, "depth": float(depth), "error": str(exc)}
                    )
                    continue
                self.last_effective_depth = depth
                report = self._sparse_requirement_report(mesh)
                report["depth"] = float(depth)
                self.last_sparse_attempts.append(report)
                if report["ok"]:
                    return mesh
                last_mesh = mesh
            if last_mesh is not None:
                return last_mesh
            if last_error is not None:
                raise last_error

        max_depth = self._effective_max_depth()
        self.last_effective_depth = max_depth
        return self._generate_at_depth(max_depth)

    def _generate_at_depth(self, max_depth: int) -> Mesh:
        min_depth = min(self.min_depth if self.adaptive else max_depth, max_depth)
        sparse_boundary_size = self._sparse_boundary_size() if self.adaptive else None
        tree = Quadtree.build(
            self.domain,
            min_depth=min_depth,
            max_depth=max_depth,
            boundary_band=self.boundary_band,
            sparse_boundary_size=sparse_boundary_size,
        )
        if self.adaptive:
            self._refine_tree_for_2ref(tree, max_depth)
        base_points = tree.collect_corners()
        grid_points = base_points
        if self.adaptive:
            grid_points = self._augment_points_for_2ref(tree, base_points)
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
                else self._interpolate_inserted_point(
                    p,
                    moved_base_points,
                    *build_side_indices(base_points),
                )
            )
            for p in grid_points
        }
        if self.adaptive:
            self._pull_vertex_cell_side_midpoints(tree, moved_points)
        moved_keys = {mesh_key(point, self.tol): key for key, point in moved_points.items()}
        x_index, y_index = build_side_indices(grid_points)

        mesh = Mesh(tol=self.tol)
        for cell in sorted(tree.leaves):
            poly = self._cell_square_polygon(tree, cell, moved_points)
            if len(poly) < 3:
                continue
            vals = np.asarray(self.domain.sdf(poly), dtype=float)
            sign_cut = bool(np.any(vals <= 0.0) and np.any(vals > 0.0))
            boundary_cut = self._cell_may_be_cut(tree, cell)
            if not (sign_cut or boundary_cut):
                region = -1 if float(np.mean(vals)) <= 0.0 else 1
                if region < 0 or self.include_exterior:
                    self._add_empty_cell(mesh, tree, cell, moved_points, x_index, y_index, region)
                continue
            pieces = self._split_cell_polygon(tree, cell, poly, moved_points, moved_keys)
            for clipped, region in pieces:
                if len(clipped) < 3:
                    continue
                if self.adaptive:
                    self._add_midpoint_subdivision(mesh, clipped, moved_points, moved_keys, region)
                else:
                    mesh.add_midpoint_subdivision(clipped, region=region)
        if self.adaptive and self.quality_relaxation:
            self._relax_free_vertices(mesh)
        return mesh

    def _uses_sparse_polyline_depth(self) -> bool:
        return (
            self.adaptive
            and self.sparse_boundary
            and hasattr(self.domain, "min_segment_length")
            and self._sparse_boundary_size() is not None
        )

    def _sparse_requirement_report(self, mesh: Mesh) -> Dict[str, float | bool]:
        quality = mesh.quality()
        by_region = mesh.quality_by_region()
        topology = mesh.topology(self.domain.sdf)
        ok = (
            quality["quads"] > 0
            and quality["min_area"] > 0.0
            and topology["nonmanifold_edges"] == 0.0
        )
        if self.include_exterior:
            ok = (
                ok
                and by_region.get("interior", {}).get("quads", 0.0) > 0.0
                and by_region.get("exterior", {}).get("quads", 0.0) > 0.0
                and topology.get("interface_edges", 0.0) > 0.0
            )
            midpoint_error = topology.get("max_interface_midpoint_error")
            if midpoint_error is not None:
                ok = ok and midpoint_error <= 1.0e-6
            endpoint_error = topology.get("max_interface_endpoint_error")
            if endpoint_error is not None:
                ok = ok and endpoint_error <= 1.0e-6
        else:
            ok = ok and by_region.get("interior", {}).get("quads", 0.0) > 0.0

        return {
            "ok": bool(ok),
            "quads": quality["quads"],
            "min_angle": quality["min_angle"],
            "max_angle": quality["max_angle"],
            "nonmanifold_edges": topology["nonmanifold_edges"],
            "interface_edges": topology.get("interface_edges", 0.0),
            "max_interface_midpoint_error": topology.get("max_interface_midpoint_error", 0.0),
        }

    def _sparse_boundary_size(self) -> float | None:
        if not self.sparse_boundary or not hasattr(self.domain, "min_segment_length"):
            return None
        return float(self.domain.min_segment_length()) * self.sparse_boundary_ratio

    def _effective_max_depth(self) -> int:
        if not self.adaptive:
            return self.max_depth
        sparse_boundary_size = self._sparse_boundary_size()
        if sparse_boundary_size is None or sparse_boundary_size <= 0.0:
            return self.max_depth
        xmin, ymin, xmax, ymax = self.domain.bounds
        span = max(xmax - xmin, ymax - ymin)
        if span <= 0.0:
            return self.max_depth
        sparse_depth = int(math.ceil(math.log2(span / sparse_boundary_size)))
        return max(self.min_depth, min(self.max_depth, sparse_depth))

    def _augment_points_for_2ref(self, tree: Quadtree, points: set[Point]) -> set[Point]:
        closed, bad_cells = self._closed_2ref_points(tree, set(points))
        if bad_cells:
            raise ValueError("quadtree still has multi-node 2-ref sides after preprocessing")
        return closed

    def _refine_tree_for_2ref(self, tree: Quadtree, max_depth: int) -> None:
        for _iteration in range(4 * max_depth + 8):
            _points, bad_cells = self._closed_2ref_points(tree, tree.collect_corners())
            if not bad_cells:
                return
            refinable = {cell for cell in bad_cells if cell.level < max_depth}
            if not refinable:
                raise ValueError("2-ref interface has multiple hanging nodes at max depth")
            tree.leaves.difference_update(refinable)
            for cell in refinable:
                tree.leaves.update(children(cell))
            tree.refine_until_conforming(
                self.domain,
                self.boundary_band,
                max_depth,
                self._sparse_boundary_size(),
            )
        raise ValueError("could not make quadtree compatible with 2-ref templates")

    def _closed_2ref_points(self, tree: Quadtree, points: set[Point]) -> Tuple[set[Point], set[Cell]]:
        augmented = set(points)
        augmented.update(self._cut_cell_midpoint_keys(tree))

        for _iteration in range(64):
            x_index, y_index = build_side_indices(augmented)
            bad_cells = {
                cell
                for cell in tree.leaves
                if any(len(side) > 1 for side in self._cell_side_node_keys(tree, cell, x_index, y_index))
            }
            if bad_cells:
                return augmented, bad_cells

            additions: set[Point] = set()
            for cell in tree.leaves:
                additions.update(self._corner_marking_2ref_points(tree, cell, x_index, y_index))
            additions.difference_update(augmented)
            if not additions:
                return augmented, set()
            augmented.update(additions)
        raise ValueError("2-ref corner marking did not converge")

    def _cut_cell_midpoint_keys(self, tree: Quadtree) -> set[Point]:
        points: set[Point] = set()
        cut_cells = {cell for cell in tree.leaves if self._cell_may_be_cut(tree, cell)}
        for cell in cut_cells:
            side_hits = self._cell_side_curve_hits(tree.bounds(cell))
            for side in range(4):
                if side not in side_hits:
                    points.add(self._side_midpoint_key(tree, cell, side))
            points.update(self._vertex_side_midpoint_keys(tree, cell))
        return points

    def _corner_marking_2ref_points(
        self,
        tree: Quadtree,
        cell: Cell,
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
    ) -> set[Point]:
        side_nodes = self._cell_side_node_keys(tree, cell, x_index, y_index)
        counts = [len(nodes) for nodes in side_nodes]
        if not any(counts):
            return set()

        if any(count > 1 for count in counts):
            raise ValueError("strongly-balanced 2-ref expects at most one hanging node per side")

        present = {side for side, nodes in enumerate(side_nodes) if nodes}
        if len(present) == 4:
            return set()

        if len(present) == 1:
            side = next(iter(present))
            marked = self._marked_corners_for_cell(cell)
            for corner in SIDE_CORNERS[side]:
                if corner in marked:
                    return {
                        self._side_midpoint_key(tree, cell, other_corner_side(corner, side))
                    }
            return {self._side_midpoint_key(tree, cell, other_corner_side(SIDE_CORNERS[side][0], side))}

        if len(present) == 2:
            sides = sorted(present)
            corner = corner_between_sides(sides[0], sides[1])
            if corner is not None and corner in self._marked_corners_for_cell(cell):
                return set()
            return {
                self._side_midpoint_key(tree, cell, side)
                for side in range(4)
                if side not in present
            }

        if len(present) == 3:
            return {
                self._side_midpoint_key(tree, cell, side)
                for side in range(4)
                if side not in present
            }

        return set()

    def _cell_side_node_keys(
        self,
        tree: Quadtree,
        cell: Cell,
        x_index: Dict[float, List[Point]],
        y_index: Dict[float, List[Point]],
    ) -> List[List[Point]]:
        x0, y0, x1, y1 = tree.bounds(cell)
        return [
            points_on_horizontal(y_index, y0, x0, x1, reverse=False)[1:-1],
            points_on_vertical(x_index, x1, y0, y1, reverse=False)[1:-1],
            points_on_horizontal(y_index, y1, x0, x1, reverse=False)[1:-1],
            points_on_vertical(x_index, x0, y0, y1, reverse=False)[1:-1],
        ]

    def _side_midpoint_key(self, tree: Quadtree, cell: Cell, side: int) -> Point:
        x0, y0, x1, y1 = tree.bounds(cell)
        points = (
            (0.5 * (x0 + x1), y0),
            (x1, 0.5 * (y0 + y1)),
            (0.5 * (x0 + x1), y1),
            (x0, 0.5 * (y0 + y1)),
        )
        return round_point(points[side])

    def _marked_corners_for_cell(self, cell: Cell) -> set[int]:
        return {0, 2} if (cell.i + cell.j) % 2 == 0 else {1, 3}

    def _cell_may_be_cut(self, tree: Quadtree, cell: Cell) -> bool:
        x0, y0, x1, y1 = tree.bounds(cell)
        if hasattr(self.domain, "segments_in_box"):
            return bool(self.domain.segments_in_box((x0, y0, x1, y1), self.tol))
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

    def _vertex_side_midpoint_keys(self, tree: Quadtree, cell: Cell) -> set[Point]:
        if not hasattr(self.domain, "vertices_in_box") or not hasattr(self.domain, "segments_in_box"):
            return set()
        bounds = tree.bounds(cell)
        vertices = self.domain.vertices_in_box(bounds, self.tol)
        if len(vertices) != 1:
            return set()

        x0, y0, x1, y1 = bounds
        side_hits = self._cell_side_curve_hits(bounds)
        side_midpoints = [
            (0, round_point((0.5 * (x0 + x1), y0))),
            (1, round_point((x1, 0.5 * (y0 + y1)))),
            (2, round_point((0.5 * (x0 + x1), y1))),
            (3, round_point((x0, 0.5 * (y0 + y1)))),
        ]
        return {point for side, point in side_midpoints if side not in side_hits}

    def _pull_vertex_cell_side_midpoints(
        self,
        tree: Quadtree,
        moved_points: Dict[Point, np.ndarray],
    ) -> None:
        if not hasattr(self.domain, "vertices_in_box"):
            return
        for cell in tree.leaves:
            vertices = self.domain.vertices_in_box(tree.bounds(cell), self.tol)
            if len(vertices) != 1:
                continue
            loop_id, vertex_id = vertices[0]
            vertex = self.domain.loops[loop_id][vertex_id]
            x0, _y0, x1, _y1 = tree.bounds(cell)
            distance = 0.125 * (x1 - x0)
            for key in self._vertex_side_midpoint_keys(tree, cell):
                point = moved_points.get(key, np.array(key, dtype=float))
                if np.linalg.norm(point - vertex) <= self.tol:
                    continue
                moved_points[key] = move_toward(point, vertex, distance)

    def _cell_side_curve_hits(self, bounds: Bounds) -> set[int]:
        if not hasattr(self.domain, "iter_segments"):
            return set()
        x0, y0, x1, y1 = bounds
        sides = [
            (np.array([x0, y0], dtype=float), np.array([x1, y0], dtype=float)),
            (np.array([x1, y0], dtype=float), np.array([x1, y1], dtype=float)),
            (np.array([x1, y1], dtype=float), np.array([x0, y1], dtype=float)),
            (np.array([x0, y1], dtype=float), np.array([x0, y0], dtype=float)),
        ]
        hits: set[int] = set()
        for side_id, (a, b) in enumerate(sides):
            side_vec = b - a
            for _loop_id, _edge_id, c, d in self.domain.iter_segments():
                hit = segment_parameters(a, side_vec, c, d - c)
                if hit is None:
                    continue
                t, u = hit
                if -self.tol <= t <= 1.0 + self.tol and -self.tol <= u <= 1.0 + self.tol:
                    hits.add(side_id)
                    break
        return hits

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
            mesh.add_quad(corners, region=region)
            return
        if any(len(side) > 1 for side in side_nodes):
            raise ValueError("strongly-balanced 2-ref expects at most one hanging node per side")
        h = [moved_points[round_point(side[0])] if side else None for side in side_nodes]
        if not any(node is not None for node in h):
            mesh.add_quad(corners, region=region)
            return
        if not self._add_2ref_template(mesh, corners, h, region):
            raise ValueError("corner marking did not produce a Fig. 7 2-ref template")

    def _add_midpoint_subdivision(
        self,
        mesh: Mesh,
        polygon: np.ndarray,
        moved_points: Dict[Point, np.ndarray],
        moved_keys: Dict[Tuple[int, int], Point],
        region: int,
        relabel_children: bool = False,
    ) -> int:
        poly = clean_polygon(polygon, self.tol)
        if len(poly) < 3:
            return 0

        center = midpoint_subdivision_center(poly)
        mids = self._midpoint_subdivision_mids(poly, moved_points, moved_keys)

        count = 0
        for i, vertex in enumerate(poly):
            prev_mid = mids[(i - 1) % len(poly)]
            next_mid = mids[i]
            quad = [vertex, next_mid, center, prev_mid]
            quad_region = self._child_quad_region(quad, region) if relabel_children else region
            if quad_region > 0 and not self.include_exterior:
                continue
            if mesh.add_quad(quad, region=quad_region):
                count += 1
        return count

    def _midpoint_subdivision_mids(
        self,
        polygon: np.ndarray,
        moved_points: Dict[Point, np.ndarray] | None = None,
        moved_keys: Dict[Tuple[int, int], Point] | None = None,
    ) -> List[np.ndarray]:
        mids: List[np.ndarray] = []
        for i, vertex in enumerate(polygon):
            nxt = polygon[(i + 1) % len(polygon)]
            midpoint = None
            if moved_points is not None and moved_keys is not None:
                a_key = moved_keys.get(mesh_key(vertex, self.tol))
                b_key = moved_keys.get(mesh_key(nxt, self.tol))
                if a_key is not None and b_key is not None:
                    mid_key = round_point((0.5 * (a_key[0] + b_key[0]), 0.5 * (a_key[1] + b_key[1])))
                    midpoint = moved_points.get(mid_key)
            if midpoint is None:
                midpoint = 0.5 * (vertex + nxt)
            mids.append(midpoint)
        return mids

    def _child_quad_region(self, points: List[np.ndarray], fallback: int) -> int:
        quad = clean_polygon(points, self.tol)
        if len(quad) != 4:
            return fallback
        center = np.mean(quad, axis=0)
        return -1 if float(self.domain.sdf(center)) <= 0.0 else 1

    def _relax_free_vertices(self, mesh: Mesh) -> None:
        if not mesh.quads:
            return
        points = np.asarray(mesh.vertices, dtype=float).copy()
        incident: Dict[int, List[int]] = {}
        neighbors: Dict[int, set[int]] = {}
        edge_use: Dict[Tuple[int, int], int] = {}
        for quad_id, quad in enumerate(mesh.quads):
            for i, vertex_id in enumerate(quad):
                incident.setdefault(vertex_id, []).append(quad_id)
                neighbors.setdefault(vertex_id, set()).update({quad[(i - 1) % 4], quad[(i + 1) % 4]})
                a = vertex_id
                b = quad[(i + 1) % 4]
                edge = (a, b) if a < b else (b, a)
                edge_use[edge] = edge_use.get(edge, 0) + 1

        distances = np.abs(np.asarray(self.domain.sdf(points), dtype=float))
        fixed = distances <= 1.0e-9
        for edge, count in edge_use.items():
            if count == 1:
                if distances[edge[0]] > 0.05:
                    fixed[edge[0]] = True
                if distances[edge[1]] > 0.05:
                    fixed[edge[1]] = True

        vertex_regions: Dict[int, set[int]] = {}
        for quad_id, quad in enumerate(mesh.quads):
            region = mesh.regions[quad_id]
            for vertex_id in quad:
                vertex_regions.setdefault(vertex_id, set()).add(region)

        for _iteration in range(max(self.quality_relaxation_iters, 0)):
            bad_vertices = self._bad_angle_vertices(mesh, points)
            changed = 0
            for vertex_id in sorted(bad_vertices):
                if fixed[vertex_id] or not neighbors.get(vertex_id):
                    continue
                if len(vertex_regions.get(vertex_id, set())) > 1:
                    continue
                current = points[vertex_id].copy()
                base_score = self._local_relaxation_score(mesh, points, incident[vertex_id])
                candidates = self._relaxation_candidates(
                    points,
                    vertex_id,
                    neighbors[vertex_id],
                    incident[vertex_id],
                    mesh,
                )
                best = current
                best_score = base_score
                for candidate in candidates:
                    if not self._candidate_stays_in_region(candidate, vertex_regions.get(vertex_id, set())):
                        continue
                    old = points[vertex_id].copy()
                    points[vertex_id] = candidate
                    score = self._local_relaxation_score(mesh, points, incident[vertex_id])
                    points[vertex_id] = old
                    if score < best_score:
                        best = candidate
                        best_score = score
                if best_score < base_score:
                    points[vertex_id] = best
                    changed += 1
            if changed == 0:
                break

        mesh.vertices = [points[i].copy() for i in range(len(mesh.vertices))]

    def _bad_angle_vertices(self, mesh: Mesh, points: np.ndarray) -> set[int]:
        bad: set[int] = set()
        for quad in mesh.quads:
            polygon = points[list(quad)]
            angles = quad_angles(polygon, self.tol)
            if angles and (min(angles) < 30.0 or max(angles) > 150.0):
                bad.update(quad)
        return bad

    def _local_relaxation_score(self, mesh: Mesh, points: np.ndarray, quad_ids: List[int]) -> Tuple[float, float, float, float]:
        min_angle = float("inf")
        max_angle = -float("inf")
        min_area = float("inf")
        for quad_id in quad_ids:
            polygon = points[list(mesh.quads[quad_id])]
            area = polygon_area(polygon)
            if area <= self.tol * self.tol:
                return (float("inf"), float("inf"), float("inf"), float("inf"))
            angles = quad_angles(polygon, self.tol)
            if not angles:
                return (float("inf"), float("inf"), float("inf"), float("inf"))
            min_angle = min(min_angle, min(angles))
            max_angle = max(max_angle, max(angles))
            min_area = min(min_area, area)
        violation = max(30.0 - min_angle, max_angle - 150.0, 0.0)
        return (violation, -min_angle, max_angle, -min_area)

    def _relaxation_candidates(
        self,
        points: np.ndarray,
        vertex_id: int,
        neighbors: set[int],
        quad_ids: List[int],
        mesh: Mesh,
    ) -> List[np.ndarray]:
        current = points[vertex_id]
        neighbor_points = [points[neighbor] for neighbor in neighbors]
        average = np.mean(neighbor_points, axis=0)
        scale = float(np.mean([np.linalg.norm(current - point) for point in neighbor_points]))
        if scale <= self.tol:
            return []

        candidates: List[np.ndarray] = []
        for alpha in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            candidates.append((1.0 - alpha) * current + alpha * average)

        directions = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
        toward_average = average - current
        if np.linalg.norm(toward_average) > self.tol:
            directions.append(toward_average / np.linalg.norm(toward_average))
        for neighbor in neighbors:
            direction = points[neighbor] - current
            length = np.linalg.norm(direction)
            if length > self.tol:
                unit = direction / length
                directions.append(unit)
                directions.append(np.array([-unit[1], unit[0]], dtype=float))
                directions.append(np.array([unit[1], -unit[0]], dtype=float))
        for quad_id in quad_ids:
            center = np.mean(points[list(mesh.quads[quad_id])], axis=0)
            direction = center - current
            length = np.linalg.norm(direction)
            if length > self.tol:
                directions.append(direction / length)

        for direction in directions:
            for step in (0.02, 0.05, 0.1, 0.2, 0.35):
                candidates.append(current + direction * scale * step)
        return candidates

    def _candidate_stays_in_region(self, candidate: np.ndarray, regions: set[int] | None) -> bool:
        if not regions:
            return True
        value = float(self.domain.sdf(candidate))
        if regions == {-1}:
            return value < -self.tol
        if regions == {1}:
            return value > self.tol
        return False

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

    def _cell_square_polygon(
        self,
        tree: Quadtree,
        cell: Cell,
        moved_points: Dict[Point, np.ndarray],
    ) -> np.ndarray:
        return clean_polygon([moved_points[round_point(p)] for p in tree.corners(cell)], self.tol)

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

    def _split_cell_polygon(
        self,
        tree: Quadtree,
        cell: Cell,
        polygon: np.ndarray,
        moved_points: Dict[Point, np.ndarray],
        moved_keys: Dict[Tuple[int, int], Point],
    ) -> List[Tuple[np.ndarray, int]]:
        if hasattr(self.domain, "vertices_in_box") and hasattr(self.domain, "iter_segments"):
            vertex_template = self._split_polyline_vertex_template_cell(tree, cell, polygon, moved_points, moved_keys)
            if vertex_template:
                return vertex_template
            vertex_chain = self._split_polyline_vertex_chain_cell(tree, cell, polygon)
            if vertex_chain:
                return vertex_chain
            exact_segments = self._split_polyline_segment_cell(tree, cell, polygon)
            if exact_segments:
                return exact_segments
            single = self._split_polyline_single_segment_cell(tree, cell, polygon)
            if single:
                return single
        return self._split_polygon(polygon)

    def _split_polyline_segment_cell(
        self,
        tree: Quadtree,
        cell: Cell,
        polygon: np.ndarray,
    ) -> List[Tuple[np.ndarray, int]]:
        segments = self._polyline_segments_in_polygon(polygon)
        if not segments:
            return []

        pieces = [polygon]
        for loop_id, edge_id in segments:
            loop = self.domain.loops[loop_id]
            a = loop[edge_id]
            b = loop[(edge_id + 1) % len(loop)]
            next_pieces: List[np.ndarray] = []
            split_count = 0
            for piece in pieces:
                intersections = self._segment_polygon_intersections(piece, a, b)
                if len(intersections) != 2:
                    next_pieces.append(piece)
                    continue
                first, second = sorted(intersections, key=lambda item: item[1] + item[2])
                if np.linalg.norm(first[0] - second[0]) <= self.tol:
                    next_pieces.append(piece)
                    continue
                split = [poly for poly in split_polygon_by_chord(piece, first, second, self.tol) if len(poly) >= 3]
                if len(split) != 2:
                    next_pieces.append(piece)
                    continue
                next_pieces.extend(split)
                split_count += 1
            pieces = next_pieces
            if split_count == 0:
                continue

        result = self._classify_polyline_pieces(pieces)
        if len(result) < 2 or len({region for _poly, region in result}) <= 1:
            return []
        return result

    def _split_polyline_single_segment_cell(
        self,
        tree: Quadtree,
        cell: Cell,
        polygon: np.ndarray,
    ) -> List[Tuple[np.ndarray, int]]:
        segments = self._polyline_segments_in_polygon(polygon)
        if len(segments) != 1:
            return []
        loop_id, edge_id = segments[0]
        loop = self.domain.loops[loop_id]
        a = loop[edge_id]
        b = loop[(edge_id + 1) % len(loop)]
        intersections = self._segment_polygon_intersections(polygon, a, b)
        if len(intersections) != 2:
            return []
        first, second = sorted(intersections, key=lambda item: item[2] + item[1])
        if np.linalg.norm(first[0] - second[0]) <= self.tol:
            return []
        split = split_polygon_by_chord(polygon, first, second, self.tol)
        pieces: List[Tuple[np.ndarray, int]] = []
        for poly in split:
            if len(poly) < 3:
                continue
            region = -1 if float(self.domain.sdf(representative_point(poly, self.tol))) <= 0.0 else 1
            if region < 0 or self.include_exterior:
                pieces.append((poly, region))
        if len(pieces) < 2:
            return []
        return pieces if len({region for _poly, region in pieces}) > 1 else []

    def _split_polyline_vertex_chain_cell(
        self,
        tree: Quadtree,
        cell: Cell,
        polygon: np.ndarray,
    ) -> List[Tuple[np.ndarray, int]]:
        vertices = self._polyline_vertices_in_polygon(polygon)
        if len(vertices) != 1:
            return []

        loop_id, vertex_id = vertices[0]
        loop = self.domain.loops[loop_id]
        vertex = loop[vertex_id]
        previous = loop[(vertex_id - 1) % len(loop)]
        nxt = loop[(vertex_id + 1) % len(loop)]

        incoming = self._segment_polygon_intersections(polygon, previous, vertex)
        outgoing = self._segment_polygon_intersections(polygon, vertex, nxt)
        incoming = [item for item in incoming if np.linalg.norm(item[0] - vertex) > self.tol]
        outgoing = [item for item in outgoing if np.linalg.norm(item[0] - vertex) > self.tol]
        if not incoming or not outgoing:
            return []

        in_hit = min(incoming, key=lambda item: np.linalg.norm(item[0] - vertex))
        out_hit = min(outgoing, key=lambda item: np.linalg.norm(item[0] - vertex))
        if np.linalg.norm(in_hit[0] - out_hit[0]) <= self.tol:
            return []

        result = self._sharp_parent_regions(polygon, vertex, in_hit, out_hit)
        result = [
            (poly, region)
            for poly, region in result
            if region < 0 or self.include_exterior
        ]
        if len(result) < 2 or len({region for _poly, region in result}) <= 1:
            return []
        return result

    def _split_polyline_vertex_template_cell(
        self,
        tree: Quadtree,
        cell: Cell,
        polygon: np.ndarray,
        moved_points: Dict[Point, np.ndarray],
        moved_keys: Dict[Tuple[int, int], Point],
    ) -> List[Tuple[np.ndarray, int]]:
        vertices = self._polyline_vertices_in_polygon(polygon)
        if len(vertices) != 1:
            return []

        loop_id, vertex_id = vertices[0]
        loop = self.domain.loops[loop_id]
        vertex = loop[vertex_id]
        previous = loop[(vertex_id - 1) % len(loop)]
        nxt = loop[(vertex_id + 1) % len(loop)]

        incoming = self._segment_polygon_intersections(polygon, previous, vertex)
        outgoing = self._segment_polygon_intersections(polygon, vertex, nxt)
        incoming = [item for item in incoming if np.linalg.norm(item[0] - vertex) > self.tol]
        outgoing = [item for item in outgoing if np.linalg.norm(item[0] - vertex) > self.tol]
        if not incoming or not outgoing:
            return []

        in_hit = min(incoming, key=lambda item: np.linalg.norm(item[0] - vertex))
        out_hit = min(outgoing, key=lambda item: np.linalg.norm(item[0] - vertex))
        if np.linalg.norm(in_hit[0] - out_hit[0]) <= self.tol:
            return []

        parent_regions = self._sharp_parent_regions(polygon, vertex, in_hit, out_hit)
        if len({region for _parent, region in parent_regions}) <= 1:
            return []

        side_midpoints = [
            point
            for point, _edge_id, _u in self._vertex_template_side_midpoints(
                tree,
                cell,
                polygon,
                moved_points,
                vertex,
                {in_hit[1], out_hit[1]},
            )
        ]
        pieces: List[Tuple[np.ndarray, int]] = []
        for parent, region in parent_regions:
            if region > 0 and not self.include_exterior:
                continue
            local_points = [
                point
                for point in side_midpoints
                if self._side_point_belongs_to_parent(point, vertex, parent, region)
            ]
            children = split_polygon_by_vertex_spokes(parent, vertex, local_points, self.tol)
            for poly in children:
                for piece in self._decompose_vertex_piece(poly, vertex):
                    if len(piece) >= 3:
                        pieces.append((piece, region))

        return pieces

    def _decompose_vertex_piece(self, polygon: np.ndarray, vertex: np.ndarray) -> List[np.ndarray]:
        poly = clean_polygon(polygon, self.tol)
        if len(poly) <= 3 or polygon_is_convex(poly, self.tol):
            return [poly]

        vertex_index = find_point_index(list(poly), vertex, self.tol)
        if vertex_index is None:
            return [poly]

        edge_midpoints: List[np.ndarray] = []
        for i, point in enumerate(poly):
            next_index = (i + 1) % len(poly)
            if i == vertex_index or next_index == vertex_index:
                continue
            nxt = poly[next_index]
            if np.linalg.norm(nxt - point) > self.tol:
                edge_midpoints.append(0.5 * (point + nxt))

        split = split_polygon_by_vertex_spokes(poly, vertex, edge_midpoints, self.tol)
        if len(split) > 1:
            pieces = [piece for piece in split if len(piece) >= 3]
            if pieces and all(len(piece) <= 3 or polygon_is_convex(piece, self.tol) for piece in pieces):
                return pieces

        ordered = list(poly[vertex_index:]) + list(poly[:vertex_index])
        pieces: List[np.ndarray] = []
        for i in range(1, len(ordered) - 1):
            tri = clean_polygon([ordered[0], ordered[i], ordered[i + 1]], self.tol)
            if len(tri) == 3:
                pieces.append(tri)
        return pieces or [poly]

    def _side_point_belongs_to_parent(
        self,
        point: np.ndarray,
        vertex: np.ndarray,
        parent: np.ndarray,
        region: int,
    ) -> bool:
        point_region = -1 if float(self.domain.sdf(point)) <= 0.0 else 1
        if point_region != region:
            return False
        if point_in_polygon(point, parent, self.tol):
            return True
        midpoint = 0.5 * (point + vertex)
        return point_in_polygon(midpoint, parent, self.tol)

    def _vertex_template_side_midpoints(
        self,
        tree: Quadtree,
        cell: Cell,
        polygon: np.ndarray,
        moved_points: Dict[Point, np.ndarray],
        vertex: np.ndarray,
        curve_sides: set[int],
    ) -> List[Tuple[np.ndarray, int, float]]:
        side_hits = self._cell_side_curve_hits(tree.bounds(cell)) | curve_sides
        points: List[Tuple[np.ndarray, int, float]] = []
        for side in range(4):
            if side in side_hits:
                continue
            key = self._side_midpoint_key(tree, cell, side)
            point = moved_points.get(key)
            if point is None:
                point = 0.5 * (polygon[side] + polygon[(side + 1) % len(polygon)])
            if np.linalg.norm(point - vertex) <= self.tol:
                continue
            points.append((point, side, 0.5))
        return points

    def _polyline_vertices_in_polygon(
        self,
        polygon: np.ndarray,
        sharp_only: bool = False,
    ) -> List[Tuple[int, int]]:
        if not hasattr(self.domain, "loops"):
            return []
        xmin = float(np.min(polygon[:, 0])) - self.tol
        xmax = float(np.max(polygon[:, 0])) + self.tol
        ymin = float(np.min(polygon[:, 1])) - self.tol
        ymax = float(np.max(polygon[:, 1])) + self.tol
        vertices: List[Tuple[int, int]] = []
        for loop_id, loop in enumerate(self.domain.loops):
            for vertex_id, point in enumerate(loop):
                if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
                    continue
                if sharp_only and not self.domain.is_sharp_vertex(loop_id, vertex_id):
                    continue
                if point_in_polygon(point, polygon, self.tol):
                    vertices.append((loop_id, vertex_id))
        return vertices

    def _polyline_segments_in_polygon(self, polygon: np.ndarray) -> List[Tuple[int, int]]:
        if not hasattr(self.domain, "iter_segments"):
            return []
        segments: List[Tuple[int, int]] = []
        for loop_id, edge_id, a, b in self.domain.iter_segments():
            if point_in_polygon(a, polygon, self.tol) or point_in_polygon(b, polygon, self.tol):
                segments.append((loop_id, edge_id))
                continue
            if self._segment_polygon_intersections(polygon, a, b):
                segments.append((loop_id, edge_id))
        return segments

    def _classify_polyline_pieces(self, pieces: List[np.ndarray]) -> List[Tuple[np.ndarray, int]]:
        result: List[Tuple[np.ndarray, int]] = []
        for poly in pieces:
            if len(poly) < 3:
                continue
            region = -1 if float(self.domain.sdf(representative_point(poly, self.tol))) <= 0.0 else 1
            if region < 0 or self.include_exterior:
                result.append((poly, region))
        return result

    def _sharp_parent_regions(
        self,
        polygon: np.ndarray,
        vertex: np.ndarray,
        in_hit: Tuple[np.ndarray, int, float],
        out_hit: Tuple[np.ndarray, int, float],
    ) -> List[Tuple[np.ndarray, int]]:
        parents = split_polygon_by_chain(polygon, in_hit, [in_hit[0], vertex, out_hit[0]], out_hit, self.tol)
        if len(parents) != 2:
            return []
        outside_parent, inside_parent = parents
        regions: List[Tuple[np.ndarray, int]] = []
        if len(outside_parent) >= 3:
            regions.append((outside_parent, 1))
        if len(inside_parent) >= 3:
            regions.append((inside_parent, -1))
        return regions

    def _sharp_piece_region(self, polygon: np.ndarray, parent_regions: List[Tuple[np.ndarray, int]]) -> int:
        center = representative_point(polygon, self.tol)
        for parent, region in parent_regions:
            if point_in_polygon(center, parent, self.tol):
                return region
        return -1 if float(self.domain.sdf(center)) <= 0.0 else 1

    def _segment_polygon_intersections(
        self,
        polygon: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> List[Tuple[np.ndarray, int, float]]:
        hits: List[Tuple[np.ndarray, int, float]] = []
        r = b - a
        n = len(polygon)
        for edge_id in range(n):
            c = polygon[edge_id]
            d = polygon[(edge_id + 1) % n]
            hit = segment_parameters(a, r, c, d - c)
            if hit is None:
                continue
            t, u = hit
            if -self.tol <= t <= 1.0 + self.tol and -self.tol <= u <= 1.0 + self.tol:
                point = a + np.clip(t, 0.0, 1.0) * r
                if not any(np.linalg.norm(point - existing[0]) <= self.tol for existing in hits):
                    hits.append((point, edge_id, float(np.clip(u, 0.0, 1.0))))
        return hits

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


def other_corner_side(corner: int, side: int) -> int:
    sides = CORNER_SIDES[corner]
    if sides[0] == side:
        return sides[1]
    if sides[1] == side:
        return sides[0]
    raise ValueError(f"corner {corner} is not incident to side {side}")


def corner_between_sides(a: int, b: int) -> int | None:
    sides = {a, b}
    for corner, incident in enumerate(CORNER_SIDES):
        if sides == set(incident):
            return corner
    return None


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def move_toward(point: np.ndarray, target: np.ndarray, distance: float) -> np.ndarray:
    direction = target - point
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-14:
        return point
    return point + direction / length * min(distance, 0.5 * length)


def mesh_key(point: np.ndarray, tol: float) -> Tuple[int, int]:
    p = np.asarray(point, dtype=float)
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))


def quad_angles(quad: np.ndarray, tol: float) -> List[float]:
    angles = []
    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= tol:
            return []
        cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
    return angles


def representative_point(polygon: np.ndarray, tol: float) -> np.ndarray:
    poly = np.asarray(polygon, dtype=float)
    centroid = polygon_centroid(poly)
    if point_in_polygon(centroid, poly, tol):
        return centroid

    n = len(poly)
    for i in range(n):
        tri = np.asarray([poly[i], poly[(i + 1) % n], poly[(i + 2) % n]], dtype=float)
        if abs(polygon_area(tri)) <= tol * tol:
            continue
        candidate = np.mean(tri, axis=0)
        if point_in_polygon(candidate, poly, tol):
            return candidate

    span = max(float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1])), 1.0)
    for i, a in enumerate(poly):
        b = poly[(i + 1) % n]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length <= tol:
            continue
        normal = np.array([-edge[1], edge[0]], dtype=float) / length
        midpoint = 0.5 * (a + b)
        for scale in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3):
            candidate = midpoint + scale * span * normal
            if point_in_polygon(candidate, poly, tol):
                return candidate

    return centroid


def point_in_polygon(point: np.ndarray, polygon: np.ndarray, tol: float) -> bool:
    p = np.asarray(point, dtype=float)
    n = len(polygon)
    inside = False
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if point_on_segment(p, a, b, tol):
            return True
        if ((a[1] > p[1]) != (b[1] > p[1])) and (
            p[0] < (b[0] - a[0]) * (p[1] - a[1]) / ((b[1] - a[1]) if b[1] != a[1] else 1.0e-300) + a[0]
        ):
            inside = not inside
    return inside


def polygon_is_convex(polygon: np.ndarray, tol: float) -> bool:
    if len(polygon) < 4:
        return True
    for i, point in enumerate(polygon):
        prev = polygon[(i - 1) % len(polygon)]
        nxt = polygon[(i + 1) % len(polygon)]
        a = point - prev
        b = nxt - point
        scale = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0)
        if cross2(a, b) < -tol * scale:
            return False
    return True


def point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    ab = b - a
    ap = point - a
    if abs(ab[0] * ap[1] - ab[1] * ap[0]) > tol * max(np.linalg.norm(ab), 1.0):
        return False
    return (
        min(a[0], b[0]) - tol <= point[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= point[1] <= max(a[1], b[1]) + tol
    )


def closest_polygon_edge(
    polygon: np.ndarray,
    point: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, int, float] | None:
    best = None
    best_distance = float("inf")
    for edge_id, a in enumerate(polygon):
        b = polygon[(edge_id + 1) % len(polygon)]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= tol * tol:
            continue
        u = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        nearest = a + u * ab
        distance = float(np.linalg.norm(point - nearest))
        if distance < best_distance:
            best = (nearest, edge_id, u)
            best_distance = distance
    return best


def split_polygon_by_spokes(
    polygon: np.ndarray,
    vertex: np.ndarray,
    boundary_points: List[Tuple[np.ndarray, int, float]],
    tol: float,
) -> List[np.ndarray]:
    boundary_points = dedupe_boundary_points(boundary_points, tol)
    if len(boundary_points) < 2:
        return []

    augmented = insert_edge_points(polygon, boundary_points, tol)
    indexed = []
    for point, _edge_id, _u in boundary_points:
        idx = find_point_index(augmented, point, tol)
        if idx is not None:
            indexed.append((idx, point))
    indexed = dedupe_indexed_points(sorted(indexed, key=lambda item: item[0]), tol)
    if len(indexed) < 2:
        return []

    polygons = []
    for (start_idx, start_point), (end_idx, end_point) in zip(indexed, indexed[1:] + indexed[:1]):
        path = polygon_path(augmented, start_idx, end_idx)
        poly = clean_polygon(path + [vertex], tol)
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def split_polygon_by_vertex_spokes(
    polygon: np.ndarray,
    vertex: np.ndarray,
    points: List[np.ndarray],
    tol: float,
) -> List[np.ndarray]:
    insertions: List[Tuple[np.ndarray, int, float]] = []
    for point in points:
        if np.linalg.norm(point - vertex) <= tol:
            continue
        nearest = closest_polygon_edge(polygon, point, tol)
        if nearest is None:
            continue
        _nearest_point, edge_id, u = nearest
        insertions.append((point, edge_id, u))
    if not insertions:
        return [polygon]

    augmented = insert_edge_points(polygon, insertions, tol)
    vertex_idx = find_point_index(augmented, vertex, tol)
    if vertex_idx is None:
        split = split_polygon_by_spokes(polygon, vertex, insertions, tol)
        return split if split else [polygon]

    indexed = [(vertex_idx, vertex)]
    for point, _edge_id, _u in insertions:
        idx = find_point_index(augmented, point, tol)
        if idx is not None:
            indexed.append((idx, point))
    indexed = dedupe_indexed_points(sorted(indexed, key=lambda item: item[0]), tol)
    if len(indexed) < 2:
        return [polygon]

    polygons = []
    for (start_idx, _start_point), (end_idx, _end_point) in zip(indexed, indexed[1:] + indexed[:1]):
        path = polygon_path(augmented, start_idx, end_idx)
        if start_idx == vertex_idx or end_idx == vertex_idx:
            poly = clean_polygon(path, tol)
        else:
            poly = clean_polygon(path + [vertex], tol)
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons if polygons else [polygon]


def split_polygon_by_chord(
    polygon: np.ndarray,
    start: Tuple[np.ndarray, int, float],
    end: Tuple[np.ndarray, int, float],
    tol: float,
) -> List[np.ndarray]:
    augmented = insert_edge_points(polygon, [start, end], tol)
    start_idx = find_point_index(augmented, start[0], tol)
    end_idx = find_point_index(augmented, end[0], tol)
    if start_idx is None or end_idx is None or start_idx == end_idx:
        return []
    return [
        clean_polygon(polygon_path(augmented, start_idx, end_idx), tol),
        clean_polygon(polygon_path(augmented, end_idx, start_idx), tol),
    ]


def split_polygon_by_chain(
    polygon: np.ndarray,
    start: Tuple[np.ndarray, int, float],
    chain: List[np.ndarray],
    end: Tuple[np.ndarray, int, float],
    tol: float,
) -> List[np.ndarray]:
    augmented = insert_edge_points(polygon, [start, end], tol)
    start_idx = find_point_index(augmented, start[0], tol)
    end_idx = find_point_index(augmented, end[0], tol)
    if start_idx is None or end_idx is None or start_idx == end_idx:
        return []

    path_a = polygon_path(augmented, start_idx, end_idx)
    path_b = polygon_path(augmented, end_idx, start_idx)
    chain_forward = dedupe_points(chain, tol)
    chain_backward = dedupe_points(list(reversed(chain)), tol)
    return [
        clean_polygon(path_a + chain_backward[1:-1], tol),
        clean_polygon(path_b + chain_forward[1:-1], tol),
    ]


def insert_edge_points(
    polygon: np.ndarray,
    insertions: List[Tuple[np.ndarray, int, float]],
    tol: float,
) -> List[np.ndarray]:
    by_edge: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for point, edge_id, u in insertions:
        by_edge.setdefault(edge_id, []).append((point, u))

    augmented: List[np.ndarray] = []
    for edge_id, point in enumerate(polygon):
        augmented.append(point)
        edge_insertions = sorted(by_edge.get(edge_id, []), key=lambda item: item[1])
        for inserted, _u in edge_insertions:
            if np.linalg.norm(inserted - point) <= tol:
                continue
            nxt = polygon[(edge_id + 1) % len(polygon)]
            if np.linalg.norm(inserted - nxt) <= tol:
                continue
            if not augmented or np.linalg.norm(inserted - augmented[-1]) > tol:
                augmented.append(inserted)
    return dedupe_points(augmented, tol)


def dedupe_boundary_points(
    points: List[Tuple[np.ndarray, int, float]],
    tol: float,
) -> List[Tuple[np.ndarray, int, float]]:
    deduped: List[Tuple[np.ndarray, int, float]] = []
    for point, edge_id, u in sorted(points, key=lambda item: (item[1], item[2])):
        if any(np.linalg.norm(point - existing[0]) <= tol for existing in deduped):
            continue
        deduped.append((point, edge_id, u))
    return deduped


def dedupe_indexed_points(points: List[Tuple[int, np.ndarray]], tol: float) -> List[Tuple[int, np.ndarray]]:
    deduped: List[Tuple[int, np.ndarray]] = []
    for idx, point in points:
        if any(idx == existing_idx or np.linalg.norm(point - existing_point) <= tol for existing_idx, existing_point in deduped):
            continue
        deduped.append((idx, point))
    return deduped


def find_point_index(points: List[np.ndarray], target: np.ndarray, tol: float) -> int | None:
    for idx, point in enumerate(points):
        if np.linalg.norm(point - target) <= tol:
            return idx
    return None


def polygon_path(points: List[np.ndarray], start: int, end: int) -> List[np.ndarray]:
    path = [points[start]]
    idx = start
    while idx != end:
        idx = (idx + 1) % len(points)
        path.append(points[idx])
    return path


def dedupe_points(points: List[np.ndarray], tol: float) -> List[np.ndarray]:
    deduped: List[np.ndarray] = []
    for point in points:
        if not deduped or np.linalg.norm(point - deduped[-1]) > tol:
            deduped.append(point)
    if len(deduped) > 1 and np.linalg.norm(deduped[0] - deduped[-1]) <= tol:
        deduped.pop()
    return deduped
