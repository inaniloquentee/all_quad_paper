from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class SDFDomain:
    """Implicit 2D domain with negative values inside and positive outside."""

    name: str
    sdf_func: Callable[[ArrayLike], ArrayLike]
    bounds: Tuple[float, float, float, float]

    def sdf(self, points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        return self.sdf_func(pts)

    def normal(self, point: ArrayLike, h: float = 1.0e-6) -> np.ndarray:
        p = np.asarray(point, dtype=float)
        dx = np.array([h, 0.0])
        dy = np.array([0.0, h])
        gx = float(self.sdf(p + dx) - self.sdf(p - dx)) / (2.0 * h)
        gy = float(self.sdf(p + dy) - self.sdf(p - dy)) / (2.0 * h)
        g = np.array([gx, gy], dtype=float)
        n = np.linalg.norm(g)
        if n < 1.0e-14:
            return np.array([1.0, 0.0])
        return g / n

    def segment_intersection(
        self,
        a: ArrayLike,
        b: ArrayLike,
        da: float | None = None,
        db: float | None = None,
        iterations: int = 60,
    ) -> np.ndarray:
        """Return a zero crossing on segment ab using bisection."""

        lo = np.asarray(a, dtype=float)
        hi = np.asarray(b, dtype=float)
        flo = float(self.sdf(lo)) if da is None else float(da)
        fhi = float(self.sdf(hi)) if db is None else float(db)
        if abs(flo) < 1.0e-14:
            return lo.copy()
        if abs(fhi) < 1.0e-14:
            return hi.copy()
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            fm = float(self.sdf(mid))
            if abs(fm) < 1.0e-14:
                return mid
            if (flo <= 0.0) == (fm <= 0.0):
                lo = mid
                flo = fm
            else:
                hi = mid
                fhi = fm
        return 0.5 * (lo + hi)

    def repel(
        self,
        point: ArrayLike,
        clearance: float,
        mode: str = "normal",
    ) -> np.ndarray:
        """Move a grid point away from the boundary if it is too close."""

        p = np.asarray(point, dtype=float)
        d = float(self.sdf(p))
        if abs(d) >= clearance:
            return p.copy()

        normal = self.normal(p)
        side = 1.0 if d >= 0.0 else -1.0
        if mode == "normal":
            return p + side * (clearance - abs(d)) * normal

        if mode != "axis":
            raise ValueError(f"unknown repelling mode: {mode}")

        axis = np.array([1.0, 0.0]) if abs(normal[0]) >= abs(normal[1]) else np.array([0.0, 1.0])
        projection = float(np.dot(normal, axis))
        if abs(projection) < 1.0e-12:
            return p + side * (clearance - abs(d)) * normal
        step = (side * clearance - d) / projection
        return p + step * axis


@dataclass
class PolylineDomain(SDFDomain):
    """Closed piecewise-linear input domain.

    The first loop is treated as the exterior boundary. Additional loops are
    holes. Loops may repeat the first point at the end; it is removed internally.
    """

    loops: List[np.ndarray]

    def __init__(self, name: str, loops: List[np.ndarray], padding: float = 0.12):
        cleaned = [clean_loop(loop) for loop in loops]
        if not cleaned:
            raise ValueError("polyline input must contain at least one loop")
        exterior = ensure_ccw(cleaned[0])
        holes = [ensure_cw(loop) for loop in cleaned[1:]]
        final_loops = [exterior] + holes
        all_pts = np.vstack(final_loops)
        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        span = max(xmax - xmin, ymax - ymin)
        pad = padding * span if span > 0.0 else padding
        bounds = (float(xmin - pad), float(ymin - pad), float(xmax + pad), float(ymax + pad))
        super().__init__(name=name, sdf_func=lambda points: self._signed_distance(points), bounds=bounds)
        self.loops = final_loops

    def _signed_distance(self, points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        flat = pts.reshape(-1, 2)
        distances = np.array([self._distance_to_segments(p) for p in flat], dtype=float)
        inside = np.array([self._contains(p) for p in flat], dtype=bool)
        signed = np.where(inside, -distances, distances)
        return signed.reshape(pts.shape[:-1])

    def normal(self, point: ArrayLike, h: float = 1.0e-6) -> np.ndarray:
        p = np.asarray(point, dtype=float)
        _closest, _distance, edge = self.closest_point(p)
        loop = self.loops[edge[0]]
        a = loop[edge[1]]
        b = loop[(edge[1] + 1) % len(loop)]
        tangent = b - a
        # Loops are oriented so the positive signed-distance side is always to
        # the right of each directed segment: exterior loop CCW, holes CW.
        n = np.array([tangent[1], -tangent[0]])
        length = np.linalg.norm(n)
        if length <= 1.0e-14:
            return np.array([1.0, 0.0])
        return n / length

    def segment_intersection(
        self,
        a: ArrayLike,
        b: ArrayLike,
        da: float | None = None,
        db: float | None = None,
        iterations: int = 60,
    ) -> np.ndarray:
        p = np.asarray(a, dtype=float)
        r = np.asarray(b, dtype=float) - p
        best_t = None
        for loop in self.loops:
            n = len(loop)
            for i in range(n):
                q = loop[i]
                s = loop[(i + 1) % n] - q
                hit = segment_parameters(p, r, q, s)
                if hit is None:
                    continue
                t, u = hit
                if -1.0e-12 <= t <= 1.0 + 1.0e-12 and -1.0e-12 <= u <= 1.0 + 1.0e-12:
                    if best_t is None or t < best_t:
                        best_t = min(max(t, 0.0), 1.0)
        if best_t is not None:
            return p + best_t * r
        return super().segment_intersection(a, b, da, db, iterations)

    def iter_segments(self) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray]]:
        for loop_id, loop in enumerate(self.loops):
            for edge_id in range(len(loop)):
                yield loop_id, edge_id, loop[edge_id], loop[(edge_id + 1) % len(loop)]

    def vertices_in_box(self, bounds: Tuple[float, float, float, float], tol: float = 1.0e-12) -> List[Tuple[int, int]]:
        x0, y0, x1, y1 = bounds
        vertices: List[Tuple[int, int]] = []
        for loop_id, loop in enumerate(self.loops):
            for vertex_id, point in enumerate(loop):
                if x0 - tol <= point[0] <= x1 + tol and y0 - tol <= point[1] <= y1 + tol:
                    vertices.append((loop_id, vertex_id))
        return vertices

    def segments_in_box(self, bounds: Tuple[float, float, float, float], tol: float = 1.0e-12) -> List[Tuple[int, int]]:
        x0, y0, x1, y1 = bounds
        hits: List[Tuple[int, int]] = []
        for loop_id, edge_id, a, b in self.iter_segments():
            if segment_intersects_box(a, b, x0, y0, x1, y1, tol):
                hits.append((loop_id, edge_id))
        return hits

    def should_refine_cell_for_polyline(
        self,
        bounds: Tuple[float, float, float, float],
        tol: float = 1.0e-12,
    ) -> bool:
        segments = self.segments_in_box(bounds, tol)
        if len(segments) <= 1:
            return False

        vertices = self.vertices_in_box(bounds, tol)
        if len(segments) == 2 and len(vertices) == 1:
            loop_id, vertex_id = vertices[0]
            loop = self.loops[loop_id]
            incoming = (loop_id, (vertex_id - 1) % len(loop))
            outgoing = (loop_id, vertex_id)
            if set(segments) == {incoming, outgoing}:
                return False
        return True

    def is_sharp_vertex(
        self,
        loop_id: int,
        vertex_id: int,
        min_turn_degrees: float = 45.0,
    ) -> bool:
        loop = self.loops[loop_id]
        vertex = loop[vertex_id]
        previous = loop[(vertex_id - 1) % len(loop)]
        nxt = loop[(vertex_id + 1) % len(loop)]
        a = previous - vertex
        b = nxt - vertex
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1.0e-14:
            return False
        angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))
        return angle <= 180.0 - min_turn_degrees

    def closest_point(self, point: ArrayLike) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        p = np.asarray(point, dtype=float)
        best_point = None
        best_distance = float("inf")
        best_edge = (0, 0)
        for loop_id, loop in enumerate(self.loops):
            n = len(loop)
            for i in range(n):
                a = loop[i]
                b = loop[(i + 1) % n]
                candidate = closest_point_on_segment(p, a, b)
                distance = float(np.linalg.norm(p - candidate))
                if distance < best_distance:
                    best_point = candidate
                    best_distance = distance
                    best_edge = (loop_id, i)
        if best_point is None:
            raise ValueError("empty polyline domain")
        return best_point, best_distance, best_edge

    def boundary_points(self) -> np.ndarray:
        return np.vstack([np.vstack([loop, loop[:1]]) for loop in self.loops])

    def _distance_to_segments(self, point: np.ndarray) -> float:
        _closest, distance, _edge = self.closest_point(point)
        return distance

    def _contains(self, point: np.ndarray) -> bool:
        inside = point_in_loop(point, self.loops[0])
        for hole in self.loops[1:]:
            if point_in_loop(point, hole):
                inside = False
        return inside

    def _nearest_segment(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        loop_id, edge_id = self.closest_point(point)[2]
        loop = self.loops[loop_id]
        return loop[edge_id], loop[(edge_id + 1) % len(loop)]


def circle(radius: float = 1.0) -> SDFDomain:
    def sdf(points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        return np.linalg.norm(pts, axis=-1) - radius

    pad = 1.2 * radius
    return SDFDomain("circle", sdf, (-pad, -pad, pad, pad))


def flower() -> SDFDomain:
    def sdf(points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        x = pts[..., 0]
        y = pts[..., 1]
        theta = np.arctan2(y, x)
        r = np.sqrt(x * x + y * y)
        boundary = 0.68 + 0.16 * np.sin(5.0 * theta) + 0.06 * np.cos(8.0 * theta)
        return r - boundary

    return SDFDomain("flower", sdf, (-1.05, -1.05, 1.05, 1.05))


def rounded_square() -> SDFDomain:
    def sdf(points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        q = np.abs(pts) - np.array([0.75, 0.75])
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
        return outside + inside - 0.08

    return SDFDomain("rounded_square", sdf, (-1.05, -1.05, 1.05, 1.05))


def star() -> SDFDomain:
    def sdf(points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        x = pts[..., 0]
        y = pts[..., 1]
        theta = np.arctan2(y, x)
        r = np.sqrt(x * x + y * y)
        boundary = 0.62 + 0.20 * np.cos(5.0 * theta)
        return r - boundary

    return SDFDomain("star", sdf, (-1.05, -1.05, 1.05, 1.05))


def two_circles() -> SDFDomain:
    def sdf(points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        left = np.linalg.norm(pts - np.array([-0.33, 0.0]), axis=-1) - 0.58
        right = np.linalg.norm(pts - np.array([0.33, 0.0]), axis=-1) - 0.58
        return np.minimum(left, right)

    return SDFDomain("two_circles", sdf, (-1.15, -0.85, 1.15, 0.85))


def make_domain(name: str) -> SDFDomain:
    key = name.lower().replace("-", "_")
    domains = {
        "circle": circle,
        "flower": flower,
        "rounded_square": rounded_square,
        "star": star,
        "two_circles": two_circles,
    }
    if key not in domains:
        choices = ", ".join(sorted(domains))
        raise ValueError(f"unknown domain '{name}'. Choices: {choices}")
    return domains[key]()


def load_polyline_domain(path: str | Path) -> PolylineDomain:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    name = str(data.get("name") or source.stem)
    if "loops" in data:
        loops = [np.asarray(loop, dtype=float) for loop in data["loops"]]
    elif "points" in data:
        loops = [np.asarray(data["points"], dtype=float)]
    else:
        raise ValueError("polyline JSON must contain either 'points' or 'loops'")
    return PolylineDomain(name=name, loops=loops)


def clean_loop(loop: np.ndarray) -> np.ndarray:
    pts = np.asarray(loop, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("each loop must be an array of [x, y] points")
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= 1.0e-12:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("each loop must contain at least three distinct points")
    return pts


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    return points.copy() if signed_area(points) >= 0.0 else points[::-1].copy()


def ensure_cw(points: np.ndarray) -> np.ndarray:
    return points.copy() if signed_area(points) <= 0.0 else points[::-1].copy()


def closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-24:
        return a.copy()
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return a + t * ab


def point_in_loop(point: np.ndarray, loop: np.ndarray) -> bool:
    x, y = point
    inside = False
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / ((y1 - y0) if y1 != y0 else 1.0e-300) + x0
        ):
            inside = not inside
    return inside


def segment_intersects_box(
    a: np.ndarray,
    b: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tol: float = 1.0e-12,
) -> bool:
    if max(a[0], b[0]) < x0 - tol or min(a[0], b[0]) > x1 + tol:
        return False
    if max(a[1], b[1]) < y0 - tol or min(a[1], b[1]) > y1 + tol:
        return False
    if point_in_box(a, x0, y0, x1, y1, tol) or point_in_box(b, x0, y0, x1, y1, tol):
        return True

    corners = (
        np.array([x0, y0], dtype=float),
        np.array([x1, y0], dtype=float),
        np.array([x1, y1], dtype=float),
        np.array([x0, y1], dtype=float),
    )
    r = b - a
    for c, d in zip(corners, corners[1:] + corners[:1]):
        hit = segment_parameters(a, r, c, d - c)
        if hit is None:
            if abs(cross2(r, c - a)) > tol * max(np.linalg.norm(r), 1.0):
                continue
            if intervals_overlap(a[0], b[0], c[0], d[0], tol) and intervals_overlap(a[1], b[1], c[1], d[1], tol):
                return True
            continue
        t, u = hit
        if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
            return True
    return False


def point_in_box(
    point: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tol: float = 1.0e-12,
) -> bool:
    return x0 - tol <= point[0] <= x1 + tol and y0 - tol <= point[1] <= y1 + tol


def intervals_overlap(a0: float, a1: float, b0: float, b1: float, tol: float = 1.0e-12) -> bool:
    return max(min(a0, a1), min(b0, b1)) <= min(max(a0, a1), max(b0, b1)) + tol


def segment_parameters(
    p: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
) -> Tuple[float, float] | None:
    denom = cross2(r, s)
    if abs(denom) <= 1.0e-14:
        return None
    qp = q - p
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    return float(t), float(u)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
