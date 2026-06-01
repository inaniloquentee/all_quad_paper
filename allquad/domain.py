from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

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
        closest, _distance, _edge = self.closest_point(p)
        n = p - closest
        length = np.linalg.norm(n)
        if length <= 1.0e-14:
            a, b = self._nearest_segment(p)
            tangent = b - a
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
