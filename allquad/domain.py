from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

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
