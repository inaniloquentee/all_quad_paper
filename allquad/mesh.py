from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    area = polygon_area(points)
    if abs(area) < 1.0e-14:
        return np.mean(points, axis=0)
    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=float)


def clean_polygon(points: Iterable[np.ndarray], tol: float = 1.0e-10) -> np.ndarray:
    cleaned: List[np.ndarray] = []
    for p in points:
        arr = np.asarray(p, dtype=float)
        if not cleaned or np.linalg.norm(arr - cleaned[-1]) > tol:
            cleaned.append(arr)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tol:
        cleaned.pop()
    if len(cleaned) < 3:
        return np.empty((0, 2), dtype=float)

    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        keep: List[np.ndarray] = []
        n = len(cleaned)
        for i, p in enumerate(cleaned):
            prev = cleaned[(i - 1) % n]
            nxt = cleaned[(i + 1) % n]
            v0 = p - prev
            v1 = nxt - p
            if np.linalg.norm(v0) <= tol or np.linalg.norm(v1) <= tol:
                changed = True
                continue
            cross = abs(v0[0] * v1[1] - v0[1] * v1[0])
            if cross <= tol * np.linalg.norm(v0) * np.linalg.norm(v1):
                # Preserve long boundary chains; only remove nearly duplicate collinear turns.
                if min(np.linalg.norm(v0), np.linalg.norm(v1)) <= 10.0 * tol:
                    changed = True
                    continue
            keep.append(p)
        cleaned = keep
    if len(cleaned) < 3:
        return np.empty((0, 2), dtype=float)
    poly = np.asarray(cleaned, dtype=float)
    if polygon_area(poly) < 0.0:
        poly = poly[::-1].copy()
    return poly


@dataclass
class Mesh:
    vertices: List[np.ndarray] = field(default_factory=list)
    quads: List[Tuple[int, int, int, int]] = field(default_factory=list)
    regions: List[int] = field(default_factory=list)
    tol: float = 1.0e-10
    _index: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False)

    def add_vertex(self, point: np.ndarray) -> int:
        p = np.asarray(point, dtype=float)
        key = (int(round(p[0] / self.tol)), int(round(p[1] / self.tol)))
        found = self._index.get(key)
        if found is not None:
            return found
        idx = len(self.vertices)
        self.vertices.append(p)
        self._index[key] = idx
        return idx

    def add_quad(self, points: Iterable[np.ndarray], region: int = 0) -> bool:
        quad = clean_polygon(points, self.tol)
        if len(quad) != 4:
            return False
        if abs(polygon_area(quad)) <= self.tol * self.tol:
            return False
        edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        if np.any(edges <= self.tol):
            return False
        ids = tuple(self.add_vertex(p) for p in quad)
        if len(set(ids)) != 4:
            return False
        self.quads.append(ids)  # type: ignore[arg-type]
        self.regions.append(region)
        return True

    def add_midpoint_subdivision(self, points: Iterable[np.ndarray], region: int = 0) -> int:
        poly = clean_polygon(points, self.tol)
        if len(poly) < 3:
            return 0
        center = polygon_centroid(poly)
        mids = 0.5 * (poly + np.roll(poly, -1, axis=0))
        count = 0
        for i, vertex in enumerate(poly):
            prev_mid = mids[(i - 1) % len(poly)]
            next_mid = mids[i]
            if self.add_quad([vertex, next_mid, center, prev_mid], region=region):
                count += 1
        return count

    def quality(self, region: int | None = None) -> Dict[str, float]:
        selected = [
            (quad, reg)
            for quad, reg in zip(self.quads, self.regions)
            if region is None or reg == region
        ]
        if not selected:
            return {
                "vertices": float(len(self.vertices)),
                "quads": 0.0,
                "min_angle": float("nan"),
                "max_angle": float("nan"),
                "min_edge_ratio": float("nan"),
                "avg_edge_ratio": float("nan"),
            }

        pts = np.asarray(self.vertices)
        angles = []
        ratios = []
        areas = []
        used_vertices = set()
        for quad, _region in selected:
            used_vertices.update(quad)
            q = pts[list(quad)]
            areas.append(abs(polygon_area(q)))
            edges = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
            ratios.append(float(np.min(edges) / np.max(edges)))
            for i in range(4):
                a = q[(i - 1) % 4] - q[i]
                b = q[(i + 1) % 4] - q[i]
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if denom <= self.tol:
                    continue
                cosv = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
                angles.append(math_degrees_acos(cosv))

        return {
            "vertices": float(len(used_vertices)),
            "quads": float(len(selected)),
            "min_angle": float(np.min(angles)),
            "max_angle": float(np.max(angles)),
            "min_edge_ratio": float(np.min(ratios)),
            "avg_edge_ratio": float(np.mean(ratios)),
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
        }

    def quality_by_region(self) -> Dict[str, Dict[str, float]]:
        labels = {-1: "interior", 1: "exterior"}
        return {
            labels.get(region, str(region)): self.quality(region=region)
            for region in sorted(set(self.regions))
        }

    def topology(self, boundary_sdf=None) -> Dict[str, float]:
        edge_use: Dict[Tuple[int, int], int] = {}
        edge_regions: Dict[Tuple[int, int], set[int]] = {}
        for quad, region in zip(self.quads, self.regions):
            for i, a in enumerate(quad):
                b = quad[(i + 1) % 4]
                edge = (a, b) if a < b else (b, a)
                edge_use[edge] = edge_use.get(edge, 0) + 1
                edge_regions.setdefault(edge, set()).add(region)

        boundary_edges = [edge for edge, count in edge_use.items() if count == 1]
        nonmanifold = sum(1 for count in edge_use.values() if count > 2)
        result = {
            "edges": float(len(edge_use)),
            "boundary_edges": float(len(boundary_edges)),
            "nonmanifold_edges": float(nonmanifold),
        }
        interface_edges = [
            edge for edge, regions in edge_regions.items() if regions == {-1, 1}
        ]
        result["interface_edges"] = float(len(interface_edges))
        if boundary_sdf is not None and interface_edges:
            pts = np.asarray(self.vertices)
            mids = np.asarray([0.5 * (pts[a] + pts[b]) for a, b in interface_edges])
            distances = np.abs(np.asarray(boundary_sdf(mids), dtype=float))
            result["max_interface_midpoint_error"] = float(np.max(distances))
            result["avg_interface_midpoint_error"] = float(np.mean(distances))
        return result

    def write_obj(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            f.write("# all-quad mesh\n")
            for p in self.vertices:
                f.write(f"v {p[0]:.12g} {p[1]:.12g} 0\n")
            current_region = None
            for q, region in zip(self.quads, self.regions):
                if region != current_region:
                    current_region = region
                    name = "interior" if region < 0 else "exterior" if region > 0 else "unknown"
                    f.write(f"g {name}\n")
                f.write("f " + " ".join(str(i + 1) for i in q) + "\n")

    def write_vtk(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("all-quad mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {len(self.vertices)} float\n")
            for p in self.vertices:
                f.write(f"{p[0]:.12g} {p[1]:.12g} 0\n")
            f.write(f"CELLS {len(self.quads)} {len(self.quads) * 5}\n")
            for q in self.quads:
                f.write("4 " + " ".join(str(i) for i in q) + "\n")
            f.write(f"CELL_TYPES {len(self.quads)}\n")
            for _ in self.quads:
                f.write("9\n")
            f.write(f"CELL_DATA {len(self.quads)}\n")
            f.write("SCALARS region int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for region in self.regions:
                f.write(f"{region}\n")


def math_degrees_acos(value: float) -> float:
    return float(np.degrees(np.arccos(value)))
