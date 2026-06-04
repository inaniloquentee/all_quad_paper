import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from allquad.domain import load_polyline_domain, make_domain, segment_parameters
from allquad.mesher import AllQuadMesher, build_side_indices
from allquad.quadtree import Quadtree


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small all-quad regression check.")
    parser.add_argument("--domain", default="circle")
    parser.add_argument("--input-json")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-angle", type=float, default=15.0)
    parser.add_argument("--max-angle", type=float, default=165.0)
    parser.add_argument("--enforce-angle-bounds", action="store_true")
    parser.add_argument("--adaptive", action="store_true", help="exercise adaptive quadtree 2-ref transitions")
    args = parser.parse_args()

    domain = load_polyline_domain(args.input_json) if args.input_json else make_domain(args.domain)
    mesher = AllQuadMesher(domain, max_depth=args.max_depth, adaptive=args.adaptive)
    mesh = mesher.generate()
    quality = mesh.quality()
    by_region = mesh.quality_by_region()
    topology = mesh.topology(domain.sdf)
    two_ref = check_two_ref(domain, mesher) if args.adaptive else {}
    boundary = check_boundary_conformance(mesh, domain) if args.adaptive else {}

    assert quality["quads"] > 0
    assert by_region["interior"]["quads"] > 0
    assert by_region["exterior"]["quads"] > 0
    assert quality["min_area"] > 0.0
    assert topology["nonmanifold_edges"] == 0.0
    assert topology["interface_edges"] > 0
    if boundary:
        assert boundary["interface_edges_not_on_input_segment"] == 0
        assert boundary["non_interface_edges_crossing_boundary"] == 0
    if args.enforce_angle_bounds:
        assert quality["max_angle"] <= args.max_angle
        assert quality["min_angle"] >= args.min_angle
    if two_ref:
        assert two_ref["bad_cells"] == 0
        assert two_ref["bad_sides"] == 0
        assert two_ref["max_side_hanging_nodes"] <= 1

    report = {
        "settings": {
            "effective_depth": mesher.last_effective_depth,
            "sparse_attempts": mesher.last_sparse_attempts,
        },
        "quality": quality,
        "quality_by_region": by_region,
        "topology": topology,
        "two_ref": two_ref,
        "boundary": boundary,
    }
    Path("outputs").mkdir(exist_ok=True)
    name = Path(args.input_json).stem if args.input_json else args.domain
    Path(f"outputs/check_{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def check_two_ref(domain, mesher: AllQuadMesher) -> dict:
    max_depth = mesher.last_effective_depth or mesher._effective_max_depth()
    tree = Quadtree.build(
        domain,
        min_depth=min(mesher.min_depth, max_depth),
        max_depth=max_depth,
        boundary_band=mesher.boundary_band,
        sparse_boundary_size=mesher._sparse_boundary_size(),
    )
    mesher._refine_tree_for_2ref(tree, max_depth)
    points, bad_cells = mesher._closed_2ref_points(tree, tree.collect_corners())
    x_index, y_index = build_side_indices(points)
    bad_sides = 0
    max_side_hanging_nodes = 0
    leaf_sizes = []
    for cell in tree.leaves:
        x0, _y0, x1, _y1 = tree.bounds(cell)
        leaf_sizes.append(x1 - x0)
        for side in mesher._cell_side_node_keys(tree, cell, x_index, y_index):
            max_side_hanging_nodes = max(max_side_hanging_nodes, len(side))
            if len(side) > 1:
                bad_sides += 1

    result = {
        "bad_cells": len(bad_cells),
        "bad_sides": bad_sides,
        "max_side_hanging_nodes": max_side_hanging_nodes,
        "leaves": len(tree.leaves),
        "min_leaf_size": min(leaf_sizes) if leaf_sizes else 0.0,
        "max_leaf_size": max(leaf_sizes) if leaf_sizes else 0.0,
    }
    if hasattr(domain, "min_segment_length") and leaf_sizes:
        result["shortest_segment"] = float(domain.min_segment_length())
        result["min_leaf_over_shortest_segment"] = result["min_leaf_size"] / result["shortest_segment"]
    return result


def check_boundary_conformance(mesh, domain) -> dict:
    if not hasattr(domain, "iter_segments"):
        return {}

    points = np.asarray(mesh.vertices, dtype=float)
    edge_regions = {}
    for quad, region in zip(mesh.quads, mesh.regions):
        for i, a in enumerate(quad):
            b = quad[(i + 1) % 4]
            edge = (a, b) if a < b else (b, a)
            edge_regions.setdefault(edge, set()).add(region)

    interface_bad = 0
    crossing_bad = 0
    interface_edges = 0
    for (a, b), regions in edge_regions.items():
        p = points[a]
        q = points[b]
        is_interface = regions == {-1, 1}
        if is_interface:
            interface_edges += 1
            if not edge_lies_on_input_segment(p, q, domain):
                interface_bad += 1
            continue
        if edge_crosses_input_boundary(p, q, domain):
            crossing_bad += 1

    return {
        "interface_edges": interface_edges,
        "interface_edges_not_on_input_segment": interface_bad,
        "non_interface_edges_crossing_boundary": crossing_bad,
    }


def edge_lies_on_input_segment(p: np.ndarray, q: np.ndarray, domain, tol: float = 1.0e-8) -> bool:
    for _loop_id, _edge_id, a, b in domain.iter_segments():
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length <= tol:
            continue
        ap = p - a
        aq = q - a
        cross_p = abs(float(ab[0] * ap[1] - ab[1] * ap[0])) / length
        cross_q = abs(float(ab[0] * aq[1] - ab[1] * aq[0])) / length
        if cross_p > tol or cross_q > tol:
            continue
        scale = float(np.dot(ab, ab))
        tp = float(np.dot(ap, ab) / scale)
        tq = float(np.dot(aq, ab) / scale)
        if -tol <= tp <= 1.0 + tol and -tol <= tq <= 1.0 + tol:
            return True
    return False


def edge_crosses_input_boundary(p: np.ndarray, q: np.ndarray, domain, tol: float = 1.0e-10) -> bool:
    r = q - p
    for _loop_id, _edge_id, a, b in domain.iter_segments():
        hit = segment_parameters(p, r, a, b - a)
        if hit is None:
            continue
        t, u = hit
        if tol < t < 1.0 - tol and -tol <= u <= 1.0 + tol:
            return True
    return False


if __name__ == "__main__":
    main()
