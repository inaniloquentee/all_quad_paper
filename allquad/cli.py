from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domain import load_polyline_domain, make_domain
from .mesher import AllQuadMesher
from .plotting import plot_mesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the all-quad meshing without cleanup algorithm.")
    parser.add_argument("--domain", default="circle", help="built-in demo: circle, flower, rounded_square, star, two_circles")
    parser.add_argument("--input-json", help="closed polyline JSON file with 'points' or 'loops'")
    parser.add_argument("--min-depth", type=int, default=2, help="minimum quadtree depth")
    parser.add_argument("--max-depth", type=int, default=6, help="maximum quadtree depth near the boundary")
    parser.add_argument("--clearance-ratio", type=float, default=0.25, help="repelling distance as a fraction of local cell size")
    parser.add_argument("--repelling", choices=["normal", "axis"], default="normal", help="paper repelling strategy")
    parser.add_argument("--boundary-band", type=float, default=0.55, help="boundary refinement band in cell-size units")
    parser.add_argument("--out", default="outputs/circle", help="output prefix or directory/name")
    parser.add_argument("--inside-only", action="store_true", help="only output the interior mesh")
    parser.add_argument("--adaptive", action="store_true", help="adaptive quadtree mode with 2-ref transition templates")
    parser.add_argument(
        "--dense-boundary",
        action="store_true",
        help="disable shortest-segment sparse stopping for adaptive polyline inputs",
    )
    parser.add_argument(
        "--sparse-boundary-ratio",
        type=float,
        default=0.5,
        help="target adaptive boundary cell size as a multiple of the shortest input segment",
    )
    parser.add_argument(
        "--no-quality-relaxation",
        action="store_true",
        help="disable local non-boundary vertex relaxation for stricter angle quality",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip PNG plot generation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    domain = load_polyline_domain(args.input_json) if args.input_json else make_domain(args.domain)
    mesher = AllQuadMesher(
        domain,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        clearance_ratio=args.clearance_ratio,
        repelling=args.repelling,
        boundary_band=args.boundary_band,
        include_exterior=not args.inside_only,
        adaptive=args.adaptive,
        sparse_boundary=not args.dense_boundary,
        sparse_boundary_ratio=args.sparse_boundary_ratio,
        quality_relaxation=not args.no_quality_relaxation,
    )
    mesh = mesher.generate()
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    mesh.write_obj(prefix.with_suffix(".obj"))
    mesh.write_vtk(prefix.with_suffix(".vtk"))
    if not args.no_plot:
        plot_mesh(mesh, domain, prefix.with_suffix(".png"))

    report = {
        "quality": mesh.quality(),
        "quality_by_region": mesh.quality_by_region(),
        "topology": mesh.topology(domain.sdf),
    }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
