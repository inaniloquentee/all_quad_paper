import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from allquad.domain import load_polyline_domain, make_domain
from allquad.mesher import AllQuadMesher


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small all-quad regression check.")
    parser.add_argument("--domain", default="circle")
    parser.add_argument("--input-json")
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    domain = load_polyline_domain(args.input_json) if args.input_json else make_domain(args.domain)
    mesh = AllQuadMesher(domain, max_depth=args.max_depth).generate()
    quality = mesh.quality()
    by_region = mesh.quality_by_region()
    topology = mesh.topology(domain.sdf)

    assert quality["quads"] > 0
    assert by_region["interior"]["quads"] > 0
    assert by_region["exterior"]["quads"] > 0
    assert quality["min_area"] > 0.0
    assert topology["nonmanifold_edges"] == 0.0
    assert topology["interface_edges"] > 0
    assert quality["max_angle"] < 180.0
    assert quality["min_angle"] > 0.0

    report = {"quality": quality, "quality_by_region": by_region, "topology": topology}
    Path("outputs").mkdir(exist_ok=True)
    name = Path(args.input_json).stem if args.input_json else args.domain
    Path(f"outputs/check_{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
