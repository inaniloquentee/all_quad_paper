# All-quad meshing without cleanup: reproduction

This workspace contains a Python reproduction of the main algorithm in
Rushdi et al., "All-quad meshing without cleanup", Computer-Aided Design 85
(2017), 83-98.

Implemented pipeline:

1. Build a quadtree and strongly balance it so corner-adjacent leaves differ by
   at most one level.
2. Repel quadtree nodes away from the input boundary by `D = s / 4`, using
   either boundary-normal or grid-axis movement.
3. Split cells cut by the geometry with closed polyline boundary intersections and
   keep both interior and exterior polygons.
4. Convert cut polygons, deformed polygons, and hanging-node polygons into
   quads with midpoint subdivision. Shared edges are subdivided consistently;
   this is a compact generic closure of the paper's local 2-ref templates.
5. Export OBJ, legacy VTK, PNG preview, region labels, and mesh quality metrics.

The code is self-contained and uses only `numpy` and `matplotlib` at runtime.
The primary input is a discrete closed polyline JSON file, matching the paper's
line-segment boundary model. Formula-based domains are still available only as
quick demos.

## Run

```powershell
python -m allquad --input-json examples/wobbly_loop.json --max-depth 7 --out outputs/wobbly_loop
python -m allquad --input-json examples/concave_polygon.json --max-depth 7 --out outputs/concave_polygon
python -m allquad --input-json examples/box_with_hole.json --max-depth 7 --out outputs/box_with_hole
```

JSON formats:

```json
{"name": "shape", "points": [[0, 0], [1, 0], [0.5, 1]]}
```

or, for holes:

```json
{
  "name": "shape_with_holes",
  "loops": [
    [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    [[-0.25, -0.25], [-0.25, 0.25], [0.25, 0.25], [0.25, -0.25]]
  ]
}
```

Loops are automatically closed, so the last point does not need to repeat the
first point.

Outputs:

- `*.obj`: quad faces grouped into `interior` and `exterior`.
- `*.vtk`: legacy VTK unstructured grid with VTK_QUAD cells and a `region`
  cell scalar (`-1` interior, `1` exterior).
- `*.png`: two-sided mesh preview with the input boundary in red.
- `*.json`: vertex/quad count, per-region quality, and boundary topology checks.

Useful parameters:

- `--repelling normal`: move along the boundary normal, as in Fig. 4(b).
- `--repelling axis`: move horizontally/vertically, as in Fig. 4(c).
- `--clearance-ratio 0.25`: paper value `D = s / 4`.
- `--min-depth` and `--max-depth`: background quadtree resolution.
- `--inside-only`: omit the exterior mesh if you only want the old one-sided
  output.
- `--domain circle`: use a built-in analytic demo instead of `--input-json`.

## Verification

```powershell
python -m compileall allquad tools
python tools/check_mesh.py --input-json examples/concave_polygon.json --max-depth 5
```

The check asserts that the output is all-quadrilateral, has positive areas, has
both interior and exterior elements, has no nonmanifold edges, has no degenerate
180 degree angles, and that detected geometry-boundary edges are shared by both
regions.

## Notes on fidelity

The paper treats input geometry as curves and vertices. This implementation now
uses piecewise-linear closed loops as the main input, so arbitrary shapes can be
meshed without writing formulas. Exact sharp-feature templates from Fig. 6 are
approximated by the same segment intersection and midpoint subdivision pipeline;
adding vertex-specific templates is the next fidelity step.
