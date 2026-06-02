# All-quad meshing without cleanup: reproduction

This workspace contains a Python reproduction of the main algorithm in
Rushdi et al., "All-quad meshing without cleanup", Computer-Aided Design 85
(2017), 83-98.

Implemented pipeline:

1. Build a background grid. The default path uses a uniform grid; adaptive
   quadtree mode is available with `--adaptive`.
2. Repel quadtree nodes away from the input boundary by `D = s / 4`, using
   either boundary-normal or grid-axis movement.
3. Split cells cut by the geometry with closed polyline boundary intersections and
   keep both interior and exterior polygons.
4. Convert cut polygons and deformed polygons into quads with midpoint
   subdivision. Cells containing input vertices use the Fig. 6 vertex treatment:
   preserve the original polyline vertex, add side midpoints on non-intersected
   square sides, pull those points by `s / 8`, and choose vertex spokes using the
   same midpoint positions that will be written to the final mesh.
5. Empty adaptive cells use Fig. 7 two-refinement transition templates with
   checkerboard corner marking to remove hanging nodes.
6. Export OBJ, legacy VTK, PNG preview, region labels, and mesh quality metrics.

The code is self-contained and uses only `numpy` and `matplotlib` at runtime.
The primary input is a discrete closed polyline JSON file, matching the paper's
line-segment boundary model. Formula-based domains are still available only as
quick demos.

## Run

```powershell
python -m allquad --input-json examples/wobbly_loop.json --adaptive --max-depth 7 --out outputs/wobbly_loop
python -m allquad --input-json examples/concave_polygon.json --adaptive --max-depth 7 --out outputs/concave_polygon
python -m allquad --input-json examples/box_with_hole.json --adaptive --max-depth 7 --out outputs/box_with_hole
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
- `--max-depth`: background grid resolution.
- `--sparse-boundary-ratio 0.5`: in adaptive polyline mode, cap the effective
  boundary depth from the shortest input segment instead of refining every
  boundary cell to `--max-depth`.
- `--dense-boundary`: disable the shortest-segment sparse cap and use the full
  requested adaptive boundary depth.
- `--no-quality-relaxation`: disable the final local relaxation pass that moves
  free non-boundary vertices to meet the stricter angle target.
- `--inside-only`: omit the exterior mesh if you only want the old one-sided
  output.
- `--domain circle`: use a built-in analytic demo instead of `--input-json`.
- `--adaptive`: adaptive quadtree mode with 2-ref transition templates for
  coarse/fine interfaces.

## Verification

```powershell
python -m compileall allquad tools
python tools/check_mesh.py --input-json examples/concave_polygon.json --max-depth 5
python tools/check_mesh.py --input-json examples/concave_polygon.json --adaptive --max-depth 7 --min-angle 30 --max-angle 150
python tools/check_mesh.py --input-json examples/wobbly_loop.json --adaptive --max-depth 7 --min-angle 30 --max-angle 150
python tools/check_mesh.py --input-json examples/box_with_hole.json --adaptive --max-depth 7 --min-angle 30 --max-angle 150
```

The check asserts that the output is all-quadrilateral, has positive areas, has
both interior and exterior elements, has no nonmanifold edges, has no degenerate
angles outside the configured quality range, and that detected geometry-boundary
edges are present and lie on the original discrete polyline boundary to floating
point tolerance. The default regression range is `15` to `165` degrees; the
adaptive polyline examples are checked with a paper-aligned sharp-feature guard:
minimum angle at least `30` degrees and maximum angle at most `150` degrees.

With the default sparse cap, the example commands still accept `--max-depth 7`,
but the effective adaptive depth is chosen from the shortest discrete input
segment. The current examples resolve to depth `6` for `wobbly_loop` and depth
`4` for `concave_polygon` and `box_with_hole`, keeping the boundary region
readable while preserving the angle and topology checks.

After the paper templates are applied, adaptive outputs run a local relaxation
pass on free same-region vertices. The pass keeps input-boundary vertices and
geometry-interface vertices fixed, only accepts moves that improve the local
angle violation, and preserves the signed side of each moved vertex. This is the
last step that brings the example outputs to the stricter `30`/`150` degree
range without moving the discrete input boundary.

## Notes on fidelity

The paper treats input geometry as curves and vertices. This implementation now
uses piecewise-linear closed loops as the main input, so arbitrary shapes can be
meshed without writing formulas. Cells cut by one or more polyline segments are
split against the actual input segments instead of an analytic formula. Cells
containing geometric vertices use the Fig. 6 vertex template: the original
polyline vertex is inserted, adjacent curve hits are connected through that
vertex, non-intersected side midpoints are pulled toward the vertex by `s / 8`,
and auxiliary edge spokes are selected using the actual midpoint-subdivision
points used by the final mesh. This keeps the interior/exterior interface on the
original discrete boundary instead of on analytic SDF chords. Vertex-chain
parents are labelled from the oriented input loop, so concave vertex cells do
not fall back to a straight SDF chord between neighboring boundary hits.

Adaptive transitions now use the paper's two-refinement (2-ref) templates for
coarse/fine interfaces. The quadtree is refined until every empty adaptive cell
has at most one hanging node on each side, then the Fig. 7 templates are applied
with checkerboard corner marking.
