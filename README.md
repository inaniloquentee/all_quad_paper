# All-quad meshing without cleanup: reproduction

This workspace contains a Python reproduction of the main algorithm in
Rushdi et al., "All-quad meshing without cleanup", Computer-Aided Design 85
(2017), 83-98.

Implemented pipeline:

1. Build a quadtree and strongly balance it so corner-adjacent leaves differ by
   at most one level.
2. Repel quadtree nodes away from the input boundary by `D = s / 4`, using
   either boundary-normal or grid-axis movement.
3. Split cells cut by the geometry with implicit boundary intersections.
4. Convert cut polygons, deformed polygons, and hanging-node polygons into
   quads with midpoint subdivision. Shared edges are subdivided consistently;
   this is a compact generic closure of the paper's local 2-ref templates.
5. Export OBJ, legacy VTK, PNG preview, and mesh quality metrics.

The code is self-contained and uses only `numpy` and `matplotlib` at runtime.
The current geometry interface is implicit signed-distance style. It reproduces
the smooth-boundary cases from the paper directly; sharp feature preservation is
represented by star-like implicit examples, but exact PSLG vertex insertion is
left as a clear extension point.

## Run

```powershell
python -m allquad --domain circle --max-depth 6 --out outputs/circle
python -m allquad --domain flower --max-depth 7 --out outputs/flower
python -m allquad --domain two_circles --max-depth 7 --out outputs/two_circles
```

Outputs:

- `*.obj`: quad faces for common mesh viewers.
- `*.vtk`: legacy VTK unstructured grid with VTK_QUAD cells.
- `*.png`: mesh preview with the input boundary in red.
- `*.json`: vertex/quad count and angle/edge-ratio quality statistics.

Useful parameters:

- `--repelling normal`: move along the boundary normal, as in Fig. 4(b).
- `--repelling axis`: move horizontally/vertically, as in Fig. 4(c).
- `--clearance-ratio 0.25`: paper value `D = s / 4`.
- `--min-depth` and `--max-depth`: background quadtree resolution.

## Verification

```powershell
python -m compileall allquad tools
python tools/check_mesh.py --domain circle --max-depth 5
```

The check asserts that the output is all-quadrilateral, has positive areas, has
no nonmanifold edges, and has no degenerate 180 degree angles.

## Notes on fidelity

The paper treats input geometry as curves and vertices, then explicitly inserts
sharp geometric vertices before midpoint subdivision. This reproduction focuses
on the same meshing pipeline for implicit 2D domains. Smooth domains such as
`circle` and `two_circles` are closest to the analyzed setting. The `star` demo
is useful for stress testing, but it is not the exact PSLG sharp-feature
handling from Fig. 6.
