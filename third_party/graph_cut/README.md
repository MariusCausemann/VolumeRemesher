# third_party/graph_cut — NOT LGPL

The legacy inside/outside cell classifier of VolumeRemesher (`mesh_generator -c graphcut`,
`src/VolumeRemesher/inOutPartition_graphcut.cpp`) is a min-cut over the dual graph of the
cells, solved with the two libraries in this directory. Their licenses are not compatible with
the LGPL the rest of this repository is under, which is why this code is compiled only when
CMake is configured with `-DVOLUMEREMESHER_WITH_GRAPHCUT=ON`, is excluded from the Python
source distribution and wheels, and why the default classifier is the winding number one in
`src/VolumeRemesher/inOutPartition.cpp`.

| Files | Library | License |
| --- | --- | --- |
| `GCoptimization.{h,cpp}`, `LinkedBlockList.{h,cpp}`, `energy.h`, `GCO_README.TXT`, `example.cpp_` | GCO 3.0 (O. Veksler, A. Delong) | Research purposes only; no commercial use; redistribution reserved to the author; a US patent on the algorithm is cited. Full text in `GCO_README.TXT`. |
| `graph.{h,cpp}`, `maxflow.cpp`, `block.h` | maxflow v3 (Y. Boykov, V. Kolmogorov) | GNU GPL (the upstream distribution's README states the terms; the headers of this copy were trimmed before it reached this repository). |

Local modifications: a memory leak and a mismatched new/delete were fixed (commit 490b2a2 in
this repository). A binary built with this code may be used for research but must not be
redistributed under the LGPL.
