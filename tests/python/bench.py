#!/usr/bin/env python3
"""Wall time of mesh_generator against the Python API on the bundled models.

    python tests/python/bench.py --binary build/mesh_generator [--models bunny.off Octocat.off]

Each case runs the tool in a scratch directory and the Python call (without and with writing the
same file) in-process, several times, and prints the best times and their ratio. The compiled code
is the same; the Python side adds thread start-up, one copy of the output into its arrays, and no
file I/O unless asked, so the ratio is expected to sit around 1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

import volume_remesher as vr

ROOT = Path(__file__).resolve().parents[2]

CASES = {
    "msh": ([], lambda m: vr.remesh(m), "volume.msh"),
    "tet": (["-t"], lambda m: vr.remesh(m, tetrahedralize=True), "volume.tet"),
    "tet -r -a": (["-t", "-r", "-a"],
                  lambda m: vr.remesh(m, tetrahedralize=True, exact=True, keep_all_cells=True),
                  "volume.tet"),
}


def best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="path to mesh_generator")
    ap.add_argument("--models", nargs="+", default=["sphere.off", "cube_subdiv.off", "100071_sf.off", "bunny.off", "Octocat.off"])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    binary = Path(args.binary).resolve()

    print(f"{'model':22s} {'case':10s} {'cli [s]':>9s} {'python [s]':>11s} {'py+save [s]':>12s} {'py/cli':>7s}")
    for name in args.models:
        model = (ROOT / "models" / name).resolve()
        for case in args.cases:
            flags, call, out = CASES[case]
            with tempfile.TemporaryDirectory() as work:
                def run_cli():
                    subprocess.run([str(binary), str(model), *flags], cwd=work,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                t_cli = best_of(run_cli, args.repeats)
                t_py = best_of(lambda: call(model), args.repeats)
                outpath = os.path.join(work, "py_" + out)
                t_py_save = best_of(lambda: call(model).save(outpath), args.repeats)
            print(f"{name:22s} {case:10s} {t_cli:9.3f} {t_py:11.3f} {t_py_save:12.3f} {t_py_save / t_cli:7.2f}")


if __name__ == "__main__":
    main()
