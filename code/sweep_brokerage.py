#!/usr/bin/env python3
"""
sweep_brokerage.py -- resumable, memory-aware full-sweep harness for run_city over
every grafo_*.gt in a directory, designed to be split across two machines.

It IMPORTS run_city (and helpers) from your existing analysis module rather than
reimplementing them, so the science is identical to the per-city runs; this file
only adds orchestration: size-balanced partitioning, a per-city memory pre-flight,
append-as-you-go resumability, and per-city failure logging.

Two-machine use (no coordination needed -- the partition is deterministic):
  natri (more RAM):  python sweep_brokerage.py --data_dir <BR> --module brokerage_constraint_pipeline \
                        --n_partitions 2 --partition 0 --mem_budget_gb 691 --out sweep_natri.csv
  wawa  (less RAM):  python sweep_brokerage.py --data_dir <BR> --module brokerage_constraint_pipeline \
                        --n_partitions 2 --partition 1 --mem_budget_gb 251 --out sweep_wawa.csv
Then concatenate sweep_natri.csv + sweep_wawa.csv.

Partitioning: cities are sorted by .gt file size (desc) and dealt snake-draft into
n_partitions, so each partition gets a comparable TOTAL byte load (not an equal city
count) -- this keeps the megacities from piling onto one machine. --max_bytes routes
graphs above a size to partition 0 only (the high-RAM machine); anything a machine
cannot fit (per --mem_budget_gb pre-flight) is skipped and logged, never OOM-killed.

Always start with --dry_run to see the assignment and per-city RAM estimate.
"""
import os, sys, glob, argparse, importlib, csv, gc, traceback
import numpy as np


# ---------------------------------------------------------------- partition + preflight
def list_cities(data_dir, pattern="grafo_*.gt"):
    paths = glob.glob(os.path.join(data_dir, pattern))
    return sorted(paths, key=os.path.getsize, reverse=True)   # biggest first


def snake_partition(paths, n_partitions):
    """Deal size-sorted paths snake-draft: 0,1,...,n-1,n-1,...,1,0,0,1,...
    Balances total bytes per partition rather than city count."""
    buckets = [[] for _ in range(n_partitions)]
    fwd = True
    order = list(range(n_partitions))
    i = 0
    for p in paths:
        buckets[order[i]].append(p)
        i += 1
        if i == n_partitions:
            i = 0; order = order[::-1] if fwd else order[::-1]; fwd = not fwd
    return buckets


def assign(paths, n_partitions, partition, max_bytes):
    """Return this partition's city paths. Graphs above max_bytes go to partition 0
    (assumed the high-RAM machine)."""
    if max_bytes:
        big = [p for p in paths if os.path.getsize(p) > max_bytes]
        small = [p for p in paths if os.path.getsize(p) <= max_bytes]
    else:
        big, small = [], paths
    buckets = snake_partition(small, n_partitions)
    mine = list(buckets[partition])
    if partition == 0:
        mine = big + mine                    # megacities pinned to the high-RAM machine
    return mine, big


def estimate_peak_gb(path, chunk):
    """Rough peak-RAM estimate for run_city on this graph, in GiB. The dominant transient
    is burt_constraint's P@P chunk product; we approximate it from graph byte size and the
    chunk fraction. Deliberately conservative (overestimates) so the guard errs safe."""
    sz = os.path.getsize(path)
    graph_gb = 3.0 * sz / 2**30              # graph-tool in-memory ~ a few x the .gt bytes
    # P@P transient scales with (chunk / N) * nnz; without loading we bound it by graph size
    matmul_gb = 2.5 * graph_gb               # empirical headroom for the sparse two-hop
    return graph_gb + matmul_gb


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("city"):
                done.add(row["city"])
    return done


def append_row(out_path, row, header_order):
    new = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush(); os.fsync(f.fileno())      # durable: crash-safe resumability


# ------------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--module", default="brokerage_constraint_pipeline",
                    help="import name of the module holding run_city (no .py)")
    ap.add_argument("--pattern", default="grafo_*.gt")
    ap.add_argument("--vp", default="quintile_nation")
    ap.add_argument("--weight_ep", default="q_calls")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--n_partitions", type=int, default=1)
    ap.add_argument("--partition", type=int, default=0, help="0-indexed partition for THIS machine")
    ap.add_argument("--max_bytes", type=int, default=0,
                    help="graphs larger than this go to partition 0 only (0 = off)")
    ap.add_argument("--mem_budget_gb", type=float, default=0,
                    help="skip+log any city whose estimated peak exceeds this (0 = off)")
    ap.add_argument("--out", default="sweep_brokerage.csv")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    paths = list_cities(a.data_dir, a.pattern)
    if not paths:
        sys.exit(f"no graphs matching {a.pattern} in {a.data_dir}")
    mine, big = assign(paths, a.n_partitions, a.partition, a.max_bytes)
    fail_log = a.out + ".failures.log"

    if a.dry_run:
        print(f"{len(paths)} cities total; partition {a.partition}/{a.n_partitions} "
              f"gets {len(mine)} cities ({sum(os.path.getsize(p) for p in mine)/2**30:.1f} GiB of .gt)")
        if a.partition == 0 and big:
            print(f"  ({len(big)} megacities > max_bytes pinned here)")
        print(f"  {'city':28s} {'.gt GiB':>8s} {'est peak GiB':>12s} {'fits?':>6s}")
        for p in mine:
            est = estimate_peak_gb(p, a.chunk)
            fits = "-" if not a.mem_budget_gb else ("yes" if est <= a.mem_budget_gb else "SKIP")
            name = os.path.basename(p)
            print(f"  {name:28s} {os.path.getsize(p)/2**30:8.2f} {est:12.1f} {fits:>6s}")
        sys.exit(0)

    mod_dir = os.path.dirname(os.path.abspath(a.module)) if a.module.endswith(".py") else None
    modname = a.module[:-3] if a.module.endswith(".py") else a.module
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__)), mod_dir):
        if d and d not in sys.path:
            sys.path.insert(0, d)
    mod = importlib.import_module(modname)
    run_city = mod.run_city
    done = already_done(a.out)
    if done:
        print(f"resuming: {len(done)} cities already in {a.out}, skipping them")

    header_order = None
    for i, p in enumerate(mine, 1):
        name = os.path.splitext(os.path.basename(p))[0].replace("grafo_", "", 1)
        if name in done:
            continue
        if a.mem_budget_gb:
            est = estimate_peak_gb(p, a.chunk)
            if est > a.mem_budget_gb:
                msg = f"SKIP {name}: est peak {est:.1f} GiB > budget {a.mem_budget_gb} GiB"
                print(f"  [{i}/{len(mine)}] {msg}")
                with open(fail_log, "a") as f:
                    f.write(msg + "\n")
                continue
        print(f"  [{i}/{len(mine)}] {name} ...", flush=True)
        try:
            row = run_city(p, vp=a.vp, weight_ep=a.weight_ep,
                           n_perm=a.n_perm, chunk=a.chunk, label=name, verbose=False)
            # provenance: this closure ratio is the class-blind (label-permutation) null
            if "q5_closure_ratio" in row:
                row["q5_R_perm"] = row.pop("q5_closure_ratio")
            if "q1_closure_ratio" in row:
                row["q1_R_perm"] = row.pop("q1_closure_ratio")
            if header_order is None:
                header_order = list(row.keys())
            append_row(a.out, row, header_order)
        except Exception as e:
            with open(fail_log, "a") as f:
                f.write(f"{name}: {e!r}\n")
                f.write(traceback.format_exc() + "\n")
            print(f"      FAILED {name}: {e!r} (logged, continuing)")
        finally:
            gc.collect()
    print(f"partition {a.partition} done -> {a.out}"
          + (f"  (failures in {fail_log})" if os.path.exists(fail_log) else ""))