# Scaph — Usage Reference

Scaph is an out-of-GPU-memory graph processing framework that pages a CSR
graph through a GPU page pool with adaptive compress / merge / full-copy
per-iteration. This document covers **how to build and run each binary**.
For the high-level algorithm design, see the code under `src/` and
`app/`.

---

## 1. Build

All binaries share one `Makefile`. Adjust the SM flag if your GPU is not
Ampere (`sm_86`).

```bash
cd Scaph
make SM=86                  # default: builds bfs, bfs64, wcc, wcc64, sssp, sssp64, pr, pr64
make SM=80                  # e.g. A100
make SM=86 DEBUG=1          # -O0 -G -g
```

Per-target builds:

```bash
make bfs          bfs64          bfs_check          bfs_check64
make wcc          wcc64          wcc_check
make sssp         sssp64         sssp_check         sssp_check64
make pr           pr64           pr_check           pr_check64
```

The three flavours per algorithm differ only in compile flags:

| Suffix      | Extra flags                                          | `vertex_t / index_t / value_t` | Purpose                                   |
| ----------- | ---------------------------------------------------- | ------------------------------ | ----------------------------------------- |
| *(none)*    | —                                                    | `unsigned int` (32-bit)        | Graphs with ≤ 4 B vertices / edges        |
| `64`        | `-DLARGESIZE`                                        | `unsigned long int` (64-bit)   | Graphs that need 64-bit vertex / edge ids |
| `_check[64]`| `-DSCAPH_KEEP_CSR_FOR_CHECK` (+ `-DLARGESIZE`)       | same as above                  | Same kernels, **plus CPU parallel verify** |

> `*_check*` binaries materialize the full CSR on the host so the CPU-side
> BFS / SSSP verifier (OpenMP parallel, see `app/bfs.cu::check` /
> `app/sssp.cu::check`) has data to compare against. They consume more
> host RAM than the plain binaries.

---

## 2. Common CLI flags

All app binaries use a shared flag parser (`src/app.cuh::scaph_parse_args`).
Both `--key value` and `--key=value` forms work, and every long option
also accepts a single-dash alias (`-input`, `-src`, …).

| Flag                              | Default | Unit / meaning                                                                                                   |
| --------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `--input PATH`                    | *(req)* | Graph file. `.bcsr64` / `.bwcsr64` suffix triggers the unified 64-bit binary loader (`src/graph_bin64.h`); any other path is treated as the legacy per-page format emitted by `tools/pagegencsr`. |
| `--src N` (`--source`)            | `0`     | Source vertex for BFS / SSSP (WCC ignores it)                                                                    |
| `--pagesize N`                    | `10`    | Sub-CSR page size in **MB of `vertex_t` slots** (≈ 40 MB for 32-bit build, 80 MB for 64-bit build)               |
| `--max_memory N`                  | `4096`  | GPU memory budget in **MB** — Scaph uses this to decide how many pages fit in the GPU page pool                  |
| `--threshold N`                   | `8`     | Adaptive-compress **α** in units of 1/32 of a page. `8` = 100 % (always full copy); `3` ≈ 37.5 %; `2` ≈ 25 %, etc. Lower = more aggressive compression + page merging |
| `--multi N`                       | `1`     | Max times a single page can be re-scheduled in one iteration (priority queue depth)                              |
| `--num_stream N`                  | `5`     | Number of CUDA execution streams                                                                                 |
| `--num_blks N`                    | `256`   | Grid size for push kernel                                                                                        |
| `--num_thds N`                    | `256`   | Block size for push kernel                                                                                       |
| `--tricknum N`                    | `0`     | Warm-up "trick" kernel footprint (see `App::trick`) — 0 disables it                                              |
| `--runs N`                        | `10`    | Number of repeat runs; reported timings are averages                                                             |
| `--check`                         | off     | Run the CPU verifier after the GPU result. **Only effective on `*_check*` binaries.** On plain binaries it prints `--check requires building with -DSCAPH_KEEP_CSR_FOR_CHECK`. |
| `--alpha F`                       | `0.85`  | **PR only** — damping factor                                                                                     |
| `--tolerance F` (`--tol`)         | `0.01`  | **PR only** — residual convergence threshold                                                                     |
| `--max_iter N` (`--iter`)         | `5000`  | **PR only** — hard iteration cap                                                                                 |

Typical max_memory values used in this repo's benchmarks: `2048`–`8192`
for 32-bit builds, `4096`–`14336` for 64-bit builds. A 16 GB A4000 can go
up to roughly `14336`.

---

## 3. Input formats

Scaph supports two input formats:

### 3a. Unified 64-bit binary (preferred)

Files with the suffix **`.bcsr64`** (unweighted) or **`.bwcsr64`**
(weighted) follow the layout below:

```
[uint64 num_nodes][uint64 num_edges]
[uint64 row_ptr[num_nodes]]                   // csr_idx
unweighted: [uint64 dst[num_edges]]           // neighbor
weighted:   [{uint64 dst, uint64 weight}[num_edges]]
```

This is the same format used by Liberator, so `.bcsr64` files produced
with `Liberator/converter` can be fed directly into Scaph's `*64` /
`*_check64` binaries. Loader: `src/graph_bin64.h::fetch_graph_bcsr64`.

### 3b. Legacy page-based format

A directory of sidecar files:

```
<name>_csr.idx          # row pointers
<name>_csr.ngh          # neighbours
<name>_csr.deg          # per-vertex degrees
<name>_csr.wgh          # (weighted only)
<name>graph.info        # GraphInfo struct
<name>page.info         # array of Page structs
<name>page0, page1, ... # packed vertex+edge per page
```

Generated by `tools/pagegencsr`:

```bash
cd Scaph/tools
g++ -O3 -o pagegencsr pagegencsr.cpp
./pagegencsr <name> <pagesize_MB> u          # unweighted
./pagegencsr <name> <pagesize_MB> w          # weighted
```

`<name>` is the common prefix of your `_csr.*` files (e.g. if you have
`foo_csr.idx`, pass `foo`). `<pagesize_MB>` must match the `--pagesize`
later used at run time.

Loader for this format: `src/graph.h::fetch_graph`.

---

## 4. Binary reference

All paths below are relative to `Scaph/`. Each binary prints a dataset
header, then per-run kernel/wall times, throughput, and iteration count.

### BFS — `bfs` / `bfs64` / `bfs_check` / `bfs_check64`

Level-synchronous BFS built on top of Scaph's asynchronous push kernel
(`app/bfs_kernel.cuh`). Result is stored as integer distance per vertex,
`MAX_VALUE>>1` meaning "unreachable".

```bash
# 32-bit graph, full copy (α=100%)
./bfs       --input=<graph>.bcsr  --src=0 --max_memory=4096 --runs=5

# 64-bit graph
./bfs64     --input=<graph>.bcsr64 --src=12 --max_memory=8192 --runs=5

# 64-bit graph + CPU parallel correctness check
./bfs_check64 --input=<graph>.bcsr64 --src=12 --max_memory=8192 --runs=1 --check
```

Notes:
* `bfs.cu` hardcodes `FLAGS_threshold = 3` (α ≈ 37.5 %) at the top of
  `main`. Pass `--threshold=8` on the command line to override.
* `--check` triggers `bfs::check()`, which runs an OpenMP level-sync BFS
  on the CPU and diffs it against the GPU result. Reports `cpu parallel
  bfs time` and `total errors` plus the first 10 mismatches.

### SSSP — `sssp` / `sssp64` / `sssp_check` / `sssp_check64`

Dijkstra-equivalent SSSP over the weighted variant of the push kernel
(`app/sssp_kernel.cuh`, relaxing `value[u] + w(u,v)` into `value[v]`).

```bash
./sssp64       --input=<graph>.bwcsr64 --src=0 --max_memory=4096 --runs=5
./sssp_check64 --input=<graph>.bwcsr64 --src=0 --max_memory=4096 --runs=1 --check
```

`*_check*` uses an OpenMP parallel Bellman-Ford-style relaxation
(`app/sssp.cu::check`) as the reference; false mismatches printed with
GPU vs CPU values.

### WCC — `wcc` / `wcc64` / `wcc_check`

Label-propagation weakly connected components — each vertex initialized
to its own id, then relaxes `value[v] = min(value[v], value[u])` via the
same push engine. `--src` is ignored.

```bash
./wcc64 --input=<graph>.bcsr64 --max_memory=4096 --runs=3

./wcc_check --input=<graph>.bcsr  --max_memory=4096 --runs=1 --check
```

> There is no `wcc_check64` in the Makefile; if you need a 64-bit CPU
> verifier for WCC, add a target mirroring `bfs_check64`.

### PageRank — `pr` / `pr64` / `pr_check` / `pr_check64`

Push-based PageRank aligned with [`EMOGI/pagerank.cu`](../EMOGI/pagerank.cu)
— same four-array state (`label`, `delta`, `residual`, `value`), same
default parameters (`alpha=0.85`, `tolerance=0.01`, `max_iter=5000`), and
the same iteration structure (push `delta → residual`, then update
`residual → value`; repeat until no vertex crosses `tolerance`). `value_t`
is `float` regardless of `LARGESIZE`, matching EMOGI; `LARGESIZE` only
widens `vertex_t / index_t` to 64-bit. `--src` is ignored.

The push kernel `gpu_push_csr_pr` (in `app/pr_kernel.cuh`) mirrors the
generic `gpu_push_csr` CTA / warp / thread load-balance layout but emits
`atomicAdd(&residual[v], delta[u])` instead of `writeMin`. A custom
`pagerank::run()` drives the convergence loop and hosts the update kernel
between Scaph push batches. Extra per-vertex arrays (`d_residual`,
`d_pr_value`) are allocated inside `App::load_graph` under `#ifdef APPPR`,
so PR does consume ≈ `8 × V` more GPU memory than the other apps.

```bash
# 64-bit graph, EMOGI-default parameters
./pr64 --input=<graph>.bcsr64 --max_memory=4096 --runs=1

# Custom parameters (alpha=0.9, tolerance=0.005, cap at 100 iters)
./pr64 --input=<graph>.bcsr64 --max_memory=4096 --runs=1 \
       --alpha=0.9 --tolerance=0.005 --max_iter=100

# 64-bit graph + CPU parallel correctness check
./pr_check64 --input=<graph>.bcsr64 --max_memory=4096 --runs=1 --check
```

`*_check*` runs an OpenMP parallel push-based PR on the CPU
(`app/pr.cu::check`) and compares vertex-wise against the GPU result with
a tolerance of `max(10 × tolerance, 5% relative)` to absorb the expected
float-ordering differences between the two parallel runs. The checker
prints `max abs diff`, `max rel diff`, and the number of errors.
`--check` only has an effect on `pr_check / pr_check64` (built with
`-DSCAPH_KEEP_CSR_FOR_CHECK`).

---

## 5. Worked examples

Run BFS on `uk-2007-05` with a tight 2 GB page-pool budget and print the
number of iterations:

```bash
./bfs64 \
    --input=/path/to/uk-2007-05.bcsr64 \
    --src=0 \
    --max_memory=2048 \
    --pagesize=10 \
    --threshold=3 \
    --runs=5
```

Run SSSP on `GAP-kron` with a source vertex known to have edges and
verify correctness against the CPU reference in a single invocation:

```bash
./sssp_check64 \
    --input=/path/to/GAP-kron.bwcsr64 \
    --src=89925859 \
    --max_memory=8192 \
    --runs=1 \
    --check
```

Expected output (tail):

```
|      Avgrage( 1 Rounds) SSSP wall     : X.XXXXXXXX seconds        |
|      Avgrage( 1 Rounds) SSSP kernel   : X.XXXXXXXX seconds        |
|===================================================================|
cpu parallel sssp time : X.XXXXX seconds
total errors           : 0
check passed!
```

A non-zero `total errors` indicates divergence from the CPU reference;
the first 10 offending vertices are printed as `vid: ... gpu: ... cpu:
...`.

---

## 6. Output fields

```
|                       graph name      : <dataset>                 |
|                       graph nodes     : <|V|>                     |
|                       graph edges     : <|E|>                     |
|                         BFS  costs    : <sum of kernel times>     |  # GPU-side only
|                         Throughput    : <|E|/time> GTEPS          |
|                         iterations    : <# push-kernel batches>   |
|      Avgrage(N Rounds) BFS wall       : <wall clock average>      |  # includes host queue
|      Avgrage(N Rounds) BFS kernel     : <kernel time average>     |
|      Avgrage(N Rounds) Throughput     : <|E|/time avg> GTEPS      |
```

The "iteration" count is the number of `gpu_push_csr` invocations, not
BFS levels — a single level can expand into several page-batch
kernels and vice versa.

---

## 7. Tooling (`tools/`)

| Tool                  | Purpose                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `pagegencsr`          | Convert a flat `_csr.idx / _csr.ngh / _csr.deg[ / _csr.wgh]` dataset into Scaph's legacy page format. Usage: `./pagegencsr <name> <pagesize_MB> w\|u` |
| `bin2ligra`           | Dump `_csr.*` files in Ligra's adjacency format                                                  |
| `bin2ligraweight`     | Same for weighted graphs                                                                         |
| `bin2pagecsr`         | Alternative legacy-format builder                                                                |
| `bin2totem` / `bin2totemweight` | Dump into Totem's format                                                               |

These are small standalone `.cpp` files; build with `g++ -O3 -o <name>
<name>.cpp`.
