# Convolution

1D and 2D convolution with a mask centered on each output element (`P[i]`
sums `N[i - maskW/2 .. i + maskW/2] * M`), boundary-checked with zero-padding
implicit — out-of-range mask taps are simply skipped rather than read.

- `conv_1d` — one thread per output element, mask applied along a single axis.
  Reads `N` directly from global memory for every mask tap; neighboring
  threads' windows overlap heavily, so a `maskW=5` mask re-reads each input
  element up to 5 times. Naive baseline.
- `conv_1d_shared` — same math as `conv_1d`, but each block first stages its
  input tile (plus halo) into `__shared__` memory once, and every thread
  reads from there instead of global memory.
- `conv_2d` — one thread per output pixel (2D grid), mask applied as a
  `maskW x maskW` square over both axes. Still reads `N` directly from
  global memory per tap (redundancy is quadratic in `maskW` here, worse than
  `conv_1d`'s linear redundancy).
- `conv_2d_shared` — same halo-tiling idea as `conv_1d_shared`, extended to
  both axes, plus two more techniques layered on top: the mask lives in
  `__constant__` memory instead of a pointer parameter, and `N`/`P` are
  `cudaMallocPitch`-allocated instead of flat `width * height` blocks. See
  below.

## `conv_1d_shared`: tiling with a halo

Each block owns an output range of `o_tile_width` elements (not `blockDim.x`
— the block is launched wider than its output range, see below). Within the
kernel:

```cpp
int index_o = blockIdx.x * o_tile_width + threadIdx.x;  // this thread's output position
int index_i = index_o - (maskW / 2);                    // shifted left by the halo
```

`index_i` is deliberately offset `maskW/2` elements before `index_o`, so
`threadIdx.x == 0` loads the *left halo* — the elements just outside this
block's output range that its edge threads still need for their mask
window — rather than the tile's first real element. Loads outside `[0, width)`
(the true left edge of `N`, or past its right edge) are zero-padded into
shared memory instead of read.

Because the load is shifted but the compute loop isn't (`d_N[threadIdx.x + i]`,
not `d_N[threadIdx.x - maskW/2 + i]`), reading shared memory starting at
`threadIdx.x` lands exactly on the `maskW`-wide window centered on `index_o` —
the shift at load time is what makes the unshifted read at compute time
correct.

This is also why the block is launched with `blockDim.x = o_tile_width + maskW - 1`
(1024 threads for a 1020-element output tile with a 5-wide mask): the extra
`maskW - 1` threads exist purely to load the halo, and only the first
`o_tile_width` threads go on to compute and write an output
(`threadIdx.x < o_tile_width`). The `index_o < width` check alongside it
matters independently — the last block's threads can still run past the end
of `N` since `o_tile_width` doesn't evenly divide `width` in general.

## `conv_2d_shared`: constant-memory mask + pitched allocation

The tiling itself mirrors `conv_1d_shared` with a halo shift on both `row` and
`col` instead of one axis (`index_i_row`/`index_i_col` each offset by
`maskW / 2`, shared tile indexed `d_N[threadIdx.y][threadIdx.x]`, compute
loop reading the unshifted `d_N[threadIdx.y + i][threadIdx.x + j]` window).
Two more techniques are layered on top of that.

**`__constant__` mask.** Instead of a `float *M` parameter, the mask lives in
a file-scope `__constant__` array:

```cpp
#define MAX_MASK_WIDTH 5
__constant__ float d_M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];
```

Every thread in the compute loop reads `d_M[i * maskW + j]` at the same index
on the same iteration, so the constant cache serves it as one broadcast to
the whole warp rather than a separate global-memory load per thread — a
strict improvement over `conv_2d`'s `M[i * maskW + j]`. The cost is that
`MAX_MASK_WIDTH` is now a compile-time cap instead of a runtime size.

**`cudaMemcpyToSymbol` to fill it.** The mask is copied in before launch:

```cpp
float *h_M2d = (float *)malloc(maskW2d * maskW2d * sizeof(float));
// ... filled with random values ...
cudaMemcpyToSymbol(d_M, h_M2d, maskW2d * maskW2d * sizeof(float));
```

Unlike `cudaMemcpy`, the destination is the `__constant__` symbol `d_M`
itself, not a device pointer from `cudaMalloc` — `cudaMemcpyToSymbol`
resolves `d_M`'s device address and copies directly into it. Nothing in the
type system enforces this running before the kernel launch that reads
`d_M`; only call order does.

This one bit while wiring it up: `main()` originally declared a local
`float *d_M` for the (at the time) still-unfilled `conv_1d` call, which
**shadowed the file-scope `__constant__ d_M` for the rest of `main()`**.
`cudaMemcpyToSymbol(d_M, ...)` then silently resolved to that local
(uninitialized) pointer instead of the constant symbol, failing with
`invalid device symbol` and leaving the mask never copied — `conv_2d_shared`
still ran and wrote *something*, just against whatever was already in
constant memory, not the intended mask. Local names that collide with a
`__constant__`/`__device__` symbol's name are worth avoiding for exactly
this reason: nothing about the resulting call is ill-typed, it just
silently binds to the wrong thing. (The 1D naive kernel's mask pointer is
`d_M1d` today, well clear of any name collision.)

**`cudaMallocPitch` for `N` and `P`.** Rather than one flat
`width * height` block where row `r` starts at `r * width`, each row is
padded to a hardware-friendly alignment, and the real per-row byte stride is
handed back through `pitchN`/`pitchP`:

```cpp
float *d_N2ds, *d_P2ds;
size_t pitchN2ds, pitchP2ds;
cudaMallocPitch(&d_N2ds, &pitchN2ds, rowBytes2d, height2d);
cudaMallocPitch(&d_P2ds, &pitchP2ds, rowBytes2d, height2d);
```

The kernel has to index rows by that reported pitch, not the logical
`width`, or it reads into row padding (or the next row) instead of the
intended data:

```cpp
const float *N_row = (const float *)((const char *)N + index_i_row * pitchN);
d_N[threadIdx.y][threadIdx.x] = N_row[index_i_col];
...
float *P_row = (float *)((char *)P + index_o_row * pitchP);
P_row[index_o_col] = pVal;
```

The cast to `char *` before adding the pitch matters: `pitchN`/`pitchP` are
byte offsets, so adding them directly to a `float *` would advance by that
many `float`s (4x too far) instead of that many bytes.

## Measured results: naive vs. shared

`main()` runs each pair — `conv_1d`/`conv_1d_shared`, `conv_2d`/`conv_2d_shared`
— on the *same* random input, so the times below are a direct,
apples-to-apples naive-vs-tiled comparison, not two separately sized runs.
It's a pure timing harness (real allocation, real random input,
`cudaEvent`-timed launches) with no CPU-reference/correctness check —
each kernel's output was verified against a CPU reference during
development, but `main()` itself doesn't re-check it on every run.

`main()` calls `warmupGPU()` before any timed section — it launches each of
the four kernels once, untimed, on tiny zeroed buffers first. Without it,
whichever kernel launches first (`conv_1d`) absorbs one-time costs (CUDA
context init, module load, GPU clocks ramping up from an idle power state)
that have nothing to do with the kernel itself, inflating its time and
making later kernels look artificially better by comparison. GPU kernel
time only (excludes H2D/D2H transfer), GTX 1650, post-warmup:

| Kernel | Input | Mask | Kernel time |
|---|---|---|---|
| `conv_1d` | 16,777,216 elements (1D) | 5 | ~0.98-1.0 ms |
| `conv_1d_shared` | 16,777,216 elements (1D) | 5 | ~1.47-1.48 ms (0.66-0.67x — *slower*) |
| `conv_2d` | 4096 x 4096 elements (2D) | 5x5 | ~3.79-3.81 ms |
| `conv_2d_shared` | 4096 x 4096 elements (2D) | 5x5 | ~3.27-3.28 ms (1.16x) |

Before `warmupGPU()` existed, `conv_1d` measured ~2.0-2.4 ms and
`conv_1d_shared` looked ~1.1-1.3x faster — that gap was mostly the cold-clock
penalty landing on `conv_1d` because it happened to launch first, not a real
tiling advantage. With clocks already warm, `conv_1d_shared` is consistently
*slower* than the naive kernel: `maskW = 5` doesn't leave much redundancy to
remove in 1D (the GTX 1650's L1/L2 already absorbs most of that reuse even in
the naive kernel), so the `__syncthreads()` + shared-memory staging overhead
in `conv_1d_shared` costs more than it saves. `conv_2d_shared` still wins
because 2D redundancy is quadratic in `maskW` (up to 25 global reads per
input element vs. 1), which is enough to outweigh the same tiling overhead.
The tiling would likely pay off more visibly in 1D too with a wider mask.

## Profiling with Nsight Systems

`nsys profile --trace=cuda,osrt --stats=true ./convolution` (GTX 1650) shows
transfer time dwarfing compute: `cuda_gpu_mem_time_sum` totals ~151 ms of
H2D+D2H copies across the run, versus ~12.3 ms of `cuda_gpu_kern_sum` kernel
execution — transfers are ~12x the compute. Overlapping memcpy with compute
(streams + pinned host memory) only hides the *smaller* of the two; whichever
side is larger still sets the floor on total time. Here that's transfers, so
the higher-leverage fix is shrinking/speeding up the copies themselves
(pinned memory, fewer/larger transfers) rather than overlap alone — overlap
becomes worth adding on top once compute is no longer trivially smaller than
transfer.

![Nsight Systems timeline showing conv_2d and conv_2d_shared as thin kernel bars sandwiched between much wider memcpy blocks](../images/Conv_V1_NsightSystems.png)

## Compute-bound or memory-bound?

Convolution with a small mask is a classic memory-bound operation: each
output element does only `maskW` (or `maskW x maskW` in 2D) multiply-adds,
but touches roughly that many input elements from memory to do it — very
little arithmetic per byte moved. The prediction, before profiling: all four
kernels here should be memory-bound, not compute-bound.

**Nsight Compute roofline, per kernel:**

```bash
sudo ncu --set full --section SpeedOfLight_RooflineChart \
  --launch-skip 4 --launch-count 4 \
  --export convolution_ncu_report ./convolution
```

(`--launch-skip 4` skips `warmupGPU()`'s 4 untimed launches, so the 4
profiled ones are the real, timed kernels.) Open the result with
`ncu-ui convolution_ncu_report.ncu-rep` (you may need
`sudo chown $USER convolution_ncu_report.ncu-rep` first, since `ncu` ran as
root), select each kernel launch, go to the **Details** page, expand
**GPU Speed Of Light Throughput**, and switch its content-selector dropdown
(top-right of that section) from "GPU Throughput Chart" to **"GPU Throughput
Roofline"** — that's the FP32/FP64 chart matching `SpeedOfLight_RooflineChart`
above (the Half/Single-hierarchical/Tensor Core options are separate sections
this command didn't collect, and don't apply here anyway — none of these
kernels use `__half` or tensor-core instructions).

## Profiling with Nsight Compute

| Kernel | Duration | % of FP32 peak | Achieved DRAM BW | % of peak BW (~192 GB/s) | `ncu`'s own verdict |
|---|---|---|---|---|---|
| `conv_1d` | 1.18 ms | 6% | 114.1 GB/s | ~60% | Latency issue |
| `conv_1d_shared` | 1.80 ms | 4% | 74.3 GB/s | ~39% | Latency issue |
| `conv_2d` | 4.71 ms | 7% | 28.5 GB/s | ~15% | Well-balanced (compute & memory both low) |
| `conv_2d_shared` | 4.07 ms | 8% | 33.3 GB/s | ~17% | Latency issue |

**`conv_1d`** — dot sits just under the memory-bandwidth diagonal, the
closest of the four to its own roofline:

![conv_1d roofline](../images/Conv_V1_1d_ncu.png)

**`conv_1d_shared`** — dot sits further below the diagonal than `conv_1d`
at a similar arithmetic intensity, despite doing less redundant global work:

![conv_1d_shared roofline](../images/Conv_v1_1dShared_ncu.png)

**`conv_2d`** — arithmetic intensity shifted well right of the 1D kernels,
but the dot is the furthest below the diagonal of all four:

![conv_2d roofline](../images/Conv_v1_2d_ncu.png)

**`conv_2d_shared`** — similarly large gap below the diagonal as `conv_2d`,
despite the tiling reducing global memory traffic further:

![conv_2d_shared roofline](../images/Conv_v1_2dShared_ncu.png)

**What the roofline model actually suggests here:** only `conv_1d` sits close
to its own roofline (the achieved point is near the diagonal directly above
its arithmetic intensity) — that one is honestly memory-bound and the
"speed it up by moving less data" prediction holds. The other three,
`conv_2d` in particular, are plotted well *below* their own diagonal despite
`conv_2d`/`conv_2d_shared` having pushed arithmetic intensity further right
than the 1D kernels (more reuse per byte from tiling/constant memory) — a
falling dot at higher intensity with no better performance is the roofline
chart's way of saying the bottleneck isn't bandwidth *or* compute, it's
latency: not enough independent warps in flight to hide stalls, regardless
of how much data movement was avoided.

Two concrete, `ncu`-flagged causes point at exactly that, rather than at
"do more math" (which is the usual roofline advice, but doesn't apply — none
of these kernels are anywhere near the compute ceiling to begin with):

- **Uncoalesced global memory access.** `SourceCounters` flags 12% excessive
  sectors on `conv_1d`, 10% on `conv_1d_shared`, **24%** on `conv_2d`, 12% on
  `conv_2d_shared` — `conv_2d`'s row-major `N[curRow * width + curCol]`
  indexing with a 16x16 thread block is the worst offender, meaning nearly a
  quarter of its memory transactions move bytes no thread actually needed.
  This directly lowers the achieved point without changing arithmetic
  intensity at all — fixing access patterns moves the dot straight up at
  the same x.
- **ALU-bound instruction mix, not FP32-bound.** `conv_1d`'s "Compute (SM)
  Throughput" reads 55%, yet only 6% of that is the FP32 FMA pipe — ncu
  separately flags "ALU is the highest-utilized pipeline (36.7%)". Most
  issued instructions are address/index arithmetic and bounds-checking
  (`ip_Start + i >= 0 && ... < width`, pitch-byte-offset casts, etc.), not
  the convolution math itself. This is a second, independent way these
  kernels fail to reach their roofline: the SM is busy, just not on the
  work the chart is measuring.

Net suggestion: for this workload, chasing arithmetic intensity (wider mask,
more work per thread, fp16) is the wrong lever — none of the four kernels
are pinned against the compute roof. The actual next steps are occupancy/
coalescing fixes (reduce boundary-check branching, coalesce `conv_2d`'s
access pattern, increase active warps per scheduler) to close the *vertical*
gap to each kernel's own roofline, not a *horizontal* move to a higher
arithmetic intensity.

## Compiling and running

`main()` runs both pairs back to back and prints the table above.

```powershell
nvcc convolution.cu -o convolution.exe
```

If `nvcc` can't find `cl.exe`, run from a "Developer Command Prompt for VS"
or add the MSVC `Hostx64\x64` bin directory to `PATH`.
