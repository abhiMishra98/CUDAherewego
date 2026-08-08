# Convolution

CUDA implementations of 1D and 2D convolution, each with a naive (global-memory)
baseline and a shared-memory tiled variant. Includes benchmarking with
`cudaEvent`s and profiling notes from Nsight Systems and Nsight Compute.

## Table of contents

- [What is convolution](#what-is-convolution)
- [Implementations](#implementations)
- [`__constant__` memory](#__constant__-memory)
- [`cudaMallocPitch`](#cudamallocpitch)
- [Measured results: naive vs. shared](#measured-results-naive-vs-shared)
- [Profiling with Nsight Systems](#profiling-with-nsight-systems)
- [Profiling with Nsight Compute](#profiling-with-nsight-compute)
- [Compiling and running](#compiling-and-running)

## What is convolution

Convolution computes each output element as a weighted sum of the input
elements in a neighborhood around it, using a small, fixed kernel of weights
(the *mask*). For 1D input `N` and mask `M` of width `maskW`:

```
P[i] = sum over k of M[k] * N[i - maskW/2 + k]
```

2D convolution applies the same idea over two axes with a `maskW x maskW`
mask. Mask taps that fall outside the input bounds are treated as zero
(zero-padding) rather than read.

Each output element requires `maskW` (1D) or `maskW x maskW` (2D)
multiply-adds but touches roughly that many input elements to do it, a low
ratio of arithmetic to data movement. This makes convolution with a small
mask a classic memory-bound workload, a property that shapes both the
optimizations below and the profiling results later in this document.

## Implementations

All four kernels live in [convolution.cu](convolution.cu).

### `conv_1d` (naive baseline)

One thread per output element. Each thread reads its `maskW`-element window
directly from global memory:

```cpp
__global__ void conv_1d(float *N, float *M, float *P, int maskW, int width)
```

Neighboring threads' windows overlap heavily. With `maskW = 5`, each input
element is re-read from global memory up to 5 times across the block.

### `conv_1d_shared` (halo-tiled)

Same math as `conv_1d`, but each block loads its chunk of input into fast
`__shared__` memory once, and all threads in that block read from there
instead of going back to global memory for every mask tap.

Each block is responsible for producing `o_tile_width` outputs, but it also
needs a few extra input elements on either side (the *halo*) to correctly
compute the outputs at the edges of its tile. So the block is launched with
a few more threads than outputs, and those extra threads' only job is to
load the halo elements:

```cpp
int index_o = blockIdx.x * o_tile_width + threadIdx.x;  // this thread's output position
int index_i = index_o - (maskW / 2);                    // shifted left by the halo
```

Every thread loads one element into shared memory at `index_i`, which is
shifted left by `maskW / 2` so the block's first few threads pick up the
halo rather than the tile's first real element. Any load that falls outside
the input (`[0, width)`) is treated as zero instead of read.

Once the load is done and threads are synced, only the first
`o_tile_width` threads compute an output, each reading its `maskW`-wide
window straight out of shared memory instead of global memory.

### `conv_2d` (naive baseline)

One thread per output pixel, mask applied as a `maskW x maskW` square over
both axes. Still reads `N` directly from global memory per tap, so
redundancy here is quadratic in `maskW`, worse than `conv_1d`'s linear
redundancy.

### `conv_2d_shared` (halo-tiled, constant-memory mask, pitched allocation)

Extends the `conv_1d_shared` tiling to both axes (`index_i_row`/`index_i_col`
each offset by `maskW / 2`, shared tile indexed `d_N[threadIdx.y][threadIdx.x]`,
compute loop reading the unshifted `d_N[threadIdx.y + i][threadIdx.x + j]`
window), and layers two further techniques on top: the mask is read from
`__constant__` memory instead of a pointer parameter, and `N`/`P` are
allocated with `cudaMallocPitch` instead of flat `width * height` blocks.
Both are covered in detail below.

## `__constant__` memory

**What it is.** A dedicated read-only memory space, backed by a cache
optimized for the case where many threads read the *same* address at the
same time (broadcast), rather than each thread reading a different address.

**Why it's used here.** In `conv_2d_shared`'s compute loop, every thread in a
warp reads `d_M[i * maskW + j]` at the same index on the same iteration.
Serving that from `__constant__` memory costs one broadcast read for the
whole warp, versus a separate global-memory load per thread for `conv_2d`'s
`M[i * maskW + j]`. The mask is small (`maskW x maskW`, capped at
`MAX_MASK_WIDTH x MAX_MASK_WIDTH`) and read-only for the duration of the
kernel, exactly the access pattern `__constant__` memory is built for.

**How it's used in this implementation.** The mask is declared file-scope
instead of passed as a kernel parameter:

```cpp
#define MAX_MASK_WIDTH 5
__constant__ float d_M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];
```

`MAX_MASK_WIDTH` becomes a compile-time cap in exchange for the broadcast
read: a runtime-sized mask isn't possible with this approach.

The mask is copied in before launch with `cudaMemcpyToSymbol`, not
`cudaMemcpy`:

```cpp
float *h_M2d = (float *)malloc(maskW2d * maskW2d * sizeof(float));
// ... filled with random values ...
cudaMemcpyToSymbol(d_M, h_M2d, maskW2d * maskW2d * sizeof(float));
```

The destination is the `__constant__` symbol `d_M` itself, not a device
pointer from `cudaMalloc`. `cudaMemcpyToSymbol` resolves `d_M`'s device
address and copies directly into it. Nothing in the type system enforces
that this call happens before the kernel launch that reads `d_M`; only call
order does.

> **Note.** During development, `main()` briefly declared a local
> `float *d_M` for the (at the time) unfilled `conv_1d` call, which shadowed
> the file-scope `__constant__ d_M` for the rest of `main()`.
> `cudaMemcpyToSymbol(d_M, ...)` then silently resolved to that local,
> uninitialized pointer instead of the constant symbol, failing with
> `invalid device symbol`. The mask was never copied, and `conv_2d_shared`
> ran against whatever was already in constant memory. The call itself was
> well-typed throughout; it simply bound to the wrong symbol. Local names
> that collide with a `__constant__`/`__device__` symbol are worth avoiding
> for exactly this reason. (The 1D naive kernel's mask pointer is `d_M1d`
> today, clear of any name collision.)

## `cudaMallocPitch`

**What it is.** A device allocator for 2D data that pads each row up to a
hardware-friendly alignment, rather than packing rows back-to-back. It
returns the real per-row byte stride (the *pitch*) alongside the pointer,
since that stride is no longer equal to `width * sizeof(float)`.

**Why it's used here.** A flat `width * height` allocation starts row `r` at
byte offset `r * width * sizeof(float)`, which may not satisfy the alignment
the memory controller wants for each row's first element. `cudaMallocPitch`
guarantees row-start alignment by padding, trading a small amount of unused
memory per row for more efficient row-aligned access.

**How it's used in this implementation.** `conv_2d_shared`'s input and
output buffers are allocated with the pitched API:

```cpp
float *d_N2ds, *d_P2ds;
size_t pitchN2ds, pitchP2ds;
cudaMallocPitch(&d_N2ds, &pitchN2ds, rowBytes2d, height2d);
cudaMallocPitch(&d_P2ds, &pitchP2ds, rowBytes2d, height2d);
```

The kernel must index rows by the reported pitch, not the logical `width`,
or it reads into row padding (or the next row) instead of the intended
data:

```cpp
const float *N_row = (const float *)((const char *)N + index_i_row * pitchN);
d_N[threadIdx.y][threadIdx.x] = N_row[index_i_col];
...
float *P_row = (float *)((char *)P + index_o_row * pitchP);
P_row[index_o_col] = pVal;
```

The cast to `char *` before adding the pitch is required: `pitchN`/`pitchP`
are byte offsets, so adding them directly to a `float *` would advance by
that many `float`s (four times too far) instead of that many bytes.
Host-side transfers into and out of pitched buffers use `cudaMemcpy2D`
rather than `cudaMemcpy`, for the same reason.

## Measured results: naive vs. shared

`main()` runs each pair (`conv_1d`/`conv_1d_shared` and
`conv_2d`/`conv_2d_shared`) on the same random input, timed with
`cudaEvent`s, so the numbers below are a direct, apples-to-apples
naive-vs-tiled comparison rather than two separately sized runs. This is a
pure timing harness (real allocation, real random input, `cudaEvent`-timed
launches); each kernel's output was verified against a CPU reference during
development, but `main()` itself does not re-check correctness on every run.

`main()` calls `warmupGPU()` before any timed section, launching all four
kernels once, untimed, on small zeroed buffers. Without this, whichever
kernel launches first (`conv_1d`) absorbs one-time costs (CUDA context
initialization, module load, GPU clocks ramping up from an idle power
state) that are unrelated to the kernel itself, inflating its time and
making later kernels look artificially better by comparison.

GPU kernel time only (excludes H2D/D2H transfer), post-warmup. See the
[repo root README](../README.md#test-environment) for GPU specs.

| Kernel | Input | Mask | Kernel time |
|---|---|---|---|
| `conv_1d` | 16,777,216 elements (1D) | 5 | ~0.98-1.00 ms |
| `conv_1d_shared` | 16,777,216 elements (1D) | 5 | ~1.47-1.48 ms (0.66-0.67x, *slower*) |
| `conv_2d` | 4096 x 4096 elements (2D) | 5x5 | ~3.79-3.81 ms |
| `conv_2d_shared` | 4096 x 4096 elements (2D) | 5x5 | ~3.27-3.28 ms (1.16x) |

Before `warmupGPU()` existed, `conv_1d` measured ~2.0-2.4 ms and
`conv_1d_shared` appeared ~1.1-1.3x faster; that gap was mostly the
cold-clock penalty landing on `conv_1d` because it launched first, not a
real tiling advantage.

With clocks already warm, `conv_1d_shared` is consistently *slower* than the
naive kernel: at `maskW = 5`, 1D redundancy is small enough that the GPU's
L1/L2 already absorbs most of the reuse even in the naive kernel, so
the `__syncthreads()` and shared-memory staging overhead in `conv_1d_shared`
costs more than it saves. `conv_2d_shared` still wins because 2D redundancy
is quadratic in `maskW` (up to 25 global reads per input element vs. 1),
which is enough to outweigh the same tiling overhead. Tiling would likely
pay off more visibly in 1D too with a wider mask.

## Profiling with Nsight Systems

```bash
nsys profile --trace=cuda,osrt --stats=true ./convolution
```

This shows transfer time dwarfing compute:
`cuda_gpu_mem_time_sum` totals ~151 ms of H2D+D2H copies across the run,
versus ~12.3 ms of `cuda_gpu_kern_sum` kernel execution: transfers are
about 12x the compute.

Overlapping memcpy with compute (streams and pinned host memory) only hides
the *smaller* of the two; whichever side is larger still sets the floor on
total time. Here that's transfers, so the higher-leverage fix is
shrinking/speeding up the copies themselves (pinned memory, fewer/larger
transfers) rather than overlap alone. Overlap becomes worthwhile on top of
that once compute is no longer trivially smaller than transfer.

![Nsight Systems timeline showing conv_2d and conv_2d_shared as thin kernel bars sandwiched between much wider memcpy blocks](../images/Conv_V1_NsightSystems.png)

## Profiling with Nsight Compute

**Compute-bound or memory-bound?** As noted above, convolution with a small
mask does little math per byte moved, so all four kernels are expected to
be memory-bound rather than compute-bound. Nsight Compute's roofline chart
checks that expectation directly: it plots each kernel's achieved
performance against the GPU's theoretical compute and memory ceilings, so
you can see at a glance which ceiling (if either) is actually limiting it.

Roofline data was collected per kernel with:

```bash
sudo ncu --set full --section SpeedOfLight_RooflineChart \
  --launch-skip 4 --launch-count 4 \
  --export convolution_ncu_report ./convolution
```

`--launch-skip 4` skips `warmupGPU()`'s 4 untimed launches, so only the 4
real, timed kernels get profiled. The report was then opened in `ncu-ui`,
and for each kernel the **GPU Speed Of Light Throughput** section was
switched to its **Roofline** view:

| Kernel | Duration | % of FP32 peak | Achieved DRAM BW | % of peak BW (~192 GB/s) | `ncu`'s verdict |
|---|---|---|---|---|---|
| `conv_1d` | 1.18 ms | 6% | 114.1 GB/s | ~60% | Latency issue |
| `conv_1d_shared` | 1.80 ms | 4% | 74.3 GB/s | ~39% | Latency issue |
| `conv_2d` | 4.71 ms | 7% | 28.5 GB/s | ~15% | Well-balanced (compute & memory both low) |
| `conv_2d_shared` | 4.07 ms | 8% | 33.3 GB/s | ~17% | Latency issue |

![conv_1d roofline](../images/Conv_V1_1d_ncu.png)
![conv_1d_shared roofline](../images/Conv_v1_1dShared_ncu.png)
![conv_2d roofline](../images/Conv_v1_2d_ncu.png)
![conv_2d_shared roofline](../images/Conv_v1_2dShared_ncu.png)

**Reading the charts.** Only `conv_1d`'s dot sits close to its own roofline
diagonal, so that one is genuinely memory-bound: the "move less data to go
faster" logic actually applies to it. The other three sit well below their
diagonal, and `conv_2d`/`conv_2d_shared` do so even though tiling and
constant memory gave them *more* reuse per byte than the 1D kernels, not
less. A kernel that does more with each byte but still doesn't go faster
isn't limited by bandwidth or compute; it's limited by latency, meaning too
few warps are in flight at once to hide memory stalls.

`ncu` points at two concrete causes, not "not enough math" (none of these
kernels are close to the compute ceiling in the first place):

- **Uncoalesced memory access.** `SourceCounters` reports excessive memory
  sectors on all four kernels (12% for `conv_1d`, 10% for `conv_1d_shared`,
  **24%** for `conv_2d`, 12% for `conv_2d_shared`). `conv_2d` is worst: its
  16x16 thread block and row-major indexing mean close to a quarter of its
  memory traffic moves bytes no thread needed. Fixing the access pattern
  would raise the achieved point without changing arithmetic intensity.
- **Instruction mix skewed toward address math, not FP32 work.** For
  `conv_1d`, `ncu` reports 55% "Compute (SM) Throughput," but only 6% of
  that is the actual FP32 FMA pipe; it separately flags the ALU pipeline as
  the busiest at 36.7%. Most of that ALU work is bounds-checking and index
  arithmetic, not the convolution itself, so the SM is busy without doing
  much of the work the roofline chart credits.

**Net takeaway.** For this workload, adding more arithmetic (a wider mask,
more work per thread, fp16) would not help; none of the four kernels are
compute-limited. The more useful next steps are occupancy and
coalescing fixes, such as simplifying boundary checks, fixing `conv_2d`'s
access pattern, and keeping more warps active per scheduler, to close the
gap to each kernel's own roofline.

## Compiling and running

`main()` runs both pairs back to back and prints the timing table above.

```powershell
nvcc convolution.cu -o convolution.exe
```

If `nvcc` can't find `cl.exe`, run from a "Developer Command Prompt for VS",
or add the MSVC `Hostx64\x64` bin directory to `PATH`.
