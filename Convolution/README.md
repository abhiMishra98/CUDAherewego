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
development, but `main()` itself doesn't re-check it on every run. GPU
kernel time only (excludes H2D/D2H transfer), GTX 1650:

| Kernel | Input | Mask | Kernel time |
|---|---|---|---|
| `conv_1d` | 16,777,216 elements (1D) | 5 | ~2.0-2.4 ms |
| `conv_1d_shared` | 16,777,216 elements (1D) | 5 | ~1.9 ms (1.1-1.3x) |
| `conv_2d` | 4096 x 4096 elements (2D) | 5x5 | ~4.7-4.8 ms |
| `conv_2d_shared` | 4096 x 4096 elements (2D) | 5x5 | ~4.1 ms (1.16x) |

The speedup from tiling is real but modest here, not the order-of-magnitude
difference the "re-reads each element up to `maskW` times" framing at the
top might suggest. `maskW = 5` doesn't leave much redundancy to remove
(`conv_2d`'s worst case is 25 global reads per input element instead of 1,
but the GTX 1650's L1/L2 caches already absorb a good chunk of that reuse
even in the naive kernel), and the wall time in both cases is fairly close
to launch/pipeline overhead. The tiling would likely pay off more visibly
with a wider mask.

## Compiling and running

`main()` runs both pairs back to back and prints the table above.

```powershell
nvcc convolution.cu -o convolution.exe
```

If `nvcc` can't find `cl.exe`, run from a "Developer Command Prompt for VS"
or add the MSVC `Hostx64\x64` bin directory to `PATH`.
