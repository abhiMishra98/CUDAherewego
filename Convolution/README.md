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
float h_M2ds[MAX_MASK_WIDTH * MAX_MASK_WIDTH] = {0};
cudaMemcpyToSymbol(d_M, h_M2ds, maskW2ds * maskW2ds * sizeof(float));
```

Unlike `cudaMemcpy`, the destination is the `__constant__` symbol `d_M`
itself, not a device pointer from `cudaMalloc` — `cudaMemcpyToSymbol`
resolves `d_M`'s device address and copies directly into it. Nothing in the
type system enforces this running before the kernel launch that reads
`d_M`; only call order does.

**`cudaMallocPitch` for `N` and `P`.** Rather than one flat
`width * height` block where row `r` starts at `r * width`, each row is
padded to a hardware-friendly alignment, and the real per-row byte stride is
handed back through `pitchN`/`pitchP`:

```cpp
float *d_N2ds, *d_P2ds;
size_t pitchN2ds, pitchP2ds;
cudaMallocPitch(&d_N2ds, &pitchN2ds, width2ds * sizeof(float), height2ds);
cudaMallocPitch(&d_P2ds, &pitchP2ds, width2ds * sizeof(float), height2ds);
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

## Measured results

GPU kernel time only (`cudaEvent` around the kernel launch, excludes
H2D/D2H transfer), GTX 1650:

| Kernel | Input | Mask | Kernel time |
|---|---|---|---|
| `conv_1d` | 16,777,216 elements (1D) | 5 | ~1.4 ms |
| `conv_2d` | 4096 x 4096 elements (2D) | 5x5 | ~4.8 ms |

Verified correct against a CPU reference implementation before timing. These
predate `conv_1d_shared`, which doesn't have numbers yet — see below.

## Compiling and running

`main()` is currently a kernel-launch sketch — it declares the block/grid
dims and calls `conv_1d`, `conv_1d_shared`, and `conv_2d`, but the device
pointers aren't allocated or filled with input. 

`conv_2d_shared` is the exception — it genuinely calls `cudaMallocPitch` and
`cudaMemcpyToSymbol` (and `cudaFree`s what it allocates), since the whole
point is demonstrating a real pitch value coming back from the allocator.
Input data still isn't filled in, so it runs without meaningful results, but
the allocation/mask-copy/launch/free sequence itself is real.

```powershell
nvcc convolution.cu -o convolution.exe
```

If `nvcc` can't find `cl.exe`, run from a "Developer Command Prompt for VS"
or add the MSVC `Hostx64\x64` bin directory to `PATH`.
