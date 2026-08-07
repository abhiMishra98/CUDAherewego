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
  global memory per tap (redundancy is quadratic in `maskW` here); a
  shared-memory version is planned next, following the same tiling approach
  as `conv_1d_shared`.

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
pointers aren't allocated or filled with input, so it compiles (with
uninitialized-variable warnings) but isn't meaningful to run yet. The host-side
setup (allocation, random/CPU-reference input, timing) that was stripped out
in favor of this skeleton needs to come back before `conv_1d_shared` can be
verified or benchmarked against `conv_1d`.

```powershell
nvcc convolution.cu -o convolution.exe
```

If `nvcc` can't find `cl.exe`, run from a "Developer Command Prompt for VS"
or add the MSVC `Hostx64\x64` bin directory to `PATH`.

## Next

- Host-side setup for `conv_1d_shared` (allocate, fill, run, verify against
  the same CPU reference used for `conv_1d`, then compare kernel times) to
  confirm the tiling actually beats the naive baseline above.
- A shared-memory version of `conv_2d` (not yet written), following the same
  tiling approach, where the payoff should be larger since the redundant
  global reads are quadratic in `maskW` rather than linear.
