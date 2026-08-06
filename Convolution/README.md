# Convolution

1D and 2D convolution with a mask centered on each output element (`P[i]`
sums `N[i - maskW/2 .. i + maskW/2] * M`), boundary-checked with zero-padding
implicit — out-of-range mask taps are simply skipped rather than read.

- `conv_1d` — one thread per output element, mask applied along a single axis.
- `conv_2d` — one thread per output pixel (2D grid), mask applied as a
  `maskW x maskW` square over both axes.

Both kernels currently read `N` directly from global memory for every mask
tap. Neighboring threads' mask windows overlap heavily (a `maskW=5` mask means
each input element is re-read up to 5 times in 1D, 25 times in 2D), so this
is the naive baseline before the planned shared-memory version, which will
stage each block's input tile (plus halo) into shared memory once and have
every thread in the block read from there instead.

## Measured results

GPU kernel time only (`cudaEvent` around the kernel launch, excludes
H2D/D2H transfer), GTX 1650:

| Kernel | Input | Mask | Kernel time |
|---|---|---|---|
| `conv_1d` | 16,777,216 elements (1D) | 5 | ~1.4 ms |
| `conv_2d` | 4096 x 4096 elements (2D) | 5x5 | ~4.8 ms |

Verified correct against a CPU reference implementation before timing.

## Compiling and running

`main()` is currently a kernel-launch sketch — it declares the block/grid
dims and calls `conv_1d`/`conv_2d`, but the device pointers aren't allocated
or filled with input, so it compiles (with uninitialized-variable warnings)
but isn't meaningful to run yet. Host-side setup is coming back with the
shared-memory version below.

```powershell
nvcc convolution.cu -o convolution.exe
```

If `nvcc` can't find `cl.exe`, run from a "Developer Command Prompt for VS"
or add the MSVC `Hostx64\x64` bin directory to `PATH`.

## Next: shared memory

Plan is to stage each block's input tile (including the halo needed by
threads at the tile's edges) into `__shared__` memory once per block, so the
`maskW`-times-redundant global reads above become one global read per element
plus fast shared-memory reads for the rest. Expect this to matter more for
`conv_2d`, where the redundancy is quadratic in `maskW`.
