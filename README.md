# CUDA-Performance-Lab
Sharing my learnings on CUDA, profiling, benchmarking, and optimization of GPU kernels

## Test environment

All benchmarks and profiling numbers in this repo (including sub-folder
READMEs) were measured on:

- **GPU:** NVIDIA GeForce GTX 1650, 14 SMs
- **Peak memory bandwidth:** ~192 GB/s

## Programs

- **clr_greyscale.cu** — Converts an RGB image to grayscale. Each thread maps to one pixel, reading the 3-channel RGB value and writing a single grayscale value using the standard luminance weights (0.21R + 0.71G + 0.07B). Not runnable yet — `main()` is a kernel-launch sketch referencing undeclared variables (`n`, `m`, `w`, `h`, `d_Pin`, `d_Pout`); host-side setup is coming in a future commit.

- **blurKernel.cu** — Applies a simple box blur to a single-channel image. Each thread averages a `(2*BLUR_SIZE+1) x (2*BLUR_SIZE+1)` neighborhood around its pixel, clamping at image boundaries. Not runnable yet — `main()` is a kernel-launch sketch referencing undeclared variables (`n`, `m`, `in`, `out`, `width`, `height`); host-side setup is coming in a future commit.

- **tileMatMul.cu** — Tiled square matrix multiplication using shared memory. Loads `TILE_WIDTH x TILE_WIDTH` tiles of the input matrices into shared memory per block to reduce global memory traffic. Assumes the matrix width is evenly divisible by `TILE_WIDTH`. Complete, runnable program with random input generation and result printing.

- **tileMatMulGeneric.cu** — Same tiled matrix multiplication approach as `tileMatMul.cu`, but generalized to handle matrix widths that are *not* evenly divisible by `TILE_WIDTH`, with boundary checks that zero-pad out-of-range shared memory loads. Complete, runnable program.

- **[histogram/](histogram/)** — Counts occurrences of `a`–`z` in a byte buffer, built up as a sequence of measured optimizations rather than a single kernel:
  - naive global `atomicAdd` per byte (`histo_kernel_naive`) as the baseline
  - block-level privatization — per-block shared-memory histogram, merged once into global memory
  - register-level privatization — per-thread counts accumulated with no atomics at all, merged into the block's shared histogram
  - launch config sized off the GPU (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) instead of the input, after that mismatch was caught making the naive kernel look faster than the privatized one
  
  Profiled with Nsight Systems: privatization measured ~2.9x faster than the naive baseline once the grid was sized correctly. That same profiling also exposed the current bottleneck — the H2D transfer fully blocks before either kernel starts — laying the groundwork for the next planned optimization (overlapping transfer with compute via pinned memory and streams). Full writeup, numbers, and timeline screenshot in [histogram/README.md](histogram/README.md). Complete, runnable program.

- **[Convolution/](Convolution/)** — 1D and 2D mask-centered convolution, one thread per output element/pixel, boundary-checked with implicit zero-padding. `conv_1d` and `conv_2d` are the naive baselines (read the input directly from global memory per mask tap); `conv_1d_shared` tiles each block's input (plus halo) into `__shared__` memory once instead. `conv_1d`/`conv_2d` kernel times measured against a CPU reference in [Convolution/README.md](Convolution/README.md); `conv_1d_shared` doesn't have numbers yet, and a shared-memory version of `conv_2d` isn't written yet either. Not runnable yet — `main()` is a kernel-launch sketch with the block/grid dims set up but no allocated input; host-side setup is coming back to verify and benchmark the shared-memory kernels.

## Compiling and running

Requires the NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU.

The runnable programs (`tileMatMul.cu`, `tileMatMulGeneric.cu`) can be built and run directly:

```bash
nvcc tileMatMul.cu -o tileMatMul
./tileMatMul

nvcc tileMatMulGeneric.cu -o tileMatMulGeneric
./tileMatMulGeneric
```

On Windows (PowerShell) with the toolkit installed, the same `nvcc` command produces a `.exe`:

```powershell
nvcc tileMatMul.cu -o tileMatMul.exe
.\tileMatMul.exe
```

`clr_greyscale.cu`, `blurKernel.cu`, and `Convolution/convolution.cu` will get the same treatment once their host-side code lands.

`histogram.cu` lives in its own folder and needs an extra flag plus a test file, since it reads input at runtime rather than generating it:

```powershell
cd histogram
nvcc -std=c++17 histogram.cu -o histogram.exe
.\histogram.exe
```

The `-std=c++17` flag is required — CUDA's `<cuda/atomic>` header won't compile without it. `input.txt` isn't checked into the repo (see [histogram/README.md](histogram/README.md) for why); generate your own before running, e.g. a few MB of random bytes.
