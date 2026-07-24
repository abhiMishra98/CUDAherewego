# CUDAherewego
Sharing my learnings on CUDA

## Programs

- **clr_greyscale.cu** — Converts an RGB image to grayscale. Each thread maps to one pixel, reading the 3-channel RGB value and writing a single grayscale value using the standard luminance weights (0.21R + 0.71G + 0.07B). Not runnable yet — `main()` is a kernel-launch sketch referencing undeclared variables (`n`, `m`, `w`, `h`, `d_Pin`, `d_Pout`); host-side setup is coming in a future commit.

- **blurKernel.cu** — Applies a simple box blur to a single-channel image. Each thread averages a `(2*BLUR_SIZE+1) x (2*BLUR_SIZE+1)` neighborhood around its pixel, clamping at image boundaries. Not runnable yet — `main()` is a kernel-launch sketch referencing undeclared variables (`n`, `m`, `in`, `out`, `width`, `height`); host-side setup is coming in a future commit.

- **tileMatMul.cu** — Tiled square matrix multiplication using shared memory. Loads `TILE_WIDTH x TILE_WIDTH` tiles of the input matrices into shared memory per block to reduce global memory traffic. Assumes the matrix width is evenly divisible by `TILE_WIDTH`. Complete, runnable program with random input generation and result printing.

- **tileMatMulGeneric.cu** — Same tiled matrix multiplication approach as `tileMatMul.cu`, but generalized to handle matrix widths that are *not* evenly divisible by `TILE_WIDTH`, with boundary checks that zero-pad out-of-range shared memory loads. Complete, runnable program.

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

`clr_greyscale.cu` and `blurKernel.cu` will get the same treatment once their host-side code lands.
