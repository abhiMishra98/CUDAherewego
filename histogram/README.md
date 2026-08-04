# Histogram

Counts occurrences of `a`–`z` in a byte buffer, bucketed into 7 groups of 4
letters each (`[a-d]`, `[e-h]`, ... `[y-z]`).

## Coalesced memory access

Each thread starts at `tIdx = blockIdx.x * blockDim.x + threadIdx.x` and
advances by `stride = blockDim.x * gridDim.x` per iteration. At every step,
consecutive threads (`threadIdx.x`, `threadIdx.x + 1`, ...) read consecutive
bytes (`buffer[tIdx]`, `buffer[tIdx + 1]`, ...). The GPU can merge these
per-thread reads into one wide memory transaction instead of issuing a
separate transaction per thread — this merging is coalescing, and it's why
threads are indexed to walk the buffer in lockstep rather than each owning a
separate contiguous chunk.

## Atomic operations

Multiple threads incrementing the same histogram bin is a classic
read-modify-write hazard: two threads can both read the old count, both add
1, and both write back the same new value — one increment is silently lost.
`atomicAdd` makes the read-modify-write indivisible at the hardware level, so
concurrent increments to the same address always accumulate correctly.

## Why the data race happens

Without atomics, `d_histoBuff[bin]++` compiles to a load, an add, and a
store as three separate steps. If thread A and thread B both target the same
bin, their loads/adds/stores can interleave in a way that drops an update.
This isn't a hypothetical here — with 256 threads per block and only 7 bins,
collisions on the same bin are frequent, not an edge case.

## Block-level privatization

Instead of every thread atomically updating a single histogram in global
memory (`histo`), each block keeps its own copy in shared memory
(`d_histoBuff[7]`). Threads within a block still contend with each other via
atomics on this shared copy, but blocks no longer contend with *each other*
over global memory — that contention is far more expensive since global
memory has much higher latency than shared memory. After each block finishes
accumulating locally, it does one atomic merge per bin into the global
`histo` array. This is why the kernel synchronizes twice: once after zeroing
the shared buffer, and once after all threads finish accumulating into it,
before the final merge.
