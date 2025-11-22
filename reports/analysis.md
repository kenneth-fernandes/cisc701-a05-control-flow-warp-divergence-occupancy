# CUDA Warp Divergence and Occupancy Analysis

## 1. What is a warp in CUDA and how many threads does it typically contain?

- A warp is a group of 32 threads that the GPU hardware schedules and executes together as the most fundamental unit of execution, meaning the GPU never runs individual threads but always processes them in these groups of 32 that move through the pipeline together.

- Think of it like a team of 32 workers who must all perform the exact same task at the exact same time because they share a single set of instructions - if even one worker needs to do something different, all 32 workers have to wait while that different task gets handled, which is why understanding warps is critical for writing efficient GPU code.

- Threads are automatically grouped into warps based on their thread IDs in a sequential manner: threads 0-31 form the first warp (warp 0), threads 32-63 form the second warp (warp 1), threads 64-95 form warp 2, and this pattern continues throughout the entire thread block.

- In the experiment, when using a block size of 256 threads, each block contained exactly 8 warps because 256 divided by 32 equals 8, and this meant that within each block, the GPU scheduler had 8 independent groups of threads it could manage and potentially switch between during execution.

- If your block size is not a multiple of 32, the GPU still allocates resources for a complete warp but the extra threads sit idle and do no useful work - for example, a block of 50 threads creates 2 warps (since you need at least 2 warps to hold 50 threads), but the second warp only has 18 active threads while the remaining 14 thread slots are wasted, which reduces your computational efficiency.

- The warp size of 32 threads is a hardware constant that remains consistent across all modern NVIDIA GPUs from different generations and different performance tiers, as confirmed by the notebook output showing `Warp Size: 32` for the Tesla T4, and this consistency means you can write warp-aware code that works well across different NVIDIA hardware.

---

## 2. What causes warp divergence, and how is it resolved in the hardware?

### What Causes It:

- Warp divergence occurs when threads within the same warp encounter a conditional statement (like an if-else) and different threads need to take different execution paths based on their individual data values, which creates a conflict because all 32 threads in a warp are supposed to execute the same instruction at the same time.

- In the experiment's threshold kernel, the code performs a simple conditional check that demonstrates this problem clearly:
  ```python
  if input_array[idx] > threshold:
      output_array[idx] = 1.0
  else:
      output_array[idx] = 0.0
  ```

- When the input array contains linearly increasing values `[0, 1, 2, 3, ... 1048575]` and the threshold is set to the middle value of 524288, approximately half of all threads process values that are greater than the threshold and need to execute the `if` branch, while the other half process values that are less than or equal to the threshold and need to execute the `else` branch.

- The critical issue is that within any single warp of 32 consecutive threads, you will have a mix of threads with values above and below the threshold - for example, in a warp containing threads processing values around 524288, some threads might have values like 524280, 524285, 524290, 524295, and these threads genuinely need to take different paths through the code.

### How Hardware Resolves It:

- The GPU hardware cannot execute both the `if` path and the `else` path simultaneously for the same warp because all 32 threads in a warp share a single program counter (instruction pointer), which means they must all be at the same point in the code at any given moment.

- To handle this conflict, the hardware first executes one branch (let's say the `if` branch) and during this phase, only the threads that actually need to take this path are active and perform their computations, while all the threads that need the `else` branch are temporarily disabled and sit idle doing nothing productive.

- After the first branch completes execution, the hardware then switches to execute the other branch (the `else` branch), and now the situation reverses - the threads that were idle before become active and perform their computations, while the threads that already did their work in the `if` branch now sit idle and wait.

- This sequential execution of both branches means that the total execution time becomes the sum of both branch times rather than just the time of a single branch, effectively serializing what could have been parallel work and reducing the GPU's throughput for this warp.

- The threads that are temporarily disabled during each phase are said to be "masked" by the hardware, which means they go through the motions of execution but their results are not written to memory and they don't affect any state, ensuring correctness while the warp works through each divergent path.

- Once both branches have completed their execution, all 32 threads in the warp "reconverge" at the instruction following the if-else block, and from that point forward they can again execute in lockstep until they encounter another divergent branch.

- To illustrate the performance impact with a concrete example: if the `if` branch requires 10 clock cycles to complete and the `else` branch also requires 10 clock cycles, then a divergent warp must spend 20 total cycles to complete the entire if-else block, whereas a non-divergent warp where all threads take the same path would only need 10 cycles - this is essentially a 2x slowdown for this portion of the code.

---

## 3. How does occupancy relate to latency hiding in GPUs?

### What is Occupancy:

- Occupancy is a metric that measures how well you are utilizing the GPU's computational resources, and it is calculated as the ratio of the number of active warps currently running on a Streaming Multiprocessor (SM) compared to the maximum number of warps that SM is capable of handling simultaneously - for example, if an SM can handle 32 warps maximum and you have 16 warps active, your occupancy is 50%.

- The Tesla T4 GPU used in the experiment has 40 Streaming Multiprocessors (SMs), and each of these SMs can manage multiple warps concurrently, which means the total computational capacity of the GPU depends on how effectively you can keep all these SMs busy with enough warps to process.

### What is Latency Hiding:

- GPU global memory access is extremely slow compared to computation speed - fetching data from the GPU's main memory can take anywhere from 200 to 800 clock cycles, while a simple arithmetic operation might only take a few cycles, which creates a massive imbalance between computation speed and data access speed.

- The GPU's solution to this problem is to switch execution to a different warp while one warp is waiting for its memory request to complete - so instead of the entire SM sitting idle for hundreds of cycles waiting for data, it immediately begins executing instructions from another warp that already has its data ready, keeping the hardware busy and productive.

- This technique is called latency hiding because from the perspective of overall throughput, the memory access latency is effectively hidden behind useful computation work - the individual warp still waits for its data, but the SM as a whole continues making progress on other work during that wait time.

### How They Connect:

- Higher occupancy directly enables better latency hiding because having more active warps on each SM means the scheduler has more options to choose from when it needs to switch away from a stalled warp - essentially, more warps equals more opportunities to find one that is ready to execute.

- Consider a scenario where you only have 2 warps active on an SM and both warps issue memory requests at around the same time - in this case, the SM has no other work to do and must sit completely idle for hundreds of cycles until at least one memory request completes, wasting valuable computational resources.

- Now consider the opposite scenario where you have 16 warps active on the same SM and 2 of them are waiting for memory - the scheduler can immediately switch to any of the remaining 14 warps that are ready to execute, keeping the SM fully utilized and productive while those memory requests complete in the background.

- In the experiment, all block size configurations produced the same total number of warps (32,768 warps) because the array size and warp size remained constant, but the way those warps were organized into blocks significantly affected how well the GPU could schedule them and achieve good occupancy on each SM.

- The occupancy analysis from the experiment demonstrated these different organizational structures clearly:
  - Block size 32: Each block contained only 1 warp, resulting in 32,768 separate blocks that needed to be scheduled
  - Block size 256: Each block contained 8 warps, resulting in 4,096 blocks with more warps grouped together
  - Block size 1024: Each block contained 32 warps, resulting in only 1,024 larger blocks

- Having more warps per block generally helps each SM reach higher occupancy levels because when a block is assigned to an SM, all of its warps become available for scheduling on that SM - this improves latency hiding by giving the scheduler more warps to choose from when it needs to switch away from a waiting warp, though there are diminishing returns and other resource constraints that limit how large blocks can effectively be.

---

## 4. From your experiment, how did thread block size affect performance?

### Experimental Results:

- The experiment systematically tested six different block sizes to understand how this configuration parameter affects kernel performance: 32, 64, 128, 256, 512, and 1024 threads per block, covering the full range from the minimum practical size (one warp) to the maximum allowed by the Tesla T4 hardware.

- The best performance was achieved with block sizes of 128 and 256 threads, which hit the sweet spot for this particular kernel and GPU combination:
  - Block size 128: Achieved 0.0963 ms for divergent execution and 0.0972 ms for non-divergent execution, representing the fastest overall times in the experiment
  - Block size 256: Achieved very similar performance with 0.0972 ms for divergent and 0.0976 ms for non-divergent execution

- The worst performance occurred at the extreme ends of the block size spectrum, demonstrating that both too-small and too-large blocks cause problems:
  - Block size 32: Required 0.1445 ms for divergent execution, which is approximately 50% slower than the optimal block sizes and represents significant wasted potential
  - Block size 1024: Required 0.1329 ms for divergent execution, which is approximately 38% slower than optimal, showing that maximum block size is not always the best choice

### Why Small Blocks (32) Performed Poorly:

- With only 1 warp per block (since 32 threads equals exactly one warp), the GPU scheduler has very limited flexibility in managing work on each SM because each block brings only a single warp to work with, which limits the options for hiding latency when that warp stalls.

- Having 32,768 separate blocks (which is what you get when dividing 1,048,576 elements by 32 threads per block) means the GPU has to deal with significant overhead in launching, scheduling, and managing all these individual blocks, and this bookkeeping work takes time away from actual computation.

- Each SM has a limit on how many blocks it can have active at once (typically 16-32 blocks depending on the GPU), and with only 1 warp per block, the SM might not be able to achieve good occupancy even if it has many blocks scheduled because of these block-count limits - you could have 16 blocks but only 16 warps, when the SM might be capable of handling 32 or more warps.

### Why Large Blocks (1024) Performed Poorly:

- Each large block of 1024 threads consumes substantial GPU resources including registers for all those threads and potentially shared memory, and these resource requirements can limit how many blocks can be active on each SM simultaneously - if each block needs too many resources, the SM can only fit one or two blocks at a time.

- With only 1,024 total blocks (calculated as 1,048,576 elements divided by 1024 threads per block), there is less work available to distribute evenly across all 40 SMs on the Tesla T4, which can lead to load imbalance where some SMs finish their assigned blocks and sit idle while others are still working.

- The combination of high per-block resource usage and limited block count can result in some SMs being overloaded with work while others have nothing to do, and this imbalance prevents the GPU from achieving its full potential throughput.

### Why Medium Blocks (128-256) Were Optimal:

- Block sizes of 128 and 256 threads provide an excellent balance between having enough warps per block (4 warps for 128 threads, 8 warps for 256 threads) to achieve good occupancy and latency hiding, while also creating enough total blocks (8,192 blocks for size 128, 4,096 blocks for size 256) to distribute work evenly across all SMs.

- With 4,096 to 8,192 blocks available, the GPU has plenty of work to assign to all 40 SMs, ensuring that no SM sits idle and that the load is well balanced - each SM gets roughly 100-200 blocks to process, which keeps them all busy throughout the kernel execution.

- The moderate number of warps per block (4-8) is enough to give the scheduler good options for hiding memory latency through warp switching, while not requiring so many resources per block that occupancy becomes limited by resource constraints.

- These block sizes represent the sweet spot specifically for this threshold kernel running on the Tesla T4 GPU - the optimal block size can vary for different kernels with different resource requirements, or for different GPU architectures with different capabilities, so it's always worth experimenting with block size when optimizing CUDA code.

---

## 5. Under what conditions is divergence most harmful? Least harmful?

### Most Harmful Conditions:

- **When the two branches have very different amounts of work to do**: If the `if` branch performs 100 arithmetic operations while the `else` branch only performs 5 operations, the threads that take the short `else` path must sit idle for 95 extra operations while waiting for the `if` threads to finish, which means most of the GPU's computational capacity is wasted during that time - the more imbalanced the branches, the worse the performance impact.

- **When divergence occurs inside tight inner loops that execute many times**: A divergent branch inside a loop that iterates 1000 times means the serialization penalty is paid 1000 separate times throughout the kernel execution, and this repeated cost adds up to a massive performance hit - for example, if each divergence costs 10 extra cycles, that's 10,000 wasted cycles per warp just from this one loop.

- **When the kernel is compute-bound rather than memory-bound**: If your kernel spends most of its time doing arithmetic calculations and rarely waits for memory, then every wasted cycle from divergence directly translates to reduced throughput because the GPU has no idle time to absorb the serialization cost - in contrast to memory-bound kernels where the GPU would have been waiting anyway.

- **When divergence splits the warp exactly 50/50 between the two branches**: When half the threads (16 out of 32) take the `if` path and the other half take the `else` path, both branches must execute completely and neither can be skipped - this is the worst-case scenario where you get zero benefit from any threads taking the same path, resulting in maximum serialization overhead.

- **When using small block sizes that limit scheduling flexibility**: The experiment showed that block size 32 had a higher divergence penalty of +2.48% compared to larger block sizes, because with fewer warps per block the scheduler has limited ability to hide the divergence cost by switching to other warps while the divergent warps work through their serialized branches.

### Least Harmful Conditions:

- **When the kernel is memory-bound and threads spend most of their time waiting for data**: If threads are already spending hundreds of cycles waiting for memory requests to complete, then the extra serialization cycles from divergence are effectively free because the warp would have been idle anyway - the divergence cost gets hidden behind the memory latency, similar to how latency hiding works for regular memory stalls.

- **When both branches perform similar amounts of work**: If the `if` branch takes 10 cycles and the `else` branch also takes 10 cycles, then it doesn't matter as much which path executes first because the total time (20 cycles) is the same regardless of how threads are distributed between branches - the overhead is still there, but at least no threads are waiting excessively for others to finish a much longer path.

- **When divergence occurs only once outside of any loops**: A single divergent if-else statement at the end of a kernel that only executes once has minimal performance impact compared to divergence inside a loop - if you only pay the serialization cost one time versus thousands of times, the overall effect on kernel runtime is negligible.

- **When the divergence pattern naturally aligns with warp boundaries**: If your data is organized such that threads 0-31 (warp 0) all take one path while threads 32-63 (warp 1) all take the other path, then there is no divergence within any warp because each warp's threads unanimously agree on which branch to take - both paths still execute, but in different warps running in parallel rather than the same warp running serially.

- **When using medium to larger block sizes that achieve good occupancy**: The experiment showed that block sizes 128 and 256 had near-zero or even slightly negative divergence penalties (-0.94% and -0.41% respectively), which indicates that with sufficient warps per block and good occupancy, the GPU scheduler is able to effectively hide the divergence cost by keeping the SM busy with other work.