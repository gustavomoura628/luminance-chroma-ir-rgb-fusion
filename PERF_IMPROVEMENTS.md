# Performance Improvements

## Prototype (stereo-gray-plus-rgb)

Baseline: parallel stereo (both eyes), 640x480/eye, async warp, rosbag source, cv2.imshow display.

### Results

| # | Change | L total | R total | Avg | Delta | Status |
|---|--------|---------|---------|-----|-------|--------|
| 0 | Baseline (Lab color space) | 3.3ms | 3.1ms | 6.5ms | — | replaced |
| 1 | YCrCb instead of Lab | 2.3ms | 2.0ms | 4.2ms | -2.3ms | carried forward |
| 2 | Parallel eyes (Thread per frame) | 2.5ms | 3.3ms | 4.6ms | +0.4ms | reverted — thread spawn overhead + cache contention |
| 3 | Parallel eyes (ThreadPoolExecutor) | 2.1ms | 2.8ms | 3.8ms | -0.4ms | replaced by GPU |
| 4 | GPU warp+colorize (torch grid_sample) | — | — | 3.4ms | -0.4ms | carried forward |
| 5 | Float16 color math (post grid_sample) | — | — | 3.5ms | ~0ms | no gain — color math is tiny vs data transfer |
| 6 | Cache warp grid (skip rebuild if M unchanged) | — | — | 3.5ms | ~0ms | no gain — grid build is fast, transfer dominates |

### Breakdown (prototype, frozen warp)

```
warp:      0.04ms   (skip — just returns cached M)
upload:    0.64ms
colorize:  2.10ms   (no ORB thread competing for GPU/CPU)
download:  0.84ms
total:     3.63ms
```

Key finding: bottleneck is CPU↔GPU data transfer (PCIe), not compute. At 640x480 the image
is too small for GPU compute to overcome the transfer overhead.

---

## ROS2 package (luminance-chroma-ir-rgb-fusion)

Baseline: ROS2 node, stereo (L+R eyes), 640x480/eye, frozen warp, rosbag source, publish rgb8.

### Results

| # | Change | Typical frame | Delta | Status |
|---|--------|---------------|-------|--------|
| 0 | Baseline (ROS2 node) | 5.5-6.5ms | — | replaced |
| 1 | Pre-alloc publish buffers (single memcpy) | 5.5-6.2ms | ~-0.4ms | replaced |
| 2 | Reuse Image msg object + split pub timing | 5.3-5.9ms | ~-0.2ms | replaced |
| 3 | torch.set_num_threads(1) | 5.3-5.9ms | ~0ms | reverted — no effect on latency or CPU usage |
| 4 | Shared RGB upload (node uploads once, both eyes reuse) | 4.5-5.1ms | ~-0.8ms | replaced |
| 5 | Fuse color math into single matmul (3x3 @ warped) | 3.9-4.3ms | ~-0.4ms | replaced |
| 6 | Pre-alloc pinned download + in-place clamp | 3.7-4.1ms | ~-0.2ms | **active** |
| 7 | Pinned RGB upload buffer | 15-35ms | regression | reverted — copy_ from non-contiguous (permuted) pinned tensor is catastrophically slow |

### Breakdown rev 0 — baseline (frozen warp)

```
decode:    0.0ms    (frombuffer + reshape, essentially free)
resize:    0.0ms    (no-op when IR and RGB are same resolution)

L pipeline: 2.0-2.5ms
  submit:  0.0ms    (frozen, just returns)
  crop:    0.1ms    (compute_crop or skip)
  to_gpu:  0.6ms    (from_numpy → .to(cuda) → permute → float — allocates every frame)
  colorize: 0.7-0.9ms  (grid_sample + YCrCb color math, queued)
  to_cpu:  0.5ms    (clamp → byte → permute → contiguous → .cpu() → .numpy(), GPU sync)

R pipeline: 1.5-1.7ms
  (same stages, faster — GPU warm, L's CUDA allocs cached)

pub_L:     1.0-1.5ms   (tobytes + array.array + rclpy publish)
pub_R:     0.8-1.3ms

total:     5.5-6.5ms typical, 8-10ms spikes
```

### Breakdown rev 2 — pre-alloc buffers + reuse msg (frozen warp)

```
decode:    0.0ms
resize:    0.0ms

L pipeline: 1.9-2.3ms   (unchanged)
R pipeline: 1.6-1.7ms   (unchanged)

msg_L:     0.2ms        (np.copyto into pre-alloc buf + header assign, reused Image obj)
dds_L:     0.7-0.8ms    (rclpy CDR serialize + DDS middleware write)
msg_R:     0.2ms
dds_R:     0.5-0.6ms

total:     5.3-5.9ms typical
```

Publish split reveals: msg construction now ~0.4ms combined (down from ~1.0ms baseline).
DDS calls are ~1.3ms combined — rclpy C-level serialize + middleware write, irreducible
from Python without zero-copy transport (Cyclone DDS + iceoryx) or a C extension.

Spikes (8-21ms) are correlated across ALL stages simultaneously — system-level contention
(GC, scheduler, other processes), not a single bottleneck.

### Breakdown rev 4 — shared RGB upload (frozen warp)

```
decode:    0.0ms
resize:    0.0ms
rgb_up:    0.6-0.7ms   (single upload shared by both eyes)

L pipeline: 1.5ms
  submit:  0.0ms
  crop:    0.1ms
  to_gpu:  0.0ms       (reuses shared rgb_t)
  colorize: 0.8ms
  to_cpu:  0.5ms

R pipeline: 1.2-1.3ms
  to_gpu:  0.0ms       (reuses shared rgb_t)
  colorize: 0.6ms
  to_cpu:  0.6ms

msg_L:     0.2ms
dds_L:     0.4ms
msg_R:     0.2ms
dds_R:     0.3ms

total:     4.5-5.1ms typical
```

### Breakdown rev 5 — fused matmul color math (frozen warp)

```
decode:    0.0ms
resize:    0.0ms
rgb_up:    0.6-0.7ms

L pipeline: 1.3ms
  submit:  0.0ms
  crop:    0.1ms
  to_gpu:  0.0ms
  colorize: 0.6ms      (down from 0.8 — matmul replaces ~13 scalar ops)
  to_cpu:  0.5ms

R pipeline: 1.0ms
  colorize: 0.4ms      (down from 0.6)
  to_cpu:  0.5ms

msg_L:     0.2ms
dds_L:     0.4ms
msg_R:     0.2ms
dds_R:     0.3ms

total:     3.9-4.3ms typical, best 3.9ms
```

### Breakdown rev 6 — pre-alloc pinned download (frozen warp)

```
decode:    0.0ms
resize:    0.0ms
rgb_up:    0.7ms

L pipeline: 1.1-1.2ms
  submit:  0.0ms
  crop:    0.1ms
  to_gpu:  0.0ms
  colorize: 1.0-1.1ms  (now includes download via pinned buffers)

R pipeline: 0.8-0.9ms
  colorize: 0.8ms

msg_L:     0.2ms
dds_L:     0.4ms
msg_R:     0.2ms
dds_R:     0.3ms

total:     3.7-4.1ms typical, best 3.7ms
```

### Top bottlenecks

1. **rgb_up (~0.7ms)** — per-frame allocation. Pinned pre-allocated buffer would help,
   BUT pinned + permute is catastrophically slow (rev 7). Would need contiguous NCHW
   pinned buffer with a manual transpose, or skip the CPU entirely.
2. **DDS publish (~1.1ms combined)** — rclpy C-level floor. Needs zero-copy transport or
   C extension to improve.
3. **colorize (~1.8ms combined)** — mostly transfer time now, compute is minimal.

### Python performance ceiling

At ~3.7-4.1ms we've hit the Python floor. All remaining time is spent at C boundaries:
PCIe data transfer (rgb upload, IR upload, result download) and DDS serialization.
Python itself adds negligible overhead — the bottleneck is inability to do zero-copy
operations and loaned DDS messages from Python.

**To go further, rewrite as a C++ ROS2 node with:**
- CUDA interop (keep tensors on GPU, no numpy↔torch shuffling)
- Loaned messages / zero-copy DDS (Cyclone DDS + iceoryx shared memory)
- Direct GPU↔DDS path if subscriber supports it
- CUDA streams to overlap upload/compute/download

This is a different project, not an incremental optimization.

---

## C++ CUDA rewrite (ir_rgb_fusion_cuda)

Baseline: C++ ROS2 node, fused CUDA kernel, no PyTorch/LibTorch, stereo 640x480/eye, frozen warp, rosbag source.

### What changed

Rewrote the entire pipeline as a C++ node (`cpp/` directory) with:
- **Fused CUDA kernel** — single kernel does inverse affine warp, bilinear RGB sampling, 3x3 color matrix multiply, IR luma addition, clamp, and uint8 output. Replaces the PyTorch chain of `grid_sample` → `mm` → `clamp_` → `copy_` → `permute` → `copy_`.
- **No PyTorch/LibTorch** — raw CUDA API, zero allocator overhead.
- **Two CUDA streams** (L/R) with `cudaEventRecord`/`cudaStreamWaitEvent` for shared RGB upload — both eyes overlap on GPU.
- **Pinned host buffers** for all transfers (`cudaMallocHost`), DMA download via `cudaMemcpyAsync`.
- **2D async memcpy** for IR crop — uploads only the ROI, handles non-contiguous stride.
- **Direct msg data access** — no cv_bridge, no intermediate copies. `sensor_msgs::msg::Image::data.data()` pointers used directly.
- **Pre-allocated output messages** — `Image` objects reused across frames.

### Results

| # | Change | Typical frame | Delta | Status |
|---|--------|---------------|-------|--------|
| 8 | C++ CUDA rewrite | 1.4-1.8ms | **-2.3ms** | carried forward |
| 9 | Event-driven 3-topic sync (drop ApproximateTime + message_filters) | 2.9-3.0ms | +1.3ms proc, but fixes RGB freshness & jitter | **active** |

### Breakdown rev 8 — C++ CUDA (frozen warp)

```
decode:       0.0ms     (zero-copy cv::Mat over msg data)
rgb_up:       0.2ms     (cudaMemcpyAsync to pre-alloc device buf + event record)
launch:       0.4-0.6ms (submit_warp + launch_async L + launch_async R)
  IR upload:  ~0.1ms    (cudaMemcpy2DAsync, ROI only)
  kernel:     ~0.05ms   (fused warp+color+luma, 16x16 blocks)
  download:   ~0.1ms    (cudaMemcpyAsync to pinned host)
L_sync+pub:   0.4-0.5ms (cudaStreamSynchronize + memcpy + DDS publish)
R_sync+pub:   0.3-0.4ms

total:        1.4-1.8ms typical, best 1.2ms, spikes to 2.4ms
```

### What the rewrite eliminated

| Overhead | Python cost | C++ cost |
|----------|------------|----------|
| PyTorch allocator (per-frame temp tensors) | ~0.3ms | 0 (pre-alloc) |
| `from_numpy` → `.to(cuda)` → `.permute` → `.float()` | ~0.6ms | 0 (direct upload) |
| `grid_sample` + `mm` + `clamp_` + cast + `permute` + download | ~1.0ms | ~0.15ms (single kernel) |
| Python GIL / interpreter overhead | ~0.1ms | 0 |
| rclpy CDR serialize vs rclcpp | ~0.3ms | ~0.2ms |

### Inter-frame jitter investigation

The C++ rewrite processes frames in 1.4-1.8ms, but subscribers see **150-340ms peak spikes**
in inter-frame timing. Investigation with rolling 2s stats on both the fusion node and a
dedicated SDL2/ImGui stream viewer confirmed:

**Fusion node stats (publisher side):**
```
30 fps | interval avg 33.3ms peak 301.6ms | proc avg 1.6ms peak 2.4ms
27 fps | interval avg 37.3ms peak 343.0ms | proc avg 1.6ms peak 2.6ms
```

- Processing time is rock solid (1.5-1.7ms avg, 2.8ms worst)
- Inter-callback interval peaks at 280-343ms — the sync callback itself fires late

**Viewer stats (subscriber side):**
```
IR       30 fps   33.3ms  peak  50.9ms
Fused    27 fps   36.5ms  peak 299.3ms
RGB      24 fps   41.8ms  peak 100.5ms
```

- IR (direct from RealSense driver): stable 30fps, peaks ~50ms
- Fused: mirrors the fusion node's interval jitter exactly
- RGB (direct from driver): 24-28fps with ~100ms peaks

**Root cause: `message_filters::ApproximateTime` synchronizer.** The fusion callback only
fires when a matching triplet (infra1 + infra2 + color) arrives within the sync slop window
(20ms default). The RGB camera arrives on a separate USB pipeline with independent timing.
When one RGB frame is delayed (USB scheduling, driver buffering), the synchronizer holds
all three streams until a match is found — causing a ~300ms gap followed by a burst of
catch-up frames.

This is upstream of the fusion node — no amount of kernel optimization can fix it.

**Resolution:** Rev 9 replaced ApproximateTime with event-driven 3-topic sync,
eliminating jitter (peak ~50ms, down from 280-343ms) while maintaining 95-100%
RGB freshness.

### Rev 9 — event-driven 3-topic sync

Replaced `message_filters::ApproximateTime` IR1+IR2 sync + decoupled RGB with CV wait
with a fully event-driven approach: three plain subscriptions on a single
`MutuallyExclusive` callback group, no mutex, no blocking waits.

**How it works:**
- All three callbacks (IR1, IR2, RGB) store their latest message and call `try_fuse()`
- `try_fuse()` checks if IR1 and RGB timestamps are within 5ms — if so, fuses immediately
- Whichever topic arrives LAST triggers fusion (no thread coordination needed)
- 5ms one-shot timer fires as fallback if RGB never matches (fuses with stale)

**Why it works:** The bag's dominant message order is `IR2 → IR1 → RGB` (77%).
With sequential callback processing, RGB callback fires ~1ms after IR1 and
triggers fusion. For the 18% where RGB arrives before IR1, the IR1 callback
finds fresh RGB immediately. This mirrors the prototype's peek-ahead logic
but handles all arrival permutations (the prototype misses `IR1 → IR2 → RGB`).

**Results:**
```
30 fps | interval avg 33.3ms peak ~50ms | proc avg 2.9-3.0ms peak 5-6ms
rgb age avg 0.0ms peak 0.0ms fresh 100%  (typical)
rgb age avg 1.7ms peak 33.7ms fresh 95%  (worst 2s window during high motion)
```

**Comparison across sync strategies:**

| Strategy | Interval peak | Proc avg | RGB fresh | Notes |
|----------|--------------|----------|-----------|-------|
| 3-way ApproximateTime (rev 8) | 280-343ms | 1.6ms | 100% | Jitter from sync holding frames |
| 2-way IR sync + CV wait 5ms | ~48ms | 3.9ms | 47-83% | Jitter fixed but RGB unreliable |
| Event-driven (rev 9) | ~50ms | 2.9ms | 95-100% | Best of both worlds |

Proc time is higher than rev 8's 1.6ms because rev 8 measured only the fusion
pipeline (sync callback), while rev 9 includes the event-driven overhead of
checking freshness after each callback. The actual GPU work is identical.

**Removed dependencies:** `message_filters` (CMakeLists.txt, package.xml).

---

## Ideas backlog

- **Zero-copy DDS** (Cyclone DDS + iceoryx shared memory) — eliminate the final memcpy + serialize
- **Keep RGB on GPU** if source can deliver it there directly (e.g. nvdec)
- **CUDA graph capture** — record the kernel launch pattern once, replay with near-zero CPU overhead
- ~~**Decouple RGB sync**~~ — done (rev 9, event-driven 3-topic sync)
