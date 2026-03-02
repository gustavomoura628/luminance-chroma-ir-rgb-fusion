# TODO

## Medium priority

- **Warp stability on low-texture scenes** — ~~With `freeze_mode: off`, the warp matrix
  goes erratic when looking at walls / low-texture surfaces (IR dot pattern dominates,
  ORB matches are garbage).~~ Partially addressed:
  - Better feature matching — filter out IR projector dots before ORB, or use a different
    detector (AKAZE, SuperPoint) that's less confused by dot patterns
  - ~~Reject bad frames~~ — done: `min_inliers` / `min_inlier_ratio` quality gate
  - ~~Constrain degrees of freedom~~ — done: `lock_rotation` parameter

- **Chroma flicker during rotation** — When the robot rotates clockwise, the chroma
  channel jumps to the right. Likely caused by stale RGB frames: the IR updates at
  full rate but the RGB used for chroma is from a previous frame, so the color overlay
  lags behind and appears to shift in the direction of rotation.

## Low priority

- **GUI application** — Add a GUI (ImGui or similar) to view streams, launch/stop the
  fusion node, and change settings (topic paths, sync slop, ORB params, freeze mode, etc.)
  at runtime. Essentially turn the fusion node + viewer into a single integrated application.
