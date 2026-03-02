# TODO

## Medium priority

- **Warp stability on low-texture scenes** — With `freeze_mode: off`, the warp matrix
  goes erratic when looking at walls / low-texture surfaces (IR dot pattern dominates,
  ORB matches are garbage). Ideas:
  - Better feature matching — filter out IR projector dots before ORB, or use a different
    detector (AKAZE, SuperPoint) that's less confused by dot patterns
  - Reject bad frames — threshold on inlier count or reprojection error, keep previous
    warp when the current match is clearly wrong
  - Constrain degrees of freedom — lock rotation and crop to the frozen values, only allow
    translation + scale to update. The camera-to-camera geometry is rigid so rotation and
    crop should never change; only depth-dependent parallax (translation/scale) varies

## Low priority

- **GUI application** — Add a GUI (ImGui or similar) to view streams, launch/stop the
  fusion node, and change settings (topic paths, sync slop, ORB params, freeze mode, etc.)
  at runtime. Essentially turn the fusion node + viewer into a single integrated application.
