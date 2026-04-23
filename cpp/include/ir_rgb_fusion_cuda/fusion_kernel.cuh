#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/// Launch the fused warp + color-matrix + luma kernel.
///
/// Each output pixel at (x_out, y_out) in the ROI:
///   1. Applies inverse affine to get source coords in RGB image
///   2. Bilinear-samples RGB (HWC uint8, zeros padding)
///   3. Multiplies by M_color (3x3) and adds IR luma
///   4. Clamps [0,255] and writes uint8 output (HWC)
///
/// @param d_rgb      RGB input, HWC uint8, shape (rgb_h, rgb_w, 3)
/// @param rgb_h      RGB image height
/// @param rgb_w      RGB image width
/// @param d_ir       IR ROI input, row-major uint8, shape (roi_h, roi_w)
/// @param ir_stride  byte stride between IR rows (for 2D memcpy crops)
/// @param d_M_inv    inverse affine matrix, 6 floats [a b tx; c d ty]
/// @param d_M_color  color matrix, 9 floats row-major (3x3)
/// @param d_output   output HWC uint8, shape (roi_h, roi_w, 3)
/// @param roi_h      ROI height
/// @param roi_w      ROI width
/// @param roi_x      ROI x offset in IR frame
/// @param roi_y      ROI y offset in IR frame
/// @param stream     CUDA stream
void launch_fuse_kernel(
    const uint8_t* __restrict__ d_rgb,
    int rgb_h, int rgb_w,
    const uint8_t* __restrict__ d_ir,
    int ir_stride,
    const float* __restrict__ d_M_inv,
    const float* __restrict__ d_M_color,
    uint8_t* __restrict__ d_output,
    int roi_h, int roi_w,
    int roi_x, int roi_y,
    cudaStream_t stream);
