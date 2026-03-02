#include "ir_rgb_fusion_cuda/fusion_kernel.cuh"

/// Fused kernel: inverse-affine warp + bilinear RGB sample + color matrix + IR luma.
///
/// Grid covers the output ROI. Each thread produces one output pixel.
/// The inverse affine maps output (roi) coords back to RGB source coords,
/// matching the Python pipeline's grid_sample(align_corners=True, zeros padding).
__global__ void fuse_kernel(
    const uint8_t* __restrict__ rgb,
    int rgb_h, int rgb_w,
    const uint8_t* __restrict__ ir,
    int ir_stride,
    const float* __restrict__ M_inv,   // [a b tx; c d ty] row-major, 6 floats
    const float* __restrict__ M_color, // 3x3 row-major, 9 floats
    uint8_t* __restrict__ output,
    int roi_h, int roi_w,
    int roi_x, int roi_y)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= roi_w || oy >= roi_h) return;

    // Output pixel position in full IR frame coordinates
    const float px = static_cast<float>(ox + roi_x);
    const float py = static_cast<float>(oy + roi_y);

    // Inverse affine → source position in RGB image
    const float sx = M_inv[0] * px + M_inv[1] * py + M_inv[2];
    const float sy = M_inv[3] * px + M_inv[4] * py + M_inv[5];

    // Bilinear sample RGB with zeros padding (matches grid_sample align_corners=True)
    float r = 0.f, g = 0.f, b = 0.f;

    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);

    // Only sample if any corner is in bounds (zeros padding for OOB)
    if (x0 >= -1 && x1 <= rgb_w && y0 >= -1 && y1 <= rgb_h) {
        float w00 = (1.f - fx) * (1.f - fy);
        float w01 = fx * (1.f - fy);
        float w10 = (1.f - fx) * fy;
        float w11 = fx * fy;

        // Top-left
        if (x0 >= 0 && y0 >= 0) {
            int idx = (y0 * rgb_w + x0) * 3;
            r += w00 * rgb[idx + 0];
            g += w00 * rgb[idx + 1];
            b += w00 * rgb[idx + 2];
        }
        // Top-right
        if (x1 < rgb_w && y0 >= 0) {
            int idx = (y0 * rgb_w + x1) * 3;
            r += w01 * rgb[idx + 0];
            g += w01 * rgb[idx + 1];
            b += w01 * rgb[idx + 2];
        }
        // Bottom-left
        if (x0 >= 0 && y1 < rgb_h) {
            int idx = (y1 * rgb_w + x0) * 3;
            r += w10 * rgb[idx + 0];
            g += w10 * rgb[idx + 1];
            b += w10 * rgb[idx + 2];
        }
        // Bottom-right
        if (x1 < rgb_w && y1 < rgb_h) {
            int idx = (y1 * rgb_w + x1) * 3;
            r += w11 * rgb[idx + 0];
            g += w11 * rgb[idx + 1];
            b += w11 * rgb[idx + 2];
        }
    }

    // Color matrix: M_color(3x3) @ (r,g,b) + IR luma
    const float ir_val = static_cast<float>(ir[oy * ir_stride + ox]);
    float out_r = M_color[0] * r + M_color[1] * g + M_color[2] * b + ir_val;
    float out_g = M_color[3] * r + M_color[4] * g + M_color[5] * b + ir_val;
    float out_b = M_color[6] * r + M_color[7] * g + M_color[8] * b + ir_val;

    // Clamp and write HWC uint8 output
    const int out_idx = (oy * roi_w + ox) * 3;
    output[out_idx + 0] = static_cast<uint8_t>(fminf(fmaxf(out_r, 0.f), 255.f));
    output[out_idx + 1] = static_cast<uint8_t>(fminf(fmaxf(out_g, 0.f), 255.f));
    output[out_idx + 2] = static_cast<uint8_t>(fminf(fmaxf(out_b, 0.f), 255.f));
}

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
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((roi_w + block.x - 1) / block.x,
              (roi_h + block.y - 1) / block.y);

    fuse_kernel<<<grid, block, 0, stream>>>(
        d_rgb, rgb_h, rgb_w,
        d_ir, ir_stride,
        d_M_inv, d_M_color,
        d_output, roi_h, roi_w,
        roi_x, roi_y);
}
