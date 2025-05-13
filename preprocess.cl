__kernel void letterbox(
    __global const uchar* input_image,
    __global uchar* output_image,
    const int input_w,
    const int input_h,
    const int target_size) {

    // For simplicity, this kernel will fill one pixel of the output per work item.
    // Output is (target_size, target_size, 3 channels)
    // Global work size should be (target_size, target_size, 1) or (target_size, target_size)
    // We will handle channels internally for each (x,y) pixel.

    const int out_x = get_global_id(0); // 0 to target_size - 1
    const int out_y = get_global_id(1); // 0 to target_size - 1

    if (out_x >= target_size || out_y >= target_size) {
        return; // Out of bounds for output
    }

    // Calculate scale factor to fit input_image into target_size x target_size
    // while maintaining aspect ratio.
    float scale_w = (float)target_size / input_w;
    float scale_h = (float)target_size / input_h;
    float scale = min(scale_w, scale_h);

    // Calculate dimensions of scaled input image
    int scaled_input_w = (int)(input_w * scale);
    int scaled_input_h = (int)(input_h * scale);

    // Calculate padding
    // (Padding is added equally to both sides)
    int pad_x = (target_size - scaled_input_w) / 2;
    int pad_y = (target_size - scaled_input_h) / 2;

    // Corresponding coordinates in the original input image
    // (Relative to the top-left of the scaled image within the padded output)
    float in_x_float = (out_x - pad_x) / scale;
    float in_y_float = (out_y - pad_y) / scale;

    // Pad color (black)
    uchar pad_r = 0;
    uchar pad_g = 0;
    uchar pad_b = 0;

    // Check if current output pixel (out_x, out_y) is in the padding area
    if (out_x < pad_x || out_x >= (pad_x + scaled_input_w) ||
        out_y < pad_y || out_y >= (pad_y + scaled_input_h)) {
        // This pixel is in the padding area
        output_image[(out_y * target_size + out_x) * 3 + 0] = pad_r; // R
        output_image[(out_y * target_size + out_x) * 3 + 1] = pad_g; // G
        output_image[(out_y * target_size + out_x) * 3 + 2] = pad_b; // B
    } else {
        // This pixel is part of the scaled input image.
        // Perform bilinear interpolation.

        // Top-left integer coordinates in the input image
        int x1 = (int)floor(in_x_float);
        int y1 = (int)floor(in_y_float);

        // Handle boundary conditions for interpolation
        x1 = max(0, min(x1, input_w - 2));
        y1 = max(0, min(y1, input_h - 2));
        
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        // Fractional parts for interpolation weights
        float wx2 = in_x_float - x1;
        float wy2 = in_y_float - y1;
        float wx1 = 1.0f - wx2;
        float wy1 = 1.0f - wy2;

        for (int c = 0; c < 3; ++c) { // Iterate over R, G, B channels
            // Values of the four neighboring pixels in the input image
            float p11 = input_image[(y1 * input_w + x1) * 3 + c];
            float p12 = input_image[(y2 * input_w + x1) * 3 + c];
            float p21 = input_image[(y1 * input_w + x2) * 3 + c];
            float p22 = input_image[(y2 * input_w + x2) * 3 + c];

            // Bilinear interpolation formula
            float interpolated_value = wx1 * wy1 * p11 +
                                       wx1 * wy2 * p12 +
                                       wx2 * wy1 * p21 +
                                       wx2 * wy2 * p22;

            output_image[(out_y * target_size + out_x) * 3 + c] = (uchar)round(interpolated_value);
        }
    }
}