__kernel void letterbox(__global uchar* input, __global uchar* output) {
    // Simplified: resize to imgsz, pad, normalize
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    // Implement letterbox logic (resize, padding)
    output[y * get_global_size(0) * 3 + x * 3 + c] = input[y * get_global_size(0) * 3 + x * 3 + c];
}