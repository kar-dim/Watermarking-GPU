__kernel void calculate_neighbors_p3(
    __read_only image2d_t image, 
    __global float* x_,
    __local float shared_neighbors[8][16][16]) //all neighbor values per block)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int x = get_group_id(1) * get_local_size(1) + ty;
    const int y = get_group_id(0) * get_local_size(0) + tx;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    const int width = get_image_width(image), height = get_image_height(image);

    if (x < width && y < height) {
        // Load 8 neighboring pixels into shared memory
        shared_neighbors[0][ty][tx] = read_imagef(image, sampler, (int2)(x - 1, y - 1)).x;
        shared_neighbors[1][ty][tx] = read_imagef(image, sampler, (int2)(x - 1, y)).x;
        shared_neighbors[2][ty][tx] = read_imagef(image, sampler, (int2)(x - 1, y + 1)).x;
        shared_neighbors[3][ty][tx] = read_imagef(image, sampler, (int2)(x, y - 1)).x;
        shared_neighbors[4][ty][tx] = read_imagef(image, sampler, (int2)(x, y + 1)).x;
        shared_neighbors[5][ty][tx] = read_imagef(image, sampler, (int2)(x + 1, y - 1)).x;
        shared_neighbors[6][ty][tx] = read_imagef(image, sampler, (int2)(x + 1, y)).x;
        shared_neighbors[7][ty][tx] = read_imagef(image, sampler, (int2)(x + 1, y + 1)).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Write 8 neighbors to global memory in a coalesced way
    if (x < width && y < height) {
        const int output_index = (x * height + y);
        //Each thread writes one value from each of the 8 neighbors (to avoid strides and uncoalesced global access)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            x_[i * width * height + output_index] = shared_neighbors[i][ty][tx];
        }
    }
}