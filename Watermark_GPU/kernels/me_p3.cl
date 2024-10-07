__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __constant int* RxMappings,
    __local float RxLocal[64][36]) //64 local threads, 36 values each (8 for rx, this is a shared memory for both Rx,rx)
 
{
    const int x = get_global_id(0), y = get_global_id(1);
    const int width = get_image_height(image); //image2d is transposed, so we read the opposite dimensions
    const int paddedWidth = get_global_size(0);
    const int localId = get_local_id(0);
    const int outputIndex = (y * paddedWidth) + x;

    //initialize shared memory, assign a portion for all threads for parallelism
    #pragma unroll
    for (int i = 0; i < 36; i++)
        RxLocal[localId][i] = 0.0f;

    int counter = 0;
    float x_[9];
    if (x < width) 
    {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        for (int i = y - 1; i <= y + 1; i++)
            for (int j = x - 1; j <= x + 1; j++)
                x_[counter++] = read_imagef(image, sampler, (int2)(i, j)).x;
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        #pragma unroll
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 8 local rx values
        #pragma unroll
        for (int i = 0; i < 8; i++) 
            RxLocal[localId][i] = x_[i] * current_pixel;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //simplified summation for rx
    //TODO can be optimized
    if (localId < 8)
    {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; i++)
            sum += RxLocal[i][localId];
        rx[(outputIndex / 8) + localId] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //calculate 36 Rx values
    if (x < width)
    {
        counter = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            for (int j = i; j < 8; j++)
                RxLocal[localId][counter++] = x_[i] * x_[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //simplified summation for Rx
    //TODO can be optimized
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++)
        sum += RxLocal[i][RxMappings[localId]];
    Rx[outputIndex] = sum;
}