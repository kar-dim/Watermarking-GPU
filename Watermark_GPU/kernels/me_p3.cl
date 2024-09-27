__kernel void me(__read_only image2d_t image,
    __global float* Rx,
    __global float* rx,
    __constant int* RxMappings,
    __local float RxLocal[64][36], //64 local threads, 36 values each
    __local float rxLocal[8][64],
    __local float rxPartial[8][8]) //helper scratch memory for rx calculation
{
    const int x = get_global_id(1), y = get_global_id(0);
    const int width = get_image_width(image);
    const int paddedWidth = get_global_size(1);
    const int localId = get_local_id(1);
    const int outputIndex = (y * paddedWidth) + x;
    const bool isPadded = paddedWidth > width;

    //initialize rx shared memory with coalesced access
    for (int i = 0; i < 8; i++)
        rxLocal[i][localId] = 0.0f;
    rxPartial[localId % 8][localId / 8] = 0.0f;

    //fix for OpenCL 1.2 where global size % local size should be 0, and local size is padded, a bound check is needed
    if (x < width) 
    {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        int counter = 0;
        float x_[9];
        for (int j = x - 1; j <= x + 1; j++)
            for (int i = y - 1; i <= y + 1; i++)
                x_[counter++] = read_imagef(image, sampler, (int2)(j, i)).x;
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 64 local Rx and 8 local rx values
        counter = 0;
        for (int i = 0; i < 8; i++) {
            rxLocal[i][localId] = x_[i] * current_pixel;
            for (int j = i; j < 8; j++)
                RxLocal[localId][counter++] = x_[i] * x_[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //each thread will calculate the reduction sums of Rx and rx and write them to global memory
    //if image is padded we don't want to sum the garbage local array values, we could zero the local array
    //but it would cost time, instead it is better to calculate what is needed directly
    const int limit = (isPadded && paddedWidth - x <= 64) ? 64 - (paddedWidth - width) : 64;
    float reductionSum_Rx = 0.0f, reductionSum_rx = 0.0f;
    for (int j = 0; j < limit; j++)
        reductionSum_Rx += RxLocal[j][RxMappings[localId]];

    //optimized summation for rx: normally we would sum 64 values per line/thread for a total of 8 sums
    //but this introduces heavy uncoalesced shared loads and bank conflicts, so we assign each of the 64 threads
    //to partially sum 8 horizontal values, and then the first 8 threads will fully sum the partial sums
    for (int i = 0; i < 8; i++)
        reductionSum_rx += rxLocal[localId / 8][((localId % 8) * 8) + i];
    rxPartial[localId % 8][localId / 8] = reductionSum_rx;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float rowSum = 0.0f;
    if (localId < 8)
    {
        for (int i = 0; i < 8; i++)
            rowSum += rxPartial[i][localId];
        rx[(outputIndex / 8) + localId] = rowSum;
    }
    Rx[outputIndex] = reductionSum_Rx;
}