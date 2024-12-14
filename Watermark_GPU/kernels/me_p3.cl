//manual loop unrolled calculation of rx in local memory
void me_p3_rxCalculate(__local float RxLocal[64][36], const int localId, const float x_0, const float x_1, const float x_2, const float x_3, const float currentPixel, const float x_5, const float x_6, const float x_7, const float x_8)
{
    RxLocal[localId][0] = x_0 * currentPixel;
    RxLocal[localId][1] = x_1 * currentPixel;
    RxLocal[localId][2] = x_2 * currentPixel;
    RxLocal[localId][3] = x_3 * currentPixel;
    RxLocal[localId][4] = x_5 * currentPixel;
    RxLocal[localId][5] = x_6 * currentPixel;
    RxLocal[localId][6] = x_7 * currentPixel;
    RxLocal[localId][7] = x_8 * currentPixel;
}

//manual loop unrolled calculation of Rx in local memory
void me_p3_RxCalculate(__local float RxLocal[64][36], const int localId, const float x_0, const float x_1, const float x_2, const float x_3, const float x_5, const float x_6, const float x_7, const float x_8)
{
    RxLocal[localId][0] = x_0 * x_0;
    RxLocal[localId][1] = x_0 * x_1;
    RxLocal[localId][2] = x_0 * x_2;
    RxLocal[localId][3] = x_0 * x_3;
    RxLocal[localId][4] = x_0 * x_5;
    RxLocal[localId][5] = x_0 * x_6;
    RxLocal[localId][6] = x_0 * x_7;
    RxLocal[localId][7] = x_0 * x_8;
    RxLocal[localId][8] = x_1 * x_1;
    RxLocal[localId][9] = x_1 * x_2;
    RxLocal[localId][10] = x_1 * x_3;
    RxLocal[localId][11] = x_1 * x_5;
    RxLocal[localId][12] = x_1 * x_6;
    RxLocal[localId][13] = x_1 * x_7;
    RxLocal[localId][14] = x_1 * x_8;
    RxLocal[localId][15] = x_2 * x_2;
    RxLocal[localId][16] = x_2 * x_3;
    RxLocal[localId][17] = x_2 * x_5;
    RxLocal[localId][18] = x_2 * x_6;
    RxLocal[localId][19] = x_2 * x_7;
    RxLocal[localId][20] = x_2 * x_8;
    RxLocal[localId][21] = x_3 * x_3;
    RxLocal[localId][22] = x_3 * x_5;
    RxLocal[localId][23] = x_3 * x_6;
    RxLocal[localId][24] = x_3 * x_7;
    RxLocal[localId][25] = x_3 * x_8;
    RxLocal[localId][26] = x_5 * x_5;
    RxLocal[localId][27] = x_5 * x_6;
    RxLocal[localId][28] = x_5 * x_7;
    RxLocal[localId][29] = x_5 * x_8;
    RxLocal[localId][30] = x_6 * x_6;
    RxLocal[localId][31] = x_6 * x_7;
    RxLocal[localId][32] = x_6 * x_8;
    RxLocal[localId][33] = x_7 * x_7;
    RxLocal[localId][34] = x_7 * x_8;
    RxLocal[localId][35] = x_8 * x_8;
}

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

    float x_0, x_1, x_2, x_3, currentPixel, x_5, x_6, x_7, x_8;
    if (x < width) 
    {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        x_0 = read_imagef(image, sampler, (int2)(y - 1, x - 1)).x;
        x_1 = read_imagef(image, sampler, (int2)(y - 1, x)).x;
        x_2 = read_imagef(image, sampler, (int2)(y - 1, x + 1)).x;
        x_3 = read_imagef(image, sampler, (int2)(y, x - 1)).x;
        currentPixel = read_imagef(image, sampler, (int2)(y, x)).x;
        x_5 = read_imagef(image, sampler, (int2)(y, x + 1)).x;
        x_6 = read_imagef(image, sampler, (int2)(y + 1, x - 1)).x;
        x_7 = read_imagef(image, sampler, (int2)(y + 1, x)).x;
        x_8 = read_imagef(image, sampler, (int2)(y + 1, x + 1)).x;
        me_p3_rxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, currentPixel, x_5, x_6, x_7, x_8);
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
        me_p3_RxCalculate(RxLocal, localId, x_0, x_1, x_2, x_3, x_5, x_6, x_7, x_8);
    barrier(CLK_LOCAL_MEM_FENCE);

    //simplified summation for Rx
    //TODO can be optimized
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++)
        sum += RxLocal[i][RxMappings[localId]];
    Rx[outputIndex] = sum;
}