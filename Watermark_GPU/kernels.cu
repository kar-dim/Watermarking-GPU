#include "kernels.cuh"

__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const int width, const int padded_width, const int height) 
{
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
	const int local_id = threadIdx.y;
	const int output_index = (y * padded_width) + x;
    const bool is_padded = padded_width > width;

    __shared__ float Rx_local[64][36];
    __shared__ float rx_local[8][64];
    __shared__ float rx_partial[8][8];

    //initialize rx shared memory with coalesced access
    for (int i = 0; i < 8; i++)
        rx_local[i][local_id] = 0.0f;
    rx_partial[local_id % 8][local_id / 8] = 0.0f;

    if (y >= height)
        return;

    if (x < width) 
    {
        int counter = 0;
        float x_[9];
        for (int j = x - 1; j <= x + 1; j++)
            for (int i = y - 1; i <= y + 1; i++)
                x_[counter++] = tex2D<float>(texObj, j, i);
        const float current_pixel = x_[4];

        //shift neighborhood values, so that consecutive values are neighbors only (to eliminate "if"s)
        for (int i = 4; i < 8; i++)
            x_[i] = x_[i + 1];

        //calculate this thread's 36 local Rx and 8 local rx values
        counter = 0;
        for (int i = 0; i < 8; i++) 
        {
            rx_local[i][local_id] = x_[i] * current_pixel;
            for (int j = i; j < 8; j++)
                Rx_local[local_id][counter++] = x_[i] * x_[j];
        }
    }

    //each thread will calculate the reduction sums of Rx and rx and write them to global memory
    //if image is padded we don't want to sum the garbage local array values, we could zero the local array
    //but it would cost time, instead it is better to calculate what is needed directly
    __syncthreads();
    const int limit = (is_padded && padded_width - x <= 64) ? 64 - (padded_width - width) : 64;
    float reduction_sum_Rx = 0.0f, reduction_sum_rx = 0.0f;
    for (int j = 0; j < limit; j++)
        reduction_sum_Rx += Rx_local[j][Rx_mappings[local_id]];

    //optimized summation for rx: normally we would sum 64 values per line/thread for a total of 8 sums
    //but this introduces heavy uncoalesced shared loads and bank conflicts, so we assign each of the 64 threads
    //to partially sum 8 horizontal values, and then the first 8 threads will fully sum the partial sums
    for (int i = 0; i < 8; i++)
        reduction_sum_rx += rx_local[local_id / 8][((local_id % 8) * 8) + i];
    rx_partial[local_id % 8][local_id / 8] = reduction_sum_rx;
    __syncthreads();

    float row_sum = 0.0f;
    if (local_id < 8) 
    {
        for (int i = 0; i < 8; i++)
            row_sum += rx_partial[i][local_id];
        rx[(output_index / 8) + local_id] = row_sum;
    }
    Rx[output_index] = reduction_sum_Rx;
}

__global__ void calculate_neighbors_p3(cudaTextureObject_t texObj, float* x_, const int width, const int height)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.y * blockDim.y + ty;
    const int y = blockIdx.x * blockDim.x + tx;
    const int output_index = (x * height + y);

    if (x < width && y < height) 
    {
        // Load 8 neighboring pixels into shared memory
        x_[0 * width * height + output_index] = tex2D<float>(texObj, x - 1, y - 1);
        x_[1 * width * height + output_index] = tex2D<float>(texObj, x - 1, y);
        x_[2 * width * height + output_index] = tex2D<float>(texObj, x - 1, y + 1);
        x_[3 * width * height + output_index] = tex2D<float>(texObj, x, y - 1);
        x_[4 * width * height + output_index] = tex2D<float>(texObj, x, y + 1);
        x_[5 * width * height + output_index] = tex2D<float>(texObj, x + 1, y - 1);
        x_[6 * width * height + output_index] = tex2D<float>(texObj, x + 1, y);
        x_[7 * width * height + output_index] = tex2D<float>(texObj, x + 1, y + 1);
    }
}