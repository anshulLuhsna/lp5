#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = call;                                                \
        if(err != cudaSuccess)                                                 \
        {                                                                      \
            cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                 << " at line " << __LINE__ << endl;                          \
            return 1;                                                          \
        }                                                                      \
    } while(0)

// ---------------- VECTOR ADDITION ----------------

#define VECTOR_SIZE 1000000

__global__ void vectorAdd(int *A, int *B, int *C)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < VECTOR_SIZE)
    {
        C[i] = A[i] + B[i];
    }
}

// ---------------- MATRIX MULTIPLICATION ----------------

#define MATRIX_SIZE 512

__global__ void matrixMul(int *A, int *B, int *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < MATRIX_SIZE && col < MATRIX_SIZE)
    {
        int sum = 0;

        for(int k = 0; k < MATRIX_SIZE; k++)
        {
            sum += A[row * MATRIX_SIZE + k] *
                   B[k * MATRIX_SIZE + col];
        }

        C[row * MATRIX_SIZE + col] = sum;
    }
}

int main()
{
    // =====================================================
    // VECTOR ADDITION
    // =====================================================

    cout << "========== VECTOR ADDITION ==========\n";

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    A = new int[VECTOR_SIZE];
    B = new int[VECTOR_SIZE];
    C = new int[VECTOR_SIZE];

    // Initialize vectors
    for(int i = 0; i < VECTOR_SIZE; i++)
    {
        A[i] = i;
        B[i] = i;
    }

    // ---------------- CPU VECTOR ADDITION ----------------

    auto cpu_start = high_resolution_clock::now();

    for(int i = 0; i < VECTOR_SIZE; i++)
    {
        C[i] = A[i] + B[i];
    }

    auto cpu_end = high_resolution_clock::now();

    auto cpu_duration =
    duration_cast<milliseconds>(cpu_end - cpu_start);

    cout << "CPU Vector Addition Time: "
         << cpu_duration.count()
         << " ms\n";

    // ---------------- GPU VECTOR ADDITION ----------------

    CUDA_CHECK(cudaMalloc((void**)&d_A, VECTOR_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, VECTOR_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, VECTOR_SIZE * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, A,
                          VECTOR_SIZE * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B,
                          VECTOR_SIZE * sizeof(int),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    vectorAdd<<<(VECTOR_SIZE + 255)/256, 256>>>(d_A, d_B, d_C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time = 0;

    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    CUDA_CHECK(cudaMemcpy(C, d_C,
                          VECTOR_SIZE * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool vector_correct = true;
    for(int i = 0; i < VECTOR_SIZE; i++)
    {
        if(C[i] != A[i] + B[i])
        {
            vector_correct = false;
            break;
        }
    }

    cout << "GPU Vector Addition Time: "
         << gpu_time
         << " ms\n";
    cout << "Vector Addition Correct: "
         << (vector_correct ? "Yes" : "No")
         << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] A;
    delete[] B;
    delete[] C;

    // =====================================================
    // MATRIX MULTIPLICATION
    // =====================================================

    cout << "\n========== MATRIX MULTIPLICATION ==========\n";

    int size = MATRIX_SIZE * MATRIX_SIZE;

    int *M1, *M2, *M3;
    int *d_M1, *d_M2, *d_M3;

    M1 = new int[size];
    M2 = new int[size];
    M3 = new int[size];

    // Initialize matrices
    for(int i = 0; i < size; i++)
    {
        M1[i] = 1;
        M2[i] = 1;
    }

    // ---------------- CPU MATRIX MULTIPLICATION ----------------

    auto cpu_start_mat = high_resolution_clock::now();

    for(int i = 0; i < MATRIX_SIZE; i++)
    {
        for(int j = 0; j < MATRIX_SIZE; j++)
        {
            int sum = 0;

            for(int k = 0; k < MATRIX_SIZE; k++)
            {
                sum += M1[i * MATRIX_SIZE + k] *
                       M2[k * MATRIX_SIZE + j];
            }

            M3[i * MATRIX_SIZE + j] = sum;
        }
    }

    auto cpu_end_mat = high_resolution_clock::now();

    auto cpu_duration_mat =
    duration_cast<milliseconds>(cpu_end_mat - cpu_start_mat);

    cout << "CPU Matrix Multiplication Time: "
         << cpu_duration_mat.count()
         << " ms\n";

    // ---------------- GPU MATRIX MULTIPLICATION ----------------

    CUDA_CHECK(cudaMalloc((void**)&d_M1, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_M2, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_M3, size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_M1, M1,
                          size * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_M2, M2,
                          size * sizeof(int),
                          cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16,16);

    dim3 blocksPerGrid(
        (MATRIX_SIZE + 15)/16,
        (MATRIX_SIZE + 15)/16
    );

    CUDA_CHECK(cudaEventRecord(start));

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(
        d_M1, d_M2, d_M3
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    CUDA_CHECK(cudaMemcpy(M3, d_M3,
                          size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool matrix_correct = true;
    for(int i = 0; i < size; i++)
    {
        if(M3[i] != MATRIX_SIZE)
        {
            matrix_correct = false;
            break;
        }
    }

    cout << "GPU Matrix Multiplication Time: "
         << gpu_time
         << " ms\n";
    cout << "Matrix Multiplication Correct: "
         << (matrix_correct ? "Yes" : "No")
         << "\n";

    // Cleanup

    CUDA_CHECK(cudaFree(d_M1));
    CUDA_CHECK(cudaFree(d_M2));
    CUDA_CHECK(cudaFree(d_M3));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] M1;
    delete[] M2;
    delete[] M3;

    return 0;
}


// Commands to run:
// nvcc cuda.cu -o cuda_program
// ./cuda_program
