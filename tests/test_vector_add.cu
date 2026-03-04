#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 声明三种实现的启动函数
extern "C" {
    void launchVectorAddBasic(const float* d_a, const float* d_b, float* d_c, int n, int blockSize);
    void launchVectorAddVectorized(const float* d_a, const float* d_b, float* d_c, int n, int blockSize);
    void launchVectorAddSharedMem(const float* d_a, const float* d_b, float* d_c, int n, int blockSize);
}

// 验证结果正确性
bool verifyResult(const float* a, const float* b, const float* c, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5) {
            printf("Error at index %d: expected %f, got %f\n", i, a[i] + b[i], c[i]);
            return false;
        }
    }
    return true;
}

// 测试单个实现
void testImplementation(const char* name,
                       void (*launcher)(const float*, const float*, float*, int, int),
                       const float* h_a, const float* h_b, float* h_c,
                       float* d_a, float* d_b, float* d_c,
                       int n, int blockSize) {
    printf("\n测试 %s 实现:\n", name);

    // 重置结果数组
    cudaMemset(d_c, 0, n * sizeof(float));

    // 执行kernel
    launcher(d_a, d_b, d_c, n, blockSize);

    // 检查kernel执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel执行失败: %s\n", cudaGetErrorString(err));
        return;
    }

    // 同步等待kernel完成
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    if (verifyResult(h_a, h_b, h_c, n)) {
        printf("✓ 结果正确\n");
    } else {
        printf("✗ 结果错误\n");
    }
}

int main(int argc, char** argv) {
    int n = 1000000;
    int blockSize = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) blockSize = atoi(argv[2]);

    printf("Vector Add 测试\n");
    printf("数组大小: %d\n", n);
    printf("Block大小: %d\n", blockSize);

    // 分配主机内存
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_c = (float*)malloc(n * sizeof(float));

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 测试三种实现
    testImplementation("Basic", launchVectorAddBasic, h_a, h_b, h_c, d_a, d_b, d_c, n, blockSize);
    testImplementation("Vectorized", launchVectorAddVectorized, h_a, h_b, h_c, d_a, d_b, d_c, n, blockSize);
    testImplementation("SharedMem", launchVectorAddSharedMem, h_a, h_b, h_c, d_a, d_b, d_c, n, blockSize);

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
