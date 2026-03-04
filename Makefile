NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -Xcompiler -fPIC

# 源文件
KERNEL_SRCS = nvidia/kernels/vector_add/basic.cu \
              nvidia/kernels/vector_add/vectorized.cu \
              nvidia/kernels/vector_add/shared_mem.cu

TEST_SRC = tests/test_vector_add.cu

# 目标文件
LIB_TARGET = libvector_add.so
TEST_TARGET = test_vector_add

all: $(LIB_TARGET) $(TEST_TARGET)

# 编译共享库（供Python调用）
$(LIB_TARGET): $(KERNEL_SRCS)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^

# 编译C++测试程序
$(TEST_TARGET): $(TEST_SRC) $(KERNEL_SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(LIB_TARGET) $(TEST_TARGET)

.PHONY: all clean
