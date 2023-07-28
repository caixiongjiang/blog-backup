---
title: "CUDA编程实战"
date: 2023-07-19T18:18:05+08:00
lastmod: 2023-07-25T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/CUDA_title.jpg"
description: "CUDA编程实战"
tags:
- CUDA编程
categories:
- HPC
series:
- CUDA
comment : true
---


## CUDA编程实战

### Hello GPU

> 编写第一个gpu程序

一般来说，CUDA程序是`.cu`结尾的程序！

hello-gpu.cu:

```c
#include <stdio.h>

void cpu() {
    printf("hello cpu\n");
}

__global__ void gpu() {
    printf("hello gpu\n");
}

int main() {
    cpu();
    gpu<<<1, 1>>>();
    // 等待cpu和gpu同步
    cudaDeviceSynchronize();
}
```

\_\_global\_\_:

* __global__关键字代表以下**函数将在GPU山运行并全局可调用。**
* 通过我们将在cpu上执行的代码称为主机代码，而在GPU上运行的代码称为设备代码。
* 注意返回类型为void。使用__global__关键字定义的函数需要返回void类型。

gpu<<<1, 1>>>():

* 通常，当调用要在GPU上运行的函数时，我们将这种函数称为`已启动的核函数`。
* 启动核函数之前必须提供执行的配置，在向核函数传递任何预期参数之前使用`<<<...>>>`语法完成配置。
* 程序员可通过执行配置为核函数启动指定线程层次结构，从而定义`线程组（也称为线程块）的数量`，以及要在`每个线程块中执行的线程数量`。这里就代表正在使用包含1线程（第二个配置参数）的1线程块（第一个配置参数）启动核函数。

cudaDeviceSynchronize():

* 与大部分c/c++代码不同，**核函数启动方式为异步：CPU代码将继续执行而无需等待核函数完成启动。**
* 调用CUDA运行时提供的函数cudaDeviceSynchronize将导致主机（cpu）代码暂停，直至设备（GPU）代码执行完成，才能在cpu上恢复执行。

> 使用nvcc编译、链接、执行

```shell
nvcc -o hello-gpu hello-gpu.cu -run
```

看到

```scss
hello cpu
hello gpu
```

说明你编译、链接、执行成功。



### 网格、块、线程

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img58.jpg)

*注意，上述结构并不是硬件结构，而是软件逻辑概念！*

一个GPU中有多个线程块，每个线程块中含有不同的线程，每个线程能执行一个程序。多个线程块组成一个网格。在实际的编程中，一个Block可以最多放1024个线程。

> 如何使用CUDA编程语言来表示这些含义

* gridDim.x 表示网格的线程块个数
* blockIdx.x 表示当前块的索引
* blockDim.x 表示块的线程数
* threadIdx.x 表示当前中线程的索引

假设执行`performWork<<<2, 4>>>()`代表线程块个数为2，块中的线程数为4

将我们的`hello-gpu.cu`文件修改执行一下，改成了`gpu<<<2, 4>>>()`，打印的结果为1个`cpu`和8个`gpu`。说明gpu的程序同时有8个线程一起执行了它。

我现在需要指定某个块中的某个CUDA线程来执行它，则需要通过两个索引值来判断：

```c
#include <stdio.h>

void cpu() {
    printf("hello cpu\n");
}

__global__ void gpu() {
  	if (blockIdx.x == 0 && threadIdx.x == 0)
    		printf("hello gpu\n");
}

int main() {
    cpu();
    gpu<<<1, 1>>>();
    // 等待cpu和gpu同步
    cudaDeviceSynchronize();
}
```

**执行的结果变为了一个cpu和一个gpu。**

**显而易见，在GPU上执行循环的复杂度由原来的O(n)变成了O(1)。**

题外话，CPU也能并发线程，那为什么CPU不好：其实CPU在软件层面上能同时开20个或者30个等线程，但这都是通过操作系统调度时间切片做到的，但实际上从物理的层面上只能同时跑8个线程，通过4个ALU同时跑4个线程，通过寄存器复制实现超线程扩展到8个。



### 显存分配

> 如何区分每个线程id

通过计算`blockIdx.x * blockDim.x + threadIdx.x`的值区分不同的线程id。

> cpu分配内存/gpu分配显存

```c
//cpu
int N = 2 << 20;
size_t size = N * sizeof(int);
int *a;
a = (int *)malloc(size);//分配内存
free(a);
//gpu
int N = 2 << 20;
size_t size = N * sizeof(int);
int *a;
cudaMallocManaged(&a, size);
cudaFree(a);
```

**需要注意的是cudaMallocManaged()分配的是统一内存，既可以被cpu使用，也可以被gpu使用。**

通过一个例子来学习显存分配：

```c
#include <stdio.h>
#include <stdlib.h>

void cpu(int *a, int N) {
  	for (int i = 0; i < N; i++) {
      	a[i] = i;
    }
}

__global__ void gpu(int *a, int N) {
  	int i = blockIdx.x * blockDim.x + threadIdx.x;
  	if (i < N) {
      	A[i] *= 2;
    }
}

bool check(int *a, int N) {
  	for (int i = 0; i < N; i++) {
      	if (a[i] != i * 2) return false;
    }
  	return true;
}

int main() {
  	const int N = 100;
  	size_t size = N * sizeof(int);
  	int * a;
  	//分配通用内存
  	cudaMallocManaged(&a, size);
  	cpu(a, N); // cpu进行操作
  	size_t threads = 256;//线程块中线程的个数
  	size_t blocks = (N + threads - 1) / threads; //向上取整计算
  	gpu<<<blocks, threads>>>(a, N); //gpu进行操作
  	cudaDeviceSynchronize(); //cpu和gpu同步(如果不进行同步check必然为error)
  	check(a, N) ? printf("ok") : printf("error");
    /* 最后输出的结果为ok */
  	cudaFree(a);
}
```

通用内存既可以被cpu调用也可以被gpu调用。

### 跨步循环

当需要并行的过程远大于线程数，便可以使用跨步循环，`跨了多少步，同一个线程就运行了多少次`：

```c
#include <stdio.h>
#include <stdlib.h>

void cpu(int *a, int N) {
  	for (int i = 0; i < N; i++) {
      	a[i] = i;
    }
}

__global__ void gpu(int *a, int N) {
  	int threadi = blockIdx.x * blockDim.x + threadIdx.x;
  	int stride = gripDim.x * blockDim.x;
  	for (int i = threadi; i < N; i += stride) {
      	a[i] *= 2;
    }
}

bool check(int *a, int N) {
  	for (int i = 0; i < N; i++) {
      	if (a[i] != i * 2) return false;
    }
  	return true;
}

int main() {
  	const int N = 100;
  	size_t size = N * sizeof(int);
  	int * a;
  	//分配通用内存
  	cudaMallocManaged(&a, size);
  	cpu(a, N); // cpu进行操作
  	size_t threads = 256;//线程块中线程的个数
  	size_t blocks = 1;
  	gpu<<<blocks, threads>>>(a, N); //gpu进行操作
  	cudaDeviceSynchronize(); //cpu和gpu同步(如果不进行同步check必然为error)
  	check(a, N) ? printf("ok") : printf("error");
  	cudaFree(a);
}
```

### 异常处理

> 普通函数异常处理

 假设一个分配内存的函数出现了错误，我们可以用cuda内置的`cudaError_t`类型来返回是否错误。

```c
cudaError_t err;
err = cudaMallocManaged(&a, size);
if (err != cudaSuccess) {
  	printf("Error: %s\n", cudaGetErrirString(err));
}
```

> Kernel函数的异常处理

核函数的没有返回值，这需要如何处理，cuda也专门提供了`cudaGetLastError()`函数来返回是否错误：

```c
gpu<<<blocks, threads>>>(a, N);
err = cudaGetLastError();
if (err != cudaSuccess) {
  	printf("Error: %s\n", cudaGetErrirString(err));
}
```

**在实际的c++编程中，我们通常会将其写成一个inline函数统一使用**：

```c++
inline cudaError_t checkCuda(cudaError_t result) {
  	if (result != cudaSuccess) {
      	fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
      	assert(result == cudaSuccess);
    }
  	return result;
}
```

那么这段程序的调用方法就显而易见了：

```c
checkCuda(cudaGetLastError());
checkCuda(cudaMallocManaged(&a, size));
```

> inline函数和普通函数的区别

普通函数：

1. 普通函数定义通常放在头文件(.h)中，实现放在源文件(.cpp)中。
2. 每次函数调用时，程序会跳转到函数的地址执行函数体中的代码。
3. 普通函数适合处理较大的代码块和复杂的逻辑。

`inline`函数：

1. `inline`关键字用于建议编译器将函数调用处的代码替换为函数体中的代码，而不是通过跳转执行函数。
2. `inline`函数通常定义放在头文件中，并且在同一编译单元内直接展开函数调用。
3. 适用于短小的、频繁调用的函数，比如简单的getter和setter函数，以避免函数调用的开销。

区别总结：

1. `inline`函数是对编译器的建议，而不是强制要求。编译器有权忽略`inline`关键字。
2. 普通函数一般用于处理复杂的逻辑，而`inline`函数用于短小的频繁调用的代码块，以节省函数调用开销。
3. `inline`函数在编译时展开代码，而普通函数是跳转执行函数体中的代码。

需要注意的是，如果函数体较大或者在多个地方调用，编译器可能会忽略`inline`关键字，将其当作普通函数处理。此外，过度使用`inline`可能导致代码膨胀，增加代码段的大小，反而可能影响性能。**因此，`inline`应谨慎使用，最好只用于短小的、频繁调用的函数。**编译器在优化方面通常会自动处理函数的内联，不需要手动添加`inline`关键字。

### 矩阵加法

> 分别通过cpu和gpu实现矩阵的加法运算

matrix.cu:

```c
#include <stdio.h>

#define N 64

__global__ void gpu(int *a, int *b, int *c_gpu) {
  	int r = blockDim.x * blockIdx.x + threadIdx.x;
  	int c = blockDim.y * blockIdx.y + threadIdx.y;
  	if (r < N && c < N) {
      	c_gpu[r * N + c] = a[r * N + c] + b[r * N + c];
    }
}

void cpu(int *a, int *b, int *c_cpu) {
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	c_cpu[r * N + c] = a[r * N + c] + b[r * N + c];
        }
    }
}

bool check(int *a, int *b, int *c_cpu, int *c_gpu) {
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	if (c_cpu[r * N + c] != c_gpu[r * N + c]) return false;
        }
    }
  	return true;
}


int main() {
  	int *a, *b, *c_cpu, *c_gpu;
  	size_t size = N * N * sizeof(int);
  	//分配globalmemory
  	cudaMallocManaged(&a, size);
  	cudaMallocManaged(&b, size);
  	cudaMallocManaged(&c_cpu, size);
 	 	cudaMallocManaged(&c_gpu, size);
  	
  	//初始化
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	a[r * N + c] = r;
          	b[r * N + c] = c;
          	c_cpu[r * N + c] = 0;
          	c_gpu[r * N + c] = 0;
        }
    }

  	dim3 threads(16, 16, 1);
  	dim3 blocks((N + threads.x - 1) / threads.x, N + threads.y - 1) / threads.y, 1);
  	gpu<<<blocks, threads>>>(a, b, c_gpu);
  	//同步到cpu上
  	cudaDeviceSynchronize();
  	
  	cpu(a, b, c_cpu);
  	check(a, b, c_cpu, c_gpu) ? printf("ok!\n") : printf("error!\n");
  
  	cudaFree(a);
  	cudaFree(b);
  	cudaFree(c_cpu);
  	cudaFree(c_gpu);
}
```

### CUDA性能分析

> 安装NVIDIA Nsight Systems

* 进入[官网](https://developer.nvidia.cn/gameworksdownload)，下载`Nsight Systems`的Linux CLI的`.deb`版本

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img64.jpg)

* 在本地进行安装：

```shell
$ sudo dpkg -i NsightSystems-linux-cli-public-2023.2.1.122-3259852.deb
```

* 在终端使用：

```shell
#application是程序，application-arguments是程序参数
$ nsys [global-options] profile [options] <application> [application-arguments]
```

参数的选择如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img65.jpg)

* 对上述矩阵加法的程序进行性能分析：

```shell
$ nsys profile --stats=true matrix
```

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img66.jpg)

**可以看到我们的程序的大部分时间都花在了显存分配这件事上，那么我们优化的目标就是让GPU计算的时间占总时间的比重更大，那么程序的效率就会越高。** 

### GPU属性

GPU内部有许多称为`SM`的**流多处理器**，一个`SM`有多个流处理器，GPU在执行Kernel函数的时候，会让SM去处理Block。

SM会在**一个名为`WARP`的线程块**内创建、管理、调度和执行**32**个线程的线程组。所以线程数选32的倍数最佳！！

> GPU信息获取

* 编写CUDA程序获取：

```c
#include <stdio.h>

int main() {
  	int id;
  	cudaGetDevice(&id);
  	
  	cudaDeviceProp props;
  	cudaGetDeviceProperties(&props, id);
  	printf("Device id: %d\n \
  	SM_num: %d\n \
  	capability major: %d\n \
  	capability minor: %d\n \
  	warp size: %d\n"\
   	, id, props.multiProcessorCount, props.major, props.minor, props.warp);
}
```

编译运行后得到结果：

```scss
Device id: 0
SM_num: 28
capability major: 8
capability minor: 6
warp size: 32
```

查询我的显卡（RTX3060）的流处理器总共有3564个，那么一个SM（流多处理器）的流处理器个数为128个。

### 显存分配（2）

前面我们使用的显存分配方式为`cudaMallocManaged()`方法，**这种方法第一时间分配的其实并不是显存，而是统一内存（UM）。分配UM时，内存尚未驻留在主机（Host）上或设备（Device）上。主机或者设备尝试访问内存时会发生缺页中断，此时主机或者设备才会批量迁移所需要的数据。也就是说CPU和GPU访问该内存都会发生上述的事件。**

如果分配的内存既被CPU调用，又被GPU调用，那么便可以使用这种分配方式。

我们使用CUDA性能分析工具对前面`hello-gpu`的程序块进行分析：

```shell
$ nsys profile --stats=true hello-gpu
```

得到的部分信息如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img67.jpg)

可以看到有`HtoD`和`DtoH`，这是由缺页触发的。

> 如何避免缺页触发内存拷贝？

我们可以通过`cudaMemPrefetchAsync`函数将托管内存异步预取到GPU设备上或CPU。那么在代码上该如何操作(在前面matrix.cu上进行修改)：

```c
#include <stdio.h>

#define N 64

__global__ void gpu(int *a, int *b, int *c_gpu) {
  	int r = blockDim.x * blockIdx.x + threadIdx.x;
  	int c = blockDim.y * blockIdx.y + threadIdx.y;
  	if (r < N && c < N) {
      	c_gpu[r * N + c] = a[r * N + c] + b[r * N + c];
    }
}

void cpu(int *a, int *b, int *c_cpu) {
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	c_cpu[r * N + c] = a[r * N + c] + b[r * N + c];
        }
    }
}

bool check(int *a, int *b, int *c_cpu, int *c_gpu) {
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	if (c_cpu[r * N + c] != c_gpu[r * N + c]) return false;
        }
    }
  	return true;
}


int main() {
  	int *a, *b, *c_cpu, *c_gpu;
  	size_t size = N * N * sizeof(int);
  	//分配globalmemory
  	cudaMallocManaged(&a, size);
  	cudaMallocManaged(&b, size);
  	cudaMallocManaged(&c_cpu, size);
 	 	cudaMallocManaged(&c_gpu, size);
  	
  	//初始化
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	a[r * N + c] = r;
          	b[r * N + c] = c;
          	c_cpu[r * N + c] = 0;
          	c_gpu[r * N + c] = 0;
        }
    }
  
  	//在GPU访问统一内存之前进行预取，防止发生缺页影响性能
  	int id;
  	cudaGetDevice(&id);
  	cudaMemPrefetchAsync(a, size, id);
  	cudaMemPrefetchAsync(b, size, id);
  	cudaMemPrefetchAsync(c_gpu, size, id);

  	dim3 threads(16, 16, 1);
  	dim3 blocks((N + threads.x - 1) / threads.x, N + threads.y - 1) / threads.y, 1);
  	gpu<<<blocks, threads>>>(a, b, c_gpu);
  
  	//数据同步到CPU之前将GPU的内存预取到CPU上
  	
  
  	//同步到cpu上
  	cudaDeviceSynchronize();
  	
  	cpu(a, b, c_cpu);
  	check(a, b, c_cpu, c_gpu) ? printf("ok!\n") : printf("error!\n");
  
  	cudaFree(a);
  	cudaFree(b);
  	cudaFree(c_cpu);
  	cudaFree(c_gpu);
}
```

对旧的`matrix.cu`和新的`matrix.cu`分别进行编译并进行性能分析：

旧的matrix.cu:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img68.jpg)

新的matrix.cu:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img69.jpg)

明显可以看到内存拷贝的开销变少了。无论是Host->Device还是Device->Host。

### 显存分配（3）

那么CUDA编程中除了分配统一内存，是否可以直接分配GPU显存，或者CPU锁页内存。

* `cudaMalloc`命令将直接为处于活动状态的GPU分配显存。这可以防止出现所有GPU分页错误，而代价是主机代码将无法访问该命令返回的指针。
* `cudaMallocHost`命令将直接为CPU分配内存。该命令可以"固定"内存（pinned memory）或"锁页"内存（page-locked memory）。它允许将内存异步拷贝至GPU或从GPU异步拷贝至内存。固定内存过多则会干扰CPU性能。释放固定内存时，应使用`cudaFreeHost`命令。
* 无论是从主机到设备还是设备到主机，`cudaMemcpy`命令均可拷贝（而非传输）内存。

我们将之前的matrix.cu再次改写：

```c
#include <stdio.h>
#include <assert>

#define N 64

__global__ void gpu(int *a_gpu, int *b_gpu, int *c_gpu) {
  	int r = blockDim.x * blockIdx.x + threadIdx.x;
  	int c = blockDim.y * blockIdx.y + threadIdx.y;
  	if (r < N && c < N) {
      	c_gpu[r * N + c] = a_gpu[r * N + c] + b_gpu[r * N + c];
    }
}

void cpu(int *a_cpu, int *b_cpu, int *c_cpu) {
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	c_cpu[r * N + c] = a_cpu[r * N + c] + b_cpu[r * N + c];
        }
    }
}

bool check(int *c_cpu, int *c_gpu) {
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	if (c_cpu[r * N + c] != c_gpu[r * N + c]) return false;
        }
    }
  	return true;
}

inline cudaError_t checkCuda(cudaError_t result) {
  	if (result != cudaSuccess) {
      	fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
      	assert(result == cudaSuccess);
    }
  	return result;
}


int main() {
  	int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_gpu2cpu, *c_cpu, *c_gpu;
  	size_t size = N * N * sizeof(int);
  	
  	//分配memory
  	cudaMallocHost(&a_cpu, size);
  	cudaMallocHost(&b_cpu, size);
  	cudaMallocHost(&c_cpu, size);
  	cudaMallocHost(&c_gpu2cpu, size);
  	cudaMalloc(&a_gpu, size);
  	cudaMalloc(&b_gpu, size);
 	 	cudaMalloc(&c_gpu, size);
  	
  	//初始化
  	for (int r = 0; r < N; r++) {
      	for (int c = 0; c < N; c++) {
          	a_cpu[r * N + c] = r;
          	b_cpu[r * N + c] = c;
          	c_cpu[r * N + c] = 0;
          	c_gpu2cpu[r * N + c] = 0;
          	//c_gpu[r * N + c] = 0; 不能直接访问GPU
        }
    }

  	cpu(a_cpu, b_cpu, c_cpu);
  
  	dim3 threads(16, 16, 1);
  	dim3 blocks((N + threads.x - 1) / threads.x, N + threads.y - 1) / threads.y, 1);
  	
  	//理论上的内存拷贝，但其实不需要，因为锁页内存在CUDA中不会被映射到硬盘上，但可以直接被GPU调用
  	cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);
  	
    gpu<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);
  	//同步到cpu上
  	cudaDeviceSynchronize();
  	cudaMemcpy(c_gpu2cpu, c_cpu, size, cudaMemcpyDeviceToHost);
  	
  	check(c_cpu, c_gpu) ? printf("ok!\n") : printf("error!\n");
  
  	cudaFreeHost(a_cpu);
  	cudaFreeHost(b_cpu);
  	cudaFreeHost(c_cpu);
  	cudaFreeHost(c_gpu2cpu);
  	cudaFree(c_gpu);
  	cudaFree(a_gpu);
  	cudaFree(b_gpu);
}
```



> 为什么GPU可以直接调用cudaMallocHost分配的锁页内存？

在CUDA编程中，使用cudaMallocHost函数分配的内存是锁页内存，这意味着该内存页不会被交换到磁盘上，从而提高了访问速度。此外，**锁页内存还可以直接与GPU内存进行数据传输，而不需要通过PCIe总线，从而进一步提高了数据传输速度。**

在CUDA编程中，GPU可以直接使用cudaMallocHost分配的锁页内存，是因为这些内存页已经被固定在物理内存中，并且可以直接映射到GPU的地址空间中。因此，在Kernel函数中可以直接访问这些内存页，而不需要进行额外的数据传输或者拷贝操作。

需要注意的是，在使用锁页内存时需要小心，因为它们会占用较多的系统内存，并且可能会导致系统变慢或者崩溃。因此，在使用锁页内存时需要谨慎考虑内存使用量，并且及时释放不再需要的内存。

### CUDA流

在CUDA编程中，流是按照顺序执行的一系列命令构成。在CUDA应用程序中，核函数的执行以及一些内存传输均在CUDA流中进行。

CUDA流行为的几项规则：

* 在给定流中所有操作回按序执行。
* 就不同非默认流中的操作而言，无法保证会按彼此之间的任何特定顺序执行
* 默认流具有阻断能力，它会等待其他已在运行的所有流完成当前操作后才运行，但其自身运行完毕之前，也会阻碍其他流的运行。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img70.jpg)

> CUDA流如何创建，运行机制是什么？

用NVIDIA官方的例程来解释一下：

```c
#include <stdio.h>

const int N = 1 << 20;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x +  blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] += sqrt(pow(3.14159, i));                                                                                                                                                                                                                                                                                                             
    }
}



int main() {
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);

        cudaMalloc(&data[i], N * sizeof(float));

        //launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        //launch a dummy kernel
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}
```

上面的`kernel<<<1, 1>>>`代表使用默认流，如果没有这句话，那么上面的每次计算几乎是并行的，因为每个Kernel函数执行的时间较长，不是默认流的情况下，新的流不断执行Kernel函数，而上一个Kernel函数还没有执行完，就会出现类似于流水线并行的方式。

如果有这个虚拟的默认流程序，就会将所有的Kernel函数分割成按序进行。**因为在非默认流有函数运行时，默认流程序会等待；在默认流执行函数时，非默认流的程序必须等待。**

> 通过流实现内存分配

`cudaMemcpyAsync`可以从主机到设备或从设备到主机异步复制内存。

**与核函数的执行类似，`cudaMemcpyAsync`在默认情况下仅相对主机是异步的。默认情况下，它在默认流中执行，因此对于GPU上发生的其他CUDA操作而言，它是阻塞操作。**

`cudaMemcpyAsync`函数将默认流作为可选的第5个参数。通过向其传递非默认流，可以将内存传输与其他默认流中发生的其他CUDA操作并发。

```c
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(&device_array[segmentOffset],
               	&host_array[segmentOffset],
                segmentSize,
                cudaMemcpyHostToDevice,
               	stream);
```

将`matrix_cudamalloc.cu`修改：

```c
#include <stdio.h>
#include <assert.h>

#define N 10000


__global__ void gpu(int *a_gpu, int *b_gpu, int *c_gpu) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < N && c < N) {
        c_gpu[r * N + c] = a_gpu[r * N + c] + b_gpu[r * N + c];
    }
}

void cpu(int *a, int *b, int *c_cpu) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            c_cpu[r * N + c] = a[r * N + c] + b[r * N + c];
        }
    }
}

bool check(int *c_cpu, int *c_gpu) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (c_cpu[r * N + c] != c_gpu[r * N + c]) {
                return false;
            }
        }
        
    }
    return true;
}


inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result != cudaSuccess);
    }
    return result;
}





int main() {
    int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu, *c_gpu2cpu;
    size_t size = N * N * sizeof(int);

    cudaMallocHost(&a_cpu, size);
    cudaMallocHost(&b_cpu, size);
    cudaMallocHost(&c_cpu, size);
    cudaMallocHost(&c_gpu2cpu, size);
    cudaMalloc(&a_gpu, size);
    cudaMalloc(&b_gpu, size);
    cudaMalloc(&c_gpu, size);
    
    //初始化
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            a_cpu[r * N + c] = r;
            b_cpu[r * N + c] = c;
            c_cpu[r * N + c] = 0;
            // c_gpu[r * N + c] = 0;
            c_gpu2cpu[r * N + c] = 0;
        }
    }

    cpu(a_cpu, b_cpu, c_cpu);



    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y, 1);

    cudaStream_t s1, s2, s3;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);

    //操纵GPU之前需要拷贝内存（新显卡不需要这项操作也是可以的）
    cudaMemcpyAsync(a_gpu, a_cpu, size, cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(b_gpu, b_cpu, size, cudaMemcpyHostToDevice, s2);
    gpu<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    checkCuda(cudaGetLastError());

    cudaMemcpyAsync(c_gpu2cpu, c_gpu, size, cudaMemcpyDeviceToHost, s3);
    //将数据同步到我们的cpu上
    cudaDeviceSynchronize();

    check(c_cpu, c_gpu2cpu) ? printf("ok\n") : printf("error\n");


    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFreeHost(a_cpu);
    cudaFreeHost(b_cpu);
    cudaFreeHost(c_cpu);
    cudaFreeHost(c_gpu2cpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
```

可以看到我们将内存拷贝的过程交给了流，通过CUDA性能分析工具分析了程序。单向的内存拷贝不会进行重叠，也就是说同一时间主机向设备或者设备向主机只能进行一次内存拷贝。