---
title: "CUDA编程实战"
date: 2023-07-19T18:18:05+08:00
lastmod: 2023-07-20T09:19:06+08:00
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
