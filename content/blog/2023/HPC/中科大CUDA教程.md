---
title: "中科大CUDA教程"
date: 2023-07-19T18:18:05+08:00
lastmod: 2023-07-25T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/CUDA_title.jpg"
description: "CUDA硬件和逻辑设计，以及部分CUDA编程和优化"
tags:
- CUDA编程
categories:
- HPC
series:
- CUDA
comment : true
---


## 中科大CUDA编程

参考资料：

* CUDA C Programming Guide，中文翻译见[here](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese)
* CUDA C++ Best Practice Guide

### CPU体系架构概述

#### 现代CPU架构和性能优化

CPU是执行指令和处理数据的器件，能完成基本的逻辑和算术指令。

> 指令

Example：

算术：add r3,r4 -> r4

访存：load [r4] -> r7

控制：jz end

对于一个编译好的程序，最优化目标：
$$
\frac{cycle}{instruction}\times \frac{seconds}{cycle}
$$
总结来说，CPI（每条指令的时钟数）& 时钟周期，注意这两个指标并不独立。

> 摩尔定律

芯片的集成密度每两年翻一番，成本下降一半。

> CPU的处理流程

取址 -> 解码 -> 执行 -> 访存 -> 写回

> 流水线

使用一个洗衣服的例子，单件衣服总时间 = wash（30min）+ dry（40min）+ fold（20min）

那么洗4件衣服需要的总时间 = 30 + 40 + 40 + 40 + 40 + 20 = 210min

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img6.jpg)

* 流水线使用的是指令级的并行，可以有效地减少时钟周期
* 增加了延迟和芯片面积（需要更多的存储）
* 带来了一些问题：具有依赖关系的指令处理，分支如何处理

> 旁路（Bypassing）

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img38.jpg)

这里的两条指令具有依赖性，按照原来的方式，需要先计算R7的结果，再进行写回，访寸取到R7的结果。有了旁路这一功能，便可以跳过这个阶段，直接取到R7的结果。

> 流水线的停滞

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img39.jpg)

如果前面的`load[R3]`没有做完，流水线便会停滞。

> 分支

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img40.jpg)

判断是否做循环`jeq loop`，我们并不知道是否要执行这个循环，计算机会通过分支预测等工作来进行处理。

具体的分支预测基于过去的分支记录，现代计算机的预测器准确率大于90%。同样它同样会增加芯片面积，增加延迟（预测的开销）。

> 分支断定

与分支预测不相同的是，它不再使用分支预测器，而是将所有分支都做一遍。

好处：不需要复杂的预测器，减少了芯片面积，减少了错误预测，在GPU中使用了分支断定。

> 增加CPU一个时钟周期能处理的指令数（IPC）

超标量——增加流水线的宽度，一个时钟周期处理多条指令。（超标量流水线将每个阶段细分为更小的微操作，并在多个功能单元上同时执行这些微操作。这样，多条指令可以在同一时钟周期内同时执行，从而提高处理器的吞吐量。）

**这需要更多的寄存器和存储器带宽。**

> 指令调度

考虑以下指令：

```scss
xor r1,r2 -> r3
add r3,r4 -> r4
sub r5,r3 -> r3
addi r3,1 -> r1 //addi代表减法
```

* `xor`和`add`是相互依赖的（读后写）
* `sub`和`addi`相互依赖（读后写）
* `xor`和`sub`不依赖（写后写）

为了让程序运行地更快，可以使用替换寄存器的方法：

```scss
xor p1,p2 -> p3
add p6,p4 -> p7
sub p5,p2 -> p8
addi p8,1 -> p9 //addi代表减法
```

*这样我们的`xor`和`sub`就可以并行执行了。*

> 乱序执行

将所有的指令重排，使其顺序更合理。

* 重排缓冲区
* 发射队列/调度器

#### 存储器架构/层次

**存储器越大越慢。**

> 缓存

利用时间临近性和空间临近性，可以使我们的处理变得更快。计算机一般有3级缓存，缓存的大小越来越大。

#### 向量运算

```c
for (int i = 0; i < N; i++) {
  	A[i] = B[i] + C[i];
}
```

可以使用`单指令多数据(SIMD)`进行加速。

```c
for (int i = 0; i < N; i += 4) {
  	//并行同时计算	
  	A[i] = B[i] + C[i];
  	A[i + 1] = B[i + 1] + C[i + 1];
  	A[i + 2] = B[i + 2] + C[i + 2];
  	A[i + 3] = B[i + 3] + C[i + 3];
}
```

> x86的向量运算

* SSE：4宽度浮点和整数指令
* AVX：8宽度浮点和整数指令

#### 线程级的并行

线程的组成：私有的寄存器、程序计数器、栈等。

**程序员可以创建和线程，OS和程序员都可以对线程进行调度。**

#### CPU的瓶颈

因为功耗墙的存在，处理器的单核性能的提升会越来越少，所以需要多核来支撑。

>  新摩尔定律

* 处理器越来越胖，核越来越多
* 单核的性能不会大幅提升

由此也带来了另外一堵墙，叫`存储器墙`，处理器的存储器带宽无法满足处理能力的提升。

###  并行程序设计概述

#### 并行计算模式

并行计算是同时应用多个计算资源解决一个计算问题：

* 涉及多个计算资源或处理器
* 问题被分解为多个离散的部分，可以同时处理（并行）
* 每个部分可以由一系列指令完成

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img41.jpg)

> Flynn矩阵

|     SISD     |     SIMD     |     MISD     |     MIMD     |
| :----------: | :----------: | :----------: | :----------: |
| 单指令单数据 | 单指令多数据 | 多指令单数据 | 多指令多数据 |

*在并行计算中，SIMD是一种很常见的方式。*

> 常见名词

* Task：任务
* Parallel Task：并行任务，该任务可以由多个并行计算的方式解决的**单个任务**。
* Serial Execution：串行执行
* Parallel Execution：并行执行
* Shared Memory：共享存储
* Distributed Memory：分布式存储
* Communications：通信
* Synchronization：同步
* Granularity：粒度
* Observed Speedup：加速比，对比Baseline，并行计算能获得的性能提升。
* Parallel Overhead：并行开销
* Scalability：可扩展性

#### 存储器架构

* 共享存储
* 分布式存储
* 分布式共享存储

#### 并行编程模型

* 共享存储模型
* 线程模型
* 消息传递模型
* 数据并行模型

具体实例：`OpenMP`，`MPI`，`Single Program Multiple Data(SPMD)`，`Multiple Program Multiple Data(MPMD)`。

> Amadahl's Law

Amadahl's Law的程序可能的加速比取决于可以被并行化的部分。
$$
\text{speedup} = \frac{1}{1-p}\\
p代表可以被并行化的部分\\
\text{speedup} = \frac{1}{\frac{P}{N} + S}\\
P代表并行部分，N代表处理器数，S代表串行部分。
$$

### CUDA开发环境搭建和工具配置

由于该教程是14年的教程，环境配置和如今已经完全不同，这部分将会在我的[博客](https://caixiongjiang.github.io/blog/2023/hpc/ubuntu%E5%AE%89%E8%A3%85cuda%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83/)上呈现。

### GPU体系架构概述

#### GPU架构

GPU是一个异构的多处理器芯片，为图形图像处理进行优化。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img52.jpg)

`Shader core`代表渲染器的核心，其组成是一个基本的ALU计算单元。

将GPU的执行单元拎出来，其结构如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img53.jpg)

从上到下分别是`地址译码单元`、`计算核心`、`执行上下文`。

现代的GPU中的ALU都共享指令集，那么为了提高效率，我们一般就通过增大ALU和SIMD来增进并行性，方便向量化的操作。

GTX480的单个架构的SM（流多处理器），一个流多处理器包含32个CUDA核心（CUDA核心本质为一个ALU）：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img54.jpg)

整个GTX480显卡可以同时承载23000个`CUDA片元`（也叫CUDA线程）。

#### GPU的存储架构

> CPU存储架构

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img55.jpg)

> GPU的存储架构

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img56.jpg)

GPU的存储是交给了专门的较大的存储，`显存`，带宽可以达到150GB/s。`访存的带宽资源`是非常宝贵的资源！

> 看一个带宽测试的例子

$$
A、B、C为三个矩阵。\\
计算 D = A\times B + C
$$

上述计算需要5个步骤：

1.Load input A[i]

2.Load input B[i]

3.Load input C[i]

4.计算A[i] * B[i] + C[i]

5.存储结果到D [i]中

如果这时候的矩阵是非常大的矩阵，那么上述几个步骤，最大的开销则发生在前3步，那么计算的效率是非常低的，这里的瓶颈是带宽。

现代的GPU通过缓存来缓解带宽受限的情况：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img57.jpg)



总结一下GPU是异构、众核的处理器，针对吞吐优化。

> 高效的GPU任务具备的条件

* 具有成千上万的独立工作
  * 尽量利用大量的ALU计算单元
  * 大量的片元（CUDA thread）切换掩藏延迟
* 可以共享指令流
  * 适用于SIMD处理
* 最好是计算密集的任务
  * 通信和计算开销比例合适
  * 不要受制于访存带宽

### CUDA/GPU编程模型

#### CPU和GPU互动模型

> cpu和gpu的交互

* cpu和gpu有各自的物理内存空间
* 它们之间通过PCIE总线相连（8G/s～16G/s）
* 交互的开销较大

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img59.jpg)

> gpu的存储架构

![](/Users/caixiongjiang/Library/Application Support/typora-user-images/image-20230721093954435.png)

> 访存速度的高低

从高到低，DRAM代表物理位置在显存中：

* Register（寄存器）- 延迟约为1个时钟周期
* Shared Memory（共享存储）-  延迟约为1个时钟周期
* Local Memory（DRAM）- 在每一个私有的线程装配的一个memory，如果寄存器放不下则装入这里，（在物理上放在显存中）速度相对较慢。
* Global Memory（DRAM）- 真正的显存，速度相对较慢
* Constant Memory（DRAM）- 相对Global和Local更慢
* Texture Memory（DRAM）- 相对Global和Local更慢
* Instruction Memory（invisible， DRAM）

#### GPU的线程组织模型

GPU的线程模型主要就是`网格`、`块`、`线程`，如下图：

![](/Users/caixiongjiang/Library/Application Support/typora-user-images/image-20230721100425159.png)

*注意上述示意图为软件逻辑上的组织，并不代表硬件层次。*

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img60.jpg)

WARP代表线程束，是一些线程的组成，一般由32个连续的线程组成。

**一个Kernel（通常是在GPU上执行的单个程序）具有大量的线程，这些线程被划分为多个线程块（Blocks），一个Block内部共享`Shared Memory`，这些Block可以进行同步。**

**线程和线程块具有唯一的标识。**

#### GPU存储模型

> gpu内存和线程的关系

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img61.jpg)

* 每个线程有一个私有的`Local Memory`
* 每个Block有多个线程，它们共享`Shared Memory`
* 整个设备拥有一个`Global Memory`
* 主机端的存储器可以跟不同的设备进行交互

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img62.jpg)

上图代表了GPU端Block内部的访问流程。

#### 编程模型

常规的GPU用于处理图形图像，操作于像素，每个像素的操作都类似，可以应用SIMD（单指令多数据）。

SIMD可以认为是数据并行的分割：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img63.jpg)

在GPU中，它被称为是SIMT：

**通过大量的线程模型获得高度并行，线程切换获得延迟掩藏，多个线程执行相同的指令流，GPU上大量线程承载和调度。**

> CUDA编程模式：Extended C

* 修饰词：global，device，shared，local，constant
* 关键词：threadIdx，blockIdx
* Intrinsics：__syncthreads
* 运行期API：Memory，symbol，execution，management
* 函数调用：例子`convolve<<<100, 10>>> (参数)`

> CUDA函数声明

|                                   | 执行位置 | 调用位置 |
| --------------------------------- | :------: | :------: |
| \_\_device\_\_ float DeviceFunc() |  Device  |  Device  |
| \_\_global\_\_ void kernelFunc()  |  Device  |   Host   |
| \_\_host\_\_ float HostFunc()     |   Host   |   Host   |

 几个需要理解的点：

* 入口函数，CPU上调用，GPU上执行
* 必须返回void
* \_\_device\_\_ 和\_\_host\_\_可以同时使用

### CUDA编程（1）

> CUDA术语

* Host：主机端，通常指cpu
* Device：设备端，通常指gpu
* Host和Device有各自的存储器
* Kernel：数据并行处理函数，也就是所谓的`核函数`，类似于OpenGL的`shader`
* Grid：一维或多维线程块
* Block：一组线程

**一个Grid的的每个Block的线程数都是一样的，Block内部的每个线程可以进行同步，并访问共享存储器。**

> 线程的层次

一个Block可以是一维，二维，甚至是三维的。（例如，索引数组、矩阵、体）

* 一维Block：Thread ID == Thread Index
* 二维Block：（Dx，Dy）

Thread ID of index(x, y) == x + y Dx

* 三维Block：（Dx，Dy，Dz）

Thread ID of index(x, y, z) == x + y Dx + z Dx Dy



看一个代码的例子：

```c
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N]) 
{
  	int i = threadIdx.x;
  	int j = threadIdx.y;
  	C[i][j] = A[i][j] + B[i][j];
}

int main() {
  	int numBlocks = 1;
  	// 对于dim3的类型，如果第三个参数不传，默认为1，这样就变成了一个二维的Block
  	dim3 threadsPerBlock(N, N);
  	//第一个参数代表1个 Thread Block，第二个参数代表一个2D的Block（相当于排列的时候变成了行和列）
  	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```

**一个线程块里的线程位于相同的处理器核，共享所在核的存储器。**

* 块索引：blockIdx
* 块的线程数：blockDim（一维，二维，三维）

使用多个Block进行矩阵的Add：

```c
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N]) 
{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
  	int j = blockIdx.y * blockDim.y + threadIdx.y;
  	if (i < N && j << N) {
      	C[i][j] = A[i][j] + B[i][j];
    }
}

int main() {
  	dim3 threadsPerBlock(16, 16);
  	dim3 numBlocks(N / threadsPerBlock.x , N / threadsPerBlock.y);
  	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```

> 线程层次结合gpu存储层次加深对代码操作的硬件理解

* Device Code：

-- 读写每个线程的的寄存器

-- 读写每个线程的local memory

-- 读写每个线程块的shared memory（线程块内线程共享）

-- 读写每个grid的global memory（不同线程块的所有线程共享）

-- 只读每个grid的constant memory（每个grid的步态变化的独立空间）

* Host Code：

-- 主机端只能读写global和constant memory，global memory代表全局的存储器，constant memory代表常量的存储器。

> CUDA内存传输

* cudaMalloc()：在设备端分配global memory
* cudaFree()：释放存储空间

分配的代码示例：

```c
float *Md;
int size = Widch * Width * sizeof(float);
//当前指针是指向设备上的存储空间
cudaMalloc((void**)&Md, size);
...
cudaFree(Md);
```

* cudaMemcpy()：内存传输，Host->Host, Host->Devicel, Device->Device, Device->Host

示例程序：

```c
// Md和Pd都是在device端的地址
cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
```

> 矩阵相乘



### CUDA编程（2）

### CUDA编程（3）

### CUDA程序分析和调试工具

### CUDA程序基本优化

### CUDA程序深入优化