---
title: "AscendCL快速入门"
date: 2023-08-08T18:18:05+08:00
lastmod: 2023-08-09T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/huawei_title.jpg"
description: "AscendCL快速入门，用于实际推理部署。"
tags:
- AscendCL编程
categories:
- HPC
series:
- 华为昇腾部署
comment : true
---


### AscendCL快速入门

ACL（Ascend Computing Language，昇腾计算语言），是华为提供的一套用于昇腾系列处理器上进行并行加速计算的API。基于这套的API，可以管理和使用昇腾软硬件计算资源，并进行机器学习相关计算。**当前ACL提供了C/C++和Python的编程接口，这和TensorRT提供的接口一致。**

鉴于之前使用TensorRT部署的语义分割模型使用的是Python接口，这次在昇腾处理器上准备使用C++进行处理。

#### ACL的主要功能

* 加载离线模型进行推理
* 加载单个算子做计算
* 对图形图像的数据进行预处理

#### 使用ResNet50进行图片推理

准备工作：需要准备一张素材图片和一个转化好的`ResNet50.om`离线模型。

先用伪代码看一下`main函数`需要做什么：

```c++
int main() {
  	char *picturePath = "./data/dog1_1024_683.bin";
  	char *modelPath = "./model/resnet50.om";
  	// 准备计算需要的资源	
  	InitResource();
  	LoadModel(modelPath);
  	LoadPicture(picturePath);
  	inference();
  	PrintResult();
  	// 卸载模型
  	UnloadModel();
  	// 卸载图片
  	UnloadPicture();
  	// 销毁资源
  	DestroyResource();
}

```

#### AscendCL接口函数

> ACL的初始化和去初始化

在调用ACL的任何接口之前，首先要做ACL的初始化。初始化的代码如下：

* `aclInit(nullptr);`:

这个接口调用会帮您准备好ACL的运行时环境。其中调用时传入的参数是一个**配置文件在磁盘上的路径**，这里暂时不需要关注。

* `aclFinalize();`：

去初始化非常机简单，在确定完成了ACL的所有调用之后，要做去初始化操作，接口调用十分简单。

> 申请和释放计算资源

使用昇腾处理器提供的加速计算，首先在运行时申请计算资源。

* `aclrtSetDevice(0);`：

这个接口会告诉运行时环境我们使用的设备，或者更具体一点在使用哪个芯片。**但是需要注意的是，芯片和我们传入的编号之间并没有物理上的一一对应关系。**

* `aclrtResetDevice(0);`：

这里传入的设备编号和申请设备的时候使用的是同一个编号。调用这个接口会将对应设备上的所有计算资源进行复位。**如果此时该设备上还有未完成的计算，则会等待该设备上的所有计算过程结束再复位设备。**

> 加载数据

要使用NPU进行加速计算，首先要申请能够被NPU直接访问到的专用内存。首先需要区分两个概念：

* Host：Host指与Device相连的`X86`服务器、`ARM`服务器，会利用Device提供的NN(Neural-Network)计算能力，完成业务
* Device：Device指安装了芯片的硬件设备，利用PCIe接口与Host侧连接，为Host提供NN计算能力

**这和NVIDIA里面的概念是相同的，只能说周斌博士，不愧是你，直接把CUDA搬过来了是吧！**

简单来说，我们的数据需要从Host端进行加载，即Host侧内存，随后将其拷贝到Device侧内存，才能进行计算。计算后的结果需要传回Host侧才能进行使用。这和模型的训练是相同的。

* 申请Host侧内存：

  * `void *pictureMemHostPtr;`// **host侧内存指针**
  * `size_t pictureMemPtrSize;`// **host侧内存大小**
  * `aclrtMallocHost(&pictureMemHostPtr, pictureMemPtrSize);`// **申请host侧内存**
  * 随后即可使用`pictureMemHostPtr`指向的内存来暂存推理输入数据。

* 申请Device侧内存：

  * `void *pictureMemDevicePtr;`// **device侧内存指针**
  * `aclrtMalloc(&pictureMemDevicePtr, pictureMemPtrSize, ACL_MEM_MALLOC_HUGE_FIRST);`// **申请device侧内存**
  * 这里我们分配了跟Host侧同样大小的内存，准备用于Device侧存放推理数据。本接口最后一个参数`ACL_MEM_MALLOC_HUGE_FIRST`是内存分配规则。

* 把数据从Host侧拷贝到Device侧：

  * `aclrtMemcpy(pictureMemDevicePtr, pictureMemPtrSize, pictureMemHostPtr, pictureMemPtrSize, ACL_MEMCPY_HOST_TO_DEVICE);`:

    参数的顺序是：目的内存地址，目的内存最大大小，源内存地址，拷贝长度，拷贝方向。（支持host->host, host->device, device->host, device->device）**哈哈哈，CUDA换层皮以为我看不出来是吧！**	

* 使用完需要释放申请过的内存：

  * `aclrtMallocHost->aclrtFreeHost;`
  * `aclrtMalloc->aclrtFree;`

> 加载模型

既然要调用模型进行推理，首先要把模型加载进来。

ACL提供了模型加载和内存管理方式，这里只选取相对简单的一种，即从磁盘上加载离线模型，并且加载后的模型内存由ACL自动管理：

* `char* modelPath = './resnet50.om';`// 模型文件在磁盘上的路径
* `unit32_t modelid;`// 加载后生成的modelId，全局唯一
* `aclmdlLoadFromFile("./resnet50.om", &modelid);`// 加载模型
* 这个`modelid`在使用模型进行推理，以及卸载模型的时候还需要用到

卸载模型的接口：

* `aclmdlUnload(modelid);`
* 在c和c++层面来说，任何使用了的资源，加载了任何素材，使用完成之后都需要进行销毁和卸载。

> 准备推理所需数据结构

模型推理所需的输入输出数据，是通过一种特殊的数据结构来组织的，这种数据结构叫`dataSet`，即所有的输入，组成了一个`dataSet`，所有的输出组成了一个dataSet。**如果模型的输入不止一个，那么所有的输入集合叫`dataSet`，其中每一个输入叫`dataBuffer`。**

* 创建`dataBuffer`：

前面申请的Device侧的内存，并且已经把数据传过去了。当时的Device侧的内存地址为`pictureMemDevicePtr`，内存的长度为`pictureMemPtrSize`。

使用上面的两个对象创建一个`dataBuffer`:

`aclDataBuffer buffer = aclCreateDataBuffer(pictureMemDevicePtr, pictureMemPtrSize);`

* 创建`dataSet`：

`aclmdlDataSet inputDataset = aclmdlCreateDataset();`

推理所需要的输入为一个即可，我们只需要一个`dataBuffer`。将创建好的`databuffer`放入`dataSet` 中：

`aclmdlAddDatasetBuffer(inputDataset, buffer);`有输入就必须有输出，输出的数据结构是dataSet，变量命名为`outputDataSet`

* 用完数据结构之后需要及时销毁：

`aclCreateDataBuffer->aclDestroyDataBuffer;`

`aclmdlCreateDataset->aclmdlDestroyDataset;`

> 推理

我们已经准备好了`modelid`,`inputDataset`,`outputDataset`。

最终的推理只需要一行代码：

`aclmdlExecute(modelid, inputDataset, outputDataset);`

**这是一个同步接口，线程会阻塞在这里直到推理结束。**推理结束后就可以提取outputDataset中的数据进行使用了。

> 整体流程

` 初始化`->`申请计算资源`->`在Host侧加载图片`->`将图片传递到Device侧`->`加载模型`->`准备输入输出数据结构`->`推理`->`将输出传回Host侧，使用模型输出`->`销毁输入，输出`->`销毁模型`->`销毁Device和Host侧内存`->`销毁计算资源`->`去初始化`

