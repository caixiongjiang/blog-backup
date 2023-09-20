---
title: "算能（Sophon）sdk部署：以SE5为例"
date: 2023-09-06T18:18:05+08:00
lastmod: 2023-09-12T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E5%AE%9E%E4%B9%A0/img_sophgo.jpg"
description: "国产化sdk（算能）算法迁移部署"
tags:
- Deep_learning
categories:
- AI部署
series:
- 国产化sdk部署
comment : true
---


## Sophon（算能）：SE5 SDK部署

算能官方提供的开发者文档：[https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/SophonSDK_doc/zh/html/index.html](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/SophonSDK_doc/zh/html/index.html)

### 设备信息

目前拿到的盒子型号为`SE5`，主要参数如下：

* 4GB A53（cpu）专用
* 4GB TPU（张量处理器）专用(BM)
* 4GB VPU（编解码）专用

盒子与加速卡不同，其走的Soc模式，其`Host Memory`代表芯片主控上的内存，而`Device Memory`则是代表划分给`TPU/VPU`的内存。

> 算力信息

* AI算力：INT8为`17.6 TOPS`，FP32为`2.2 TFLOPs`。

* 支持的视频解码能力：`H.264 & H.265: 1080P @960fps`

* 支持的视频解码分辨率：`8192 * 8192 / 8K / 4K / 1080P / 720P / D1 / CIF`

* 支持的视频编码能力： `H.264 & H.265: 1080P @50fps`

* 视频编码分辨率：`4K / 1080P / 720P / D1 / CIF`
* 图片解码能力：`JPEG:480张/秒 @1080P`

### 环境配置（Soc）

tpu-nntc环境供用户在x86主机上进行模型的编译量化，提供了libsophon环境供用户进行应用的开发部署。PCIe用户可以基于tpu-nntc和libsophon完成模型的编译量化与应用的开发部署； **SoC用户可以在x86主机上基于tpu-nntc和libsophon完成模型的编译量化与程序的交叉编译，部署时将编译好的程序拷贝至SoC平台（SE微服务器/SM模组）中执行。**

#### 开发环境配置

因为最后运行的环境为`soc`平台，以及需要完成对摄像头的拉流、模型推理。开发平台需要的开发环境为交叉编译环境。需要先在`x86_64`架构上的主机配置交叉编译环境。

环境要求：Linux系统（Ubuntu）、x86_64架构、能连接外网（互联网）

* 下载交叉编译工具链：

```shell
$ apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu 
```

* 验证工具链：

```shell
$ which aarch64-linux-gnu-g++
# 终端输出 /usr/bin/aarch64-linux-gnu-g++
```

* 配置pipeline交叉编译需要的`libsophon`，`sophon-opencv`，`sophon-ffmpeg`:

具体参考[https://github.com/sophgo/sophon-pipeline/blob/release/docs/docs_zh/arm_soc.md](https://github.com/sophgo/sophon-pipeline/blob/release/docs/docs_zh/arm_soc.md)

但是按照这个教程集成的`soc-sdk`有问题，联系了厂商那边，给出了它们配置的`soc-sdk`包，看到里面除了上面需要的库，还增加了厂商封装的AI加速库`sail`。

这样开发环境就配置好了!

#### 运行环境配置

SE5属于soc平台，内部已经集成了`libsophon`，`sophon-opencv`，`sophon-ffmpeg`环境，在`/opt/sophon/`下。

配置环境变量即可：

```shell
$ vim ~/.bashrc
# 加入以下内容保存 x.y.z代表版本，不知道的可以自己进目录看一下
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv_<x.y.z>/opencv-python 
$ source ~/.bashrc
```

### 网络模型移植

> BModel

* `BModel`是一种面向算能TPU处理器的深度神经网络模型文件格式，其中包含目标网络的权重（weight）、TPU指令流等等。

* **`Stage`是支持将同一个网络的不同`batch size`的模型combine为一个BModel；同一个网络的不同`batch size`的输入对应着不同的`stage`，推理时BMRuntime会根据输入shape的大小自动选择相应stage的模型**。也支持将不同的网络combine为一个BModel，通过网络名称来获取不同的网络。

* **动态编译和静态编译：** 支持模型的动态编译和静态编译，可在转换模型时通过参数设定。动态编译的`BModel`，在Runtime时支持任意小于编译时设置的shape的输入shape；静态编译的`BModel`，在Runtime时只支持编译时所设置的shape。

> 模型转化工具链

算丰系列TPU平台只支持BModel模型加速。所以需要先将模型转化为BModel。

常用的`Pytorch`和`ONNX`模型的转化为BModel都有版本的要求：

* ONNX要求：onnx==1.7.0（Opset Version == 12）onnxruntime == 1.3.0 protobuf >= 3.8.0

* Pytorch要求：pytorch >= 1.3.0, 建议1.8.0

`TPU-NNTC工具链`帮助实现模型迁移。对于BM1684平台来说，它既支持float32模型，也支持int8量化模型。其模型转换流程以及章节介绍如图:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E5%AE%9E%E4%B9%A0/img4.jpg)

*如果需要运行in8 BModel，需要先准备`量化数据集`、将原始模型转换为fp32 UModel、再使用量化工具量化为int8 UModel、最后使用`bmnetu`编译为int8 BModel。*

>  具体的`INT8`流程如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E5%AE%9E%E4%B9%A0/img5.jpg)

* Parse-Tools：

解析各深度学习框架下已训练好的网络模型，生成统一格式的网络模型文件—umodel， 支持的深度学习框架包括： Caffe、TensorFlow、MxNet、PyTorch、Darknet、ONNX以及PaddlePaddle。

* Calibration-Tools：

分析float32格式的umodel文件，默认基于熵损失最小算法（可选MAX等其他算法），将网络系数定点化成8bit，最后 将网络模型保存成int8格式的umodel文件。

* U-FrameWork：

自定义的深度学习推理框架，集合了各开源深度学习框架的运算功能，提供的功能包括：

1. 作为基础运算平台，为定点化提供基础运算。

2. 作为验证平台，可以验证fp32，int8格式的网络模型的精度。

3. 作为接口，通过bmnetu，可以将int8umodel编译成能在SOPHON运算平台上运行的bmodel。

**更为具体的，如量化数据集的格式要求、如何进行精度测试和调优可以详细参考官方开发文档！**

### 算法移植

**对于基于深度学习的视频/图片分析任务来说，通常都包括如下几个步骤：**

1. 视频/图片解码
2. 输入预处理
3. 模型推理
4. 输出后处理
5. 视频/图片编码

实际任务中，算法往往还会包含多个不同神经网络模型，因此，步骤2-4会根据需要反复执行多次。

> 硬件加速支持

SophonSDK提供了`视频/图片解码`、`输入预处理`、`模型推理`、`输出后处理`、`视频/图片编码`五个阶段的加速。

为了提高算法的效率，在编写代码和使用接口的时候需要注意这几个方面：

* 内存零copy
* 申请物理连续内存
* 将多个预处理步骤进行合并
* 凑4batch进行推理

> 编程接口

目前支持C/C++/Python三种编程接口。BMRuntime模块支持C/C++接口编程，BMCV、BMLib支持C接口编程；Python/C++编程接口基于SAIL库实现。**目前， SAIL 模块中所有的类、枚举、函数都在 “sail” 命名空间下。**

常见算法的开发示例可以参考官方提供的demo:[https://github.com/sophgo/sophon-demo/](https://github.com/sophgo/sophon-demo/)

常见的处理视频流的开发示例可以参考官方提供的pipeline:[https://github.com/sophgo/sophon-pipeline](https://github.com/sophgo/sophon-pipeline)

### 编译运行实例

参考的代码:[https://github.com/sophgo/sophon-pipeline](https://github.com/sophgo/sophon-pipeline)

* 运行环境编译整个repo的代码：

```shell
$ git clone https://github.com/sophgo/sophon-pipeline
$ cd sophon-pipeline
# 编译需要SoC平台上运行的程序 soc-sdk填入开发环境配置阶段配置好的soc-sdk包的绝对地址
$ ./tools/compile.sh soc ${soc-sdk}
# 这样我们就可以在`sophon-pipeline/release`中找到所要所要算法生成的可执行文件
# 以yolov5s为例，其路径为`sophon-pipeline/release/yolov5s_demo/`下
```

* 需要拷贝的文件包括：`模型`(必须是Bmodel的格式)、流视频地址或者`.h264`或者`.h265`格式的视频文件、在编译生成的目录下的`json配置文件`和包含可执行程序的`soc`文件夹。
* 将其全部放到盒子内部的同一目录下 ，修改`json`配置文件:

主要配置两个参数，第一个是视频的地址、第二个是模型的地址，其他可以按需配置。具体配置含义信息可参考[https://github.com/sophgo/sophon-pipeline/blob/release/docs/docs_zh/yolov5.md](https://github.com/sophgo/sophon-pipeline/blob/release/docs/docs_zh/yolov5.md)

*  注意修改可执行文件的权限，然后运行demo：

```shell
$ ./soc/yolov5s_demo --config=./cameras_yolov5.json
# 打印下面的信息就说明成功了
...
[2022-10-13:16:00:26] total fps =nan,ch=0: speed=nan
[2022-10-13:16:00:27] total fps =24.0,ch=0: speed=24.0
[2022-10-13:16:00:28] total fps =24.0,ch=0: speed=24.0
[2022-10-13:16:00:29] total fps =25.4,ch=0: speed=25.4
...
```

### RTSP视频流输出画面

算能为`sophon_pipeline`准备了可视化工具`pipeline_client`。由于本次是在ubuntu上启动客户端，编译相对简单。

* 下载`pipeline_client`代码：

```shell
git clone https://github.com/sophon-ai-algo/pipeline_client.git
```

* 下载相关依赖：

```shell
 sudo apt install qtbase5-dev
 sudo apt install libopencv-dev
 sudo apt install ffmpeg
 sudo apt install libavfilter-dev
```

* 在Linux下能找到默认的opencv库位置，不需要修改CMakeLists.txt，Windows端需要按照说明操作：[https://github.com/sophon-ai-algo/pipeline_client](https://github.com/sophon-ai-algo/pipeline_client)
* 编译：

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ..
$ ./build/bin/pipeline_client # 启动客户端
```

系统会出现类似于如下的客户端画面：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E5%AE%9E%E4%B9%A0/img6.jpg)

* RTSP需要配合一个`rtsp server`使用，先推流到`rtsp server`，`pipeline_client`再拉流获取画面。使用官网推荐的`rtsp server`:[https://github.com/bluenviron/mediamtx/releases](https://github.com/bluenviron/mediamtx/releases)，注意Linux x86_64机器需要选择下面这个包：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E5%AE%9E%E4%B9%A0/img7.jpg)

* 现在`rtsp server`，`pipeline_client`，`sophon_pipeline`都已经准备完毕，接下来就可以测试输出画面了（注意`sophon_pipeline`在盒子端，`rtsp server`和`pipeline_client`在交叉编译端）：
  * 在交叉编译端启动`rtsp server`，解压下好的文件，并执行`./mediamtx`，默认端口为`8544`，如果修改端口，则需要修改其中的配置文件`mediamtx.yml`文件，必须要确保启动的端口没有被其他应用占用。
  * 运行`sophon-pipeline`例程程序，推流到指定rtsp地址：将编译好的配置文件`cameras_yolov5.json`的参数配置改为`rtsp://${ip}:8544/abc`，这里的ip是指交叉编译机器的ip地址。
  * 运行`pipeline_client`，指定拉流地址为`rtsp://${ip}:8554/abc`，如果`pipeline_client`和`rtsp server`在一个机子上，ip可以直接用`127.0.0.1`或者`//localhost`代替。
  * 稍等片刻，则可以看见画面。