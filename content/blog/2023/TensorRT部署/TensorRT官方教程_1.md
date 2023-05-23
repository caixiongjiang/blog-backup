---
title: "NVIDIA官方教程：第一节"
date: 2023-05-17T18:18:05+08:00
lastmod: 2023-05-17T09:19:06+08:00
draft: False
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/tensorrt_title.jpg"
description: "TensorRT学习笔记"
tags:
- Deep_learning
categories:
- 深度学习
series:
- TensorRT部署
comment : true
---

## TensorRT简介

* TensorRT是用于高效实现已经训练好的深度学习模型的推理过程的SDK。
* TensorRT内含`推理优化器`和`运行时环境`。
* TensorRT使Deep Learning模型能以更高的吞吐量和更低的延迟运行。
* 包含C++和python的API，完全等价可以混用。

一些reference：
TensorRT文档:[https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
C++ API文档:[https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)
python API文档:[https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
TensorRT下载:[https://developer.nvidia.com/nvidia-tensorrt-download](https://developer.nvidia.com/nvidia-tensorrt-download)
该教程配套代码:[https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook)

### TensorRT基本特性

#### TensorRT的基本流程

[示例代码](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/01-SimpleDemo/TensorRT8.0/main-cudart.py)

基本流程：
* 构建期：
    * 建立Buider（构建引擎器）
    * 创建Network（计算图内容）
    * 生成SerializedNetwork（网络的TRT内部表示）
* 运行期：
    * 建立Engine和Context
    * Buffer相关准备（Host端 + Device端 + 拷贝操作）
    * 执行推理（Execute）

#### TensorRT工作流
* 使用框架自带的TRT接口(TF-TRT、Torch-TensorRT)
    * 简单灵活、部署仍然在原框架中，无需书写插件。
* 使用Parser(TF/Torch/... -> ONNX -> TensorRT)
    * 流程成熟，ONNX通用性好，方便网络调整，兼顾性能效率
* 使用TensorRT原生API搭建网络
    * 性能最优，精细网络控制，兼容性最好
