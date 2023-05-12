---
title: "TensorRT部署方案介绍"
date: 2023-05-11T18:18:05+08:00
lastmod: 2023-05-11T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/tensorrt_title.jpg"
description: "介绍一些TensorRT部署实现的方案和优缺点，并正确使用onnx使用c++中推理！"
tags:
- Deep_learning
categories:
- 深度学习
series:
- TensorRT部署
comment : true
---

### TensorRT部署方案介绍

#### 为每个模型写硬代码（c++）

仓库地址:[https://github.com/wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/img112.jpg)

看图可知，也就是通过c++源代码直接调用`TensorRT API`，对每个不同的网络模型进行重写，再调用`TensorRT Builder`生成`TensorRT Engine`。

原理：
> 1.使用作者定义的gen_wts.py的存储权重。
> 2.使用C++硬代码调用TensorRT C++API构建模型，加载gen_wts.py产生的权重组成完整模型。

优点：
> 1.可以控制每个layer的细节和权重，直接面对TensorRT API。
> 2.这种方案不存在算子问题，如果存在不支持的算子可以自行增加插件。

缺点：
> 1.新模型需要对每个layer重写C++代码。
> 2.过于灵活，需要控制的细节多，技能要求很高。
> 3.部署时无法查看网络结构进行分析和排查。

#### 为每个算子写Converter

仓库地址:[https://github.com/NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/img113.jpg)

原理：
> 1.作者为每一个算子（比如ReLU，Conv等），为每一个操醉的forward反射到自定义函数
> 2.通过反射torch的forward操作获取模块的权重，调用Python API接口实现模型

优点：
> 1.直接集成了Python、Pytorch，可以实现Pytorch模型到TensorRT模型的无缝转换。

缺点：
> 1.提供了Python的方案并没有提供c++的方案。
> 2.新的算子需要自己实现converter，需要维护新的算子库
> 3.直接用Pytorch赚到tensorRT存储的模型是TensorRT模型，如果跨设备必须在设备上安装pytoch，灵活度差，不利于部署。
> 4.部署时无法查看网络结构进行分析和排查。

#### 基于ONNX路线提供C++和Python的接口

仓库地址:[https://github.com/shouxieai/tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/img114.jpg)

原理：
> 通过Pytorch官方和NVIDIA官方对`torch->onnx`和`onnx->TRT`算子库的支持。

优点：
> 1.集成工业级推理方案，支持TensorRT从模型导出到应用到项目中的全部工作
> 2.案例有YoloV5、YoloX、AlphaPose、RetinaFace、DeepSORT等，每个应用均为高性能工业级。
> 3.具有简单的模型导出方法和onnx问题的解决方案
> 4.具有简单的模型推理接口，封装tensorRT细节。支持插件。
> 5.依赖onnx，有两大官方进行维护。

缺点：
> onnx存在各种兼容性问题。

### 如何正确导出onnx（避坑指南）

* 1.对于任何用到shape、size返回值的参数时，例如`tensor.view(tensor.size(0), -1)`，避免直接使用tensor.size的返回值，而是加上int转换，`tensor.view(int(tensor.size(0)), -1)`。（这里的tensor值的是一个具体的张量）
* 2.对于`nn.Upsample`或者`nn.fucntional.interpolate`函数，使用`scale_factor`指定倍率，而不是使用size参数指定大小。
* 3.对于`reshape`、`view`操作时，-1请指定到batch维度，其他维度计算出来即可。
* 4.`torch.onnx.export`指定`dynamic_axes`参数，并只指定batch维度，不指定其他维度。我们只需要动态batch，相对动态的宽高有其他方案。

> 这些做法的必要性体现在简化过程的复杂度，去掉gather、shape类的节点。

Example：
```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=True)
        self.conv.weight.data.fill_(0.3)
        self.conv.bias.data.fill_(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        """
        x.size(0)代表batch的大小
        x.numel()代表元素的数量
        那么int(x.numel() // x.size(0))就代表剩下维度的大小
        原则是batch的维度需要变为自动推理，这样可以实现动态batch的方案
        """
        # return x.view(x.size(0), -1) 
        return x.view(-1, int(x.numel() // x.size(0)))

model = Model().eval()
x = torch.full((1, 1, 3, 3), 1.0)
y = model(x)

torch.onnx.export(
    model, (x, ), "model_name.onnx", verbose=True
)
```

c++模型编译：
```c++
/** 模型编译，onnx到trtmodel**/
TRT::compile(
    TRT::Model::FP32,      /** 模式：fp32, fp16, int8 **/
    1,                     /** 最大batch size **/
    "model.onnx",          /** onnx文件 输入**/
    "model.fp32.trtmodel"  /** trt模型文件，输出 **/
);

/** 加载编译好的引擎 **/
auto infer = TRT::load_infer("model.fp32.trtmodel");

/** 设置输入的值 **/
infer->input(0)->set_to(1.0f);

/** 引擎进行推理 **/
infer->forward();

/** 取出引擎的输出并打印 **/
auto out = infer->output(0);
for(int i = 0; i < out->channel(); ++i)
    INFO("%f", out->at<float>(0, i));
```
