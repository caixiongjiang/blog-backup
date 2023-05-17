---
title: "TensorRT动态Batch和动态宽高的实现"
date: 2023-05-16T18:18:05+08:00
lastmod: 2023-05-16T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/tensorrt_title.jpg"
description: "学习一下TensorRT中动态宽高和动态Batch的实现方式。"
tags:
- Deep_learning
categories:
- 深度学习
series:
- TensorRT部署
comment : true
---

### TensorRT之动态Batch和动态宽高

#### 动态Batch

该特性的需求主要源于`TensorRT`编译时对batch的处理，若静态batch则意味着无论你有多少图，都按照固定大小batch推理。耗时是固定的。

实现动态Batch的注意点：
> 1.onnx导出模型是，注意view操作不能固定batch维度数值，通常写-1。
> 2.onnx导出模型是，通常可以指定`dynamic_axes`（通常用于指定动态维度），实际上不指定也没关系。

#### 动态宽高

该特性需求来自`onnx`导出时指定的宽高是固定的，`TensorRT`编译时也需要固定大小引擎，若你想得到另外一个不同大小的`TensorRT`引擎（一个eng模型只能支持一个输入分辨率）时，就需要动态宽高的存在。而直接使用`TensorRT`的动态宽高（一个eng模型能支持不同输入分辨率的推理）会带来不必要的复杂度，所以使用中间方案：在编译时修改`onnx`输入实现相对动态（一个onnx模型，修改参数可以得到不同输入分辨率大小的eng模型），避免重回`Pytorch`再做导出。

实现动态宽高的注意点：
> 1.不建议使用`dynamic_axes`指定Batch以外的维度为动态，这样带来的复杂度太高，并且存在有的layer不支持。
> 2.如果`onnx`文件已经导出，但是输入的shape固定了，此时希望修改`onnx`的输入shape：
> &nbsp;&nbsp;&nbsp;步骤一：使用`TRT::compile`函数的`inputsDimsSetup`参数重新定义输入的shape。
> &nbsp;&nbsp;&nbsp;步骤二：使用`TRT::set_layer_hook_reshape`钩子动态修改reshape的参数实现适配。

动态Batch demo：
```c++
int max_batch_size = 5;
/** 模型编译，onnx到trtmodel **/
TRT::compile(
    TRT::Model::FP32,
    max_batch_size,               //最大batch size
    "model_name.onnx",
    "model_name.fp32.trtmodel"
);

/** 加载编译好的引擎 **/
auto infer = TRT::load_infer("model_name.fp32.trtmodel");

/** 设置输入的值 **/
/** 修改input的0维度为1，最大可以是5 **/
infer->input(0)->resize_single_dim(0, 2);
infer->input(0)->set_to(1.0f);

/** 引擎进行推理 **/
infer->forward();

/** 取出引擎的输出并打印 **/
auto out = infer->output(0);
INFO("out.shape = %s", out->shape_string());
```

动态宽高 demo：
```c++
/** 这里的动态宽高是相对的，仅仅调整onnx输入大小为目的 **/
static void model_name() {
    //钩子函数
    TRT::set_layer_hook_reshape([](const string& name, const vector<int64_t>& shape)->vector<int64_t>{
        INFO("name: %s, shape: %s", name.c_str(), iLogger::join_dims(shape).c_str());
        return {-1, 25}; //25代表5*5的宽高，-1代表的是Batch的维度
    });

    /** 模型编译 **/
    TRT::compile(
        TRT::Model::FP32,
        1,
        "model_name.onnx",
        "model_name.fp32.trtmodel",
        {{1, 1, 5, 5}}             //对输入的重定义
    );

    auto infer = TRT::load_infer("model_name.fp32.trtmodel");
    auto model = infer->output(0);
    INFO("out.shape = %s", out->shape_string());
} 
```

