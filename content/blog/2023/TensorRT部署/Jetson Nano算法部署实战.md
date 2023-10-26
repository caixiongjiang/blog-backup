---
title: "Jetson Nano算法部署实战"
date: 2023-05-31T18:18:05+08:00
lastmod: 2023-06-02T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/tensorrt_title.jpg"
description: "在边缘计算设备上完成算法部署"
tags:
- Deep_learning
categories:
- 深度学习
series:
- TensorRT部署
comment : true
---

### Jetson Nano算法部署实战

#### 硬件和环境准备
* Jetson Nano B01开发板
* CSI摄像头模块
* Wifi模块

> 我选择的是亚博智能的wife进阶套餐开发套件，TF卡自带镜像，自带一些所需的Package:
* deepstream-app version: 6.0.1
* DeepStream SDK: 6.0.1
* JetPack: 4.6
* Python: 3.6.9
* CUDA Driver version: 10.2
* CUDA Runtime version: 10.2
* TensorRT version: 8.2
* cuDNN version: 8.2
* ONNXRuntime-gpu: 1.6.0(自行下载)

*ONNXRuntime-gpu的下载：Jetson Nano为arm64架构，ONNXRuntime-gpu不能直接通过pip下载，需要手动编译。好在官方已经帮我们完成了，需要根据Jetpack版本和Python版本进行选择！* [下载地址](https://elinux.org/Jetson_Zoo#ONNX_Runtime)

下载完成之后，打开`Terminal`,进入下载地方的地址，使用`pip`安装：
```shell
$ pip3 install https://nvidia.box.com/shared/static/49fzcqa1g4oblwxr3ikmuvhuaprqyxb7.whl
```

#### 连接工具

我们需要在Jetson Nano内部写代码，需要使用较为方便的编辑器，我这里选择的是vscode远程连接Jetson Nano。

**vscode远程配置连接：**
* 首先在vscode中添加扩展`Remote - SSH`
* 启动Jetson Nano，并连接wifi，打开Terminal输入`ifconfig`，将最下方的ip地址记下。
* 在PC端的`Remote - SSH`连接到刚刚的IP地址，并输入Jetson Nano账户和密码（亚博智能的为 账户：Jetson 密码：yahboom）需要注意的是PC和Jetson Nano必须连接到同一个wifi。
* 在vscode端为jetson nano内部配置Python扩展。
* 文件传输工具，因为我使用的是MAC端，所以我使用`Transmit`工具，连接方式和vscode连接是一样的。

#### 算法准备
关于算法你需要的文件就只有一个`ONNX`文件，因为`ONNX`既包含了模型的权值参数也包含了计算图。我在这里使用的算法是我开发的脐橙缺陷检测分割算法`FastSegFormer-P`。
* ONNX文件导出（需要pth文件）：
```python
#--*-- coding:utf-8 --*--
import torch
from models.fastsegformer.fastsegformer import FastSegFormer

# 替换成自己需要的模型
net = FastSegFormer(num_classes=4, pretrained=False, backbone='poolformer_s12', Pyramid='multiscale', cnn_branch=True).cuda()
net.eval()

# 训练时就使用了FP16进行训练
export_onnx_file = "weights/FastSegFormer_P_224_FP16.onnx"
x=torch.onnx.export(net,  # 待转换的网络模型和参数
                    torch.randn(1, 3, 224, 224, device='cuda'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                    export_onnx_file,  # 输出文件的名称
                    verbose=False,      # 是否以字符串的形式显示计算图
                    input_names=["images"],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                    output_names=["output"], # 输出节点的名称
                    opset_version=11,   # onnx 支持采用的operator 
                    do_constant_folding=True, # 是否压缩常量
                    dynamic_axes=None)
```

#### ONNXRuntime-gpu单帧处理逻辑
视频检测最重要的就是单帧的处理逻辑，将该部分完成了，程序内核就完成。该部分主要的流程就是`输入前处理->算法推理->掩码预测->图像合并->图像后处理`。

**单帧处理代码:**
```python
import cv2
import copy
import numpy as np
import time

import onnxruntime

import torch
import torch.nn.functional as F


def blend_images(old_image, new_image, alpha):
    """
    使用cv2.addWeighted()函数混合两个图像
    """
    blended_image = cv2.addWeighted(old_image, alpha, new_image, 1 - alpha, 0)

    return blended_image

def process_frame(model, img, name_classes = None, num_classes = 21, count = False, input_shape = (224, 224), device = 'cpu', weight_type = None, inputs=None, outputs=None, bindings=None, stream=None, context=None):
    
    # torch.cuda.synchronize()
    since = time.time()
    
    # 转化为彩色图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 对输入图像做一个备份
    old_img = copy.deepcopy(img)
    original_h  = np.array(img).shape[0]
    original_w  = np.array(img).shape[1]
    # 将图像转化为模型输入的分辨率
    if original_h != input_shape[0] or original_w != input_shape[1]: 
        image_data = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
    # 添加Batch维度
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    # 将内存不连续的数组转成连续存储
    image_data = np.ascontiguousarray(image_data)

    if weight_type == '.onnx':
        ort_inputs = {'images': image_data}
        pred = model.run(['output'], ort_inputs)[0]
        pred = pred[0]
        # 转化为张量
        pred = torch.tensor(pred)
        pred = F.softmax(pred.permute(1,2,0),dim = -1).cpu().numpy()
        pred = cv2.resize(pred, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        pred = pred.argmax(axis=-1)
    
    # 用于掩码的比例组成
    if count:
            classes_nums        = np.zeros([num_classes])
            total_points_num    = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(num_classes):
                num     = np.sum(pred == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    # 不同缺陷的类别使用的色彩      
    if num_classes <= 21:
        colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                        (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                        (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                        (128, 64, 12)]

    # 转回原图尺寸    
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pred, [-1])], [original_h, original_w, -1])
    # 分割图像和原图结合
    image = blend_images(old_image=old_img, new_image=seg_img, alpha=0.6)
    # 转回RGB图像    
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # torch.cuda.synchronize()
    end = time.time()

    # 计算FPS并且放到图像的左上角
    fps = 1 / (end - since)
    image = cv2.putText(image, "FPS " + str(int(fps)), (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
    fps = int(fps)
    
    # 返回分割图像、合并图像、fps指标
    return seg_img, image, fps
```

#### TensorRT单帧处理逻辑

该部分最主要的流程为就是`从ONNX文件构建engine->从engine构建context->从engine中获取size并分配buffer->单帧处理逻辑`

**从ONNX构建TensorRT的Engine：**
```python
# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(TRT_LOGGER, max_batch_size=1, onnx_file_path="", engine_file_path="",fp16_mode=True, save_engine=False):
    """
    params max_batch_size:      预先指定大小好分配显存
    params onnx_file_path:      onnx文件路径
    params engine_file_path:    待保存的序列化的引擎文件路径
    params fp16_mode:           是否采用FP16
    params save_engine:         是否保存引擎
    returns:                    ICudaEngine
    """
    # 如果已经存在序列化之后的引擎，则直接反序列化得到cudaEngine
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, \
            trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化
    else:  # 由onnx创建cudaEngine
        
        # 使用logger创建一个builder 
        # builder创建一个计算图 INetworkDefinition
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network,  \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            builder.create_builder_config() as config, \
            trt.Runtime(TRT_LOGGER) as runtime: # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
            
            config.max_workspace_size = 1<<30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
            builder.max_batch_size = max_batch_size # 执行时最大可以使用的batchsize
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            # 解析onnx文件，填充计算图
            if not os.path.exists(onnx_file_path):
                quit("ONNX file {} not found!".format(onnx_file_path))
            print('loading onnx file from path {} ...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model: # 二值化的网络结果和参数
                print("Begining onnx file parsing")
                parser.parse(model.read())  # 解析onnx文件
            #parser.parse_from_file(onnx_file_path) # parser还有一个从文件解析onnx的方法

            print("Completed parsing of onnx file")
            # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
            print("Building an engine from file {}' this may take a while...".format(onnx_file_path))

            #################
            # print(network.get_layer(network.num_layers-1).get_output(0).shape)
            # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan) 
            print("Completed creating Engine")
            if save_engine:  #保存engine供以后直接反序列化使用
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  # 序列化
            return engine

# 构建TensorRT所需的engine，上下文Context，分配buffer
TRT_LOGGER = trt.Logger()
eng_path = model_path.split(".")[0] + ".trt"
# 构建engine
engine = get_engine(TRT_LOGGER, onnx_file_path=model_path, engine_file_path=eng_path)
# 构建Context
context = engine.create_execution_context()
# 从engine中获取size并分配buffer（尤其是动态模式）
inputs, outputs, bindings, stream = allocate_buffers(engine)
```
*由于我的模型代码中存在双线性插值等操作，onnx版本算子不支持，选择不序列化engine保存，而选择在线构建引擎的方式，所以会比较耗时。（序列化后无法进行反序列化读取engine）*

**单帧处理代码：**
```python
import cv2
import copy
import numpy as np
import time

import onnxruntime

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 

import torch
import torch.nn.functional as F


def blend_images(old_image, new_image, alpha):
    """
    使用cv2.addWeighted()函数混合两个图像
    """
    blended_image = cv2.addWeighted(old_image, alpha, new_image, 1 - alpha, 0)

    return blended_image

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def do_inference_v2(context, bindings, inputs, outputs, stream):
    """
    用于TensorRT的推理
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def process_frame(model, img, name_classes = None, num_classes = 21, count = False, input_shape = (224, 224), device = 'cpu', weight_type = None, inputs=None, outputs=None, bindings=None, stream=None, context=None):
    """
    需要注意的是此时传入的model是TensorRT构建的引擎
    """
    # torch.cuda.synchronize()
    since = time.time()
    
    # 转化为彩色图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 对输入图像做一个备份
    old_img = copy.deepcopy(img)
    original_h  = np.array(img).shape[0]
    original_w  = np.array(img).shape[1]
    # 将图像转化为模型输入的分辨率
    if original_h != input_shape[0] or original_w != input_shape[1]: 
        image_data = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
    # 添加Batch维度
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    # 将内存不连续的数组转成连续存储
    image_data = np.ascontiguousarray(image_data)
    
    if weight_type == '.trt':
        # 输出为一个列表
        shape_of_output = (1, 4, 224, 224)
        # Load data to the buffer
        # host代表cpu，将数据展平成为1维
        inputs[0].host = image_data.reshape(-1)
        # TensorRT推理
        trt_outputs = do_inference_v2(context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # 
        pred = postprocess_the_outputs(trt_outputs[0], shape_of_output)
        # 由于是批量推理，所以要取第一个值
        pred = pred[0]
        # 转化为张量
        pred = torch.tensor(pred)
        pred = F.softmax(pred.permute(1,2,0),dim = -1).cpu().numpy()
        pred = cv2.resize(pred, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        pred = pred.argmax(axis=-1)

    if count:
            classes_nums        = np.zeros([num_classes])
            total_points_num    = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(num_classes):
                num     = np.sum(pred == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        
    if num_classes <= 21:
        colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                        (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                        (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                        (128, 64, 12)]

    # 转回原图尺寸    
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pred, [-1])], [original_h, original_w, -1])
    # 分割图像和原图结合
    image = blend_images(old_image=old_img, new_image=seg_img, alpha=0.6)
    # 转回RGB图像    
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # torch.cuda.synchronize()
    end = time.time()

    # 计算FPS并且放到图像的左上角
    fps = 1 / (end - since)
    image = cv2.putText(image, "FPS " + str(int(fps)), (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
    fps = int(fps)
            
    return seg_img, image, fps
```

#### DeepStream视频处理框架

DeepStream是一种高性能、低延迟的边缘智能视频分析和处理平台，由NVIDIA开发。它结合了深度学习推理、多流视频分析和传统计算机视觉技术，旨在为各种实时视频分析应用提供可扩展的解决方案。主要用于智能安防监控、智能交通、零售分析和工业检测等行业。

以下是DeepStream的一些主要特点：

1. 高性能推理：DeepStream利用GPU的并行计算能力，提供高性能的深度学习推理，可以实时处理多个视频流并进行实时分析和推断。

2. 多流处理：DeepStream支持并行处理多个视频流，可以同时处理来自多个摄像头或视频源的数据。这使得DeepStream非常适合大规模监控系统、智能交通和视频分析等应用场景。

3. 灵活的插件架构：DeepStream采用了灵活的插件架构，可以轻松集成各种不同的算法和模型。这使得开发人员可以根据应用需求选择适合的算法，例如目标检测、跟踪、人脸识别等。

4. 实时分析和响应：DeepStream提供实时的视频流分析和处理，能够在几乎实时的条件下检测、跟踪和分析视频中的对象。这对于需要即时响应的应用非常重要，例如安防监控和实时决策系统。

下面是DeepStream的工作流程：

> 数据源接入 -> 数据解码 -> 对数据流做混合 -> 建立推理模块 -> 处理后的数据进行转换 -> 数据输出 -> 数据后处理以及在原视频上加入外来部分(分割：掩码；检测：检测框)

**DeepStream使用Python接口程序的逻辑：**
`创建Pipeline -> 构建不同部分的处理模块 -> 将所有处理模块都加入Pipeline -> 将所有模块按照需要的顺序连接起来 -> 设置不同模块的参数以及配置模型的配置文件 -> 编写需要对画面进行处理的程序 -> 启动Pipeline进行视频处理`

使用DeepStream-Python进行视频处理的示例程序（图像分割），示例程序使用的都是Caffe框架的模型：
图像分割：[deepstream-segmentation](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-segmentation) 
视频流实例分割：[deepstream-segmask](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-segmask)
[官方模型config文件参数设置](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications)

#### Jetson-FastSegFormer

这是我在Jetson Nano上使用`ONNXRuntime-gpu`和`TensorRT`以及`DeepStream`，可以在这里访问：[https://github.com/caixiongjiang/FastSegFormer-pyqt/tree/main/jetson-FastSegFormer](https://github.com/caixiongjiang/FastSegFormer-pyqt/tree/main/jetson-FastSegFormer)


下面是我测试的结果：
<table>
	<tr>
	    <th colspan="8">Jetson-FastSegFormer</th>
	</tr >
	<tr>
	    <td style="text-align: center;">Task</td>
	    <td style="text-align: center;">Video/Stream input</td>
	    <td style="text-align: center;">Inference input</td>  
      <td style="text-align: center;">Inference framework</td>
      <td style="text-align: center;">GPU computing capability</td>
      <td style="text-align: center;">Quantification</td>
      <td style="text-align: center;">Video processing</td>
      <td style="text-align: center;">Average FPS</td>
	</tr >
	<tr >
	    <td rowspan="5" style="text-align: center;">Video Detection</td>
	    <td rowspan="5" style="text-align: center;">$512\times 512$</td>
	    <td rowspan="10" style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="10" style="text-align: center;"> 0.4716 TFLOPS</td>
      <td rowspan="10" style="text-align: center;">FP16</td>
      <td rowspan="2" style="text-align: center;">Single frame</td>
      <td style="text-align: center;">10</td>
	</tr>
	<tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">15</td>
	</tr>
	<tr>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">Multi-thread</td>
      <td style="text-align: center;">~</td>
	</tr>
	<tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">DeepStream</td>
      <td style="text-align: center;">23</td>
	</tr>
  <tr>
	    <td rowspan="5" style="text-align: center;">CSI Camera Detection</td>
      <td rowspan="5" style="text-align: center;">$1280\times 720$</td>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">Single frame</td>
      <td style="text-align: center;">8</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">12</td>
	</tr>
  <tr>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">Multi-thread</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">DeepStream</td>
      <td style="text-align: center;">20</td>
	</tr>
</table>
~:Can't run dual-threaded acceleration on Jetson nano (4G) because of lack of memory.






