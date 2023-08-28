---
title: "MMDeploy部署分割模型"
date: 2023-08-27T18:18:05+08:00
lastmod: 2023-08-28T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/mmseg_title.jpg"
description: "将MMSegmentation训练好的模型转化为离线模型部署端侧。"
tags:
- Deep_learning
categories:
- 深度学习
series:
- 《MMCV系列》
comment : true
---
## MMDeploy教程
一图看懂MMDeploy的作用：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/img2.jpg)
**如果模型的训练是使用mmcv系列工具生成的，那么使用MMDeploy是最好的！**

### MMDeploy安装

#### 安装MMDeploy
```shell
# pip install mmdeploy --upgrade
pip install mmdeploy==1.2.0
```

#### 下载MMDeploy源码
```shell
git clone https://github.com/open-mmlab/mmdeploy.git
```

#### 验证安装配置成功
```shell
$ python3
$ >>>import mmdeploy
$ >>>print('MMDeploy 版本', mmdeploy.__version__)
# MMDeploy 版本 1.2.0
```
成功输出上述信息则安装成功。

### MMDeploy-模型转换

#### 在线模型转换工具

官方提供了一个在线模型转换工具：[https://platform.openmmlab.com/deploee](https://platform.openmmlab.com/deploee)

点击`模型转化`，再点击`新建转化任务`之后，会进入这样一个画面：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/img3.jpg)

然后将需要的pth文件和config的python文件上传就可以开始转化了！

#### 在线模型测试工具
官方提供了一个在线模型测试工具：[https://platform.openmmlab.com/deploee/task-profile-list](https://platform.openmmlab.com/deploee/task-profile-list)

点击`模型测试`，再点击`新建测试任务`之后，会进入这样一个画面：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/img4.jpg)


#### 使用Python API进行模型转换

* 进入主目录：
```shell
cd mmdeploy
```
* 下载ONNX包：
```shell
$ pip install onnxruntime
$ import onnxruntime as ort
$ print('ONNXRuntime 版本', ort.__version__)
```
* Pytorch模型转ONNX模型：
```shell
python tools/deploy.py \
        configs/mmseg/segmentation_onnxruntime_dynamic.py \
        ../mmsegmentation/Zihao-Configs/ZihaoDataset_FastSCNN_20230818.py \
        ../mmsegmentation/checkpoint/Zihao_FastSCNN.pth \
        ../mmsegmentation/data/watermelon_test1.jpg \
        --work-dir mmseg2onnx_fastscnn \
        --dump-info
```
转换完成的模型，导出在`mmdeploy/mmseg2onnx_fastscnn`。

* 验证转化成功（非必需）：

output_onnxruntime.jpg：用ONNX Runtime推理框架的预测结果，应与output_pytorch.jpg几乎相同

output_pytorch.jpg：用原生Pytorch的预测结果，应与output_onnxruntime.jpg几乎相同

detail.json：模型转ONNX的信息，用于追溯bug

* 转换得到的模型及信息（必需）：

deploy.json：模型描述，用于MMDeploy Runtime推理

end2end.onnx：ONNX模型

pipeline.json：模型输入、预处理、推理、后处理，每一步骤的输入输出信息

**转化其他模型也是一样的，只需要改变`configs/mmseg/`文件夹下转化配置文件即可。注意`dynamic`代表模型输入尺寸为动态，如果是不变的则为`static`。`--dump-info`代表打印转换过程的信息。**