---
title: "MMSegmentation初探"
date: 2023-08-16T18:18:05+08:00
lastmod: 2023-08-16T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/mmseg_title.jpg"
description: "用于快速训练现成算法的语义分割算法库，在论文中进行模型对比时简单高效。"
tags:
- Deep_learning
categories:
- 深度学习
series:
- 《MMcv系列》
comment : true
---

### MMSegmentation环境配置

#### 安装Pytorch

这部分比较常规，就不细讲了，一般来说就是安装虚拟环境再安装pytorch的cuda训练包，这里要求的环境是`Pytorch 1.10.0` + `CUDA 11.3`。

#### 使用MIM安装MMCV

`mmcv`是`MMsegmentation`的基础，也是商汤框架所有算法库的母框架，这里我们使用MIM来安装MMCV：
```shell
$ pip install -U openmim
$ mim install mmengine
$ mim install 'mmcv==2.0.0' 
```

#### 安装其他工具包

这些工具包都是图像处理常用的包，可以自行选择。

```shell
pip install opencv-python pillow matplotlib seaborn tqdm 'MMdet>=3.1.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这里需要下载MMdet是因为后面可能部分语义分割算法依赖于MMdet

#### 安装MMSegmentation

这里推荐使用源码安装的方式，这样也可以参考部分源码获取信息：
* 下载源码：
```shell
git clone https://github.com/open-mmlab/mmsegmentation.git -b v1.1.1
```
*-b代表该仓库的分支下载！*

* 进入主目录：
```shell
cd mmsegmentation
```
* 源码安装：
```shell
pip install -v -e
```

#### 检查安装是否成功
* 检查Pytorch:
```python
import torch, torchvision

print("Pytorch 版本", torch.__version__)
print("CUDA 是否可用", torch.cuda.is_available())
```
* 检查mmcv:
```python
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print("MMCV版本", mmcv.__version__)
print("CUDA版本", get_compiling_cuda_version())
print("编译器版本", get_compiler_version())
```
* 检查mmsegmentaion：
```python
import mmseg
from mmseg.utils import register_all_modules
from mmseg.apis import inference_model, init_model

print("mmsegmentaion", mmseg.__version__)
```
