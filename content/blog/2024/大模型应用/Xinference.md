---
title: "Xinference：为LLM app赋能"
date: 2024-07-11T18:18:05+08:00
lastmod: 2024-07-11T09:19:06+08:00
draft: true
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/xinference_title.jpg"
description: "使用一套统一的框架集成LLM、embedding、reranker、图像模型和语音模型。"
tags:
- Deep_learning
categories:
- 大模型
series:
- 《LLM》
comment : true
---

### Xinference：为LLM app赋能


* Xinference Github官网：[https://github.com/xorbitsai/inference](https://github.com/xorbitsai/inference)
* Xinference 官方中文文档：[https://inference.readthedocs.io/zh-cn/latest/index.html](https://inference.readthedocs.io/zh-cn/latest/index.html)

Xinference 集成了 LLM、embedding、image、audio、rerank模型。关于LLM，则集成了Transformer、vLLM、Llama.cpp、SGLang引擎。该框架主要是制作了一个UI界面来自主控制集成好的模型，为应用开发提供底层能力。

#### 环境安装

* 本地环境安装：可根据不同的引擎来安装不同的版本，具体看[https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)
* Docker镜像：由于Docker管理起来比较方便，我这里就使用了Docker的部署方式。
官方提供了方便下载的镜像：
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/xprobe_xinference/xinference:<tag>
```
我选择了`v0.13.0`的版本。

#### Xinference启动

使用Docker启动的方式比较简单，使用官方提供的命令：
```bash
docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --gpus all xprobe/xinference:v<your_version> xinference-local -H 0.0.0.0 --log-level debug
```
*由于公司的网下载大模型比较慢，所以通常我们会选择在本地下好的模型进行加载，所以需要对命令做一点改造*

现在本地新建文件夹，并把需要的模型传入。
```bash
mkdir xinference_hub
```
开始启动镜像：
```bash
# 这里使用了我实际的镜像名，并将需要的模型映射进了镜像内
docker run -v /data/caixj/Interests/xinference_hub:/model_hub -e XINFERENCE_HOME=/model_hub -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --name xinference --gpus all caixj/xinference:v0.13.0 xinference-local -H 0.0.0.0 --log-level debug
```
#### 模型启用

按照上述镜像启动后，容器已经在后台运行，此时可以在浏览器输入相应`https://<ip>:9998`进行访问了。

在本地开发环境，可以使用界面来操作配置模型，管理模型，启动模型等。

在框架内本身集成了很多现有的模型，如下图所示，你可以直接点击启动，框架会自动去下载模型。 
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img54.jpg)

但很多时候网络不好的时候，我们则需要将自己映射进去的模型给注册了并使用。
以`Qwen2-7B-Instruct`模型为例:
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img55.jpg)

主要还是填写一些基本信息，以及模型的格式，模型的架构等，**注意名字要加前缀，防止更系统中自带的模型名字不同而冲突**。

同样的也可以注册embedding、和rerank模型：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img56.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img57.jpg)

注册好之后，就可以在`Launch Model`这里找到自己注册的模型：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img58.jpg)

然后就可以启动自己的模型了：

