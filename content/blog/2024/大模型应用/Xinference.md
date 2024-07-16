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
# 这里使用了我实际的镜像名，并将需要的模型映射进了镜像内，并在后台运行
docker run -v /data/caixj/Interests/xinference_hub:/model_hub -e XINFERENCE_HOME=/model_hub -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 -itd --name xinference --gpus all caixj/xinference:v0.13.0 xinference-local -H 0.0.0.0 --log-level debug
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

注册好之后，就可以在`Launch Model`的`custom models`这里找到自己注册的模型：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img58.jpg)

然后就可以启动自己的模型了!

**先看LLM Chat模型**：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img59.jpg)
以Qwen2-7B-instruct为例子：
* model_engine可以选择`Transformers`、`vllm`和`llama.cpp`。
* `Transformers`为例，`Model Format`参数主要是选择`pytorch`模型还是`gptq`，`awq`模型，后两者为不同量化方法的模型。
* `Model Size`在注册模型时便指定，以B为单位。
* `Quantization`代表量化的位数，可以选择`none`，`4-bit`，`8-bit`。
* `N-GPU`代表使用的GPU数量，这里能选择的参数会根据docker映射进来的GPU数量自动决定选择上限。
* `Optional Configurations`中的`Model UID`是你调用该运行的模型时的名字，后续会用到。
* `Worker Ip`一般都直接写`0.0.0.0`
* `GPU Idx`是你要在哪块GPU上进行启动推理。
* `Addtional parameters`则需要参考Transformers

查询Xinference可以启动的、某种类型的模型：
```bash
xinference registrations --model-type <MODEL_TYPE> 
```

可以使用`xinference launch --help`来查看命令

上述参数对应的命令为：
```bash
xinference launch --model-name qwen2-instruct \
                  --model-engine Transformers \
                  --size-in-billions 7 \
                  --model-format pytorch \
                  --quantization none \
                  --model-uid qwen2-7b-instruct \
                  --n-gpu 1 \ 
                  --worker-ip 0.0.0.0 \
                  --gpu-idx 0
```

运行成功后就可以在`Running Models`里面找到它了!

验证方法：
```bash
curl -X 'POST' \
  'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<MODEL_UID>",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the largest animal?"
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": true # 流式
  }'
```
`XINFERENCE_HOST`根据启动的机子的ip来，`XINFERENCE_PORT`默认为9998，`Model_UID`使用的上述启动模型时设置的`model-uid`。

**Embedding模型启动**：

界面启动：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img60.jpg)

对应命令：
```bash
xinference launch --model-name caixj-bge-m3 \ 
                  --model-uid bge-m3 \
                  --model-type embedding \
                  --gpu-idx 2 \
                  --worker-ip 0.0.0.0
```

验证方法：
```bash
curl -X 'POST' \
  'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<MODEL_UID>",
    "input": "What is the capital of China?"
  }'
```
输出内容：
```
{
"object":"list",
"model":"bge-m3-1-0",
"data":[{"index":0,
    "object":"embedding",
    "embedding":[-0.03103026933968067,0.03556380420923233, ... ,-0.0026659774594008923,-0.006471091415733099,-0.0057240319438278675]}],
    "usage":{
        "prompt_tokens":23,
        "total_tokens":23
    }
}
```

**Rerank模型启动**：

界面启动：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img61.jpg)

对应命令：
```bash
xinference launch --model-name caixj-bge-reranker-large \
                  --model-uid bge-reranker-large \
                  --model-type rerank \
                  --gpu-idx 0 \
                  --worker-ip 0.0.0.0
```

验证方法：
```bash
curl -X 'POST' \
  'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<MODEL_UID>",
    "query": "A man is eating pasta.",
    "documents": [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ],
    "return_documents": true
  }'
```

输出内容：
```
{'id': '09e431fa-3f38-11ef-a3c2-0242ac11003a', 
'results': [{'index': 0, 'relevance_score': 0.9999258518218994, 'document': {'text': 'A man is eating food.'}}, 
            {'index': 1, 'relevance_score': 0.048283521085977554, 'document': {'text': 'A man is eating a piece of bread.'}}, 
            {'index': 2, 'relevance_score': 7.636439841007814e-05, 'document': {'text': 'The girl is carrying a baby.'}}, 
            {'index': 4, 'relevance_score': 7.636331429239362e-05, 'document': {'text': 'A woman is playing violin.'}}, 
            {'index': 3, 'relevance_score': 7.617334631504491e-05, 'document': {'text': 'A man is riding a horse.'}}], 
            'meta': {'api_version': None, 'billed_units': None, 'tokens': None, 'warnings': None}}
```

有时候界面启动了并不一定启动了，你需要结合`nvidia-smi`显存占用和本地映射文件中的日志来确定，日志一般在logs文件夹下，会根据每次启动单独生成一个文件夹。

#### 启动训练好的微调模型

TODO: 完成训练好的Qwen2-7B-Instruct模型的部署。
