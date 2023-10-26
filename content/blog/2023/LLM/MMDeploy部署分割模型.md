---
title: "大模型高效微调技术（PEFT）"
date: 2023-10-24T18:18:05+08:00
lastmod: 2023-10-25T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/LLM_title.jpg"
description: "学习并介绍一下高效的大模型微调技术（PEFT）"
tags:
- Deep_learning
categories:
- 深度学习
series:
- 《LLM》
comment : true
---

## 大模型高效微调技术（PEFT）

### Adapter Tuning

#### 技术原理

论文：**Parameter-Efficient Transfer Learning for NLP** 
论文链接：[http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)

该方法设计了一个`Adapter`结构，嵌入`Transformer`结构中。针对一个`Transformer Block`，增加两个Adapter结构，增加都放在残差结构之前。训练时，固定住原来的预训练模型参数不变，只对`Adapter`结构和`Layer Normal`层进行微调。Adapter层是一个类似于`Squeeze-and-Excitation`层思想的结构，首先使用较高维度的特征投影到较低维度的特征，中间通过一个非线性层，再将低维特征重新映射回原来的高维特征。其参数量主要低维特征的维度决定。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img1.jpg)

#### 方法效果

对$BERT_{LARGE}$在各个任务上进行全量微调和Adapter-Tuning的结果如下，在不同的任务上，低维特征的维度m不同时效果不同。将m固定在64时，性能会略微下降。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img2.jpg)

**Adapter通过引入额外的0.5%~5%参数可以媲美全量微调的效果，缺点是引入的参数在Transformer Block内部，在面对不同的任务时，需要保存不同的权重副本。**

### Adapter Fusion

#### 技术原理

#### 方法效果

### Adapter Drop

#### 技术原理

#### 方法效果

### BitFit

#### 技术原理

论文：**BitFit: Simple Parameter-efﬁcient Fine-tuning for Transformer-based Masked Language-models**

论文链接：[https://arxiv.org/pdf/2106.10199.pdf](https://arxiv.org/pdf/2106.10199.pdf)

`Bitfit`是一种稀疏微调的方法，它冻结了大部分`transformer-encoder`参数，只`更新bias参数`跟`特定任务的分类层参数`。涉及到的bias参数有attention模块中计算`query,key,value`跟`合并多个attention结果时涉及到的bias`，`MLP层中的bias`，`Layernormalization层的bias参数`。方法的技术原理非常简单，且需要微调的参数量极小。

#### 方法效果

使用$BERT_{LARGE}$进行全量微调，以及一些对比的高效微调方法，在不同的任务数据集上进行了实验。可以看到仅微调bias的`Bitfit`方法也能达到略低于3.6%参数的`Adapter`和使用0.5%参数的`Diff-Prune`方法的水平，而需要微调的参数量只有0.08%。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img3.jpg)

而且通过消融实验也可以看出，bias参数的变化并不是在所有结构的部分都比较大。从结果来看，计算query和FFN层的bias参数变化最为明显，只更新这一部分参数也能达到较为不错的效果。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img4.jpg)

**Bitfit只引入了0.03%~0.09%的参数，达到了较好的微调效果，其参数量比其他model tuning的方法少了将近10倍，缺点是更新bias参数依旧需要根据不同的任务保存不同模型的副本。**

### Prefix Tuning

#### 技术原理

论文：**Preﬁx-Tuning: Optimizing Continuous Prompts for Generation**

论文链接：[https://arxiv.org/pdf/2101.00190.pdf](https://arxiv.org/pdf/2101.00190.pdf)

`Prefix Tuning`是一种优化前缀提示的微调方法。与前文对模型参数的微调不同，前缀微调的方法对预训练的模型没有任何改动，其目标是通过微调输入的向量参数来改变模型的生成行为。前缀被添加到输入文本的开头，并且通过调整前缀的内容和长度，可以引导模型生成符合用户意图的输出。但是直接更新前缀表示通常会对模型产生较大的负面影响，所以该方法优化了输入向量参数，即前缀的表示，以控制模型的生成结果。

正如下图表示，对于原有的Transformer预训练模型不需要进行改动，而是针对不同的任务训练不同的前缀提示。

为了实现这个目标，文中使用了多层感知机（MLP）来将前缀与模型输入进行连接。通过训练MLP的参数，可以将前缀映射到与模型输入相同的维度，并且学习前缀和输入之间的复杂映射关系。这样还解决了前缀的长度可能与模型预期的输入长度不匹配的问题，通过使用MLP可以将前缀映射到与模型输入相同的维度。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img5.jpg)

*Prefix Tuning与构造Prompt比较相似，构造Prompt是人为的，叫不稳定，不同的Prompt可能会产生极大的差异。而Prefix可以视为隐藏式的Prompt，且该Prompt是可以训练的！*

#### 方法效果

使用$GPT-2_{MEDIUM}$和$GPT-2_{LARGE}$作为基础模型进行全量微调和一些常用的高效微调方法，在不同任务上都进行了实验。可以看到在很多任务上的微调效果都很不错，引入额外的参数为0.1%，该部分参数不在模型内部。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img6.jpg)

通过消融实验证明，在部分实验上仅仅对Embedding层添加前缀表现力不够，性能将会远不如Full Fine-Tuning，因此在每个Layer都增加了一个可训练的提示前缀，尽量逼近全量微调的效果，参数量上涨为2%。

![](/Users/caixiongjiang/Library/Application Support/typora-user-images/image-20231026175645046.png)

论文还对不同的前缀提示长度做了消融研究，发现提示的长度越长，MLP进行转化的效果越好，当然这种增长并不是线性的，随着前缀变长，增长的幅度会越来越小。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img7.jpg)

### Prompt Tuning

### P-Tuning

### P-Tuning v2

### LoRA

### AdaLoRA

### QLoRA



