---
title: "RAG实战（四）- 关键词检索"
date: 2024-06-24T18:18:05+08:00
lastmod: 2024-06-24T09:19:06+08:00
draft: true
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/rag_title.jpg"
description: "文本检索中常用的关键词检索模块"
tags:
- RAG
categories:
- NLP
series:
- 《RAG进阶》
comment : true
---

## RAG实战（四）- 关键词检索


### BM25检索器

BM25 是一种基于概率的排名函数，用于信息检索系统。**BM25原理是根据查询词在文档中的出现频率以及查询词在整个文本集中的出现频率，来计算查询词和文档的相似度**。BM25模型的主要思想是：如果一个词在一份文档（这里的文档一般是指分块之后的document）中出现的频率高，且在其他文档中出现的频率低，那么这个词对于这份文档的重要性就越高，相似度就越高。BM25模型对于长文档和短文档有一个平衡处理，防止因文档长度不同，而导致的词频偏差。

#### 基本原理
BM25基于这样一个假设：对于一个特定的查询项，它在相关文档中出现的频率高于在非相关文档中的频率。算法通过结合词项频率（TF）和文档频率（DF）来计算文档的得分。

**TF（词项频率）**
词项频率是指一个词项在文档中出现的次数。BM25对传统TF的计算方法进行了调整，引入了饱和度和长度归一化，以防止长文档由于包含更多词项而获得不公平的高评分。

**IDF（逆文档频率）**
逆文档频率是衡量词项稀有程度的指标。它的计算基于整个文档集合，用来降低常见词项的权重，并提升罕见词项的权重。

**计算公式**：
$$
\text{BM25}(D, Q) = \sum_{i=1}^n\text{IDF}(q_i)\cdot\frac{f(q_i, D)\cdot(k_1 + 1)}{f(q_i, D) + k_1\cdot(1-b+b\cdot \frac{\text{len}(D)}{\text{avg\_len}})}
$$

其中：

* $n$是查询中的词项数。
* $q_i$是查询中的第$i$个词项。
* $\text{IDF}(q_i)$是逆文档频率，计算方式通常是$\text{log}\frac{N-n(q_i)+0.5}{n(q_i)+0.5}$，其中$N$是文档总数，$n(q_i)$ 是包含词项$q_i$的文档数。
* $\text{len}(D)$是文档$D$的长度。
* $\text{avg\_len}$是所有文档的平均长度。
* $k_1$和$b$是调整参数，通常设置为$k_1=1.5$和$b=0.75$

#### 代码使用

在LangChain中已经集成了该方法，模块名叫`BM25Retriever`。

示例使用代码：
```python
from typing import (
    List
)
from langchain.schema import Document
import jieba
import jieba.posseg as pseg

# jieba_dictionnary为分词内部自带的分词文件地址
# jieba_user_dict为自己定义的特殊分词表
def func(text: str) -> List[str]:
    jieba.set_dictionary(jieba_dictionary)
    jieba.load_userdict(jieba_user_dict)

    jieba_list = jieba.lcut(text, cut_all=True)
    words = pseg.cut(text)
    for word, flag in words:
        if flag == "nr":
            jieba_list.append(word)

    return jieba_list

# 该部分为通过文本切分之后的chunk块
texts: List[Document] = [...]
# BM25检索器，获取前5个最相关的结果
bm25_retriever = BM25Retriever(docs=texts, k=5) 
# BM25检索器有默认的前处理分词策略，你也可以实现自己的分词策略，这里我使用jieba分词
my_bm25_retriever = BM25Retriever(docs=texts, k=5, preprocess_func=func)
```

其中词汇的内容通常用以下格式：
```txt
# 名称 词频 词性（词频越高，优先级越高）
苹果 3 n
iPhone 5 nz
Python 10 n
深度学习 8 n
jieba 5 nr
```
除了上述的格式，词频和词性都是可以省略的，当然，标注完整效果会更佳。

### ElasticSearch检索器