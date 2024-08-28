---
title: "Advanced-RAG: RAG进阶使用技巧"
date: 2024-08-25T18:18:05+08:00
lastmod: 2024-08-27T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/rag_title.jpg"
description: "RAG在工程中的进阶使用技巧"
tags:
- RAG
categories:
- NLP
series:
- 《RAG进阶》
comment : true
---

# Naive RAG

前言：目前下述的代码都基于Langchain 0.1的版本进行，目前Langchain已经更新到0.2，还有在构建RAG应用的时候还是不要过分依赖框架，降低灵活性，这有时候会让你的工程开发陷入被动！

首先先介绍一下最简单的RAG的流程，其主要的流程如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img7.jpg)

将知识库文本拆分为块，然后使用一些Transformer Encoder模型将这些块嵌入向量中，将所有这些向量放入索引中，最后为LLM创建一个提示，告诉模型根据我们在搜索步骤中发现的上下文，回答用户的查询。

在运行时，我们使用相同的编码器模型矢量化用户的查询，然后针对索引执行此查询矢量的搜索，找到top-k结果，从我们的数据库中检索相应的文本块，并将其作为上下文输入LLM提示符。

示例的RAG提示词模版如下：
```python
# 英文提示词
prompt_template = """Give the answer to the user query delimited by triple backticks ```{query}```\
                using the information given in context delimited by triple backticks ```{context}```.\
                If there is no relevant information in the provided context, try to answer yourself, 
                but tell user that you did not have any relevant context to base your answer on.
                Be concise and output the answer of size less than 80 tokens.
"""

# 中文版本
prompt_template_zh = """给出由三重反引号分隔的用户查询的答案````{query}```\

使用由三重反引号```{context}```分隔的上下文中给出的信息。\

如果提供的上下文中没有相关信息，请尝试回答自己，

但告诉用户，您没有任何相关上下文来作为答案的基础。

简明扼要，并输出大小小于80个代币的答案。
"""
```

# Advanced-RAG: RAG进阶使用技巧

做过RAG应用的都知道一句话叫“RAG demo5分钟，上线上一年。”

使用简单的RAG流程必然不可能有很好的检索效果，大模型的回答也一定是一塌糊涂。

放一张国外的博客的图：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img8.jpg)
绿色的部分下面我们将会深入展开，蓝色部分则是处理好的文本元素。

下面将会记录的所有RAG相关的技术都或多或少与上面这个图相关。我们将会介绍所有与RAG相关的步骤中相关的进阶技术，以及RAG效果测试数据集的建立、评估、测试。

## 文档加载

## 文本分块

**分块的目的主要是为了确保在内容向量化时尽可能减少噪声，同时保持语义相关性。**例如，在**语义搜索**(semantic search)中为一个包含特定主题信息的文档语料库建立索引。可以通过应用有效的分块策略，确保搜索结果准确捕捉用户查询的真实意图。如果分块太小或太大，可能导致搜索结果不精确或错过表面相关内容。作为经验之谈，如果文本块在没有周围上下文的情况下对人类来说可读可理解，它对语言模型也同样有意义。因此，找到语料库中文档的最佳块大小对于确保搜索结果的准确性和相关性至关重要。


### 长短不一的文本切分

**短内容向量化**：当一个句子被向量化时，生成的向量专注于该句子的具体含义，在与其他句子向量进行比较时也会更多关注句子层面的上。这也意味着**短内容向量化可能会丢失句子在段落或全文中所包含的上下文信息。**

**长内容向量化**：当整个段落或文档被向量化时，向量化过程考虑了整体上下文以及文本内部句子和短语之间的关系，这样向量化时包含的内容可能更全面，捕捉到的文本含义和主题更准确。但是，**较大的文本块大小可能更容易引入噪声，或稀释某些句子或短语的重要性，使得在查询索引时的精确匹配变得更加困难。**

> 较短的查询，如单个句子或短语，将集中于具体细节，并可能更适合与句子级别的向量匹配。较长的查询，跨越多个句子或段落，可能与段落或文档级别的向量更为协调。

索引可能也是非均质的，包含不同大小的块向量。这可能在查询结果相关性方面带来挑战，但也可能带来一些积好处。一方面，由于**长短内容之间语义表示的差异可能会导致查询与结果相关性的波动**。另一方面，**非均质索引可能捕捉到更全面的上下文信息，因为不同大小的块代表了文本中不同层次的细节**。这样可能会有利于更灵活地适应不同类型的查询。

### 分块（chunking）考虑因素

* **内容的性质**：处理的是长文档（如文章或书籍）还是较短的内容（如推文或即时消息）？这将决定哪种模型更适合您的目标，从而影响应用的分块策略。
* **使用的向量模型**：接下来要考虑的是应该使用的向量模型，以及它在哪些块大小上表现最佳。这部分主要是探究模型在多少个数量的令牌（token）上表现更好，以此来辅助决定分块的大小。
* **用户查询的预期长度和复杂度**：RAG是与用户相关的应用问题，一般来说都会预先定义一些查询结果样例，这些**样例的形式和长度**在文本分块中起到重要的作用。
* **检索结果在特定应用中的使用方式**：如果检索结果要被放入大模型里使用，则需要检索结果的内容长度作一定的限制，因为大模型支持的令牌（token）数是有限的。（按照经验来说，带多轮对话的RAG一般上下文长度限定在8K个token左右）

### 分块方法

在文本分块方面，有几种不同的方法可供选择，每种方法适用于不同的情况。以下是一些主要的分块方法，以及它们的优势和劣势。

#### 固定大小分块（Fixed-size Chunking）

这是最常见且直接的分块方法，简单地决定每个块中的token数量，并可选择性地决定它们之间是否应该有重叠。通常用户会希望在块之间保持一些重叠，以确保上下文语义不会在块之间丢失。**在多数情况下，固定大小分块将是最优方法。**

Langchain中执行的例子：
```python
text = "..." # your text
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([text])
```


#### “内容感知”分块（“Content-aware” Chunking）

1、**原始分割法**：一种按照简单的标点符号或者换行符来分割文本。这种方法快速且易于实现，但可能不会考虑到所有可能的边界情况，如缩写、省略号或带有句号的引用等。所以虽然朴素分割在某些情况下足够有效，但它可能无法准确地处理**相对复杂的文本结构**。

```python
text = "..." # your text
docs = text.split(".")
```

2、 **自然语言工具包**（Nature Language Toolkit, NLTK）：NLTK是一个比较流行的Python库，用于处理人类语言数据。其中有一个文本分割器，可以将文本分割成单独的句子，NLTK句子分词器考虑了更多的语言规则和边界情况，更加精确。

```python
text = "..." # your text
from langchain.text_splitter import NLTKTextSplitter
text_splitter = NLTKTextSplitter()
docs = text_splitter.split_text(text)
```

3、**spaCy**是另一个强大的Python库，用于各种NLP任务。它提供了一个高级的句子分割功能，**能够有效地将文本划分为单独的句子，从而在生成的块中更好地保留上下文**。spaCy的句子分割功能使用复杂的算法和模型来理解文本结构，因此它能够更准确地处理各种文本格式和风格。

```python
text = "..." # your text
from langchain.text_splitter import SpacyTextSplitter
text_splitter = SpaCyTextSplitter()
docs = text_splitter.split_text(text)
```

#### 递归分块

递归分块使用一组分隔符以分层和迭代的方式将输入文本分割成更小的块。**如果最初的分割尝试没有产生所需大小或结构的块，该方法会递归地在结果块上使用不同的分隔符或标准进行调用**，直到达到所需的块大小或结构。这意味着虽然块的大小不会完全相同，但它们仍然会“努力”达到类似的大小。

```python
text = "..." # your text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 256,
    chunk_overlap  = 20
)

docs = text_splitter.create_documents([text])
```

#### 专用分块

这部分的内容主要是针对文本的类型来决定的，比如特殊的文本类型，`markdown`和`latex`等。
这两种文本类型都具有高度的结构化属性，它们可以通过语法来生成高质量的结构化文本。前者往往用于工程代码的说明书和文档，后者则一般用于论文的编写。

```python
from langchain.text_splitter import MarkdownTextSplitter
markdown_text = "..."
markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])
```

```python
from langchain.text_splitter import LatexTextSplitter
latex_text = "..."
latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
docs = latex_splitter.create_documents([latex_text])
```

#### 语义分割

Greg Kamradt首次引入了一种新的接近块状的实验技术。在他的笔记本中，Kamradt正确地指出，一个事实是，即全局块大小可能太微不足道，无法考虑文档中段的含义。如果我们使用这种类型的机制，我们无法知道我们是否在组合彼此有任何关系的片段。

幸运的是，如果您正在使用LLM构建应用程序，您很可能已经有能力创建嵌入——嵌入可用于提取数据中存在的语义。这种语义分析可用于创建由谈论相同主题或话题的句子组成的块。

以下是使语义分块的步骤：
1、将文档分解成句子。

2、创建句子组：对于每个句子，创建一个包含给定句子前后一些句子的组。该组本质上被用于创建它的句子“安置”。您可以决定每个组之前或之后的具体数字，但组中的所有句子都将与一个“anchor”句子相关联。

3、为每个句子组生成嵌入，并将其与他们的“anchor”句子相关联。

4、按顺序比较每个组之间的距离：当您按顺序查看文档中的句子时，只要主题或主题相同-为给定句子嵌入的句子组和它前面的句子组之间的距离将很低。另一方面，更高的语义距离表明主题或话题已经改变。这可以有效地从下一个块中划定一个块。

Langchain中已经将这种方法集成进去了，但它是一个实验性的方法：
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())
docs = text_splitter.create_documents([state_of_the_union])
print(docs[0].page_content)
```
### 确定适用于应用的最佳块大小

说了这么多，我们还没有研究如何确定最佳的分块大小（比如是固定大小分块），只是一些通用的建议。

* 数据预处理：首先拿到语料需要对数据进行一定的数据预处理确保数据的质量，然后才能确定块的大小。（去掉一些不合理的标签，文档数据的不合理读取）。
* **选择一系列块大小**：一旦数据预处理完成，下一步是选择一系列潜在的块大小进行测试。如文章前面提到的，选择时应考虑**内容的性质（例如，短消息或长文档）**、将要使用的**向量模型及其能力（例如，token数量限制）**。目标是在保留上下文语境和维持准确性之间找到平衡点。开始时可以探索各种块大小，包括较小的块（例如，128或256个token）以捕获更细粒度的语义信息，然后再探索较大的块（例如，512或1024个token）以保留更多上下文语境。
* **评估每种块大小的性能**：可以使用多个索引或一个带有多个命名空间的单一索引测试不同的块大小的性能。使用代表性数据集为想要测试的块大小创建向量，并将它们保存在一个或多个索引中。然后运行一系列查询以评估质量，并比较各种块大小的性能。这很可能是一个迭代过程，需要测试不同的块大小针对不同的查询，直到能够确定最适合内容和预期查询的块大小。


## 向量检索

## 关键词检索


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


## 混合检索


## 分级检索

## 结果排序

## 指标评估

## 知识图谱在RAG中的应用

## 问题改写
