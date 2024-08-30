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

#### 语义分块

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


## 向量化

在简单的RAG流程中，不管是知识库的建立还是问题的查询，首先都要经过向量化。向量化的模型排行榜可以在[leaderboard](https://huggingface.co/spaces/mteb/leaderboard)中找到，向量化模型的选择需要根据业务类型和文本分块的长度来联合确定，并不是使用排行越高的的模型效果就一定更好。

Langchain中使用向量化，使用Chroma、faiss举例：
```python
from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_instance = HuggingFaceEmbeddings(model_name=model_path,
                                           model_kwargs={"device": device},
                                           encode_kwargs={"normalize_embeddings": True})
vector_path = "./vector_store"

Chroma.from_documents(
    documents=texts,
    embedding=embedding_instance,
    persist_directory=vector_path
)

vs = FAISS.from_documents(
    documents=texts,
    embedding=embedding_instance
)
vs.save_local(vector_path)
```

### 搜索索引

#### 向量存储索引

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img9.jpg)

RAG 管道的关键部分是搜索索引，存储我们在上一步获得的矢量化内容。最简单的实现使用平面索引——查询向量和所有块向量之间的暴力距离计算。

一个适合在10000+元素规模上进行高效检索的搜索索引是向量索引(vector index)，如[faiss](https://faiss.ai/)、[nmslib](https://github.com/nmslib/nmslib)或[annoy](https://github.com/spotify/annoy)，使用一些近似最近邻(Approximate Nearest Neighbours, ANN)实现，如聚类、树或HNSW(Hierarchical Navigable Small World)算法。

还有一些托管解决方案，如`openSearch`或`ElasticSearch`，以及向量数据库，它们在后台处理了前面描述的数据摄取管道，如Pinecone、Weaviate或Chroma。

**根据索引选择、数据和搜索需求，用户还可以在存储向量的同时存储元数据(metadata)**，然后使用元数据filter来搜索某些日期或来源内的信息。主要的元数据包括页码，上级标题，自定义文档索引等等，这些都有助于检索，也有助于后续建立测试数据集。

#### 多层索引

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img10.jpg)

如果知识库中的文档很多，那么我们必须更加高效地搜索其中的信息，更快更好地找到相关的文档。**这部分在检索时必须存储引用来源的单一答案**。在处理大型数据库时，高效做到这个点的方法是**创建两个索引——一个由摘要组成，另一个由文档块组成**。如图所示，进行两步搜索，首先通过摘要筛选出相关的满足要求的文档，然后再在这些文档中继续检索。

**具体实现方式**：
1、对整体文本内容分块，向量化，入库
2、相关文档进行摘要总结，添加对应关系，向量化，入库
3、输入问题在摘要库中进行检索，通过对应的摘要-文本映射表找到文本库中相关的部分。
4、在相关文本库的部分重新进行二次检索，找到相关的内容

上述方法在由多个文件组成的内容知识库中尤其适用，每个文件代表的内容是主题比较明确。这会在对检索到的文本进行过滤也会更加轻松搞笑。假设知识库为一整本书的内容，可能需要人工总结摘要，或者使用LLM来总结摘要，形成结构化信息，提供更加精准的检索。

#### 假设问题和假设向量文档（Hypothetical Questions and HyDE）

另一种方法是**让LLM为每个块生成问题，并将这些问题以向量形式嵌入，在运行时对这些问题向量索引执行查询搜索（在我们的索引中用问题向量替换块向量），然后在检索后路由到原始文本块**，并将它们作为上下文发送给LLM以获取答案。这相比多层索引的方案粒度更细，对于能命中的问题，精度较高，命中不了的问题则完全错误，需要根据场景来谨慎选择。

实现步骤：
1、为每个相关的向量生成对应的问题，向量化，
2、在问题库里面进行检索，命中后根据映射表拿到对应上下文，送给大模型回答。（由于这里按照问题进行检索本身在生成问题时已经比较精确，不需要二次检索了）

这种方法通过在查询和假设问题之间的更高语义相似性来提高搜索质量，与我们对实际块所拥有的相比。

还有一种逆向逻辑方法称为**假设向量文档(Hypothetical Document Embeddings, HyDE)——用户可以让LLM为给定的查询生成一个假设性响应，然后使用它的向量以及查询向量来增强搜索质量。**

实现步骤：
1、使用提示词工程让大模型回答问题，尽量简短，向量化
2、向量化的回答做为内容进行检索，将检索到的内容拼装到RAG的提示词中回答


#### 上下文增强

这里的上下文增强是检索较小的块以提高搜索质量，但增强上下文内容语境以供LLM推理。

上下文增强有两套方案，一是利用在较小检索块附件的句子来扩展上下文，二是递归地将文档分割成包含更小子块的更大父块。

> 句子窗口检索

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img11.jpg)

在这个方案中，**文档中的每个句子都被单独嵌入，这为查询与上下文的余弦距离(cosine distance)搜索提供了极高的准确性。**为了在获取最相关的单个句子后更好地推理所找到的上下文，我们通过在检索到的句子前后各扩展k个句子来扩展上下文窗口，然后将这个扩展的上下文发送给LLM。

> 自动合并检索（父文档检索器）

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img12.jpg)

这里的想法与句子窗口检索器非常相似——搜索更颗粒度更细的信息，然后在将检索到的上下文送给LLM推理前扩展上下文窗口。文档被递归地分割成更大父块中更小的子块。

这种方法需要在文本分块的时候做出记录结构化的父子块信息，通常通过记录metadata来实现。



### 关键词检索

BM25是一种比较常用的关键词检索组件

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

#### ElasticSearch检索器


### 混合检索

一个简单的想法，就是将传统搜索行业的`关键词检索`（稀疏检索算法）和最新的`语义搜索`，`向量检索`结合起来，生成一个最优的检索结果。但是一般不通过简单的融合来实现，这里关键的技巧是正确地结合具有不同相似度得分的检索结果——这个问题通常通过互惠排名融合(Reciprocal Rank Fusion, RRF)算法解决。最终的实现方式如图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img13.jpg)

在Langchain中，这是在`Ensemble Retriever`类中实现的，它结合了用户定义的一系列检索器，例如基于faiss的向量索引和基于BM25的检索器，并使用RRF进行重新排序。

示例demo:
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

doc_list_1 = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
]

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

doc_list_2 = [
    "You like apples",
    "You like oranges",
]

embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_texts(
    doc_list_2, embedding, metadatas=[{"source": 2}] * len(doc_list_2)
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```

但在实际的检索过程中，这种合并的过程并不总是理想。举一个自己在做RAG的一个例子，下图是自己做的产品库的信息检索方案：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img14.jpg)

因为并不能简单通过权重来评判不同的检索器的好坏。我个人的做法是通过reranker Model（下面会讲到）先对不同的检索结果进行排序，然后进行合并去重，最后取`top_n`的检索结果。**将reranker模型前置的好处是，在各自的检索器检索到的内容，其相似度得分并不一定准确，可以减少多检索器在RRF排序过程的关键检索信息损失。**

合并和去重检索结果的位置排布变得很重要，一般会有两种排布方式：

* 按照各自的排序结果交替排序并进行去重：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img15.jpg)

这种方式将不同检索器得到的结果交替进行排序，每次将结果压入是需要先判断是否存在。

* 按照各自的排序结果头尾排序并进行去重：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img16.jpg)

这种方式是将不同的结果分别放在头和尾，需要注意的是尾部的顺序需要倒过来。这种做法是根据大模型对较长的文本中头和尾的信息更加敏感的特点，所以这种方法一般针对检索结果较多以及最终的文本较长的情况。

## 重排序和过滤

重排序是指通过一个`reranker`模型对最终检索到的结果重新进行排序，过滤则是通过一些明显的业务属性来过滤明显错误的结果，或者通过`metadata`中的一些信息进行排序，过滤，增加，删减等一系列操作，这个主要看业务的属性来决定。

## 查询转换

查询转换是一系列技术的总称，使用**LLM作为推理引擎**来修改用户输入，以提高检索质量。有几种不同的方法可以做到这一点。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img17.jpg)

查询转换有几种方式：

* 一种是通过根据提问的问题转化为更加通用的主题的方式来降低检索的难度。例如，我在问办理某个产品的需要多少钱？通常的做法是让LLM将其转化为带关键词的更加高效的检索。

```python
# 这里的资费通常知识库常用的关键词
办理<某产品>的需要多少钱？ --> <某产品>的资费
```

* 另一种查询转换则是将复杂的查询分解为子查询。例如，如果用户问：“在Github上，Langchain和LlamaIndex哪个更优秀？”，而我们不太可能在语料库中的某些文本中找到直接的比较，所以将这个问题分解为两个子查询是有意义的，假设更简单、更具体的信息检索：“Langchain在Github上有多少星星？”“Llamaindex在Github上有多少星星？”它们将并行执行，然后将检索到的上下文合并到一个提示中，供LLM合成对初始查询的最终答案。这两个库都实现了这个功能——在Langchain中作为多查询检索器(Multi Query Retriever)，在Llamaindex中作为子问题查询引擎(Sub Question Query Engine)。

Langchain demo:
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

question = "What are the approaches to Task Decomposition?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)
```

* 回退提示(Step-back prompting)使用LLM生成一个更普适的查询，检索它我们获得一个更普适或更高层次的上下文语境，有助于支撑初始查询的答案。初始查询的检索同时被执行，两个上下文都被提交到LLM以便在最后一步生成答案。

## 参考引用

如果用户使用多个检索来源生成答案，无论是因为初始查询的复杂性（用户不得不执行多个子查询，然后将检索到的上下文合并为一个答案），还是因为在不同文档中找到了某个查询的相关上下文，那么就会出现一个问题，即Generator是否能准确地显示所使用的引用来源。

有几种方法可以做到这一点：

1、将这个**显示引用源的任务插入到prompt中**，要求LLM说明所使用的引用源编号。

例子： 
```python
prompt = """
...
[
    retrieve_results_1, source: A
    retrieve_results_2, source: B
]
...
请你说明你的回答使用了哪些检索结果，标注出相应的引用源。
"""
```

2、将生成的答案部分与索引中的原始文本块匹配——LlamaIndex为这种情况提供了一种高效的模糊匹配解决方案(fuzzy matching based solution)(模糊匹配(fuzzy matching)是一种非常强大的字符串匹配技术)。

**这种方法是通过算法来解决文本块使用的问题。**

上述两种方法各有优劣，需要根据实际情况进行灵活选择。


## 聊天引擎（多轮对话）

单体的RAG流程仅仅是针对单次问答的情况，并不会考虑上下文。如果要构建多次问答的RAG系统，则需要支持历史对话，但是多轮历史对话往往会很长，而大模型支持的token数比较有限。

要构建一个能够对单个查询多次执行操作的高效RAG系统，下一个重要步骤是实现聊天逻辑，这与经典聊天机器人在LLM时代之前处理对话上下文的方式类似。采用这种方式是为了**支持后续问题、指代解析或与之前对话上下文相关的任何用户指令**。解决这一问题的方法是通过**查询压缩(query compression)技术**，同时考虑聊天上下文和用户查询。

在上下文压缩方面，有几种方法可供选择：

* 一种比较流行且相对简单的方法是使用`ContextChatEngine`，它**首先检索与用户查询相关的上下文，然后将其连同聊天历史记录从内存缓冲区取出发送给LLM**，以便LLM在生成下一个答案时能够了解之前的上下文。
* 另一种相对复杂的情况是`CondensePlusContextMode`，在这种模式下，**每次交互中的聊天记录和最后一条消息被压缩成一个新的查询，然后这个查询被发送到索引里**，并把检索到的上下文连同原始用户消息一起传递给LLM以生成答案。这种技术需要LLM将最近几条的聊天记录和最新的一条消息进行压缩，保证整体的语义与用户的真实意图相同。

在实际的生产环境中，`ContextChatEngine`往往是不work的，因为你无法保证用户在输入问题时使用书面化的问法，**大部分口语化的问答方式，关键信息会保留在上一次问题中，而在本次提问中仅仅进行追问。在本次的查询中缺少关键信息，检索效果往往不佳。**

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img18.jpg)

聊天引擎是RAG系统中的一个关键组件，它不仅处理用户的直接查询，还能够理解和响应与之前对话上下文相关的内容。通过这种方式，RAG系统能够提供更连贯、更自然的对话体验。

## 查询路由

查询路由是指针对用户的查询，由LLM来决定下一步操作的决策步骤。通常的选择包括总结信息、对某些数据索引进行搜索，或尝试多种不同的路径并将它们的输出合成为唯一的答案。实现方式便是`Agent智能体`。

**查询路由器(Query Routers)**还用于选择索引，或者更通俗地说是选择执行用户查询命令的数据源。无论是拥有多个数据源（比如经典的向量存储、图形数据库或关系数据库），还是拥有多层索引结构（比如处理在多文档存储的时候，一个典型的索引创建方案很可能是一个由摘要组成的索引和另一个由文档块向量组成的索引），都需要进行查询路径选择。

其中所有的查询选择，结果判断，都需要大模型自己判断，而不是人工定制好一个路径。

### 多文档智能体方案


> Agents的概念：一个能够进行推理、提供一系列工具和一个完成特定任务的LLM

智能体提供的工具可能包括如各种语言代码功能，各种外部API，甚至各种其他智能体——这种LLM链式的想法是LangChain名字的由来。目前市面上的大模型都适配了工具调用能力，也就是将语言文本转化为使用API调用外部工具或数据库查询的能力。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img19.jpg)

上述图片中的`多文档智能体`的方案，将多个文档检索的方式都制作为智能体，每个智能体都能自行判断需要进行哪些子查询，自行决定查询完了之后需要使用哪个查询出来的结果。顶层的智能体则单纯使用路由的能力，来选择使用哪些查询智能体。

这种复杂方案的缺点基本可以从上图中推测出来——由于涉及智能体内部的多次LLM迭代，所以执行效率比较低下。**需要注意的是，LLM调用总是RAG管道中最费时的操作**。因此，对于大型多文档存储，建议考虑对这个方案进行一些简化，以便提高其可扩展性。最简单的方法就是将所有的文档智能体使用一套通用的固定查询路径。

## 响应合成器

这是每个RAG管道的最后一步——基于用户精心检索到的所有上下文和初始用户查询生成答案。最简单的方法是将所有检索到的上下文（高于某个相关性阈值）与查询一起连续地提交给LLM。但总有其他更复杂的方案，比如多次执行LLM调用来细化检索到的上下文，以生成更完美的答案。

响应合成的主要方法包括：

1、通过将检索到的上下文逐块提交给LLM来迭代优化答案。
2、摘要检索到的上下文以匹配prompt。
3、基于不同上下文块生成多个答案，然后将它们融合或摘要。

## RAG相关的微调

* embedding模型微调：[LLamaIndex](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/)提供了embedding模型微调的方式，根据国外博主的实测，bge-large-en-v1.5的embedding模型微调能够对RAG系统带来2%的性能提升。
* reranker模型微调：另一个好的旧选择是，如果您不完全信任基础编码器，则有一个交叉编码器来重新排名检索到的结果。它的工作方式如下——您将查询和每个检索到的前k个文本块传递给交叉编码器，由SEP令牌隔开，并微调为相关块输出1，不相关块输出0。这种调整过程的一个很好的例子可以在[这里](https://docs.llamaindex.ai/en/latest/examples/finetuning/cross_encoder_finetuning/cross_encoder_finetuning.html#)找到，结果显示，通过交叉编码器微调，配对得分提高了4%。

## 指标评估

### 基于人工的数据集评测

基于人工的RAG评测需要人工构建问题和对应目标文本的数据集。这是一个示例的数据集格式：
```json
{
    # relevant_docs_source是在整个知识库中的document-ID（存储在metadata信息中）
    "query_id": 1,
    "query_text": "<某产品>的优势是什么？",
    "relevant_docs_source": [40]
}
```
关于不同的类别，可以在其上层继续构建分类，用于不同的类别。
关于定量指标，通常选择一些常用的：`recall`, `precison`, `f1_score`, `hit_rate`。

* 召回率（Recall）：召回率是指系统正确检索到的相关文档数与所有相关文档数的比例。


$$Recall = \frac{检索到的相关文档数}{所有相关文档数}$$


* 精确率（Precision）：精确率是指系统检索到的相关文档数与检索到的所有文档数的比例。

$$Precision = \frac{检索到的相关文档数}{检索到的所有文档数}$$

* F1分数（F1 Score）：F1分数是精确率和召回率的调和平均值，用于综合衡量系统的性能。

$$ F1\-Score = 2 \times \frac{Precision * Recall}{Precision + Recall}$$

* 命中率（Hit Rate）：命中率是指系统在检索到的文档中至少有一个相关文档的比例。

$$Hit Rate = \frac{至少有一个相关文档的查询数}{总查询数}$$


**上述所有指标都需要结合最终检索的格式`top_n`结合起来看，在上面几个指标中，最重要的是`召回率`和`命中率`。这是因为你即便召回了部分噪声，大模型本身具有判断的能力，但如果没有正确的没有召回，大模型则没有信息来源，是不可能回答正确的。**

### 基于大模型的数据集评测

《Evaluating RAG Applications with RAGAs》文章介绍了一个用于评估RAG应用的框架，称为RAGAs，这篇文章详细介绍了RAGAS框架，它的核心目标是提供一套综合性的评估指标和方法，以量化地评估**RAG管道(RAG Pipeline)**在不同组件层面上的性能。RAGAs特别适用于那些结合了**检索（Retrieval）和生成（Generation）**两个主要组件的RAG系统。

**无参考评估**：RAGAs最初设计为一种“无参考”评估框架，意味着它不依赖于人工注释的真实标签，而是利用大型语言模型（LLM）进行评估。

**组件级评估**：RAGAs允许对RAG管道的两个主要组件——检索器和生成器——分别进行评估。这种分离评估方法有助于精确地识别管道中的性能瓶颈。

**综合性评估指标**：RAGAs提供了一系列评估指标，包括**上下文精度(Context Precision)、上下文召回(Context Recall)、忠实度(Faithfulness)和答案相关性(Answer Relevancy)**。这些指标共同构成了RAGAs评分，用于全面评估RAG管道的性能。

具体的评估流程和代码示例：[https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a](https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a)

**Reference**:[https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)
