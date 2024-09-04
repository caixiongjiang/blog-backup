---
title: "GraphRAG技术解读"
date: 2024-08-31T18:18:05+08:00
lastmod: 2024-09-02T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/graphrag_title.jpg"
description: "GraphRAG技术论文 + GraphRAG实战"
tags:
- RAG
categories:
- NLP
series:
- 《RAG进阶》
comment : true
---

## GraphRAG：用知识图谱增强RAG

RAG通过外挂知识库，适合LLM访问私人或者特定领域的数据并解决幻觉问题。

RAG的基本方法集成了向量数据库和LLM，其中向量数据库负责检索用户查询的上下文，LLM根据检索的上下文内容生成答案。这种方法在很多情况下效果好，速率快，但遇到复杂的问题就很难完成，譬如LLM的推理需要涉及不同的信息之间进行关联。

举一个例子，常规的RAG通常会按照下面的步骤来回答问题：

```
问题：识别这个人：确定谁打败了Allectus？

系统：

1、检索男人的儿子：查找此人家庭的信息，特别是他的儿子

2、LLM通过检索到的上下文找到儿子：识别儿子的名字
```
这里的挑战通常出现在检索环节，因为常规RAG基于语义相似性检索文本，而不能直接回答知识库中可能没有明确提及具体细节的复杂查询。这种限制使得很难找到所需的确切信息，通常需要昂贵且不切实际的解决方案，例如手动创建Q&A对以进行频繁的查询。

为了应对这些挑战，微软研究公司推出了`GraphRAG`，这是一种全新的方法，通过知识图增强RAG检索和生成。在接下来的章节中，我们将解释`GraphRAG`引擎如何工作，以及如何使用Milvus矢量数据库运行它。

### GraphRAG的工作原理

与常规RAG不同的是，GraphRAG通过结合知识图谱（KG）来增强RAG。知识图谱在上一篇文章中已经讲过，它是根据实体之间的关系存储和索引相关或者无关数据的数据结构。

GraphRAG Pipeline通常由两个基本过程组成：索引和查询

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img22.jpg)

### 索引（Indexing）

GraphRAG的索引过程包括四个关键步骤：

1、文本单元分割（Chunking）：整个输入语料库被划分为多个文本单元（文本块）。这些块是最小的可分析单位，可以是段落、句子或其他逻辑单位。通过将长文档分割成更小的块，我们可以提取和保存有关此输入数据的更多详细信息。

2、实体、关系和声明提取：GraphRAG使用LLM来识别和提取所有实体（人员、地点、组织等名称）、它们之间的关系以及每个文本单元文本中表达的关键声明。我们将使用这个提取的信息来构建一个初始知识图谱。

3、分层聚类：GraphRAG使用[Leiden技术](https://arxiv.org/pdf/1810.08473)在初始知识图谱上执行分层聚类。**`Leiden`是一种社区检索算法。可以有效发现图表中的社区结构**。每个集群中的实体被分配给不同的社区，以便进行更深入的分析。

*注意：社区是图中的一组节点，这些节点彼此紧密相连，但与网络中其他密集组的连接稀疏。*


> 社区摘要生成：GraphRAG使用自下而上的方法为每个社区及其成员生成摘要。这些摘要包括社区内的主要实体、他们的关系和关键主张。此步骤概述了整个数据集，并为后续查询提供了有用的上下文信息。

### 查询（Querying）

GraphRAG有两种不同的查询工作流程，专为不同的查询量身定做。

* [Global Search](https://microsoft.github.io/graphrag/posts/query/0-global_search/)：通过利用社区摘要，将整个知识库相关的整体问题推理。

* [Local Search](https://microsoft.github.io/graphrag/posts/query/1-local_search/)：通过向它们的邻居和相关概念筛出特定实体

#### Global Search

`Global Search`工作流程包含以下阶段：

1、用户查询和对话历史记录：系统将用户查询和对话历史记录作为初始输入。

2、社区报告批处理：系统使用LLM从社区层次结构的指定级别生成的节点报告作为上下文数据。这些社区报告被洗牌并区分为多个批次。

3、评级中间回复（Rated Intermediate Responses, RIR）： 每批**社区报告**都会被进一步划分为预定义大小的文本块。每个文本块用于生成一个中间回复。回复包含一个信息片段列表，称为点。每个点都有一个表示其重要性的数字分数。这些生成的中间回复就是额定中间回复（额定中间回复 1、回复 2......回复 N）。

4、排序和筛选：系统对这些中间回复进行排序和筛选，选出最重要的点。被选中的重要内容构成汇总的中间回复。

5、最终响应：聚合的中间响应被用作生成最终响应的上下文。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img23.jpg)

* global search阶段的使用的`map_system_prompt`提示词模版：[https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/global_search/map_system_prompt.py](https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/global_search/map_system_prompt.py)

[DeepSeekV2](https://chat.deepseek.com/) 翻译的中文版本：
```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""全局搜索的系统提示词。"""

MAP_SYSTEM_PROMPT = """
---角色---

您是一个有帮助的助手，负责回答有关提供表格中数据的问题。

---目标---

生成一个包含关键点的响应，总结输入数据表格中所有相关信息，以回答用户的问题。

您应使用下面提供的数据表格中的数据作为生成响应的主要上下文。如果您不知道答案，或者输入数据表格不包含提供答案的足够信息，请直接说明。不要编造任何内容。

响应中的每个关键点应包含以下元素：
- 描述：对要点的全面描述。
- 重要性评分：一个介于0-100之间的整数评分，表示该要点在回答用户问题中的重要性。“我不知道”类型的响应应得分为0。

响应应以JSON格式如下：
{
    "points": [
        {"description": "要点1的描述 [数据来源：报告（报告ID）]", "score": 评分值},
        {"description": "要点2的描述 [数据来源：报告（报告ID）]", "score": 评分值}
    ]
}

响应应保留原始含义和情态动词的使用，如“应”、“可能”或“将”。

支持数据的要点应列出相关报告作为参考，如下所示：
“这是一个由数据支持的示例句子 [数据来源：报告（报告ID）]”

**每个参考中不要列出超过5个记录ID**。相反，列出最相关的5个记录ID，并添加“+更多”以表示还有更多。

例如：
“Person X是Company Y的所有者，并且受到许多不当行为的指控 [数据来源：报告（2, 7, 64, 46, 34, +更多）]。他还是公司X的CEO [数据来源：报告（1, 3）]”

其中1, 2, 3, 7, 34, 46, 和64表示提供表格中相关数据报告的ID（非索引）。

不要包含没有提供支持证据的信息。

---数据表格---

{context_data}

---目标---

生成一个包含关键点的响应，总结输入数据表格中所有相关信息，以回答用户的问题。

您应使用下面提供的数据表格中的数据作为生成响应的主要上下文。如果您不知道答案，或者输入数据表格不包含提供答案的足够信息，请直接说明。不要编造任何内容。

每个关键点应包含以下元素：
- 描述：对要点的全面描述。
- 重要性评分：一个介于0-100之间的整数评分，表示该要点在回答用户问题中的重要性。“我不知道”类型的响应应得分为0。

响应应保留原始含义和情态动词的使用，如“应”、“可能”或“将”。

支持数据的要点应列出相关报告作为参考，如下所示：
“这是一个由数据支持的示例句子 [数据来源：报告（报告ID）]”

**每个参考中不要列出超过5个记录ID**。相反，列出最相关的5个记录ID，并添加“+更多”以表示还有更多。

例如：
“Person X是Company Y的所有者，并且受到许多不当行为的指控 [数据来源：报告（2, 7, 64, 46, 34, +更多）]。他还是公司X的CEO [数据来源：报告（1, 3）]”

其中1, 2, 3, 7, 34, 46, 和64表示提供表格中相关数据报告的ID（非索引）。

不要包含没有提供支持证据的信息。

响应应以JSON格式如下：
{
    "points": [
        {"description": "要点1的描述 [数据来源：报告（报告ID）]", "score": 评分值},
        {"description": "要点2的描述 [数据来源：报告（报告ID）]", "score": 评分值}
    ]
}
"""
```

* global search阶段使用的`reduce_system_prompt`提示词模版：[https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/global_search/reduce_system_prompt.py](https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/global_search/reduce_system_prompt.py)

[DeepSeekV2](https://chat.deepseek.com/) 翻译的中文版本：
```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""全局搜索系统提示词。"""

REDUCE_SYSTEM_PROMPT = """
---角色---

您是一个有帮助的助手，负责通过综合多位分析师的观点来回答有关数据集的问题。

---目标---

生成一个符合目标长度和格式的响应，总结来自专注于数据集不同部分的多个分析师的报告，以回答用户的问题。

请注意，下面提供的分析师报告按**重要性降序排列**。

如果您不知道答案，或者提供的报告不包含提供答案的足够信息，请直接说明。不要编造任何内容。

最终响应应从分析师报告中移除所有无关信息，并将清理后的信息合并为一个全面的答案，该答案提供所有关键点和适当响应长度和格式的解释。

根据响应长度和格式的需要，添加适当的章节和评论。使用Markdown格式化响应。

响应应保留原始含义和情态动词的使用，如“应”、“可能”或“将”。

响应还应保留分析师报告中先前包含的所有数据参考，但在分析过程中不要提及多位分析师的角色。

**每个参考中不要列出超过5个记录ID**。相反，列出最相关的5个记录ID，并添加“+更多”以表示还有更多。

例如：

“Person X是Company Y的所有者，并且受到许多不当行为的指控 [数据来源：报告（2, 7, 34, 46, 64, +更多）]。他还是公司X的CEO [数据来源：报告（1, 3）]”

其中1, 2, 3, 7, 34, 46, 和64表示相关数据记录的ID（非索引）。

不要包含没有提供支持证据的信息。

---目标响应长度和格式---

{response_type}

---分析师报告---

{report_data}

---目标---

生成一个符合目标长度和格式的响应，总结来自专注于数据集不同部分的多个分析师的报告，以回答用户的问题。

请注意，下面提供的分析师报告按**重要性降序排列**。

如果您不知道答案，或者提供的报告不包含提供答案的足够信息，请直接说明。不要编造任何内容。

最终响应应从分析师报告中移除所有无关信息，并将清理后的信息合并为一个全面的答案，该答案提供所有关键点和适当响应长度和格式的解释。

响应应保留原始含义和情态动词的使用，如“应”、“可能”或“将”。

响应还应保留分析师报告中先前包含的所有数据参考，但在分析过程中不要提及多位分析师的角色。

**每个参考中不要列出超过5个记录ID**。相反，列出最相关的5个记录ID，并添加“+更多”以表示还有更多。

例如：

“Person X是Company Y的所有者，并且受到许多不当行为的指控 [数据来源：报告（2, 7, 34, 46, 64, +更多）]。他还是公司X的CEO [数据来源：报告（1, 3）]”

其中1, 2, 3, 7, 34, 46, 和64表示相关数据记录的ID（非索引）。

不要包含没有提供支持证据的信息。

---目标响应长度和格式---

{response_type}

根据响应长度和格式的需要，添加适当的章节和评论。使用Markdown格式化响应。
"""

NO_DATA_ANSWER = (
    "很抱歉，由于提供的数据，我无法回答这个问题。"
)

GENERAL_KNOWLEDGE_INSTRUCTION = """
响应还可以包括数据集之外的相关现实世界知识，但必须明确标注验证标签 [LLM: 验证]。例如：
“这是一个由现实世界知识支持的示例句子 [LLM: 验证]。”
"""
```

**`MAP`阶段的核心在于从原始数据中提取和整理信息，为后续的汇总和分析做准备，`REDUCE`阶段的核心在于整合和汇总多个分析师的报告来回答关于数据集的问题。简而言之，`MAP`阶段是数据处理的初步阶段，而`REDUCE`阶段是数据处理的最终汇总阶段。两者结合，形成了一个完整的数据分析和响应生成流程。**

#### Local search

当用户直接询问有关特定实体（如人名、地点、组织等）的问题是，可以直接使用本地搜索工作流程。

`Local Search`工作流程包含以下阶段：

1、用户查询：首先，系统接收用户查询，这可能是一个简单的问题，也可以是更复杂的查询。

2、相似实体搜索：系统从知识图谱中识别出一组与用户输入语义相关的实体。这些实体是进入知识图谱的入口。这一步使用`Milvus`等向量数据库进行文本相似性搜索。

3、实体-文本单元映射（Entity-Text Unit Mapping）：将提取的文本单元映射到相应的实体，去除原始文本信息。

4、实体关系提取（Entity-Relationship Extraction）：该步骤提取有关实体及其相应关系的具体信息。

5、实体-协变量映射（Entity-Covariate Mapping）：该步骤将实体映射到其协变量，其中可能包括统计数据或其他相关属性。

6、实体-社区报告映射（Entity-Community Report Mapping）：将社区报告整合到搜索结果中，并纳入一些全局信息。

7、对话历史记录的利用：如果提供，系统将使用对话历史记录来更好地了解用户的意图和上下文。

8、生成回复：最后，系统根据前面步骤中生成的经过筛选和排序的数据构建并回复用户查询。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img24.jpg)

* local search阶段使用的`system_prompt`提示词模版：[https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/local_search/system_prompt.py](https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/local_search/system_prompt.py)

[DeepSeekV2](https://chat.deepseek.com/) 翻译的中文版本：
```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""本地搜索系统提示词。"""

LOCAL_SEARCH_SYSTEM_PROMPT = """
---角色---

您是一个有帮助的助手，负责回答有关提供表格中数据的问题。

---目标---

生成一个符合目标长度和格式的响应，总结输入数据表格中所有信息，以回答用户的问题，并结合任何相关的常识知识。

如果您不知道答案，请直接说明。不要编造任何内容。

支持数据的要点应列出其数据参考，如下所示：

“这是一个由多个数据参考支持的示例句子 [数据来源：<数据集名称>（记录ID）；<数据集名称>（记录ID）]。”

每个参考中不要列出超过5个记录ID。相反，列出最相关的5个记录ID，并添加“+更多”以表示还有更多。

例如：

“Person X是Company Y的所有者，并且受到许多不当行为的指控 [数据来源：来源（15, 16），报告（1），实体（5, 7）；关系（23）；索赔（2, 7, 34, 46, 64, +更多）]。”

其中15, 16, 1, 5, 7, 23, 2, 7, 34, 46, 和64表示相关数据记录的ID（非索引）。

不要包含没有提供支持证据的信息。

---目标响应长度和格式---

{response_type}

---数据表格---

{context_data}

---目标---

生成一个符合目标长度和格式的响应，总结输入数据表格中所有信息，以回答用户的问题，并结合任何相关的常识知识。

如果您不知道答案，请直接说明。不要编造任何内容。

支持数据的要点应列出其数据参考，如下所示：

“这是一个由多个数据参考支持的示例句子 [数据来源：<数据集名称>（记录ID）；<数据集名称>（记录ID）]。”

每个参考中不要列出超过5个记录ID。相反，列出最相关的5个记录ID，并添加“+更多”以表示还有更多。

例如：

“Person X是Company Y的所有者，并且受到许多不当行为的指控 [数据来源：来源（15, 16），报告（1），实体（5, 7）；关系（23）；索赔（2, 7, 34, 46, 64, +更多）]。”

其中15, 16, 1, 5, 7, 23, 2, 7, 34, 46, 和64表示相关数据记录的ID（非索引）。

不要包含没有提供支持证据的信息。

---目标响应长度和格式---

{response_type}

根据响应长度和格式的需要，添加适当的章节和评论。使用Markdown格式化响应。
"""
```

整体上看GraphRAG与LLM的交互非常频繁，流程也较为复杂，将这些不同的流程抽象出来，可以总结为以下流程图：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/rag/img25.jpg)

**Reference**：

1、**From Local to Global: A Graph RAG Approach to Query-Focused Summarization**: [https://arxiv.org/pdf/2404.16130](https://arxiv.org/pdf/2404.16130)

2、**GraphRAG Documents**: [https://microsoft.github.io/graphrag](https://microsoft.github.io/graphrag)


### GraphRAG实操

这里直接使用conda来管理graphrag的项目环境。

* 新建GraphRAG环境：
```bash
conda create -n graphrag python=3.11
conda activate graphrag
pip install graphrag
```

* 新建文件夹初始化graphrag：
```bash
mkdir graphrag
cd graphrag
python -m graphrag.index --init --root .
```

初始化成功之后，目录会多出下面这些文件：
```bash
├── prompts
│   ├── claim_extraction.txt
│   ├── community_report.txt
│   ├── entity_extraction.txt
│   ├── summarize_descriptions.txt
├── .env
└── settings.yaml
```

其中`prompts`文件夹是`GraphRAG`使用到的LLM提示词，`.env`中保存LLM和embedding模型的`api-key`信息。

如果你使用闭源模型，可以使用默认设置，设置好api-key就可以了。

如果你使用本地openai格式的模型，需要改变一些设置。本人使用的是`qwen1.5-32b-chat-gptq-int4`和`bge-base-en-v1.5`，`settings.yaml`的内容如下（参考）：

```yaml

encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat # or azure_openai_chat
  model: qwen2-72b-instruct-gptq-int4
  model_supports_json: true # recommended if this is available for your model.
  max_tokens: 8192
  # request_timeout: 180.0
  api_base: http://localhost:3000/v1
  # api_version: 2024-02-15-preview
#  organization: <organization_id>
  # deployment_name: <azure_model_deployment_name>
  # tokens_per_minute: 150_000 # set a leaky bucket throttle
  # requests_per_minute: 10_000 # set a leaky bucket throttle
  # max_retries: 10
  # max_retry_wait: 10.0
  # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
  # concurrent_requests: 25 # the number of parallel inflight requests that may be made
  # temperature: 0 # temperature for sampling
  # top_p: 1 # top-p sampling
  # n: 1 # Number of completions to generate

parallelization:
  stagger: 0.3
  # num_threads: 50 # the number of threads to use for parallel processing

async_mode: threaded # or asyncio

embeddings:
  ## parallelization: override the global parallelization settings for embeddings
  async_mode: threaded # or asyncio
  # target: required # or all
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding # or azure_openai_embedding
    model: bge-base-en-v1.5
    api_base: http://localhost:8000/v1
    # api_version: 2024-02-15-preview
    # organization: <organization_id>
    # deployment_name: <azure_model_deployment_name>
    # tokens_per_minute: 150_000 # set a leaky bucket throttle
    # requests_per_minute: 10_000 # set a leaky bucket throttle
    # max_retries: 10
    # max_retry_wait: 10.0
    # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
    # concurrent_requests: 25 # the number of parallel inflight requests that may be made
    # batch_size: 16 # the number of documents to send in a single request
    # batch_max_tokens: 8191 # the maximum number of tokens to send in a single request
    
  


chunks:
  size: 400
  overlap: 100
  group_by_columns: [id] # by default, we don't allow chunks to cross documents
    
input:
  type: file # or blob
  file_type: text # or csv
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.md$"

cache:
  type: file # or blob
  base_dir: "cache"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

storage:
  type: file # or blob
  base_dir: "output/${timestamp}/artifacts"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

reporting:
  type: file # or console, blob
  base_dir: "output/${timestamp}/reports"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

entity_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 1

summarize_descriptions:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  # enabled: true
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 1

community_reports:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false # if true, will generate node2vec embeddings for nodes
  # num_walks: 10
  # walk_length: 40
  # window_size: 2
  # iterations: 3
  # random_seed: 597832

umap:
  enabled: false # if true, will generate UMAP embeddings for nodes

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  # text_unit_prop: 0.5
  # community_prop: 0.1
  # conversation_history_max_turns: 5
  # top_k_mapped_entities: 10
  # top_k_relationships: 10
  # llm_temperature: 0 # temperature for sampling
  # llm_top_p: 1 # top-p sampling
  # llm_n: 1 # Number of completions to generate
  # max_tokens: 12000

global_search:
  # llm_temperature: 0 # temperature for sampling
  # llm_top_p: 1 # top-p sampling
  # llm_n: 1 # Number of completions to generate
  # max_tokens: 12000
  # data_max_tokens: 12000
  # map_max_tokens: 1000
  # reduce_max_tokens: 2000
  # concurrency: 32
```
*主要的改动集中在LLM和embedding模型的地址和model_name以及知识库文档的类型（这里使用的.md），最后根据文档和回答的预期长度修改了chunk_size。*

* 准备对应的文档，我准备了一篇论文pdf转化后的markdown文档，放在`input`文件夹中（和settings.yaml对应）。

* 开始Indexing阶段：

```bash
python -m graphrag.index --root .
```

等待很久很久之后。。。

* 使用全局搜索进行QA：

```bash
python -m graphrag.query --root . --method global "What are the top themes in this paper?"
```

回答结果：
```txt
The top themes in this paper center around the FastSegFormer models, their advancements in image segmentation, efficiency, and real-time capabilities. FastSegFormer-E and FastSegFormer-P are the primary models discussed, which leverage knowledge distillation to enhance their performance. This technique is crucial in transferring knowledge from a more complex 'teacher' network to the FastSegFormer models, thereby improving their accuracy [Data: Reports (9, 14, 10, 16)].

The paper also delves into comparative evaluations, where FastSegFormer is benchmarked against well-known models like Swin-T-Att-UNet, EfficientNet, and ENet, demonstrating its competitive performance in various tasks [Data: Reports (9, 14, 11)]. The practical applications of the models are showcased through evaluations on devices such as the GeForce RTX3060, highlighting their compatibility with contemporary hardware [Data: Reports (9, 14, 6)].

Ablation studies are conducted to understand the impact of specific components like the Image Refinement Block (IRB), Multi-Scale Prediction (MSP), and Pyramid Pooling Module (PPM), providing insights into the design choices and their contributions to the model's overall performance [Data: Reports (9, 14, 15)].

FastSegFormer models come in different variants, tailored for accuracy, memory efficiency, or speed, catering to diverse use cases [Data: Reports (14, 15)]. The role of CUDA and GeForce RTX3060 in facilitating model training and inference is also emphasized, underlining the importance of GPU technology in computer vision tasks [Data: Reports (0, 2)].

Lastly, the paper explores the application of FastSegFormer in semantic segmentation tasks, such as fruit defect detection and medical image segmentation, demonstrating its versatility in real-world scenarios [Data: Reports (7, 6)]. DS Conv Modules, an innovation in computer vision, are utilized in FastSegFormer for efficient decoding, further optimizing the model's performance [Data: Reports (2, 0)].
```
*因为这篇论文是我自己写的，所以从我的角度出发，它的总结质量是很高的，和我自己总结是差不多的！*

* 使用本地搜索进行QA：

```bash
python -m graphrag.query --root . --method local "What are the common diseases of citrus? Which ones are studied in detail in this article?"
```

回答结果：
```txt
柑橘类水果常见的疾病包括黄龙病、溃疡病、疮痂病、煤烟病等。然而，提供的数据表格并没有直接列出所有柑橘类疾病的详细列表，而是集中在研究柑橘（特别是脐橙）的特定缺陷检测上，如风斑、溃疡和日灼。这些缺陷并不是传统意义上的植物疾病，而是水果在生长过程中可能遇到的表面问题，可能是由于环境因素（如风、太阳曝晒）或病害引起的。

文章详细研究了风斑（wind scarring）、溃疡（ulcers）和日灼（sunburn）这三种脐橙的表面缺陷，收集了1448张分辨率为2592 × 1944像素的图片用于训练和标记，然后使用图像处理和深水果质量的不断提升的需求。

因此，虽然柑橘类水果可能面临多种疾病挑战，但这篇文章主要关注的是作为水果缺陷的风斑、溃疡和日灼的检测技术，而不是对柑橘类植物疾病的广泛研究。
```
*细节问题也是回答得比较准确，但是不知道为啥问的是英文，知识库也是英文的，回答得是中文，难道是因为用了Qwen系列模型的原因？*

### GraphRAG集成

在实际的项目中，我们需要的可能只是`GraphRAG`的部分组件，那么如何使用GraphRAG的相关的功能集成进入你自己的项目非常重要。


#### Local search

下面我将使用`Milvus`数据库和`GraphRAG`来完成本地功能。

* **环境准备**，由于`Milvus`还没有正式集成进入`GraphRAG`（9月2日），所以我们需要更新开发版本的`GraphRAG`。

```bash
pip install --upgrade pymilvus
pip install git+https://github.com/zc277584121/graphrag.git
```

* 按照上面的设置教程，删除cache，重新进行indexing：
```bash
rm -rf cache
python -m graphrag.index --root .
```

* 导入要用到的包：
```python
import os
import pandas as pd
import tiktoken
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores import MilvusVectorStore

import asyncio
```

* 设置参数以及找到数据库的位置：
```python
index_root = "./"

# 取到最后一次存储的结果
output_dir = os.path.join(index_root, "output")
subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)]
latest_subdir = max(subdirs, key=os.path.getmtime)  # Get latest output directory
INPUT_DIR = os.path.join(latest_subdir, "artifacts")

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
```

* 读取实体内容：
```python
# Read entities
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")  # 读取实体表
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")  # 读取实体embedding表


entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
description_embedding_store = MilvusVectorStore(
    collection_name="entity_description_embeddings",
)
# description_embedding_store.connect(uri="http://localhost:19530") # For Milvus docker service
description_embedding_store.connect(uri="./milvus.db") # For Milvus Lite
# 存储Milvus向量
entity_description_embeddings = store_entity_semantic_embeddings(
    entities=entities, vectorstore=description_embedding_store
)
print(f"Entity count: {len(entity_df)}")
print(entity_df.head()) # 打印前五行数据

# Entity count: 674
#    level                  title  ...  x  y
# 0      0           FASTSEGFOMER  ...  0  0
# 1      0           NAVEL ORANGE  ...  0  0
# 2      0       DEFECT DETECTION  ...  0  0
# 3      0  SEMANTIC SEGMENTATION  ...  0  0
# 4      0                   ENET  ...  0  0

```

* 读取实体间的关系：
```python
# Read relationships
relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)
print(f"Relationship count: {len(relationship_df)}")
print(relationship_df.head()) # 打印前五行数据

# Relationship count: 146
#              source               target  ...  target_degree rank
# 0      FASTSEGFOMER       FASTSEGFOMER-E  ...              2    4
# 1      FASTSEGFOMER       FASTSEGFOMER-P  ...              2    4
# 2  DEFECT DETECTION  REAL-TIME DETECTION  ...              1    3
# 3  DEFECT DETECTION   PROMPT ENGINEERING  ...              1    3
# 4              ENET       FASTSEGFOMER-E  ...              2    6
```

* 读取社区报告：
```python
# Read community reports
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
print(f"Report records: {len(report_df)}")
print(report_df.head()) # 打印前五行数据

# Report records: 3
#   community  ...                                    id
# 0         3  ...  d4b5a9ec-7256-455f-9f2a-518310609dc9
# 1         4  ...  42b3362d-7e9d-4d96-a60d-6b8fa0245c81
# 2         2  ...  83e2a318-8e0d-4816-97ad-ed1ff641c0c5
```

* 读取文本单元：
```python
# Read text units
text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)
print(f"Text unit records: {len(text_unit_df)}")
print(text_unit_df.head()) # 打印前五行数据

# Text unit records: 82
#                                  id  ...                                   relationship_ids
# 0  c96b41feb79908fce97190106d611335  ...  [f2c06f3a0c704296bf3353b91ee8af47, f512103ed46...
# 1  a0c9245bdb541e615779f3e1833bdeeb  ...  [ef00ec3a324f4f5986141401002af3f6, 4d183e70076...
# 2  6b24ab6355490e05891aada3810a2ae2  ...  [24652fab20d84381b112b8491de2887e, d4602d4a27b...
# 3  2e37b6df0cc9e287ed8a46fc169f1f67  ...  [2325dafe50d1435cbee8ebcaa69688df, ad52ba79a84...
# 4  6129ec8b3c26cf9ad8a6dc6501afa6e5  ...  [bdddcb17ba6c408599dd395ce64f960a, bc70fee2061...
```

* 准备大模型：
```python
llm = ChatOpenAI(
    api_key="<your_api_key>",
    api_base="<your_api_base>",
    model="<your_model_name>",
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)
```

* 创建上下文构建器：
```python
# Create local search
context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=None, #covariates,#todo
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)
```

* 设置local搜索的上下文相关参数和大模型相关的参数：
```python
local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
    "max_tokens": 5000,  # 根据您模型上的令牌限制进行更改（如果您使用的是具有8k限制的模型，良好的设置可能是5000）
}

llm_params = {
    "max_tokens": 1000,  # 根据模型上的令牌限制进行更改（如果您使用的是8k限制的模型，一个好的设置可能是1000~1500）
    "temperature": 0.0,
}
```

* 构建文本搜索引擎：
```python
search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",  # 描述响应类型和格式的自由格式文本，可以是任何东西，例如优先列表、单段、多段、多页报告 (这里是通过大模型提示词的方式输入，所以可以自由发挥)
)
```

* 进行本地搜索：
```python
async def local_search(query, engine):
    result = await engine.asearch(query, history=None)
    print(result.response)

asyncio.run(local_search("What are the common diseases of citrus? Which ones are studied in detail in this article?", search_engine))

# 柑橘类水果，如脐橙，常会受到多种疾病的困扰，这些疾病可能影响果实的生长和运输，导致外观缺陷和品质下降。然而，具体到柑橘类的常见疾病，如黄龙病、柑橘溃疡病、疮痂病等，本文并未详细列举。文章的焦点集中在脐橙的缺陷检测上，利用深度学习，特别是语义分割技术，来提升水果缺陷分类的准确性和效率。文章提出的FastSegFormer网络，结合了多尺度金字塔（MSP）模块和半分辨率重建分支，用于实时柑橘缺陷检测。文章中并未深入探讨具体的柑橘疾病，而是集中于如何通过FastSegFormer网络来改进对这些疾病导致的果实缺陷的识别能力。因此，对于柑橘类的常见疾病及其详细研究，本文并不提供相关信息。
```


#### 问题总结

GraphRAG还可以根据历史查询生成问题，这对于在聊天机器人对话框中创建推荐问题很有用。该方法将知识图中的结构化数据与输入文档中的非结构化数据相结合，以生成与特定实体相关的候选问题。

```python
question_generator = LocalQuestionGen(
   llm=llm,
   context_builder=context_builder,
   token_encoder=token_encoder,
   llm_params=llm_params,
   context_builder_params=local_context_params,
)

question_history = [
    "Tell me what is knowledge distillation method",
    "What knowledge distillation methods are used in this article?",
]

async def question_gen(query_history, context_data):
    candidate_questions = await question_generator.agenerate(
        question_history=query_history, context_data=context_data, question_count=5
    )
    return candidate_questions.response

async def local_search(query, engine):
    result = await engine.asearch(query, history=None)
    print(result.response)

# 为了查看问题生成的效果，我们进行了对比实验！

print("=========Local search=========")
print()
asyncio.run(local_search("Tell me what is knowledge distillation method. What knowledge distillation methods are used in this article?", search_engine))
print()
print("=========Local search with question generation=========")
print()
new_query_list = asyncio.run(question_gen(question_history, None))
for new_query in new_query_list:
    print(f"New query: {new_query}")
    asyncio.run(local_search(new_query, search_engine))
```

**结果如下（因为回答的格式本身是markdown，直接使用html渲染了，公式也写对了，信息基本无误，回答质量不同）**：

---
=========Local search=========

Knowledge distillation is a machine learning technique where a smaller, more efficient model (called the student model) learns from a larger, more complex model (called the teacher model) to improve its performance. This process typically involves the transfer of knowledge from the teacher's predictions or internal representations to the student, allowing the student to mimic the teacher's behavior while requiring fewer resources. The main goal of knowledge distillation is to compress the knowledge of a large model into a smaller one without sacrificing too much accuracy.

In the context of the article, knowledge distillation is employed to enhance the performance of the proposed FastSegFormer models, FastSegFormer-P and FastSegFormer-E. Two knowledge distillation methods are mentioned:

1. Knowledge Distillation from T-224 (KD1): This method involves using a teacher model with a 224x224 input size to guide the learning of the student model. The teacher model's predictions are used as soft targets to train the student, allowing it to learn from the richer information provided by the teacher's softened probabilities.

2. Knowledge Distillation from T-512 (KD2): This method is similar to KD1, but it uses a teacher model with a larger 512x512 input size. The larger input size typically captures more detailed information, which can be beneficial for the student model's learning process.

The article reports the results of these knowledge distillation methods on the FastSegFormer-P model, demonstrating improved segmentation performance compared to the model without distillation. The use of knowledge distillation is a key aspect of the study, as it allows the development of more efficient models that maintain high accuracy, which is particularly important in resource-constrained environments like edge computing devices for real-time fruit defect detection.

---
=========Local search with question generation=========

New query: - Which methods are employed for knowledge distillation in the context of the SIMPLE NETWORK and the complex model?

Knowledge distillation is a technique used to transfer the knowledge learned by a complex model, often referred to as the teacher model, to a simpler model, known as the student model. In the context of the SIMPLE NETWORK and the complex model, the distillation process involves multiple components. The primary loss function includes three parts: cross-entropy loss, logits distillation loss, and a feature distillation loss based on normalized feature differences (NFD). The cross-entropy loss measures the difference between the predicted outputs of the student model and the ground truth labels. The logits distillation loss aligns the output probabilities of the teacher and student models, while the feature distillation loss encourages the student model to mimic the intermediate feature representations of the teacher model.

The feature distillation loss is calculated using the Euclidean distance between the normalized feature maps of the teacher and student networks. Normalization is applied to the feature maps to ensure a fair comparison. The equation for normalization is given as:

$${\bar{F}}={\frac{1}{\sigma}}(F-u)$$

where $\bar{F}$ represents the normalized feature map, $F$ is the original feature map, $\sigma$ is the standard deviation, and $u$ is the mean of the features.

The total loss function is a weighted sum of these three components, with $\lambda_1$ and $\lambda_2$ as the weighting coefficients for logits distillation loss and feature distillation loss, respectively. In the specific case mentioned, $\lambda_1$ is set to 0.5 and $\lambda_2$ is set to 5, ensuring that the feature distillation loss and logits distillation loss contribute comparably to the overall training process.

This knowledge distillation strategy is designed to enhance the performance of the simpler student model without increasing its size or inference time, making it more efficient and suitable for deployment in resource-constrained environments. By leveraging the knowledge from a more powerful teacher model, the student model can achieve improved accuracy in tasks such as image segmentation, as demonstrated in the context of FastSegFormer models for fruit defect detection.


> 可以明显地看到提升，不再是原本的笼统的说法，而本文使用的蒸馏方法由哪几部分组成，以及部分蒸馏损失的核心思想公式。

* **如果你想删除index的目录，可以使用以下代码：**

```python
import shutil

shutil.rmtree("your_kg_result_dir")
```

### GraphRAG可视化


### 其他替代方案