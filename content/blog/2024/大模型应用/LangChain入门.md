---
title: "LangChain:LLM应用框架"
date: 2024-01-14T18:18:05+08:00
lastmod: 2024-01-15T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/LangChain_title.jpg"
description: "学习大模型应用的整体流程，以及应用技巧。"
tags:
- Deep_learning
categories:
- 大模型
series:
- 《LLM》
comment : true
---

### LangChain：LLM应用框架

下面是一个简单场景（LangChain结合本地知识库对用户的提问进行回答）的整体流程：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/LLM/img29.jpg)



#### 本地知识库处理

首先需要加载本地知识库文件，将其转化为知识向量库。

* 文本嵌入模型：它本身用于将高维的稀疏数据（文本）转化为低维稠密数据。主流的嵌入模型有**Word2Vec**、**GloVe**、**BERT**、**ERNIE**、**text2vec**。
* 向量索引库：它在不同领域中用于存储和高效查询向量数据，主要应用于图像、音频、视频、自然语言等领域的相似性搜索任务。主流的库有**faiss**、**pgvector**、**Milvus**、**pinecone**、**weaviate**、**LanceDB**、**Chroma**等。

```python
from langchain.document_loaders import DirectoryLoader, unstructured
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, faiss

import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
}

def load_documents(directory="books"):
    """
    Load documents
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # text: abcdefg, chunk_size:3, chunk_overlap: 1. result：abc cde efg
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)

    return split_docs


def load_embedding_model(model_name="ernie-tiny"):
    """
    Load embedding model
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}

    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name], # 这里如果不能在线下载，可以使用本地下载好的文件
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()

    return db




if __name__ == '__main__':
    embeddings = load_embedding_model("text2vec3")
    if not os.path.exists("VectorStore"):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)
```

#### 本地大模型加载

这里是使用ChatGLM-6B作为演示，且没有经过微调，使用预训练模型。本地模型推理服务在端口8000上启动，模型文件可以通过本地指定文件进行加载。

#### 用户输入处理

* RetrievalQA 是用于执行检索式问答任务的函数。检索式问答是一种基于信息检索的问答方法，它通过在大规模文本语料库中检索相关信息来回答用户提出的问题。RetrievalQA 函数的主要作用是执行以下任务：

  1. **问题理解：**RetrievalQA 函数接收用户提出的问题作为输入，并进行问题理解的处理。这包括对问题进行语法和语义分析，以确定问题的意图、关键信息和相关特征。

  2. **文本检索：**一旦问题被理解，RetrievalQA 函数使用预定义的文本检索方法来在大规模文本语料库中检索相关的文本片段或文档。这可以使用各种技术，如关键词匹配、向量相似度计算或者基于语义的匹配算法。

  3. **答案生成：**在完成文本检索后，RetrievalQA 函数从检索到的文本片段中提取或生成可能的答案。这可能涉及到答案抽取、文本摘要、实体识别等技术，以生成最终的答案。

* gradio是用户对话的一个UI框架，用于对话式大模型问答的界面生成。

```python
from langchain.document_loaders import DirectoryLoader, unstructured
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, faiss
from langchain.llms import ChatGLM
from langchain.chains import RetrievalQA

import gradio as gr

import os


embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
}

def load_documents(directory="books"):
    """
    Load documents
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # text: abcdefg, chunk_size:3, chunk_overlap: 1. result：abc cde efg
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)

    return split_docs


def load_embedding_model(model_name="ernie-tiny"):
    """
    Load embedding model
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}

    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name], # 这里如果不能在线下载，可以使用本地下载好的文件
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()

    return db




if __name__ == '__main__':
    embeddings = load_embedding_model("text2vec3")
    if not os.path.exists("VectorStore"):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)

    # 需要根据ChatGLM项目本地启动大模型服务在8000端口
    llm = ChatGLM(
        endpoint="http://127.0.0.1:8000",
        max_token=80000,
        top_p=0.9
    )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever
    )


    def chat(question, history):
        response = qa.run(question)
        return response

    demo = gr.ChatInterface(chat)
    demo.launch(inbrowser=True)
```

#### 支持文档上传功能的对话问答

除了提前在代码库里提供文档，我们同样可以临时上传文档进行知识库的更新。

代码如下：

```python
from langchain.document_loaders import DirectoryLoader, unstructured
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, faiss
from langchain.llms import ChatGLM
from langchain.chains import RetrievalQA

import gradio as gr

import time
import os


embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
}

def load_documents(directory="books"):
    """
    Load documents
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # text: abcdefg, chunk_size:3, chunk_overlap: 1. result：abc cde efg
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)

    return split_docs


def load_embedding_model(model_name="ernie-tiny"):
    """
    Load embedding model
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}

    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name], # 这里如果不能在线下载，可以使用本地下载好的文件
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()

    return db


# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video).
# Plus shows support for streaming text.


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    # 取到文件夹（临时文件夹）
    directory = os.path.dirname(file.name)
    documents = load_documents(directory)
    store_chroma(documents, embeddings)
    history = history + [((file.name,), None)]
    return history


def bot(history):
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "文件上传成功"
    else:
        response = qa.run(message)
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


if __name__ == "__main__":
    embeddings = load_embedding_model("text2vec3")
    if not os.path.exists("VectorStore"):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)

    # 需要根据ChatGLM项目本地启动大模型服务在8000端口
    llm = ChatGLM(
        endpoint="http://127.0.0.1:8000",
        max_token=80000,
        top_p=0.9
    )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever
    )
    
    with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["txt"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)
    
		demo.queue()
    demo.launch()
```


