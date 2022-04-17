---
title: "《STL源码剖析》第六讲 学习笔记"
date: 2022-04-08T18:18:05+08:00
lastmod: 2022-04-08T18:18:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/STL.png"
description: "主要用于了解stl各种容器的内部构造，熟悉使用c++容器，并了解其底层原理。"
tags:
- c++
categories:
- 《STL源码剖析》学习笔记
series:
- 《STL源码剖析》学习笔记
comment : true
---

## 第六讲：迭代器设计的原则&iterator traits的作用与设计

### iterator traits
iterator traits是为了提取出迭代器的特点，迭代器必须有能力回答algorithm提出的问题。

c++标准库设计了5种回答方式,集成在class中（迭代器相关的类型，迭代器模版中本身必须定义出来）：
* iterator_category：迭代器的类型，具体表现在能否++或者--或者一下加减很多步
* difference_type：代表两个iterator的距离应该用什么type来表现（随机访问：unsign interger，实现链表的都用ptrdiff_t类型）
* value_type：iterator所指元素的类型
* reference：引用（暂未使用）
* pointer：指针（暂未使用）

![5种associated type](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img6_1.jpg)

但如果iterator本身不是class，例如native pointer（退化的iterator），如何回答算法的5个问题？

![iterator traits](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img6_2.jpg)


![iterator traits](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img6_3.jpg)
iterator traits用于分离class iiterators和non-class iterators

