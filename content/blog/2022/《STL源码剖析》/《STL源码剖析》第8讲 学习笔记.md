---
title: "第八讲：array和forward_list深度探索"
date: 2022-04-19T18:14:05+08:00
lastmod: 2022-04-19T18:14:06+08:00
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

## 第八讲：array和forward_list深度探索

### 容器array

*array在c中本身就存在，为什么array要成为一个容器？*

array成为容器的好处：array可以使用迭代器，可以使用标准算法，更加地快捷。

使用标准array的方法：
```c++
array<int, 10> myArray;
auto it = myArray.begin();
//array<int, 10>::iterator it = myArray.begin();
it += 3;
cout << *it;
```
源码如下图，较简单(TR1版本)：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img8_1.jpg)

### forward_list
单向链表的结构与list双向链表的结构类似，可以类推。（c++11以前为slist）
