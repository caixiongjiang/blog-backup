---
title: "《STL源码剖析》第三讲 学习笔记"
date: 2022-04-07T15:21:05+08:00
lastmod: 2022-04-07T15:21:06+08:00
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

## 第三讲:泛型编程和模版技术

### 源代码之分布

* VC的标准库文件夹:

![vc文件夹](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_1.jpg)
* GNU c++标准库文件夹:

![gnu c++文件夹](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_2.jpg)

### OOP(面向对象编程) vs GP(泛型编程)

* OOP将datas和methods关联在一起（成员变量和成员函数都放在类里面）
    demo:list内部本身存在有sort()算法，使用语法如下：
    ```c++
    c.sort()
    ```
* GP是将datas和methods分开来
    
    demo：vector和deque内部都没有sort()函数，sort()函数是放在算法里面的，使用语法如下：
    ```c++
    ::sort(c.begin(), c.end());
    ```
    注：*list本身是链式容器，无法支持随机访问，所以需要本身自己重新定义sort()。*

* 采用GP的好处：
    * 容器和算法团队可以各自写自己的东西，其间用迭代器来进行关联。
    * 算法通过迭代器确定操作范围，并通过迭代器取用容器的元素。
    * 两个石头如何比较大小不需要类来决定，而是用仿函数来决定。
* max()算法的demo：
  
  ![max()算法的两个版本](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_3.jpg)
  
  第二个版本的参数3接收的是**函数对象**或者是**仿函数**，**用于自定义的比较规则制定**！

### 操作符重载复习
操作符操作单个数或者多个数，以及能否成为类内部的成员函数

![操作符重载的类型](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_4.jpg)

### 模版Template复习

* 类模版:

![类模版举例](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_5.jpg)
* 函数模版:

![函数模版举例](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_6.jpg)
* 成员模版

#### 模版中泛化 vs 特化
* 泛化的特点是可以接收大多数type的数据结构
* 特化的特点是为了某些特殊的数据（数据有不同的特点）进行的模版中特殊化的写法

demo:

![泛化和特化](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_7.jpg)

![泛化和特化](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_8.jpg)

#### 特化中的偏特化（局部特化）

* 个数的偏特化:

![特化中的偏特化](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_9.jpg)
* 范围的偏特化：

![特化中的偏特化](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/stl/img3_10.jpg)