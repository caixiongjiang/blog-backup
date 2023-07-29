---
title: "M1 Mac安装LabelImg及使用"
date: 2023-07-28T18:18:05+08:00
lastmod: 2023-07-29T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E5%AE%9E%E4%B9%A0/shixi_tiltle.jpg"
description: "解决M1 Mac下安装LabelImg的依赖问题，pyqt5使用pyqt6代替，以及LabelImg的使用"
tags:
- 实习笔记
categories:
- 实习
series:
- 实习笔记
comment : true
---


## LabelImg

### M1 Mac 安装LabelImg

如果你直接使用命令行安装：
```shell
$ pip install labelimg
```

你会发现你的M1笔记本会报错，查了一些资料，找到了解决方法。

总体思路就是使用pyqt6代替pyqt5，从源文件进行安装：

* 1.找到问题的解决方案：[https://github.com/HumanSignal/labelImg/tree/pyside6](https://github.com/HumanSignal/labelImg/tree/pyside6)
* 2.下载分支文件：必须安装该分支的文件（download zip且解压），不能使用git clone，这样会变成下载主分支。或者可以直接clone分支：
```shell
$ git clone -b pyside6 https://github.com/HumanSignal/labelImg.git
```
* 3.进入文件目录
```shell
$ cd labelImg-pyside6
```
* 建立虚拟环境并进行安装：
```shell
$ conda create -n LabelImg python=3.9
$ conda activate LabelImg
$ pip3 install pyside6 lxml
$ make pyside6
$ python3 labelImg.py
```

OK!启动成功！

### LabelImg导出不同格式的标签文件
