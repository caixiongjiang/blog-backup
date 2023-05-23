---
title: "Docker配置深度学习环境"
date: 2023-05-17T18:18:05+08:00
lastmod: 2023-05-17T09:19:06+08:00
draft: True
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/img_title.jpg"
description: "为深度学习不同端侧的部署提供一个较好的方案。"
tags:
- Deep_learning
categories:
- 深度学习
series:
- 《深度学习》学习笔记
comment : true
---

## 深度学习环境搭建

### 租用云GPU服务

这里我使用的是[AutoDL](https://www.autodl.com/)，主要是价格比较便宜。

> 在租用GPU时，要注意所需项目的环境，特别是Pytorch和CUDA版本。服务器可以选择你需要的环境，如果没有你想要的环境，可以只选择`Miniconda`环境，选择一个使用的Unbantu和Python版本，然后自行下载需要的环境。

#### vscode连接服务器

- 打开vscode，下载`Remote - SSH`插件。
- 打开最左边的`Remote - SSH`模块，点击`+`，添加远程ssh登录指令，并输入`AutoDL`实例页面的登录指令，并按回车，再次按回车写入配置文件。
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/deep-learning%26computer-vision/img115.jpg)
- 点击右下角的`connect`，然后输入`AutoDL`示例页面提供的密码，回车，成功连接。
- 在vscode的插件商店中选择`Python`,并为**服务器端**下载插件。
- 在vscode中按下`command+shift+p`，输入python，选择`Python: Select Interpreter`，选择服务器中自动配置好的`base`环境。如果在配置实例时只选择了`Miniconda`，可以重新建立一个虚拟环境，安装`Pytorch大礼包`。

#### 配置服务器免密登录

这里我使用的是Mac操作系统进行配置，Windows具体流程相似。

操作流程：
- 在本机创建新的ssh key:
```shell
$ cd ~/.ssh
$ ssh-keygen -t rsa -C "ssh key的注释"
```
此时在`.ssh`目录下就会生成两个文件`id_rsa`和`id_rsa.pub`，分别是私钥和公钥。
- 复制公钥的内容到服务器端`/root/.ssh`目录下的`authorized_keys`（有时也是`authorized_keys2`）:
```shell
$ cat ~/.ssh/id_rsa.pub
$ # 然后选中复制，并粘贴到服务器端的目标位置
```
- 配置完成之后，本地进入服务器端就不需要一直输ssh密码了。

#### 上传文件

- 如果我们的文件可以在Github中找到，直接使用`Terminal`进行克隆：
```shell
$ git clone 你的repo的https地址
```
- 如果文件在本机，可以直接将文件拖拽到vscode目录下，也可以通过实例进入`JupyterLab`上传文件。

#### 谷歌Colab白嫖GPU资源

首先登录[Colab官网](https://colab.research.google.com/)，点击`新建笔记本`，点击`代码执行程序`，点击下拉列表里的`更改运行时类型`，选择GPU，保存。然后就可以开心地白嫖了！

### Docker配置深度学习环境

