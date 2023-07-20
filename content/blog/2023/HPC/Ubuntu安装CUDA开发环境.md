---
title: "Ubuntu CUDA编程环境配置"
date: 2023-07-14T18:18:05+08:00
lastmod: 2023-07-15T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img42.jpg"
description: "CUDA开发环境配置"
tags:
- CUDA编程
categories:
- HPC
series:
- CUDA
comment : true
---


## Ubuntu下CUDA环境配置

为了学习CUDA编程，我们需要一套Linux下的CUDA编程环境，需要注意的是我们的Linux下需要直通显卡，所以记住不能使用虚拟机！不能使用虚拟机！不能使用虚拟机！

那为了方便学习，又不能完全抛弃Windows，毕竟我还要写毕业论文。那么`双系统`就是最好的方案了。

### Windows & Ubuntu 双系统

首先，Linux发行版的选择，我建议还是用Ubuntu，毕竟群体多，社区才能维护。

[双系统安装参考视频](https://www.bilibili.com/video/BV11k4y1k7Li/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=841bd3506b40b195573d34fef4c5bdf7)

### Ubuntu安装显卡驱动、cuda、cudnn

我使用的配置如下：
* Ubuntu：20.04
* 显卡：RTX3060
* CUDA版本：11.7
* cudnn版本：8.9.0
* 显卡驱动版本：535

#### 显卡驱动安装

[Ubuntu安装显卡驱动，CUDA，cudnnn参考视频](https://www.bilibili.com/video/BV16Y411M7SC/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=841bd3506b40b195573d34fef4c5bdf7)

**注意，安装完驱动必须重启！**


#### CUDA和cudnn安装

[Ubuntu22.04安装CUDA环境参考文章](http://www.xbhp.cn/news/160895.html)

> CUDA工具包安装

首先进入[NVIDIA CUDA Toolkit Archive](https://developer.nvidia.cn/cuda-toolkit-archive)下载你想要的cuda工具版本，**需要根据你系统的版本和显卡驱动版本进行选择，只要支持就ok**。

在Ubuntu终端输入:
```shell
$ nvidia-smi
```
在显示的CUDA版本代表当前驱动支持的最高版本CUDA，我的最高版本为12.2，所以选择安装的CUDA版本需要小于12.2。

我选择了CUDA11.7的版本：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img42.jpg)

依次输入命令：
```shell
$ wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
```
下载完包之后，我们要按照第二条命令装CUDA，但是在安装这个之前，你需要安装一下`gcc的环境`，不然就会缺少依赖报错：
```shell
$ sudo apt update
$ sudo apt install gcc
# 顺便安装一下g++
$ sudo apt install g++
```
安装cuda包：
```shell
$ sudo sh cuda_11.7.0_515.43.04_linux.run
```
安装过程中间会跳出`Abort`和`continue`的选项，不要理会，选择`continue`。

后续又会跳出接受不接受，选择`accept`，之后便会跳出需要`install`的选项，盗用一下别人的截图：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img43.jpg)
*需要注意上述出现的Driver是CUDA11.7配套的驱动，但不一定适合显卡，但只需要我们的Driver版本比它高，都是可以用的。选中Driver，按空格去掉x，然后再选择install。*

> CUDA环境变量配置

打开终端，输入：
```shell
$ gedit ~/.bashrc
```
此时会出现一个记事本来编辑你的环境变量：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img44.jpg)
像上图那样，或者像下面那样都ok，输入：
```scss
# CUDA Environment
export PATH=$PATH:/usr/local/cuda-11.7/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
```
保存退出，然后输入：
```shell
$ source ~/.bashrc
```
或者直接重启终端也是OK的。

检查环境是否配置成功：
```shell
$ nvcc -V
```
如果出现了CUDA版本，那么就说明你已经配置成功了！

> cudnn安装

首先进入[NVIDIA cuDNN Archive](https://developer.nvidia.cn/rdp/cudnn-archive)下载你想要的cuDNN版本，**需要根据你的CUDA版本进行选择，只要支持就ok**。

所以我选择了`cuDNN v8.9.0, for CUDA 11.x`:
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img45.jpg)

中间需要注册NVIDIA官方开发者的账号，这个我就不展示了！

* 下载完成之后，在下载好的目录进入终端：
```shell
$ sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
```
* import CUDA GPG key:
```shell
# 注意这里需要根据终端提示你的命令进行下载，每个人的设备提示可能不一样
$ sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/cudnn-local-2063C34E-keyring.gpg /usr/share/keyrings/
```
* 刷新库
```shell
$ sudo apt-get update
```
* 安装runtime library：
```shell
# 注意，这里的libcudnn8和cuda版本的配对是指定的
# 可通过apt-cache policy libcudnn8命令查看。
# 我这里应该使用libcudnn8=8.9.0.131-1+cuda11.8
$ sudo apt-get install libcudnn8=8.9.0.131-1+cuda11.8
```
* 安装develop library：
```shell
$ sudo apt-get install libcudnn8-dev=8.9.0.131-1+cuda11.8
```
* 安装 the code samples and the cuDNN library documentation：
```shell
sudo apt-get install libcudnn8-samples=8.9.0.131-1+cuda11.8
```

> 验证cudnn是否安装成功
```shell
$ cp -r /usr/src/cudnn_samples_v8/ $HOME
$ cd $HOME/cudnn_samples_v8/mnistCUDNN
# 如果没有make包，则输入sudo apt-get make下载
$ make clean && make
$ ./mnistCUDNN
```
注：如果上面的make命令提示缺少`FreeImage.h`，运行：
```shell
$ sudo apt-get install libfreeimage3 libfreeimage-dev
```
如果出现下面的情况说明cudnn测试成功了：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img46.jpg)

#### Ubuntu配置编程工具
下载一个`vscode`，选择官网的`.deb`版本。

下载完成之后安装：
```shell
$ sudo dpkg -i code_1.80.1-1689183569_amd64.deb
```
启动`vscode`：
```shell
$ code
```

`vscode`添加扩展程序：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img47.jpg)

环境配置成功。
