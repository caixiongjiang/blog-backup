---
title: "从零制作昇腾开发环境Docker镜像"
date: 2023-08-21T18:18:05+08:00
lastmod: 2023-08-22T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/huawei_title.jpg"
description: "介绍适配昇腾环境的Docker镜像制作"
tags:
- Deep_learning
categories:
- AI部署
series:
- 华为昇腾部署
comment : true
---

## 昇腾开发环境Docker镜像制作

### 昇腾计算服务器信息

登录服务器后，查看计算卡信息：
```shell
npu-smi info
```
显示如下信息：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img94.jpg)
**可以发现服务器npu信息可以正常获取，说明服务器的npu卡驱动和固件已经安装完毕，不需要我们继续安装！**

查看服务器的架构：
```shell
uname -m
```
发现为`x86_64`架构。

查看上述npu为8张`昇腾910B`，每张卡有`16G的显存`。

### 下载nnrt和Toolkit
要在计算平台上进行开发，需要四要素：
* 驱动+固件
* nnrt
* toolkit
* MindStudio

*由于我们的服务器已经下好了驱动和固件，所以我们已经不需要驱动和固件了，而且在服务器上只能使用CLI界面进行操作，所以不需要MindStudio软件了！*

前往昇腾社区的相应下载界面:[https://www.hiascend.com/zh/software/cann/community](https://www.hiascend.com/zh/software/cann/community)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img95.jpg)

### 从零制作Docker镜像

#### 制作基础镜像
* 检查Docker是否有效：
```shell
docker images
```
* 拉取一个Ubuntu的Docker镜像：
```shell
docker pull ubuntu:18.04
```
* 通过镜像启动一个容器并进入容器：
```shell
docker run -it ubuntu:18.04 /bin/bash
```

#### 容器内安装常用工具

* 容器内安装`vim`工具：
```shell
$ apt-get update
$ apt-get install vim
```

* 更换Ubuntu的apt源：
```shell
$ cd /etc/apt
$ cp sources.list sources.list.bak # 制作一个备份
$ >sources.list # 清空内容
$ vim sources.list
```
加入以下内容，换中科大源：
```scss
##中科大源

deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```
* `apt-get`索引重新生效:
```shell
apt-get update 
```

* 容器内安装`wget`工具：
```shell
$ apt-get install wget
```

#### 容器内安装依赖
按照`昇腾社区`开发环境需求：[https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha001/softwareinstall/instg/instg_000026.html](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha001/softwareinstall/instg/instg_000026.html)

*注意不同的nnrt和toolkit版本的安装手册不同，请根据自己的版本进行选择！*

* 安装`gcc`，`cmake`，`g++`，`unzip`等：
```shell
apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3
```

*ibsqlite3-dev需要在python安装之前安装，如果用户操作系统已经安装满足版本要求的python环境，在此之后再安装libsqlite3-dev，则需要重新编译python环境。*

* 安装Python3.7.5：
```shell
$ cd /downloads # 没有该目录则创建一个
$ wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
$ tar -zxvf Python-3.7.5.tgz # 解压
$ cd Python-3.7.5 # 进入目录
$ ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared
$ make
$ make install
```
* 设置Python3.7.5环境变量：
```shell
vim ~/.bashrc
```
加入以下内容：
```scss
#用于设置python3.7.5库文件路径
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
#如果用户环境存在多个python3版本，则指定使用python3.7.5版本
export PATH=/usr/local/python3.7.5/bin:$PATH
```
* 环境变量生效：
```shell
source ~/.bashrc
```
* 检查Python和pip环境：
```shell
$ python3 --version
$ pip3 --version
```
* 更新pip版本:
```shell
pip3 install --upgrade pip
```

* 安装一些常用的Python包：
```shell
$ pip3 install attrs
$ pip3 install numpy
$ pip3 install decorator
$ pip3 install sympy
$ pip3 install cffi
$ pip3 install pyyaml
$ pip3 install pathlib2
$ pip3 install psutil
$ pip3 install protobuf
$ pip3 install scipy
$ pip3 install requests
$ pip3 install absl-py
```
* 更新保存为新的镜像:
```shell
$ exit # 退出容器
$ docker commit -m="Update Packages" -a="caixj" <容器id> caixj/ascend910b:infer
```

### 容器内安装nnrt和Toolkit

* 通过更新后的镜像重新启动容器，并映射文件地址,进入容器：
```shell
docker run -v /data:/data -it caixj/ascend910b:infer /bin/bash
```
* 找到下载好的nnrt包，进入目录并进行安装：
```shell
$ cd 目录地址
$ ./Ascend-cann-nnrt_7.0.RC1.alpha002_linux-x86_64.run --install
# 安装完成后将提示信息要求的环境配置信息按照上述方式在.bashrc中更新
```

* 找到下载好的toolkit包，进入目录并进行安装：
```shell
$ cd 目录地址
$ ./Ascend-cann-toolkit_7.0.RC1.alpha002_linux-x86_64.run --install
# 安装完成后将提示信息要求的环境配置信息按照上述方式在.bashrc中更新
```
* 更新的容器重新保存为新的镜像：
```shell
$ exit
$ docker commit -m="Update Packages" -a="caixj" <容器id> caixj/ascend910b:infer
```
* 重新启动并通过另外的方式进入容器：
```shell
$ docker run -v /data:/data -itd caixj/ascend910b:infer /bin/bash
$ docker exec -it <容器id> /bin/bash
# 这种方法进入容器后，即使exit退出容器后容器仍然在运行
# 如果要停止容器运行： docker stop <容器id>
```

到此，对于一个华为昇腾开发环境的基础开发环境已经构建成功了！

#### 验证命令行环境

如果容器内的环境配置正确之后，我们在命令行输入`atc`是会返回信息的：
```scss
ATC start working now, please wait for a moment.
...
ATC run failed, Please check the detail log, Try 'atc --help' for more information
E10007: [--framework] is required. The value must be [0(Caffe) or 1(MindSpore) or 3(TensorFlow) or 5(Onnx)].
```

但是目前返回的信息为：
```scss
/usr/local/Ascend/ascend-toolkit/7.0.RC1.alpha001/x86_64-linux/bin/atc.bin: error while loading shared libraries: libascend_hal.so: cannot open shared object file: No such file or directory
```

网上找到了解决方案：[https://bbs.huaweicloud.com/blogs/344623](https://bbs.huaweicloud.com/blogs/344623)

按照它的方法操作了之后，成功了！

#### 8月28日更新
对于上述命令行转化工具`atc`的环境变量问题，在华为官方issues维护人员的帮助下，解决了环境问题。

**首先需要纠正的是nnrt和toolkit包是有重叠的。toolkit包含了离线推理和模型转换，而nnrt是只有包含离线推理的地方。**

其次，在我自制的Docker镜像中发现需要设置驱动的库文件的环境变量。

所以容器中`~/.bashrc`关于toolkit的配置为：
```bashrc
# toolkit
. /usr/local/Ascend/ascend-toolkit/set_env.sh
# driver
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/stub:＄LD_LIBRARY_PATH 
export LD_LIBRARY_PATH-/usr/local/Ascend/driver/lib64/driver:＄LD_LIBRARY_PATH
```

顺便记录一下我的Docker启动命令：
```shell
sudo docker run -it -u root \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /var/log/npu:/var/log/npu \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/slog:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/tools/:/usr/local/Ascend/driver/tools/ \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /home/foot/caixj:/work_dir \
caixj/ascend910b:python \
/bin/bash
```


