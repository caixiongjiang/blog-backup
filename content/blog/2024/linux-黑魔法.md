---
title: "给你的Linux网络装上黑魔法"
date: 2024-07-08T18:18:05+08:00
lastmod: 2024-07-08T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/clash_title.jpg"
description: "给ubuntu系统网络上黑魔法"
tags:
- clash
categories:
- 黑魔法
series:
- 《Linux网络》
comment : true
---

### Clash在ubuntu上的使用

网络魔法是日常工作中不可或缺的，本人是做算法的，经常要逛一逛`HuggingFace`，黑魔法网络必不可少。在Windows和mac上配置黑魔法较为简单，现在需要在`Ubuntu22.04`系统上实现黑魔法，“睁眼看世界”。

#### 下载Clash内核

由于官方的clash作者已经删库跑路了，所以需要从第三方下载clash内核，下载地址：[https://github.com/DustinWin/clash_singbox-tools](https://github.com/DustinWin/clash_singbox-tools)。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/img1.jpg)
这里我使用的是clash的p核，`clashpremium-linux-amd64`，请自行选择与cpu架构合适的内核。

在任意一个地方新建文件夹：
```bash
mkdir clash
cd clash
# 然后将你下载好的内核移动到clash文件夹下

# 重命名clash内核
mv clashpremium-linux-amd64 clash
# 赋予执行权限
chmod +x clash
```
上述任何操作如何没有权限，可以进root之后再操作。

#### 下载机场配置文件

从购买的机场下载`yml`或者`yaml`配置文件，放入到clash目录下面与内核齐平。将配置文件的标题改为`config.yaml`，并修改里面的部分内容。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/img2.jpg)
将allow-lan修改为`true`，你也可以额外添加ui界面控制，解开注释`external-ui`，然后后面的`folder`改为你的ui界面文件夹位置。ui界面可以在[https://github.com/DustinWin/clash_singbox-tools](https://github.com/DustinWin/clash_singbox-tools)下载解压就可以。（也可以使用第三方网页配置，这样就不需要动external-ui相关的配置，后面会说）

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/img3.jpg)

除了`config.yaml`，我们还需要Country.mmdb文件，虽然网上有很多资源，但是不一定和你的适配，建议从windows clash文件夹中的`Country.mmdb`拷贝过来，放在clash目录下和内核与配置文件同级别。

#### 启动clash

在clash文件夹下执行命令，启动：
```bash
./clash -d .
```
如果看到如下画面则启动成功：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/img4.jpg)
**我这里是放的网图，命令必须按照上述来，因为我们并没有把clash配置进系统命令中。**

在浏览器打开`127.0.0.1:9090`，就能看到控制面板，对应的是你下载的UI界面。这里我没有使用自己的ui界面，使用了第三方的ui界面，配置自己的ip和端口。
访问地址：[https://clash.razord.top/#/settings](https://clash.razord.top/#/settings)
打开后点击`编辑`，配置相应的ip和端口，然后点击代理，则可以查看自己的网络配置。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/img5.jpg)

最后一步，在系统中配置端口（设置->网络->设置代理）：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/clash/img6.jpg)

这里面对应的端口号和`config.yaml`是一一对应的，如果你因为端口被占用，可以选择kill掉无关的进程，也可以修改相应的端口号。如果修改了`config.yaml`中的端口号，在系统设置的时候也要进行相应的修改。

配置好之后就可以打开你的google进行搜索了。

#### 配置docker自启动

虽然上面的步骤已经配置好了clash，但是每次都需要打开终端进入文件夹下启动clash，非常麻烦，这时候我们可以使用docker来配置这个轻量化的服务。

编写dockerfile建立一个基础环境：
```bash
touch dockerfile
vim dockerfile
```
在dockerfile中编写如下内容：

```dockerfile
FROM ubuntu:22.04
# 设置环境变量 非交互式
ENV DEBIAN_FRONTEND=noninteractive
# 指定工作目录
WORKDIR /clash

# 替换为阿里云的 Ubuntu 源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更新并安装 Python 3.9 和其他必需的包
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    curl

# 指定容器位置和时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone

# 设置容器的编码为UTF-8
RUN echo 'LANG=en_US.UTF-8' > /etc/locale.conf && \
    echo 'LC_ALL=en_US.UTF-8' >> /etc/locale.conf
```

在dockerfile文件的同级别目录下，运行命令构建docker镜像：
```bash
# 这里的镜像名可以自己替换
docker build -t caixj/clash-linux:beta .
```

构建好镜像之后，我们要使用镜像启动相应的容器，简单启动进行源文件复制。
```bash
# 查看镜像信息
docker images
# 启动
docker run -itd --names clash caixj/clash-linux:beta 
# 查看是否启动成功
docker ps -a
```
然后我们需要写一个shell脚本自动执行， `start.sh`：
```shell
#!/bin/bash
echo "Container is starting..."

# 启动clash
./clash -d .
```
写好之后加入执行权限：
```bash
chmod +x start.sh
```
然后我们要把需要的文件全部复制到容器中，需要的文件有`clash`(之前下载的内核)， `config.yaml`， `Country.mmdb`，写好的sh脚本`start.sh`：
```bash
# 查看容器id
docker ps 
# 复制文件进入容器（需要复制的文件都需要挨个复制）
docker cp ${文件名} ${容器id}:/clash
```
特别说明，如果你在前面配置了额外的ui界面，ui界面的文件夹也需要复制进容器中，文件夹复制的命令如下：
```bash
docker cp ${ui文件夹路径} ${容器id}:/clash
```
还需要修改config.yaml中的ui界面的文件夹位置为`/clash/你的ui文件夹名`。

将容器更新为新的镜像：
```bash
# -a代表作者，-m代表更新内容
docker commit -a="caixj" -m="增加clash相应文件" ${容器id} caixj/clash-linux:beta
```
最后一步，重新从镜像构建容器，并开启脚本自动执行。由于我们要使用到本机的7890～7892端口和9090端口，则需要在容器启动时做端口映射：
```bash
# 在镜像最后加入了脚本代表容器启动时会自动执行start.sh
docker run -itd -p 7890:7890 -p 7891:7891 -p 7892:7892 -p 9090:9090 --name clash caixj/clash-linux:beta /clash/start.sh
```

启动成功便可以在之前的`127.0.0.1:9090`进行配置网络，或者是前文说的第三方界面进行配置。

如此一来，每次重新启动之后，只要启动暂时退出的容器就可以了：
```bash
docker ps -a
# 这里的clash是因为之前docker启动时配置了 --name clash  
docker start clash
```

然后在系统中开启之前配置好的代理，就可以访问google搜索了。




