---
title: "Docker快速入门"
date: 2022-06-24T18:18:05+08:00
lastmod: 2022-06-24T18:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/docker/img_title.jpg"
description: "目前因为虚拟机的高消耗资源的特性，然后VM Fusion还不太适配M1系列的芯片，所以入门一下容器技术，用来平替。"
tags:
- Docker
categories:
- 容器技术
comment : true
---

## Docker一小时快速入门

### Docker简介和安装

#### Docker 开发部署流程

自己在 Windows/Mac 上开发、测试 --> 打包为 Docker 镜像（可以理解为软件安装包） --> 各种服务器上只需要一个命令部署好。

> 优点:确保了不同机器上跑起来都是一致的运行环境，不会出现不同机器之间切换出现bug的问题。

#### Docker通常来做什么

* 应用分发、部署，方便传播给他人安装。特别是开源软件和提供私有部署的应用
* 快速安装测试/学习软件，用完就丢（类似小程序），不把时间浪费在安装软件上。例如 Redis / MongoDB / ElasticSearch / ELK
* 多个版本软件共存，不污染系统，例如 Python2、Python3，Redis4.0，Redis5.0
* Windows/mac上体验/学习各种 Linux 系统

#### 重要概念：镜像、容器

镜像：可以理解为软件安装包，可以方便进行传播和安装

容器：软件安装后的状态，每个软件运行环境都是独立的、隔离的、称之为容器

#### 安装

桌面版：[https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

mac版安装无障碍，Windows版安装还挺麻烦的，可以看一下这个[在线文档](https://docker.easydoc.net/doc/81170005/cCewZWoN/lTKfePfP#nav_6)

服务器版：[https://docs.docker.com/engine/install/#server](https://docs.docker.com/engine/install/#server)

#### 镜像加速源

| 镜像加速源          | 镜像加速器地址                          |
| ------------------- | --------------------------------------- |
| Docker 中国官方镜像 | https://registry.docker-cn.com          |
| DaoCloud 镜像站     | http://f1361db2.m.daocloud.io           |
| Azure 中国镜像      | https://dockerhub.azk8s.cn              |
| 科大镜像站          | https://docker.mirrors.ustc.edu.cn      |
| 阿里云              | https://<your_code>.mirror.aliyuncs.com |
| 七牛云              | https://reg-mirror.qiniu.com            |
| 网易云              | https://hub-mirror.c.163.com            |
| 腾讯云              | https://mirror.ccs.tencentyun.com       |

> 我配置的是阿里云的镜像源，不过阿里云的镜像源需要自己配置一个账号，也可以配置其他的镜像源。

配置方法如下，在`Docker Desktop`中点击设置，找到`Docker Engine`中，在`registry-mirrors`字段中加上镜像源restart即可。

### Docker快速安装软件

#### Docker安装的优点

* 一个命令就可以安装好，快速方便
* 有大量的镜像，可直接使用
* 没有系统兼容问题，Linux 专享软件也照样跑
* 支持软件多版本共存
* 用完就丢，不拖慢电脑速度
* 不同系统和硬件，只要安装好 Docker 其他都一样了，一个命令搞定所有

#### 演示Docker安装Redis

首先，Docker官方镜像仓库查找Redis：[https://hub.docker.com](https://hub.docker.com)

查看文档之后，使用一个命令跑Redis,在`terminal`输入：

```shell
# -d： 表示在后台运行 
# -p：表示把容器的端口暴露到宿主机 
# --name：容器的命名
# redis：后面加的是版本号
$ docker run -d -p 6379:6379 --name redis redis:6.0
```

然后查看`Docker Desktop`就可以看到redis在容器里运行了。

Docker命令参考：[https://docs.docker.com/engine/reference/commandline/run](https://docs.docker.com/engine/reference/commandline/run)

### 制作自己的镜像

#### 使用示例Web项目构建镜像

项目示例代码：[https://github.com/gzyunke/test-docker](https://github.com/gzyunke/test-docker)

这是一个Nodejs + \+ Koa2 写的 Web 项目，提供了简单的两个演示页面。
软件依赖：[nodejs](https://nodejs.org/zh-cn/)
项目依赖库：koa、log4js、koa-router

#### 编写Dockerfile

项目所需要的环境配置，依赖，软件写成一个dockerfile：

```dockerfile
FROM node:11
MAINTAINER easydoc.net

# 复制代码
ADD . /app

# 设置容器启动后的默认运行目录
WORKDIR /app

# 运行命令，安装依赖
# RUN 命令可以有多个，但是可以用 && 连接多个命令来减少层级。
# 例如 RUN npm install && cd /app && mkdir logs
RUN npm install --registry=https://registry.npm.taobao.org

# CMD 指令只能一个，是容器启动后执行的命令，算是程序的入口。
# 如果还需要运行其他命令可以用 && 连接，也可以写成一个shell脚本去执行。
# 例如 CMD cd /app && ./start.sh
CMD node app.js
```

#### Build为镜像（安装包）和运行

编译`docker build -t test:v1 .`

> `-t`设置镜像名称和版本号

运行`docker run -p 8080:8080 --name test-hello test:v1`

> `-p`映射容器内端口到主机
>
> `--name`容器名字
>
> `-d`后台运行

#### 更多相关命令

`docker ps` 查看当前运行中的容器
`docker images` 查看镜像列表d
`docker rm container-id` 删除指定 id 的容器
`docker stop/start container-id` 停止/启动指定 id 的容器
`docker rmi image-id` 删除指定 id 的镜像
`docker volume ls` 查看 volume 列表
`docker network ls` 查看网络列表

### 目录挂载

#### 现存问题

* 使用 Docker 运行后，我们改了项目代码不会立刻生效，需要重新`build`和`run`，很是麻烦。
* 容器里面产生的数据，例如 log 文件，数据库备份文件，容器删除后就丢失了。

> 目录挂载解决以上问题

#### 几种挂载方式

* `bind mount` 直接把宿主机目录映射到容器内，适合挂代码目录和配置文件。可挂到多个容器上
* `volume` 由容器创建和管理，创建在宿主机，所以删除容器不会丢失，官方推荐，更高效，Linux 文件系统，适合存储数据库数据。可挂到多个容器上
* `tmpfs mount` 适合存储临时文件，存宿主机内存中。不可多容器共享

挂载示意图：

 ![](https://sjwx.easydoc.xyz/46901064/files/kv96dc4q.png)

#### 挂载演示

`bind mount`方式用绝对路径`-v D:/code:/app`
`volume` 方式，只需要一个名字`-v db-data:/app`

示例：

```shell
# 注意： D：/code是你的程序文件的绝对地址，需要更改
$ docker run -p 8080:8080 --name test-hello -v D:/code:/app -d test:v1
```

>   使用这种方式，如果本地的代码进行改变，容器里的代码也会进行实时改变。运行的项目需要在容器重启之后生效，从而发生变化。

### 多容器通信

#### 学习目标

项目往往都不是独立运行的，需要数据库，缓存这些东西配合运作。

我们将前面的Web项目增加一个Redis依赖，多跑一个Redis容器，演示如何进行多容器之间的通信。

#### 创建虚拟网络

要想多容器之间互通，从 Web 容器访问 Redis 容器，我们只需要把他们放到同个网络中就可以了。

文档参考：[https://docs.docker.com/engine/reference/commandline/network/](https://docs.docker.com/engine/reference/commandline/network/)

#### 演示

创建一个名为`test-net`的网络

```shell
$ docker network create test-net
```

运行Redis在`test-net`网络中，别名为`redis`：

```shell
# --network:制定网络的名字
# --network-alias:起一个网络别名
# 最后是镜像来源版本号
docker run -d --name redis --network test-net --network-alias redis redis:6.0
```

修改代码中访问`redis`的地址为网络别名

```js
const redis = require('redis');
let rds = redis.createClient({url: "redis://redis:6379"}); // 这里的路径上需要写上上面写的网络别名
rds.on('connect', ()=> console.log('redis connect ok'))
rds.connect();
```

运行 Web 项目，使用同个网络

```shell
# 这里还是用本地目录挂载，我就直接把我mac电脑上的目录写进去了
# 这里最后的test：v1是代表使用之前已经部署好的镜像（安装包）
$ docker run -p 8080:8080 --name test -v /Users/caixiongjiang/github仓库/test-docker:/app --network test-net -d test:v1
```

查看数据

`http://localhost:8080/redis`
容器终端查看数据是否一致

### Docker-Compose

#### 现存问题

在上节，我们运行了两个容器：Web 项目 + Redis
如果项目依赖更多的第三方软件，我们需要管理的容器就更加多，每个都要单独配置运行，指定网络。
这节，我们使用 docker-compose 把项目的多个服务集合到一起，一键运行。

#### 安装Docker Compose

* 如果你是安装的桌面版 Docker，不需要额外安装，已经包含了。
* 如果是没图形界面的服务器版 Docker，你需要单独安装 [安装文档](https://docs.docker.com/compose/install/#install-compose-on-linux-systems)
* 运行`docker-compose`检查是否安装成功

#### 编写脚本

要把项目依赖的多个服务集合到一起，我们需要编写一个`docker-compose.yml`文件，描述依赖哪些服务
参考文档：https://docs.docker.com/compose/

```
version: "3.7"

services:
  app:
    build: ./
    ports:
      - 80:8080
    volumes:
      - ./:/app
    environment:
      - TZ=Asia/Shanghai
  redis:
    image: redis:5.0.13
    volumes:
      - redis:/data
    environment:
      - TZ=Asia/Shanghai

volumes:
  redis:
  
```

> 容器默认时间不是北京时间，增加TZ=Aisa/Shanghai可以改为北京时间

#### RUN

在`docker-compose.yml`文件所在目录，执行`docker-compose up`就可以运行起来了。

命令参考：https://docs.docker.com/compose/reference/up/

> 这里执行`docker-compose up`相当于批量执行配置，包括前面的依赖，目录挂载，环境配置等。
>
> 这里没有指定网络是因为，docker会默认给所有的东西指定同一个网络，可以直接互相通信。

在后台运行只需要加一个 -d 参数`docker-compose up -d`
查看运行状态：`docker-compose ps`
停止运行：`docker-compose stop`
重启：`docker-compose restart`
重启单个服务：`docker-compose restart service-name`
进入容器命令行：`docker-compose exec service-name sh`
查看容器运行log：`docker-compose logs [service-name]`

### 发布和部署

#### 镜像仓库介绍

镜像仓库用来存储我们 build 出来的“安装包”，Docker 官方提供了一个 [镜像库](https://hub.docker.com/)，里面包含了大量镜像，基本各种软件所需依赖都有，要什么直接上去搜索。

我们也可以把自己 build 出来的镜像上传到 docker 提供的镜像库中，方便传播（跟项目发布到github是一个道理）。
当然你也可以搭建自己的私有镜像库，或者使用国内各种大厂提供的镜像托管服务，例如：阿里云、腾讯云

#### 上传镜像

* 首先你要先 [注册一个账号](https://hub.docker.com/)
* 创建一个镜像库

 ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/docker/img1.png)

* 命令行登录账号：
  `docker login -u username`
* 新建一个tag，名字必须跟你注册账号一样
  `docker tag test:v1 username/test:v1`
* 推上去
  `docker push username/test:v1`
* 部署试下
  `docker run -dp 8080:8080 username/test:v1`

docker-compose 中也可以直接用这个镜像了

```
version: "3.7"

services:
  app:
#    build: ./
    image: jarsoncai/test:v1
    ports:
      - 80:8080
    volumes:
      - ./:/app
    environment:
      - TZ=Asia/Shanghai
  redis:
    image: redis:5.0.13
    volumes:
      - redis:/data
    environment:
      - TZ=Asia/Shanghai

volumes:
  redis:

```

### 备份和迁移

容器中的数据，如果没有用挂载目录，删除容器后就会丢失数据。
前面已经了解了如何 [挂载目录](doc:kze7f0ZR)
如果你是用`bind mount`直接把宿主机的目录挂进去容器，那迁移数据很方便，直接复制目录就好了
如果你是用`volume`方式挂载的，由于数据是由容器创建和管理的，需要用特殊的方式把数据弄出来。

#### 备份和导入 Volume 的流程

备份：

- 运行一个 ubuntu 的容器，挂载需要备份的 volume 到容器，并且挂载宿主机目录到容器里的备份目录。
- 运行 tar 命令把数据压缩为一个文件
- 把备份文件复制到需要导入的机器

导入：

- 运行 ubuntu 容器，挂载容器的 volume，并且挂载宿主机备份文件所在目录到容器里
- 运行 tar 命令解压备份文件到指定目录

### 备份 MongoDB 数据演示

- 运行一个 mongodb，创建一个名叫`mongo-data`的 volume 指向容器的 /data 目录
  `docker run -p 27018:27017 --name mongo -v mongo-data:/data -d mongo:4.4`
- 运行一个 Ubuntu 的容器，挂载`mongo`容器的所有 volume，映射宿主机的 backup 目录到容器里面的 /backup 目录，然后运行 tar 命令把数据压缩打包
  `docker run --rm --volumes-from mongo -v d:/backup:/backup ubuntu tar cvf /backup/backup.tar /data/`

最后你就可以拿着这个 backup.tar 文件去其他地方导入了。

### 恢复 Volume 数据演示

- 运行一个 ubuntu 容器，挂载 mongo 容器的所有 volumes，然后读取 /backup 目录中的备份文件，解压到 /data/ 目录
  `docker run --rm --volumes-from mongo -v d:/backup:/backup ubuntu bash -c "cd /data/ && tar xvf /backup/backup.tar --strip 1"`
