---
title: "Linux环境搭建：虚拟机上运行CentOS"
date: 2022-01-30T20:55:05+08:00
lastmod: 2022-01-30T21:43:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/CentOS.png"
description: "c++服务器开发，通常运行在Linux端，所以必须要有Linux环境，结合国内互联网公司对于Linux发行版的使用情况，这里选择使用CentOS"
tags:
- Linux
- Linux环境配置
categories:
- Linux环境配置
series:
- 环境配置&博客搭建
comment : true
---

## 为什么选择CentOS环境而不选择Ubuntu

原本我的windows笔记本上已经装了`Ubuntu`,已经有了`Linux`环境。

最近听了网上大佬的说法，说国内的互联网公司基本都在使用另一个`Linux`操作系统CentOS,加之其比`Ubuntu`更加稳定，最重要的是可视化界面更少，这一点是我选择它的原因。



## windows端部署

大家有mac就用mac，没有条件就用windows，毕竟macbook这么贵。

不得不说mac系统真的比windows好用很多，特别是配置环境和下载软件这方面真的有些操蛋！

### 软件准备

1.虚拟机软件`VMware workstation`

社区免费版下载地址：	[https://customerconnect.vmware.com/en/downloads/info/slug/desktop_end_user_computing/vmware_workstation_player/16_0](https://customerconnect.vmware.com/en/downloads/info/slug/desktop_end_user_computing/vmware_workstation_player/16_0)

2.CentOS 7 镜像文件：

复制这段内容后打开百度网盘手机App，操作更方便哦
链接：https://pan.baidu.com/s/1tmxews_LiAWS9DXLIi8egA 
提取码：6wZd 

3.SSH终端连接工具：

因为`SecurCRT`软件不是免费的，需要破解，windows端找软件真的找破天，非常多站点下的都有问题，甚至有病毒。我在这里提供两种下载方式，安装版和解压版：

* Securecrt安装版破解版 v8.5下载地址：[https://www.32r.com/soft/49437.html](https://www.32r.com/soft/49437.html)

  此种方式的破解方法视频：[https://www.bilibili.com/video/BV1QC4y1H7zR?spm_id_from=333.1007.top_right_bar_window_history.content.click](https://www.bilibili.com/video/BV1QC4y1H7zR?spm_id_from=333.1007.top_right_bar_window_history.content.click)

  注意注册机文件的放入位置和patch，以及各种信息的填写。如果这种方式无效，就直接去下解压版吧。

* Securecrt解压版下载地址：

  复制这段内容后打开百度网盘手机App，操作更方便哦
  链接：https://pan.baidu.com/s/1ebeOqNtbdEkGVdJOCOvAQQ 
  提取码：6wZd

  解压完就可以使用了。

4.SFTP传输工具`WinSCP`:

这个软件可以在windows端和Linux端传文件的工具，下载地址：[https://winscp.net/eng/docs/guide_install#downloading](https://winscp.net/eng/docs/guide_install#downloading)

### CentOS安装

首先说一下用虚拟机安装系统的好处：

* 如果系统被你玩坏了，重新装就好了，不会让你的电脑的文件丢失，而如果装双系统极有可能会丢失文件。
* 而虚拟机可以设置多个节点，可以做服务器集群，而双系统就只能用一个。
* 如果你觉得虚拟机装Linux没得灵魂，记住你这句话，你会说“真香“的！



1.安装好虚拟机软件之后，打开，点击新建虚拟机，浏览找到安装镜像源的位置，点击下一步，输入节点的名字

2.设置好登录的用户名和密码以及root密码（root密码要记住，以后在命令行中使用root操作需要输入此密码）

3.进入一段黑屏绿字的安装时间之后进入设置安装内容界面。

安装设置界面：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/install_set1.jpg)

语言支持选择中文，安装位置选择自动配置分区，然后再点击软件选择，考虑到以后可能用到很多服务器开发有关的东西，我选择`开发及生成工作站`，勾选如下选项。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/install_set2.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/install_set3.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/install_set4.jpg)

然后开始你漫长的安装过程，出现如下画面,就代表装好了。

初始界面：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/Desktop.png)



### 检查连通性以及SSH连接

在windows端一般来说都是直接配好的，可以通过命令行检查与外网和宿主机的连通性：

* 与外网连通性：

  ```shell
  $ ping www.baidu.com
  ```

  如果出现跳出毫秒信息等信息，说明已经连通了，按`Ctrl+c`取消。

  如果跳出`未知的名称或服务`，说明不连通

* 与宿主机连通性：

  * Linux连接宿主机：

    1.宿主机按`Ctrl+R`，输入`cmd`运行，输入：

    ```shell
    ipconfig/all
    ```

    找到`ipv4`地址，复制，在命令行输入：

    ```shell
    $ ping 192.168.31.11 
    ```

    查看连通性。

  * 宿主机连接Linux：

    1.在Linux命令行输入：

    ```shell
    ifconfig
    ```

    出现如下画面：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/ping.png)

    复制第一个`inet`后面的ip地址，*注意这个ip地址虽然是DHCP动态分配的，但在windows上基本每次分配都是同一个ip地址*

    然后打开windows终端，输入：

    ```shell
    ping 192.168.130.129
    ```

    测试其连通性。

* 配置地址使其连通

  一般来说windows端一般是不用配置的，如果需要，*我会写在mac端的配置里面*。

我们配好连通性之后需要新建SSH连接，这是为了之后的SFTP文件传输。

* 打开`SecurCRT`,点击`文件`，点击`快速连接`，出现如下画面

  SSH设置：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/p1.png)

* 主机名为Linux端之前输入`ifconfig`的ip地址
* 用户名一般设置为`root`。
* 连接之后输入root密码就ok了。

### SFTP连接及文件传输

注意：SFTP连接只有SSH连接配好之后才能进行传输。

* 打开`WinSCP`,同样输入Linux的ip地址，也就是`ifconfig`输入之后显示的第一个ip地址。
* 输入用户名`root`，密码就是root密码
* 如果每次打开怕嫌麻烦，可以保存这个站点的信息（windows端每次dhcp分配的ip地址基本是一样的），如下图。

SFTP连接配置：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/p2.png)



* 验证文件传输：

  我传了一张图片到Linux桌面，进入Linux端二进制命令行，进入桌面目录，查找文件

  ```shell
  $ pwd
  $ cd Desktop
  $ ls -l
  ```

  在桌面找到了文件，如下：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/p3.png)

  成功了！

### 设置节点以及克隆节点

* 选择节点，点击编辑虚拟机设置：里面可以选择内存大小和cpu核心数。如果你的电脑cpu性能好，运行内存大，可以选择2个核心数和2G的内存；如果不行就1个核心数和1G的内存。
* 如果你要做服务器之间的通信或者服务器集群，就可以选择克隆节点，不知道为啥16版本我找不到克隆选项，其他版本听说有。实在不行就再装一个一模一样的。



## MAC端部署

*注意：mac端只有在安装软件和连通性有区别，其他步骤均类似！所以下面只讲不一样的部分*

参考视频：[codesheep程序羊关于mac端Linux环境部署](https://www.bilibili.com/video/BV1bA411b7vs?spm_id_from=333.788.top_right_bar_window_history.content.click)

### mac端软件准备

* CentOS镜像源：这个与windows端是一样的
* 虚拟机：VM Fusion
* SSH连接工具：`SecureCRT` for mac
* SFTP传输工具：`Transmit`

### 静态ip配置

mac端默认是不能上网的，无论是外网还是宿主机连通都不行，如果不信就可以按windows端的方法ping一下。

* 查看ip地址：

  输入：

  ```shell
  ifconfig
  ```

  会发现`ens33`文件下没有ip地址

* 点击左上角的`虚拟机`，出现如下图，选择你需要的模式：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mac1.jpg)

* 继续配置静态ip

  进入Linux二进制命令行，

  * 进入root模式：

    ```shell
    $ su root
    ```

    然后输入密码

  * 分配网络类可用的ip地址，并将IP地址固定下来。

    ```shell
    $ dhclient
    $ ifconfig
    ```

    复制分配下来的ip地址

  * 配置网卡内容，输入：

    ```shell
    $ vim /etc/sysconfig/network-scripts/ifcfg-ens33
    ```

    这里参考视频内容里的配置，配置前后的画面如下：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mac2.jpg)

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mac3.jpg)

  * 具体修改的内容：

    ```shell
    $ BOOTPROTO=static
    $ ONBOOT=yes
    $ IPADDR=192.168.31.100   //这个地址为Linux为你分配的地址，也就是前面复制的那个地址
    $ NETMASK=255.255.255.0   //子网掩码
    $ GATEWAY=192.168.31.1    //网关地址，根据ip地址可以计算（计网的知识）
    $ DNS1=119.29.29.29       //公网DNS，这里使用的是腾讯的
    ```

    最后输入`Esc`，再输入`:wq`修改保存

* 最后再进行外网和宿主机的连通性：与windows端的ping方法一样。
