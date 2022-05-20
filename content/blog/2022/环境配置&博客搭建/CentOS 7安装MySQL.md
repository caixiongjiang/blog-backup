---
title: "CentOS7安装MySQL"
date: 2022-05-20T18:18:05+08:00
lastmod: 2022-05-20T18:20:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/title_img.jpg"
description: "因为企业使用mysql的环境基本在Linux下使用，所以需要在Linux下安装MySQL"
tags:
- Linux
- MySQL
categories:
- Linux环境配置
series:
- 环境配置&博客搭建
comment : true

---

## Linux下MySQL安装&远程访问配置

### 下载安装包

社区免费版下载地址：[https://downloads.mysql.com/archives/community/](https://downloads.mysql.com/archives/community/)

选择正确的版本：

* 需要安装Linux版本的MySQL，所以可以选择两个版本：一个是`Red Hat Enterprice Linux`,另一个是`Linux-Generic`。我选择的是前者，因为CentOS和它是同一个公司出的！
* 选择OS version：如果你是CentOS 7，则选择Linux 7；如果为CentOS 8，则选择Linux 8。至于x86，ARM如何选取，取决于你的cpu：
  * 如果你是mac OS系统，intel芯片，就选择x86 64bit位（现在的电脑基本都是64bit位）；如果是M1系列的芯片，请选择ARM 
  * 如果你是Windows系统，就选x86 64bit位

### Linux-虚拟机三件套

此次我是在mac OS下操作虚拟机，使用三个软件来部署环境，至于如何配置环境，我在之前的博客里已经讲过了！[点击查看](https://caixiongjiang.github.io/blog/2022/%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E5%8D%9A%E5%AE%A2%E6%90%AD%E5%BB%BA/linux%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/)

* 虚拟机软件：VMware Fusion
* 连接工具：nuoshell
* 文件传输工具：Transmit

### 安装MySQL

将下载好的包文件使用`Transmit`传输到虚拟机的系统。

正式开始安装：

* 解压包到mysql文件夹：

  ```shell
  # 新建文件夹
  mkdir mysql
  # 把包解压到mysql文件夹下
  tar -xvf mysql-8.0.28-1.el7.x86_64.rpm-bundle.tar -C mysql
  # 进入文件
  cd mysql
  # 查看解压后的文件
  ll
  ```

  会得到如下的结果：

  ```shell
  total 823244
  -rw-r--r--. 1 7155 31415  55199948 Dec 18 05:12 mysql-community-client-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415   5933684 Dec 18 05:12 mysql-community-client-plugins-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415    645388 Dec 18 05:12 mysql-community-common-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415   7763684 Dec 18 05:13 mysql-community-devel-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415  23637584 Dec 18 05:13 mysql-community-embedded-compat-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415   2215928 Dec 18 05:13 mysql-community-icu-data-files-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415   4935572 Dec 18 05:13 mysql-community-libs-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415   1265072 Dec 18 05:13 mysql-community-libs-compat-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415 473116268 Dec 18 05:14 mysql-community-server-8.0.28-1.el7.x86_64.rpm
  -rw-r--r--. 1 7155 31415 268279684 Dec 18 05:16 mysql-community-test-8.0.28-1.el7.x86_64.rpm
  ```

* 配置依赖：

  * perl：CentOS 7下自带

  * MySQL安装不能存在mariadb的依赖！mariadb是CentOS自带的数据库，所以要将其删除

    ```shell
    # 利用管道查询mariadb
    rpm -qa|grep mariadb
    # 出现：
    # mariadb-libs-5.5.64-1.el7.x86_64
    # mariadb-devel-5.5.64-1.el7.x86_64
    
    # 强制删除这几个文件
    rpm -e --nodeps mariadb-libs-5.5.64-1.el7.x86_64
    rpm -e --nodeps mariadb-devel-5.5.64-1.el7.x86_64
    ```

* **按顺序**安装包（因为包有依赖关系，必须按顺序安装）

  ```shell
  # 安装依赖
  rpm -ivh mysql-community-common-8.0.28-1.el7.x86_64.rpm
  # 安装依赖
  rpm -ivh mysql-community-client-plugins-8.0.28-1.el7.x86_64.rpm 
  # 安装依赖
  rpm -ivh mysql-community-libs-8.0.28-1.el7.x86_64.rpm 
  # 安装客户端
  rpm -ivh mysql-community-client-8.0.28-1.el7.x86_64.rpm
  # 安装依赖
  rpm -ivh mysql-community-icu-data-files-8.0.28-1.el7.x86_64.rpm
  # 安装服务器
  rpm -ivh mysql-community-server-8.0.28-1.el7.x86_64.rpm
  ```

* 启动mysql并配置密码：

  * 启动mysql：

    ```shell
    # 初始化
    mysqld --initialize --console
    # 修改安装目录的所有者和所属组 使得mysql这个用户可以有权限使用
    chown -R mysql:mysql /var/lib/mysql/
    # 启动mysql服务
    systemctl start mysqld
    ```

  * 利用日志文件查看随机密码并利用SQL语句重设密码：

    ```shell
    # 利用管道搜索查看临时密码
    cat /var/log/mysqld.log|grep localhost
    # 结果：localhost: =sFEphHAH8Sk
    # 利用临时密码登录
    mysql -u root -p =sFEphHAH8Sk
    # 修改密码 
    alter user 'root'@'localhost' identified by '你要设置的密码';
    # 然后再重新登录试试就ok了
    ```

### 创建远程访问用户并分配权限

```sql
# 创建远程访问用户
create user 'root'@'%' identified with mysql_native_password by '你的远程访问密码';
# 分配所有权限
grant all on *.* to 'root'@'%';
```

### 关闭防火墙并使用Data Grip远程访问mysql

```shell
# 查看防火墙状态 如果是active则代表开着
systemctl status firewalld.service
# 关闭防火墙
systemctl stop firewalld
# 然后输入密码就OK了
```



