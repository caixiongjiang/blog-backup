---
title: "Docsify：最简易的知识库生成器"
date: 2022-01-28T16:15:05+08:00
lastmod: 2022-01-28T00:43:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/docsify.jpg"
description: "Docsify是一款轻量化的知识库搭建工具，搭建简单，可以将自己学过的又容易忘的放在上面，当成一个知识库来用"
tags:
- Docsify
- 知识库搭建
categories:
- 知识库搭建
series:
- 环境配置&博客搭建
comment : true
---

## 我的个人知识库搭建：蜜蜂网

docsify是一款比Hugo更加轻便的网站部署软件，它比Hugo少了生成html文件这一步，而且少了许多配置文件，真正的本地文件也就只有几个。

许多人用它来搭建博客，但我认为正因为它参数配置很少，不利于搭建自己想要的风格，反而作为自己的知识库会更加便利。我们都不想哪里忘了去翻那厚重的书本吧！

下面是一些参考网站：

* [docsify官网](https://docsify.js.org/)
* [Codesheep程序羊关于个人知识库搭建的b站视频](https://www.bilibili.com/video/BV1eu411m797)

**为了书写方便，后文中提到的命令在windows端在Git中运行，在mac端在terminal中运行。**

### 环境配置

#### Node.js安装

`Node.js`是为你使你的电脑上确保有`npm`工具，因为docsify的安装需要用到`npm`工具。

* [Node.js官网下载地址](https://nodejs.org/en/)

下载安装好之后，可以查看一下`npm`是否已经安装好了：

* windows端：在桌面点击右键，选择`Git bash here`，如果出现了，其实就已经安装好了。如果不放心还是可以输入：

  ```shell
  npm -v
  ```

  如果出现了版本号信息，则表示已经安装好了。

* mac端：打开终端`terminal`同样输入上述的命令：

  ```shell
  npm -v
  ```

  如果出现了版本号信息，则表示已经安装好了。

#### docsify安装

这里的docsify安装其实安装的是它的二进制命令行工具。

windows/mac端输入：

```shell
npm install -g docsify-cli
```

安装完成后输入：

```shell
docsify -v
```

如果出现版本号则说明安装好了。

### 新建站点

这里用我的站点名称`jarson-cai-blog`作为例子：

1.新建知识库的文件夹`jarson-cai-blog`

2.使用命令行工具进入知识库文件夹（根据你的路径不同而不同），并对其进行初始化，输入：

```shell
cd Desktop/jarson-cai-blog
docsify init
y  //提示你是否初始化，选择y
```

你的站点已经生成了，进入之后会有两个文件，一个是html文件，一个是Markdown文件。

### 本地启动服务端口（https）

*这个功能主要用于修改配置参数时，查看预览效果的功能，是非常重要的。*

输入：

```shell
docsify serve
```

会提示`Web Server is available at //localhost:3000/`，复制`//localhost:3000/`在网页端预览。

### 写知识库文件

docsify统一采用Markdown格式进行书写。而且其中的语法与markdown基本一致，没有个人特殊的配置。

### 新建Github仓库，作为部署网站的仓库

注意事项：**新建的仓库名必须与你的github用户名相同**，以我的github名字caixiongjiang为例。

我的仓库名为：`caixiongjiang.github.io`，点击创建，注意先不要生成`README.md`文件，否则你在第一次推送文件前还需要先拉取文件。

如果你没有`VPN`，不能稳定访问github，可以使用`Gitee`，这个仓库名就没有限制了

### 生成知识库站点

这里特别要说的一点就是docsify不需要生成站点，也就是没有Hugo中有.md文件生成.html文件的过程。

**也就是说，你只要将本地的markdown文件修改了，上传到服务器上就OK了。**是不是特别方便？哈哈哈。

### 在Github Pages或者Gitee Pages部署站点

* 使用Github配置：

  步骤：

  * 进入站点根目录

  * 创建git本地仓库，并上传github
  
    ```shell
    git init		//git初始化
    git add .
    git commit -m "我的第一次博客提交"   //将文件提交至本题仓库
    git remote add origin https://github.com/caixiongjiang/caixiongjiang.github.io.git //将本题仓库与远端的github仓库进行关联
    git push -u origin master //将文件上传到远端的github仓库的master分支
    ```
  
  * 输入你的github用户名和github密码
  
    当然你也可以提前配置`SSH`钥匙

    ```shell
    git config --global user.name "用户名"
    git config --global user.email "邮箱地址"
    ```
  
    然后生成SSH公钥，并配置到github上，**具体步骤可自行百度**，如果嫌麻烦可以每次输入用户名和密码。
  
  * 最后你就可以使用`caixiongjiang.github.io`的域名进行访问你的静态知识库了。

在github上的步骤就是如此了。

* 使用Gitee配置：

  前面的步骤和github一样，需要多一步手动生成`Gitee Pages`：

  **进入远端仓库，点击`服务`，点击`Gitee Pages`，勾选`强制使用HTTPS`，点击生成，就可以访问了。

### 更改博客内容后上传Github（和第一次上传有所不同）

1.在本地知识库的根目录。

2.更改文件之后上传。

```shell
git add .
git commit -m "你要备注的内容"
git push -u origin master //如果你只有master分支也可以直接使用 git push
```

## 参数配置

这里所有的参数配置均在站点的`index.html`上更改，它相当于网站的入口文件。

为了更加清晰地知道参数添加在哪里，我附上我的配置参数：

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="description" content="Description">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0">
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/lib/themes/vue.css">//主题的更换在这里
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: 'Jarson Cai',
      repo: '',
      coverpage: true,
      loadNavbar: true,
      count:{
        countable:true,
        fontsize:'0.9em',
        color:'rgb(90,90,90)',
        language:'chinese'
      }
    }
  </script>
  <!-- Docsify v4 -->
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
  <script src="//unpkg.com/docsify-count/dist/countable.js"></script>
</body>
</html>
```



* 添加封面：

  ```html
  coverpage:true
  ```

  添加完了封面之后，要在站点根目录下添加一个名为`_coverpage.md`的文件，里面写一些封面的文字和按钮。

  这里我也将我的文件内容分享给大家：

  ```markdown
  [![logo.png](https://i.postimg.cc/286V2b8s/logo.png)](https://postimg.cc/s1kj22Wm)
  
  - 本站名为“**蜜蜂**”，意为像蜜蜂一样勤奋学习！这里聚集了自学编程以来所用资源和分享内容的大聚合。**网站内容会持续保持更新，欢迎收藏品鉴！**
  
  ## Jarson Cai的知识库
  [**Github**](https://github.com/caixiongjiang)
  [**开启阅读**](README.md)
  ```

  

* 添加导航栏

  ```html
  loadNavbar: true
  ```

  添加完了导航栏的参数之后，要在站点根目录下添加一个名为`_navbar.md`的文件，里面写一些导航栏的内容。

* 文章记数器：可以计算本篇文章有多少字，大约需要读多少分钟

  ```html
  count:{
          countable:true,
          fontsize:'0.9em',
          color:'rgb(90,90,90)',
          language:'chinese'
        }
  ```

  在底部脚本区域添加语句：

  ```html
  <script src="//unpkg.com/docsify-count/dist/countable.js"></script>
  ```

* 添加脚本：官网提供了非常多好用的脚本，比如搜索框等等，可以参考官方的文档[官方插件预览](https://docsify.js.org/#/zh-cn/plugins)

## 蜜蜂网预览

![蜜蜂网封面](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/docsify_example.png)  
![蜜蜂网内容预览](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/content_example.png)
