# Jarson Cai's Blog


目前新的博客采用 Github 托管源代码+Github Pages静态页面托管的方式运行。

地址为：caixiongjiang.github.io

---

主题链接：https://github.com/AmazingRise/hugo-theme-diary

评论系统：https://twikoo.js.org/

* 评论系统Vercel部署流程：可以参考这个[b站视频](https://www.bilibili.com/video/BV1Fh411e7ZH?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=841bd3506b40b195573d34fef4c5bdf7)，但是视频比较久远了，现在一些界面已经不同了，具体可以配合[中文文档](https://twikoo.js.org/quick-start.html#vercel-%E9%83%A8%E7%BD%B2)以来看。

*特别需要注意的是：Vercel 部署的环境需配合 1.4.0 以上版本的 twikoo.js 使用！我就是没看到这句话导致我半天配置不成功。*

hugo模板开发教程：

- https://hugo.aiaide.com/post/
- https://www.jianshu.com/p/0b9aecff290c
- http://blog.wikty.com/post/hugo-tutorial/


# 构建方法

```bash
# 克隆项目
git clone https://github.com/caixiongjiang/blog-backup.git

# 初始化本地配置文件
git submodule init
# 拉数据
git submodule update

# 启动服务器
hugo server
```

# 维护

本地部署博客（https端口预览）命令(这里使用Makefile配置了，所以跟官方的方法不一样)：

```shell
hugo server
```

根据markdown文件生成具体的html页面：

```shell
hugo --theme=diary --baseUrl="https://caixiongjiang.github.io/" --buildDrafts
```

此后便会生成相应的一个public文件，以此仓库作为git的根目录，也就是github的master的分支。

* [图标下载地址](https://feathericons.com/)

如果你使用url推送有问题，也可以使用ssh方式进行推送：
```shell
git remote add origin_ssh git@github.com:caixiongjiang/blog-backup.git # 使用ssh连接远程仓库
git push origin_ssh master #进行推送
```
