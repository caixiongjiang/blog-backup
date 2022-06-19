# Jarson Cai's Blog


目前新的博客采用 Github 托管源代码+Github Pages静态页面托管的方式运行。

地址为：caixiongjiang.github.io

---

主题链接：https://github.com/AmazingRise/hugo-theme-diary

评论系统：https://twikoo.js.org/(要钱暂时不用)

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
