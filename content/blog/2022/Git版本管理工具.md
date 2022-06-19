---
title: "Git：版本管理工具"
date: 2022-06-18T18:18:05+08:00
lastmod: 2022-06-19T18:20:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/git/img_title.jpg"
description: "Git代码版本管理，学习一下分支的概念,分支合并时如何解决merge冲突。"
tags:
- Git
categories:
- 版本管理工具
comment : true
---

## Git版本管理工具

### Git安装

mac下有两种安装方法：

* 官网安装：[https://git-scm.com/downloads](https://git-scm.com/downloads)

* 命令行通过`Homebrew` 源安装：

  Homebrew官网下载：[https://brew.sh](https://brew.sh)

  如果官网提供的脚本下载太慢了，可以试试下面这个脚本：

  ```shell
  $ /bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
  ```

  Homebrew下载完之后，使用brew安装Git：

  ```shell
  $ brew install git
  ```

### Git简介

#### Git管理的文件

* 文本文件（.txt）
* 脚本文件（.py）
* 各种基于文本信息的文件

#### Git不能管理的文件

* 图片文件（.jpg）
* MS word（.doc）

### 创建第一个版本库

#### 创建版本库（init）

首先在桌面创建一个文件夹`gitQAQ`，在Terminal中打开，并进入该目录:

```shell
$ cd ~/Desktop/gitQAQ
```

为了记录每一个对版本记录施加修改的人`user.name`，我们在git中添加用户名和用户email`user.email`：

```shell
$ git config --global user.name "Jarson Cai"
$ git config --global user.email "nau_cxj@163.com"
```

然后，我们就可以建立git的管理文件：

```shell
$ git init
Initialized empty Git repository in /Users/caixiongjiang/Desktop/gitQAQ/.git/
```

这样，我们就建立了一个空的Git版本管理库。

#### 添加文件管理（add）

通常执行`ls`就能看见文件夹中所有的文件，不过git创建的管理库文件`.git`会被隐藏起来，执行下面语句才能看到：

```shell
$ ls -a
.	..	.git
```

新建一个`1.py`文件：

```shell
$ touch 1.py
```

现在我们可以通过`status`来查看版本库的状态：

```shell
$ git status
On branch master    # 在 master 分支

Initial commit

Untracked files:    
  (use "git add <file>..." to include in what will be committed)

    1.py        # 1.py 文件没有被加入版本库 (unstaged)

nothing added to commit but untracked files present (use "git add" to track)
```

现在`1.py`没有加入版本库，我们需要使用`add `把它加入版本库：

```shell
$ git add 1.py
# 再次查看状态 status
$ git status
On branch master

Initial commit

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)

    new file:   1.py    # 版本库已识别 1.py (staged)
```

如果想要一次性添加文件夹中所有未被添加的文件，可以输入如下语句：

```shell
$ git add .
```

#### 提交更改（commit）

我们已经添加完了，就要提交这次的改变，并使用`-m`自定义这个改变，也就是给提交写一个备注。**而且每次提交更改都是时间记录的节点，也就是你什么时候提交就什么时候生成修改的记录。**

#### 流程图

用一张图来表示上述过程：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/git/img1.jpg)

### 记录修改（log&diff）

正如前面所说，每一次提交（commit）的修改，都会被单独的保存起来。也就是说git的中的所有文件都是一次次修改累积起来的，整个commit的过程都会被记录下来。

#### 修改记录（log）

之前我们以`Jarson Cai`的名义对版本库进行了修改，添加了一个`1.py`的文件。

我们可以来看看版本库修改的过程，可以看到`Author`那里已经有了我的名字和email信息。

```shell
$ git log
commit a7cee9ba1286517a6c64c16452e42c52a1fc6b98 (HEAD -> master)
Author: Jarson Cai <nau_cxj@163.com>
Date:   Sat Jun 18 20:34:13 2022 +0800

    create 1.py
```

如果我们对`1.py`文件进行一次修改，添加这行代码：

```python
a = 1
```

然后我们能在`status`中看到修改还没被提交的信息。

```shell
$ git status

On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   1.py    # 这里显示有一个修改还没被提交

no changes added to commit (use "git add" and/or "git commit -a")
```

所以我们把这次修改添加（add）到可被提交（commit）的状态，然后提交这次修改：

```shell
$ git add 1.py
$ git commit -m "change 1"

[master 5d66d74] change 1
 1 file changed, 1 insertion(+)
```

再次查看`log`，现在我们就能看到`create 1.py`和`change 1`这两条修改信息了，而且做出这两条`commit`的ID，修改的`Author`，修改`Date`也被显示在上面。

```shell
$ git log

Author: Jarson Cai <nau_cxj@163.com>
Date:   Sat Jun 18 21:29:28 2022 +0800

    change 1

commit a7cee9ba1286517a6c64c16452e42c52a1fc6b98
Author: Jarson Cai <nau_cxj@163.com>
Date:   Sat Jun 18 20:34:13 2022 +0800

    create 1.py
```

如果删除部分代码或者再添加部分代码，也会被记录上。

```python
a = 1
b = 2
```

#### 查看unstaged

如果想要查看这次还没`add`的修改部分和上个已经`commit`的文件有何不同，我们可以使用`$ git diff`：

```shell
$ git diff

diff --git a/1.py b/1.py
index 1337a53..223ca50 100644
--- a/1.py
+++ b/1.py
@@ -1 +1,2 @@
 a = 1
+b = 2
```

#### 查看staged（cached）

如果你已经`add`了这次修改，文件变成了“可提交状态”（staged），我们可以在`diff`中添加参数`--cached`来查看修改：

```shell
$ git add . # add 全部修改文件
$ git diff --cached
diff --git a/1.py b/1.py

index 1337a53..223ca50 100644
--- a/1.py
+++ b/1.py
@@ -1 +1,2 @@
 a = 1
+b = 2
```

#### 查看staged&unstaged(HEAD)

还有中方法让我们可以查看`add`过（staged）和没`add` （unstaged）的修改，比如我们再修改一下`1.py`但不`add`：

```python
a = 2
b = 1
c = b
```

目前`a = 2`和`b = 1`已被`add`，`c = b`是新的修改，还没`add`：

```shell
# 对比三种不同 diff 形式
$ git diff HEAD     # staged & unstaged

@@ -1 +1,3 @@
 a = 1  # 已 staged  
 b = 2  # 已 staged
+c = b  # 还没 add 去 stage (unstaged)
-----------------------
$ git diff          # unstaged

@@ -1,2 +1,3 @@
 a = 1  # 注: 前面没有 +
 b = 2  # 注: 前面没有 +
+c = b  # 还没 add 去 stage (unstaged)
-----------------------
$ git diff --cached # staged

@@ -1 +1,2 @@
 a = 1  # 已 staged 
+b = 2  # 已 staged

# 可以看出在两个@符号之间的+后面的数字代表当前命令查询出来的修改数量，具体的规则仔细看一下就知道了！
```

为了下节的内容，我们保持这次修改，全部`add`变成`staged`状态，并`commit`。

```shell
$ git add .
$ git commit -m "change 2"

[master 2418b0e] change 2
 1 file changed, 2 insertions(+)
```

### 回到从前（reset）

#### 修改已commit的版本

有时，我们已经提交了`commit`却发现在`commit`中忘了附上另一个文件，接下来我们模拟这种情况。

上节内容最后一个`commit`是`change 2`。所以我们把`1.py`这个文件，改名为`2.py`。并把`2.py`变成`staged`，然后使用`--amend`将这次合并到之前的`change 2`中。

```shell
$ mv 1.py 2.py # 修改文件名
$ git add 2.py
$ git commit --amend --no-edit  # "--no-edit":不编辑，直接合并到上一个commit
$ git log --oneline  # "--oneline":每个commit内容显示在一行

c8e5f0f (HEAD -> master) change 2 # 合并过的 change 2
5d66d74 change 1
a7cee9b create 1.py
```

#### reset回到add之前

有时我们添加`add`了修改，但是又后悔，并想补充一些内容再`add`。这时我们有一种方法可以回到`add`之前。比如在`2.py`文件中添加这一行：

```python
d = 3
```

然后`add`去`staged`再返回到`add`之前：

```shell
$ git add 2.py
$ git status -s # "-s": status的缩写模式

D  1.py
M  2.py  # staged
---------------------------
$ git reset 2.py

Unstaged changes after reset:
M   2.py
---------------------------
$ git status -s

D  1.py
 M 2.py  # unstaged
```

#### reset回到commit之前

在回到过去的`commit`之前，我们必须了解git如何一步步累加更改的。如下所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/git/img2.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/git/img3.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/git/img4.jpg)

每个`commit`都有自己的`id`数字号，`HEAD`是一个指针，指引当前的状态是在哪个`commit`。最近一次`commit`在最右边，我们如果要回到过去，就是让`HEAD`回到过去并`reset`此时的`HEAD`到过去的位置。

```shell
# 不管我们之前有没有做一些 add 工作，这一步都会让我们回到 上一次的 commit
$ git reset --hard HEAD

HEAD 现在位于 c8e5f0f change 2
----------------------------
# 查看所有的log
$ git log --oneline
# 已经回到上一次的commit了
c8e5f0f (HEAD -> master) change 2
5d66d74 change 1
a7cee9b create 1.py
----------------------------
# 回到 5d66d74 change 1
# 方式1: "HEAD^"
$ git reset --hard HEAD^

# 方式2: "commit id"
$ git reset --hard 5d66d74
----------------------------
# 查看现在的 log
$ git log --oneline

5d66d74 (HEAD -> master) change 1
a7cee9b create 1.py
```

那消失的`change 2`怎么办，还可以恢复吗？我们可以查看`$ git reflog`里面最近做的所有`HEAD`改动，并选择想要挽救的`commit id`:

```shell
$ git reflog

5d66d74 (HEAD -> master) HEAD@{0}: reset: moving to 5d66d74
c8e5f0f HEAD@{1}: reset: moving to HEAD
c8e5f0f HEAD@{2}: commit (amend): change 2
2418b0e HEAD@{3}: commit: change 2
5d66d74 (HEAD -> master) HEAD@{4}: commit: change 1
a7cee9b HEAD@{5}: commit (initial): create 1.py
```

重复`reset`步骤就能回到`commit (amend):change 2`（id=c8e5f0f）这一步：

```shell
$ git reset --hard c8e5f0f
$ git log --oneline

c8e5f0f (HEAD -> master) change 2
5d66d74 change 1
a7cee9b create 1.py
```

我们又回到了`change 2`。

### 回到从前（checkout针对单个文件）

#### 改写文件checkout

其实`checkout`最主要的用途不是让单个文件回到过去，后面会介绍`checkout`在分支`branch`中的应用，这一节主要讲使用`checkout`让文件回到过去。

我们现在版本库中的有两个文件：

```shell
- gitQAQ
	- 1.py
	- 2.py
```

我们只对`1.py`进行回到过去操作，回到`c6762a1 change 1`这一个commit。使用`checkout`+id`c6762a1`+`--`+文件目录`1.py`的指针`HEAD`放在这个时刻`c6762a1`:

```shell
$ git log --oneline

904e1ba change 2
c6762a1 change 1
13be9a7 create 1.py
---------------------
$ git checkout c6762a1 -- 1.py
```

这时`1.py`文件的内容就变成了：

```python
a = 1
```

我们在`1.py`加上一行内容`# I went back to change 1`然后`add`并`commit``1.py`：

```shell
$ git add 1.py
$ git commit -m "back to change 1 and add comment for 1.py"
$ git log --oneline

47f167e back to change 1 and add comment for 1.py
904e1ba change 2
c6762a1 change 1
13be9a7 create 1.py
```

可以看出，和`reset`不同，我们的`change 2`并没有消失，但是`1.py`却已经回去了过去。

### 分支

我们之前编辑的所有的改变都是在一条主分支`master`上进行的。通常我们会把`master`当成最终的版本，而开发新版本或者新属性的时候，需要在另一个分支上进行。这样开发和使用就互不干扰了。

#### 使用graph观看分支

我们之前的文件夹中，只有一条`master`分支，我们可以通过`--graph`来观看分支：

```shell
$ git log --oneline --graph

* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

#### 使用branch创建dev分支

接着我们建立另一个分支`dev`,并查看所有分支：

```shell
$ git branch dev # 建立dev分支
$ git branch     # 查看当前分支

 dev       
* master    # * 代表了当前的 HEAD 所在的分支
```

当我们想把`HEAD`切换去`dev`分支的时候，我们可以用到上次说的`checkout`：

```shell
$ git checkout dev

Switched to branch 'dev'
--------------------------
$ git branch

* dev       # 这时 HEAD 已经被切换至 dev 分支
  master
```

#### 使用checkout创建dev分支

使用`checkout -b`+分支名，就能直接创建和切换到新建的分支：

```shell
$ git checkout -b dev

Switched to a new branch 'dev'
--------------------------
$ git branch

* dev       # 这时 HEAD 已经被切换至 dev 分支
  master
```

#### 在dev分支中修改

`dev`分支中的`1.py`和`2.py`和`master`中的文件是一模一样的。因为当前的指针`HEAD`在`dev`分支上，所以现在对文件夹中的文件进行修改不会影响到`master` 分支。

我们在`1.py`上加上这一行`# I was changed in dev branch`，然后再`commit`：

```shell
$ git commit -am "change 3 in dev" # "-am":add所有改变并直接 commit
```

#### 将dev的修改推送到master

这时如果我们的开发版`dev`已经更新好了，我们要将`dev`中的修改推送到`master` 中，大家就能使用正式版中的新功能了。

首先我们要切换到`master`，再将`dev` 推送过来。

```shell
$ git checkout master # 切换到 master 才能把其他分支合并过来

$ git merge dev       # 将 dev merge 到 master 中
$ git log --oneline --graph

* f30de52 (HEAD -> master, dev) change 3 in dev
* c8e5f0f change 2
* 5d66d74 change 1
* a7cee9b create 1.py
```

需要注意的是，如果直接`git merge dev`，git会默认采用`Fast forward`格式进行`merge`，这样`merge`的这次操作不会有`commit`信息。`log`中也不会有分支的图案。我们可以采取`--no-ff`这种方式保留`merge`的`commit`信息。

```shell
$ git merge --no-ff -m "keep merge info" dev  # 保留merge信息
$ git log --oneline --graph

*   c60668f keep merge info
|\  
| * f9584f8 change 3 in dev         # 这里就能看出, 我们建立过一个分支
|/  
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

### merge分支冲突

是想一下这种情况，不仅有人在做开发版`dev`的更新，还有人在修改`master`中的一些bug。当我们再`merge dev`的时候，就会产生冲突。因为git不知道应该怎么处理这个`merge`，因为`master` 和`dev`进行的不同修改。

当创建了一个新的分支后，我们对两个分支都进行了修改。

比如在：

* `master` 中的`1.py`加上`# edited in master`。
* `dev`中的`1.py` 加上`# edited in dev`。

在下面可以看出`master`和`dev` 中不同的`commit`：

```shell
# 这是 master 的 log
* 3d7796e change 4 in master # 这一条 commit 和 dev 的不一样
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
-----------------------------
# 这是 dev 的 log
* f7d2e3a change 3 in dev   # 这一条 commit 和 master 的不一样
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

当我们想要`merge``dev`到`master`的时候：

```shell
$ git branch

  dev
* master
-------------------------
$ git merge dev

Auto-merging 1.py
CONFLICT (content): Merge conflict in 1.py
Automatic merge failed; fix conflicts and then commit the result.

```

git发现的我们的`1.py`在`master`和`dev`上的版本是不同的，所以提示`merge`有冲突。具体的冲突，git已经帮我们标记出来，我们打开`1.py`就能看到：

```python
a = 1
# I went back to change 1
<<<<<<< HEAD
# edited in master
=======
# edited in dev
>>>>>>> dev
```

所以我们只要手动合并一下两者的不同就OK了。我们将当前`HEAD`（也就是`master`）中的描述和`dev`中的描述合并一下。

```python
a = 1
# I went back to change 1

# edited in master and dev

```

然后再`commit`现在的文件，冲突就解决了。

```shell
$ git commit -am "solve conflict"
```

再来看看`master`的`log`：

```shell
$ git log --oneline --graph

*   7810065 solve conflict
|\  
| * f7d2e3a change 3 in dev
* | 3d7796e change 4 in master
|/  
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

### rebase分支冲突

#### 什么是rebase

和上节内容一样，我们来使用更高级的合并方式`rebase`。同样是合并，`rebase`的做法和`merge`不一样。

假设共享的branch是`branch B`，而我在`branch A`上工作，有一天我发现`branch B`已经有了一些小更新，我也想试试我的程序和这些小更新兼容程度，这时就可以使用`rebase`来补充我的分支`branch B`的内容。补充完以后，和后面那张图的`merge` 不同，我还是继续在`C3`上工作，不过此时的`C3` 本质却不一样了，因为吸收了那些小更新，我们用`C3`来代替。

**也就是说上面的步骤是将主线中更新的内容提前补充到目前工作的分支！**

可以看出`rebase`已经改变了`C3`的属性，`C3`已经不是`C1`衍生而来的了。这一点和`merge`不一样。`merge`在合并的时候创建了一个新的`C5` `commit`。这一点不同，使得在共享分支中使用`rebase`变得危险。如果是共享分支的历史被改写。别人之前共享内容的`commit`就被你`rebase` 修改掉了。

所以需要强调的是**只能在你自己的分支中使用rebase，和别人共享的部分是不能用**。没事，我们还能用在`reset 这一节`提到的`reflog`恢复原来的样子。为了验证在共享分支上使用`rebase`的危险性，我们在下面的例子中也验证一下。

#### 使用rebase

初始的版本库还是和上回一样，在`master`和`dev`分支中都有自己的独立修改。

```shell
# 这是 master 的 log
* 3d7796e change 4 in master # 这一条 commit 和 dev 的不一样
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
-----------------------------
# 这是 dev 的 log
* f7d2e3a change 3 in dev   # 这一条 commit 和 master 的不一样
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

当我们想要用`rebase`合并`dev`到`master`的时候：

```shell
$ git branch

# 输出
  dev
* master
-------------------------
$ git rebase dev 

# 输出
First, rewinding head to replay your work on top of it...
Applying: change 3 in dev
Using index info to reconstruct a base tree...
M   1.py
Falling back to patching base and 3-way merge...
Auto-merging 1.py
CONFLICT (content): Merge conflict in 1.py
error: Failed to merge in the changes.
Patch failed at 0001 change 3 in dev
The copy of the patch that failed is found in: .git/rebase-apply/patch

When you have resolved this problem, run "git rebase --continue".
If you prefer to skip this patch, run "git rebase --skip" instead.
To check out the original branch and stop rebasing, run "git rebase --abort".
```

git发现的我们的`1.py`在`master`或者`dev`上的版本是不同的，所以提示`merge`有冲突，具体的冲突，git已经帮我们标记出来了，我们打开`1.py` 就能看到：

```python
a = 1
# I went back to change 1
<<<<<<< f7d2e3a047be4624e83c1265a0946e2e8790f79c
# edited in dev
=======
# edited in master
>>>>>>> change 4 in master
```

这时`HEAD`并没有指向 `master`或者`dev`，而是停在了`rebase`模式上：

```shell
$ git branch 
* (no branch, rebasing master) # HEAD 在这
  dev
  master
```

所以我们打开`1.py`，手动合并一下两者的不同。

```python
a = 1
# I went back to change 1

# edited in master and dev
```

然后执行`git add`和`git rebase --continue`就完成了 `rebase`的操作了。

```shell
$ git add 1.py
$ git rebase --continue
```

再来看看`master`的`log`：

```shell
$ git log --oneline --graph

# 输出
* c844cb1 change 4 in master    # 这条 commit 原本的id=3d7796e, 所以 master 的历史被修改
* f7d2e3a change 3 in dev       # rebase 过来的 dev commit
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

**注意，这个例子也说明了使用`rebase`要万份小心，千万不要在共享的branch中`rebase`，`master`的历史已经被`rebase`改变了。**`master`当中别人提交的`change 4`被修改了，所以不要在共享分支中使用`rebase`。

**从上面的例子可以看出，如果共享分支和其他分支都进行修改了，此时merge会冲突。解决方案就是其他分支先进行rebase共享分支的修改，千万不能在共享分支上进行rebase**

### 临时修改

如果我正在改进代码，突然接到老板的电话要改之前的一个程序。虽然还需要很久时间才能改进自己的代码，但如果我不想把要改的程序和自己代码改进的部分一起`commit`，这时`stash`就是我的救星。使用`stash`能先将我的改进的部分放在一边分隔开来，先处理老板的任务。

#### 暂存修改

假设我们在`dev`分支上改代码：

```shell
$ git checkout dev
```

在`dev`中的`1.py`中加上一行`# feel happy`，然后有临时的任务，我还没修改完代码。所以可以用`stash`将这些改变暂时放一边。

```shell
$ git status -s
# 输出
 M 1.py
------------------ 
$ git stash
# 输出
Saved working directory and index state WIP on dev: f7d2e3a change 3 in dev
HEAD is now at f7d2e3a change 3 in dev
-------------------
$ git status
# 输出
On branch dev
nothing to commit, working directory clean  # 干净得很
```

#### 做其他任务

然后我们建立另一个`branch`用来完成老板的任务：

```shell
$ git checkout -b boss

Switched to a new branch 'boss' # 创建并切换到 boss
```

然后完成老板的任务，比如添加`# lovely boss`去`1.py`。然后`commit`，完成老板的任务。

```shell
$ git commit -am "job from boss"
$ git checkout master
$ git merge --no-ff -m "merged boss job" boss # 保留分支的 commit 信息
```

`merge`如果有冲突的话，可以像之前那样解决。

通过以下步骤完成老板的任务，并观察一下`master`的log：

```shell
$ git commit -am "solve conflict"
$ git log --oneline --graph
*   1536bea solve conflict
|\  
| * 27ba884 job from boss
* | 2d1961f change 4 in master
|/  
* f7d2e3a change 3 in dev
* 47f167e back to change 1 and add comment for 1.py
* 904e1ba change 2
* c6762a1 change 1
* 13be9a7 create 1.py
```

#### 恢复暂存

完成任务之后，可以继续改代码了。

```shell
$ git checkout dev
$ git stash list   # 查看在 stash 中的缓存

stash@{0}: WIP on dev: f7d2e3a change 3 in dev
```

上面说明`dev`中，我们有`stash`的工作。现在可以通过`pop`来提取这个并继续工作。

```shell
$ git stash pop

On branch dev
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   1.py

no changes added to commit (use "git add" and/or "git commit -a")
Dropped refs/stash@{0} (23332b7edc105a579b09b127336240a45756a91c)
----------------------
$ git status -s
# 输出
 M 1.py     # 和最开始一样了
```

### Github在线代码管理

#### 建立github版本库

注册一个github账户，然后添加一个repository。

#### 连接本地版本库

将之前的例子推送到github上：

```shell
$ git remote add origin https://github.com/caixiongjiang/git-demo.git
$ git push -u origin master     # 推送本地 master 去 origin
$ git push -u origin dev        # 推送本地 dev  去 origin
```

 这样github仓库里就有你推上去的版本库了，你还可以看到你之前的`commit`具体做了什么。

#### 推送修改

如果在本地进行修改，比如在`1.py`文件中加上`# happy github`，然后`commit`并推上去：

```shell
$ git commit -am "change 5"
$ git push -u origin master
```

github上就可以看到你的修改了。

