---
title: "Python中的import"
date: 2023-05-06T18:18:05+08:00
lastmod: 2023-05-05T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%9F%A5%E8%AF%86%E6%9D%82%E8%B0%88/python_title.jpg"
description: "解决python项目中的import路径错误问题。"
tags:
- Python
categories:
- 语言基础
series:
- 《知识杂谈》
comment : true
---

## Import的常用方法
最初的项目相关的代码都在Windows系统下运行，写这个也是因为了在Linux系统下运行代码的需求。我们会发现在Pycharm里能运行的代码，在Linux下就会报错，大部分错误都来自import的错误。

### 场景及使用方法

#### 绝对路径import

> Note：在Python3的新版本中文件夹下面不需要`__init__.py`这个文件，也可以视为一个`Package`，不过也可以用该文件来实现一些额外的功能。

> 在下面的例子中大写字母开头的英文代表文件夹，小写字母开头的代表文件。
 

* 同级目录下的单个脚本引用

目录：
```
ROOT
- example.py
- test.py
```

如果你想要在`test.py`文件中引用`example.py`的方法，你可以使用以下的代码：
```python
# 引入example.py中的所有内容作
import test
# 引入example.py中的方法
from test import '方法名'
# 使用.引入的文件中的类
x = test.Class_A
```


* 同级目录下的单个文件夹引用

目录:
```
ROOT
- Package
-- example.py
- test.py
```

如果想要在`test.py`中引入`Package`整个包，你可以使用如下代码：
```python
# 引入整个包
import Package
# 引入包中的一个文件
import Package.example # 这样import的方式会让Package这个文件夹加入系统的路径
# 将某个文件赋值给某个变量
import Package.example as ep # 这样ep的内容就变成了example这个文件
# 从Package里引入某个文件中的某个方法或者类
from Package.example import function/Class_A
```

> Note:如果你的Package里有`__init__.py`文件，那么Python在引入包后还会额外运行`__init__.py`。如果没有，Python并不会知道`Package`文件夹下有什么文件。


#### 相对路径import

其实上面的绝对路径import大家一般都不会弄错，而这个相对路径import就非常容易出错了。

在进行场景演示前，先要说一下Python的import机制。在运行Python脚本时的import机制是：首先Python会寻找当前文件所在的文件夹，那么这个文件夹下的所有子文件夹和文件都会被加入系统路径，但是子文件下的文件则不会被Python知道。

举一个例子：
```
ROOT
- Pacakge1
- Package2
- utils.py
- train.py
```
如果在运行train.py，那么其他同级目录下的文件夹和文件都会加入系统的路径。

虽然相对路径引用非常容易引起bug，但是这在一些项目里是必须的。比如在AI中训练的所需的权重一般都使用相对路径存放。因为如果这个项目被别人下载运行时，绝对路径会造成大量报错。

* 在一个Package下的文件相互引用(它们之间的关系是稳定的，且该Package经常需要被移植)

目录：
```
ROOT
- Package
-- example.py
-- test.py
```

如果想在`test.py`里引用`example.py`,可以使用如下代码：
```python
from .test import example 
# 注意哦上述代码在test.py运行时，Python中先被将其转化为绝对路径，也就是Package.test
```

举一个在AI中经常使用的例子：
```
ROOT
- Models
-- model.py
-- backbone.py
- train.py
```

我们在`model.py`中需要引入`backbone.py`作为部分结构，使用相对路径引用会出现两种情况：

在使用`if __name__ == '__main__':`进行测试`model.py`是否能跑通时，会遇到如下问题：

1.如果你使用下面的代码进行导入，在测试时会报错
```python
# 会提示找不到backbone
from .backbone import function
```

2.如果你使用下面的代码导入，外部的`train.py`在调用`model.py`时会报错
```python
# model.py可以测试功能
from backbbone import function
# train.py 提示找不到backbone文件，
# 这是因为当前的路径已经变成了上一级的路径，
# 而不是在model.py运行时的路径
from Models.model import Class_model
```

3.在Pycharm中能运行`model.py`的测试，这是因为Pycharm会自动将根目录下的所有文件和文件夹加入系统路径，而在Linux中或者vscode中就会提示找不到Models文件夹。**这也是许多代码在Pycharm文件下能运行，而Linux下会出错的主要原因。**

那么解决方法就是将整个项目需要运行或者测试的代码都放到根目录或者引入根目录.而在Package内部使用相对路径引用进行相互引用。

目录如下：
```
ROOT
- Models
-- model.py
-- backbone.py
- train.py
- test.py
```

使用如下代码：
```python
# model.py
import .backbone import functin
# train.py
from Models.model import Class_model
# test.py 在test.py中进行测试网络
from Models.model import Class_model
if __name__ == '__main__':
    """
    测试代码
    """
```

* 在一个Package下的子文件和子子文件的引用

还是相同的，我们只能将Package内部的测试功能放到外部来。

```
ROOT
- Models
-- utils.py
-- Nets
--- model.py
--- backbone.py
- train.py
- test.py
```

使用如下代码：
```python
# model.py
from ..utils import function
# train.py
from Models.Nets.model import Class_model
# test.py
from Models.Nets.model import Class_model
if __name__ == '__main__':
    """
    测试代码
    """
```

* 两个文件夹的子文件进行相互引用

目录：
```
ROOT
- Configs
-- config.py
- Tools
-- train.py
```

使用如下代码，这种情况`train.py`是无法直接运行的,最好是在最外头写一个`train.sh`来运行train.py

```python
# train.py
from Config.config import function
# 最后在根目录下调用train.py
```


