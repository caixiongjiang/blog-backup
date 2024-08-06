---
title: "Python的编程艺术"
date: 2024-07-30T18:18:05+08:00
lastmod: 2024-07-30T09:19:06+08:00
draft: true
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/python_advance/title.jpg"
description: "学习一些较为常用的编程设计思想"
tags:
- python
categories:
- Python语法
series:
- 《Python进阶》
comment : true
---

## Python的编程艺术


### 策略模式

当前我们需要写一个处理文件的类：
```python
from abc import abstractmethod, ABC


class ProcessStrategy(ABC):
    # 约束子类必须实现process_file方法
    @abstractmethod
    def process_file(self, filepath):
        pass


class ExcelProcessStrategy(ProcessStrategy):
    def process_file(self, filepath):
        print("Processing excel file")


class CsvProcessStrategy(ProcessStrategy):
    def process_file(self, filepath):
        print("Process csv file")

class TxtProcessStrategy(ProcessStrategy):
    def process_file(self, filepath):
        print("Process txt file")


class FileProcessor:
    def __init__(self, file_path: str, strategy: ProcessStrategy) -> None:
        self.file_path = file_path
        self.strategy = strategy

    def process_file(self) -> None:
        self.strategy.process_file(self.file_path)
```

在上述代码中，`@abstractmethod`是一个装饰器，使用该装饰器的类方法，在子类继承其类的时候必须实现该方法。`ABC`的全称是`abstract base class`(抽象类)，当我们定义仅作为基类而不应该被实例化的类时，就会用到`ABC`。当前`ABC`和`@abstractmethod`使得基类不能被实例化。然后再FileProcessor中传入了策略类，这样避免了反复更改FileProcessor类的代码。

### EAFP：先斩后奏的编程哲学

EAFP全名叫`it's easier to ask for forgiveness than permission`，请求原谅比请求许可更容易。

来看一个例子：
```python
profile1 = {"name": "Tom", "age": 33}

def print_profile(profile):
    name = profile["name"]
    age = profile["age"]
    print(f"This is {name}, {age} years old.")

print_profile(profile1)


profile2 = {"name": "Jerry"}
print_profile(profile2)
```

可以根据逻辑明显看到，在`print_profile(profile2)`这句代码会报错。

假设使用`请求许可`的方式规避：
```python
def print_profile(profile):
    if "name" in profile and "age" in profile:
        name = profile["name"]
        age = profile["age"]
        print(f"This is {name}, {age} years old.")
    else:
        print("Missing keys!")
```

使用`请求原谅`的方式进行规避：
```python
def print_profile(profile):
    try:
        name = profile["name"]
        age = profile["age"]
        print(f"This is {name}, {age} years old.")
    except KeyError as e:
        print(f"Missing {e} keys!")
```

通常来说，在工程代码上，请求原谅的写法比请求许可是更好的：
* 请求许可中，需要先进行读取两次对象来确认是否存在问题，请求原谅则只需要读取一次。
* 可读性更强，可以让别人迅速知道代码在规避什么问题

再看一个读文件的例子：
```python
import os

def read_file(filepath: str) -> None:
    if os.path.exists(filepath):
        with open(filepath) as f:
            print(f.read())
    else:
        print("File not exists!")
```

它的请求原谅版本：
```python
def read_file(filepath: str) -> None:
    try:
        with open(filepath) as f:
            print(f.read())
    except FileNotFoundError as e:
        print(e)
    else:
        with f:
            print(f.read())
# 这里的else代表try的语句没有出现报错时，会执行的语句
```
