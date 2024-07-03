---
title: "Python进阶用法（二）"
date: 2024-07-01T18:18:05+08:00
lastmod: 2024-07-01T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/python_advance/title.jpg"
description: "使用Python的高级特性更加高效地完成任务"
tags:
- python
categories:
- Python语法
series:
- 《Python进阶》
comment : true
---


## Python进阶笔记（二）

### Python中“*”的功能

Python中`*`最常用的便是作为一个运算符号使用，但其实`*`也可用于实现重复、打包、解包功能。

* `*`的重复功能，看一个例子：
```python
print("hahaha" * 3)
# hahahahahahahahaha
print([1, 2, 3] * 3)
# [1, 2, 3, 1, 2, 3, 1, 2, 3]
```
* `*`的打包功能，它可以将多个值捆绑到一个容器中
```python
numbers = [1, 2, 3, 4, 5]
first, *rest = numbers
print(first)
print(rest)
# 1
# [2, 3, 4, 5]
```

还有一个典型的例子便是返回yaml文件的值时，如果不需要用到全部值，可以使用这个方法：
```python

import yaml

def config_read():
    config_path = "./configs/config.yaml"
    fo = open(config_path, 'r', encoding='utf-8')
    res = yaml.load(fo, Loader=yaml.FullLoader)

    return res

def return_config():
    config = config_read()
    return (config["LOG"], config["SERVER"], config["DEVICE"])

first, *_ = return_config()
print(first)
print(_)
# {'LOG_LEVEL': 'DEBUG', 'LOG_FOLDER': './logs', 'LOG_FILENAME': 'service.log', 'BACKUPCOUNT': 5}
# [{'IP': '0.0.0.0', 'PORT': 10004, 'WORKERS': 1}, 'mps']
```

可以看到当前不需要的参数暂时会被存入`_`中。

在函数中有一个常用的作法是使用`args`，所有传入的参数都会被打包成一个列表：
```python
def print_values(*args):
    for arg in args:
        print(arg)

print(1, "2", "都是借口")
# 1 2 都是借口
```

那么说到`args`，就不得不说`kwargs`了，它通常配合`**`使用：
```python
def example(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

example(a=1, b=2, c=3)
# a = 1
# b = 2
# c = 3

example(1, 2, 3) 
# 注意这种写法会报错，因为kwargs必须传入字典
```

* `*`的解包功能，与打包相反，它会将一系列的值打开（类似于解压），比如列表可以使用`*`解包，字典可以使用`**`解包。
```python
def greet(name, age):
    print(f"hello {name}, you are {age} years old.")

person = ("Alice", 30)
greet(*person)
# hello Alice, you are 30 years old.

a = [1, 2, 3]
b = (3, 4, 5)
c = [*a, *b]
print(c)
# [1, 2, 3, 3, 4, 5]
```
可以看到它甚至可以合并元组和列表，因为`*`对列表和元组的解包都生效。再看`**`的例子：
```python
def create_profile(name, age, email):
    print(f"name: {name}")
    print(f"age: {age}")
    print(f"email: {email}")

dict_1 = {
    "name": "Jarson Cai",
    "age": 18,
    "email": "jarsoncai@qq.com"
}
create_profile(**dict_1)
# name: Jarson Cai
# age: 18
# email: jarsoncai@qq.com

dict_2 = {"tall": 180}
dict_3 = {**dict_1, **dict_2}
list_3 = [*dict_1, *dict_2]
set_3 = {*dict_1, *dict_2}
print(dict_3)
# {'name': 'Jarson Cai', 'age': 18, 'email': 'jarsoncai@qq.com', 'tall': 180}
print(list_3)
# ['tall', 'name', 'email', 'age']
print(set_3)
# {'tall', 'name', 'email', 'age'}
```
`**`在传入参数重可以解包，在合并字典中也可以解包，同样在使用`*`对字典进行解包时，默认会将所有的key解包出来作为一个值。