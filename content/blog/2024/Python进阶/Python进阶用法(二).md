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

### 装饰器

装饰器（Decorator）是Python中一种非常强大且灵活的工具，用于修改或增强函数或方法的行为。装饰器本质上是一个函数，它接受一个函数作为参数，并返回一个新的函数。

举一个简单的例子：
```python
def square(x):
    return x*x

def print_running(f, x):
    print(f"{f.__name__} is running")
    return f(x)

result = print_running(square, 2)
print(result)
# square is running
# 4
```
它在这里的作用是不改变函数的情况下，增加了函数运行的提示。同样的功能可以使用装饰器来实现，我们在原有的基础上再增加一个测量时间的功能：
```python

def square(x):
    return x*x

import time

def decorator(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} is running")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time}")
        return result
    return wrapper

decorator_square = decorator(square)
decorator_square(10)
# square is running
# square execution time: 9.5367431640625e-07
```
而python中定义了一个更为简单的方式来使用装饰器吗，就是直接给函数戴一个装饰器的帽子：
```python 
import time

def decorator(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} is running")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time}")
        return result
    return wrapper

@decorator
def square(x):
    return x*x

square(10)
```
上述的写法效果与前面相同。
但是判断不同函数运行时间是否超过阈值的功能虽然常见，但它往往是变化的，也就是说需要的阈值会不同，这样我们可以再套一层定义一个装饰器生成器，来看下面的例子：
```python
import time

def timer_decorator(threshold):
    def time_calculate(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if end_time - start_time > threshold:
                print(f"{func.__name__} took longer than {threshold}")
            return result
        return wrapper
    return time_calculate


@timer_decorator(0.3)
def time_sleep():
    time.sleep(0.4)

time_sleep()
print(time_sleep.__name__)
# time_sleep took longer than 0.3
# wrapper
```
上述代码将装饰器变为了可定义的装饰器，便于使用，但是使用了装饰器后的函数名会发生改变，我们需要使用一种特殊的方法进行继承。
```python
import time
import functools

def timer_decorator(threshold):
    def time_calculate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if end_time - start_time > threshold:
                print(f"{func.__name__} took longer than {threshold}")
            return result
        return wrapper
    return time_calculate


@timer_decorator(0.3)
def time_sleep():
    time.sleep(0.4)

time_sleep()
print(time_sleep.__name__)
# time_sleep took longer than 0.3
# time_sleep
```
上述代码中使用python中自带的装饰器`@functools.wraps`。

总的来说，装饰器有许多优点：
* 提升代码复用性，避免冗余。
* 使用装饰器可以保证一个复杂的函数逻辑清晰，减少代码查看量。
* 通过装饰器，可以扩展别人的函数，在添加额外的行为时，不会修改原函数的逻辑。

### 使用dotenv模块存储敏感信息

在项目中，我们通常会用到一些数据库的密码，以及大模型相关的key信息，然而如果我们将这些信息直接写在代码中，很容易造成泄露。

我们可以使用`python-dotenv`模块来解决这个问题。首先下载这个模块：
```python
pip install python-dotenv
```

在项目目录下新建一个`.env`文件，将敏感信息存入其中：
```scss
OPENAI_API_KEY = "FAKE_OPENAI_API_KEY"
DB_PASSWORD = "FAKE_DB_PASSWORD"
```
通过代码来读取：
```python
from dotenv import load_dotenv
# 将文件中的环境变量变为进程中的环境变量
load_dotenv()

import os 
openai_api_key = os.getenv("OPENAI_API_KEY")
db_password = os.getenv("DB_PASSWORD")
```

当然也可以将项目的名称加入.env文件中来进行区分：
```python
from dotenv import load_dotenv
# 读取的时候也需要加入改变后的名字
load_dotenv("projectA.env")
```

如果需要上传git仓库，则需要加好`.gitignore`：
```.gitignore
*.env
```