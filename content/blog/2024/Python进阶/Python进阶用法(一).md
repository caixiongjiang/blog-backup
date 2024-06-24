---
title: "Python进阶用法（一）"
date: 2024-06-23T18:18:05+08:00
lastmod: 2024-06-23T09:19:06+08:00
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


## Python进阶笔记（一）

### 面向对象编程

在之前的Python类编程中，只知道类似的self调用类内方法，今天来学习一下类变量、类方法、静态方法。

#### 类变量

来看一个类的例子：
```python
class Student:
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex

s1 = Student("Caicai", "male")
```
可以看到name和sex属性都代表了单个实例的信息，这时，假设我需要统计学生类的人数，也就是学生类实例的个数，我们就可以使用类变量的方法。

```python
class Student:
    student_num = 0

    def __init__(self, name, sex):
        self.name = name
        self.sex = sex
        Student.student_num += 1

s1 = Student("Caicai", "male")
print(f"Student.student_num: {Student.student_num}")
# Student.student_num: 0
```

在类下面添加的变量就是类变量，类变量的访问需要通过类直接进入。

假设我们将上述代码改成了如下形式：
```python
class Student:
    student_num = 0

    def __init__(self, name, sex):
        self.name = name
        self.sex = sex
        # 执行该句时，python需要先取值再写入 
        self.student_num += 1 

s1 = Student("Caicai", "male")
print(f"Student.student_num: {Student.student_num}")
print(f"s1.student_num:{s1.student_num}")
# Student.student_num: 0
# s1.student_num:1
```
上述代码在执行`self.student_num += 1`时，python需要先取值再写入，而实例中并不存在student_num这个属性。因此在取值的时候，python实际上是在类里面获取了`student_num`的值，写入时给实例写入了这一属性，没有改变类内属性的值。**因此我们需要统一通过类名进行访问类属性！！！**

#### 类方法

修正上述方法的表达后，我们增加一个添加学生数量的类方法。
```python
class Student:
    student_num = 0
    # 构造方法
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex
        Student.student_num += 1
    # 类方法
    @classmethod
    def add_students(cls, add_num):
        cls.student_num += add_num
    # 类方法
    @classmethod
    def from_string(cls, info):
        name, sex = info.split(" ")
        return cls(name, sex) # 返回类方法


s1 = Student("Caicai", "male")
s2 = Student.from_string("Caicai male")
print(f"Student.student_num: {Student.student_num}")
# Student.student_num: 2
```

类方法需要使用装饰器`@classmethod`来装饰，类方法的第一个参数通常为类本身class的缩写`cls`，它可用于访问类变量。

上述的类通过`from_string`类方法解析不同的格式来返回构造函数，增加了不同的初始化方法。**使用类方法来替代构造方法是一种很常见的方法。**

#### 静态方法

静态方法需要使用`@staticmethod`来装饰，静态方法不需要传入`cls`或者`self`，它同样不能访问类和实例里面的私有属性。静态方法的适用场景：该静态方法不需要类里面的内容，但在逻辑上这个静态方法需要在类的里面，同样也适用于一系列功能相关的函数封装在一起。

来看一个例子：
```python
class Student:
    student_num = 0
    # 构造方法
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex
        Student.student_num += 1
    # 类方法
    @classmethod
    def add_students(cls, add_num):
        cls.student_num += add_num
    # 类方法
    @classmethod
    def from_string(cls, info):
        name, sex = info.split(" ")
        return cls(name, sex) # 返回类方法

    # 静态方法
    @staticmethod
    def name_len(name):
        return len(name)


s1 = Student("Caicai", "male")
s2 = Student.from_string("Caicai male")
print(f"s2.name：{s2.name}, s2.name.len：{Student.name_len(s2.name)}")
# s2.name：Caicai, s2.name.len：6
```

静态方法在外部必须使用类本身来访问，在类内则可以通过`self`来访问。



