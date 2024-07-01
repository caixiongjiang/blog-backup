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

### 推导式

推导式是一种高效创建列表、字典、集合或者其他可迭代的对象的方式，极大了压缩了代码的长度。

#### 列表
先看一个列表复制的例子：
```python
nums = [0, 1, 2, 3, 4, 5]

my_list = []
for i in nums:
    my_list.append(nums[i])
print(my_list)
```
上述为普通的写法，如果使用推导式，它的代码将简化为：
```python
my_list = [i for i in nums]
print(my_list)
```
再来看一个略微复杂的例子：
```python
nums = [0, 1, 2, 3, 4, 5]

my_list = []
for i in nums:
    my_list.append(i**2)
print(my_list)
```
它的推导式写法可变为：
```python
my_list = [i**2 for i in nums]
print(my_list)
```

在循环中加入条件判断
```python
# 1
nums = [0, 1, 2, 3, 4, 5]

my_list = []
for i in nums:
    if i % 2 == 0:
        my_list.append(i**2)
print(my_list)

# 2
nums = [0, 1, 2, 3, 4, 5]

my_list = [] 
for i in nums:
    if i % 2 == 0:
        my_list.append(i**2)
print(my_list)
```
**注意只有if和有if-else的写法是不一样的**
```python
# 1
nums = [0, 1, 2, 3, 4, 5]
my_list = [i**2 for i in nums if i%2 == 0]
print(my_list)

# 2
nums = [0, 1, 2, 3, 4, 5]
my_list = [i**2 if i%2 == 0 else i**3 for i in nums]
print(my_list)
```

再加大一点难度，加入双层for循环：
```python
letters = ["a", "b", "c"]
nums = [1, 2, 3]

my_list = []
for i in letters:
    for j in nums:
        my_list.append((i, j))
print(my_list)
```
推导式的写法为：
```python
letters = ["a", "b", "c"]
nums = [1, 2, 3]

my_list = [(i, j) for i in letters for j in nums]
print(my_list)
```
#### 字典

将列表中的元素一一配对：
```python
letters = ["a", "b", "c"]
nums = [1, 2, 3]

my_dict = {}
for i, j in zip(letters, nums):
    my_dict[i] = j
print(my_dict)
```
推导式写法为：
```python
# 注意切换为字典后写法的变化
my_dict = {i: j for i, j in zip(letters, nums)}
```
#### 集合
集合的特点是**无序，不重复**。
举一个例子，使用频率没有前两个那么高：
```python
l = [1,2,3,4,5,6,7,8,8]
my_set = set()
for i in l:
    my_set.add(i)
print(my_set)
```

```python
l = [1,2,3,4,5,6,7,8,8]
my_set = set()
my_set = {i for i in l}
print(my_set)
```
#### 总结
Python推导式非常适用于从零创建有序的一些迭代对象。但是推导式使用循环最多两层，再多不建议使用，因为比较容易造成代码可读性较差的问题，以及在代码调试上的一些难题。


### 生成器

生成器是一个在处理大量数据时较为有效的工具。它与列表是一种类似的工具，但两者的性能却完全不同。它们的主要区别：

Python中的生成器（Generator）和列表（List）是两种不同的数据结构，它们在内存使用、性能和使用场景上有显著的区别。以下是生成器和列表的主要区别：

1. **内存使用**：
   - **列表**：列表在内存中存储所有的元素。如果列表很大，会占用大量的内存。
   - **生成器**：生成器是惰性计算的，它只在需要时生成下一个元素。这意味着生成器在内存中只存储生成下一个元素所需的状态，因此占用的内存非常少。

2. **性能**：
   - **列表**：列表在创建时会立即计算并存储所有元素，因此访问列表中的元素非常快。
   - **生成器**：生成器在每次迭代时才计算下一个元素，因此生成器的创建和迭代速度可能比列表慢。

3. **迭代行为**：
   - **列表**：列表可以多次迭代，因为所有元素都存储在内存中。
   - **生成器**：生成器只能迭代一次，因为元素是按需生成的，一旦生成并使用，生成器就无法再次生成相同的元素。

4. **语法**：
   - **列表**：列表使用方括号 `[]` 来定义，例如 `[1, 2, 3, 4]`。
   - **生成器**：生成器使用圆括号 `()` 来定义生成器表达式，或者使用 `yield` 关键字来定义生成器函数。例如：
     ```python
     # 生成器表达式
     gen = (x for x in range(10))
     
     # 生成器函数
     def gen_func():
         for i in range(10):
             yield i
     ```

5. **使用场景**：
   - **列表**：适用于需要多次访问、修改或查找元素的场景。
   - **生成器**：适用于需要处理大量数据或无限序列，且不需要多次访问所有元素的场景。

**生成器适用于超大数据场景的读取上，还有大语言模型的流式输出也使用到了生成器的思想。**


来看一个例子来学习生成器：
```python
def square_numbers(nums):
    result=[]
    for i in nums:
        result.append(i * i)
    return result


def gen_numbers(nums):
    for i in nums:
        yield i * i

my_nums = square_numbers([1, 2, 3, 4, 5])
print(my_nums)
# [1, 4, 9, 16, 25]

my_gens = gen_numbers([1, 2, 3, 4, 5])
print("Running next function:")
print(next(my_gens))
print("Then running for loop:")
for gen in my_gens:
    print(gen)
print("Then running for loop:")
for gen in my_gens:
    print(gen)
# Running next function:
# 1
# Then running for loop:
# 4
# 9
# 16
# 25
# Then running for loop:
```

**可以看到先使用next读取之后，for loop的读取便从4开始了，由此可见生成器生成的内容只允许读取一遍，所以再第二次遍历的时候已经变为了空值。**

再举一个大语言模型的流式输出例子：
```python
def stream_output(data, chunk_size=10):
    """
    生成器函数，用于逐个生成数据块
    :param data: 要处理的数据
    :param chunk_size: 每个数据块的大小
    """
    start = 0
    end = chunk_size
    while start < len(data):
        yield data[start:end]
        # 使用时间停顿来模拟大模型处理过程
        time.sleep(0.1)
        start = end
        end += chunk_size

# 示例数据
data = "This is a long piece of text that we want to stream out in chunks."

# 使用生成器进行流式输出
for chunk in stream_output(data):
    print(chunk)
```

