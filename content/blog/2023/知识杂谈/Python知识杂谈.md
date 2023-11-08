---
title: "Python知识杂谈"
date: 2023-10-16T18:18:05+08:00
lastmod: 2023-11-07T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%9F%A5%E8%AF%86%E6%9D%82%E8%B0%88/c%2B%2B_tools_title.jpg"
description: "使用Python编程时遇到的一些语法及知识以及一些注意点"
tags:
- python
categories:
- 知识点总结
series:
- 《知识杂谈》
comment : true
---

### Python知识杂谈

#### Python中的赋值

当使用:

```python
a = "Python"
```

Python解释器干的事情：

1.创建变量a

2.创建一个对象(分配一块内存)，来存储值 'python'

3.将变量与对象，通过指针连接起来，从变量到对象的连接称之为引用(变量引用对象)

* 赋值：**只是复制了新对象的引用，不会开辟新的内存空间。**

并不会产生一个独立的对象单独存在，只是将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生变化，另一个标签也会随之改变。

* 浅拷贝：**创建新对象，其内容是原对象的引用。**

浅拷贝有三种形式： `切片操作`，`工厂函数`，`copy模块中的copy函数`。

Example： lst = [1,2,[3,4]]

切片操作：lst1 = lst[:] 或者 lst1 = [each for each in lst]

工厂函数：lst1 = list(lst)

copy函数：lst1 = copy.copy(lst)

*浅拷贝只拷贝了一层，拷贝了最外围的对象本身，内部的元素都只是拷贝了一个引用*

1）当浅拷贝的值是不可变对象（字符串、元组、数值类型）时和“赋值”的情况一样，对象的id值与浅拷贝原来的值相同。

2）当浅复制的值是可变对象（列表、字典、集合）时会产生一个“不是那么独立的对象”存在。有两种情况：

* *第一种情况：拷贝的对象中无复杂子对象，原来值的改变并不会影响浅拷贝的值，同时浅拷贝的值改变也并不会影响原来的值。原来值的id值与浅拷贝原来的值不同。*
* *第二种情况：拷贝的对象中有复杂子对象（例如列表中的一个子元素是一个列表），如果不改变其中拷贝子对象，浅拷贝的值改变并不会影响原来的值。 但是改变原来的值中的复杂子对象的值会影响浅拷贝的值。*

* 深拷贝：**与浅拷贝对应，深拷贝拷贝了对象的所有元素，包括多层嵌套的元素。深拷贝出来的对象是一个全新的对象，不再与原来的对象有任何关联。**

Example：

```python
import copy

a = 1
b = a
b = 2
print(f"a:{a}")
print(f"b:{b}")
print("------------------------")
c = copy.copy(a)
print(f"a:{a}")
print(f"c:{c}")
print("------------------------")
x = {
    "cards": [
        {
            "devid": 0,
            "cameras": [
                {
                    "cameraId": "S000001",
                    "address": "rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0",
                    "chan_num": 1,
                    "model_names": [
                        "ex1"
                    ]
                }
            ]
        }
    ]
}

print(f"json-x-origin:{x}")
# 改变简单子对象
print("------------------------")
y = copy.copy(x)
y["example"] = "card"
print(f"json-x:{x}")
print(f"json-y:{y}")
# 改变复杂子对象
print("------------------------")
z = copy.copy(x)
z["cards"][0]["cameraId"] = "S000002"
print(f"json-x:{x}")
print(f"json-z:{z}")
# 深拷贝
print("------------------------")
k = copy.deepcopy(x)
k["cards"][0]["cameraId"] = "S000003"
print(f"json-x:{x}")
print(f"json-k:{k}")
```

*上面的测试代码包含了赋值，浅拷贝的三种情况以及深拷贝，测试结果如下*：

```shell
a:1
b:2
------------------------
a:1
c:1
------------------------
json-x-origin:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}]}]}
------------------------
json-x:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}]}]}
json-y:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}]}], 'example': 'card'}
------------------------
json-x:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}], 'cameraId': 'S000002'}]}
json-z:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}], 'cameraId': 'S000002'}]}
------------------------
json-x:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}], 'cameraId': 'S000002'}]}
json-k:{'cards': [{'devid': 0, 'cameras': [{'cameraId': 'S000001', 'address': 'rtsp://admin:1qaz@WSX@10.73.135.42:6554/cam/realmonitor?channel=1&subtype=0', 'chan_num': 1, 'model_names': ['ex1']}], 'cameraId': 'S000003'}]}
```

#### Type Hint入门

Python语言非常灵活，对输入参数和输出参数没有特定的类型约束。但在正式项目中，当代码量比较大的时候，由于类型检查导致的错误会慢慢增加，降低Debug的效率，尤其是在需要合作开发代码的时候会更加低效。

在`Python3.5`之后就逐渐引入了`Type Hint`的特性。

例1:

```python
def f(a, b):
    return a + b
print(f(1, 2))
```

加上具体的类型后：

```python
def f(a: int, b: int) -> int:
    return a + b
print(f(1, 2))
```

*这样做的好处是在IDE中放到f函数上会出现入参和出参的类型。*

我们也可以使用`mypy`静态分析工具来分析：

```shell
(base) ➜  paper_figure mypy test.py 
test.py:5: error: Argument 1 to "f" has incompatible type "str"; expected "int"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
(base) ➜  paper_figure 
```

当然我们可以在`vscode`工作区自动加入`mypy`的检查，`.vscode/settings.json`里面加入：

```json
{
	"python.linting.mypyEnabled": true
}
```

这样在类型传参错误的时候，就会显示报错了。

* 自定义数据类型：

来看一个例子：

```python
class A:
    name = "A"

def get_name(o: A) -> str:
    return o.name

get_name(A) # 在mypy检查下报错信息： error: Argument 1 to "get_name" has incompatible type "type[A]"; expected "A"  [arg-type]
get_name(A()) # 正确
```

有一种特殊情况：

```python
class Node:
    def __init__(self, prev: Node): # Node在mypy中会报错
        self.prev = prev
        self.next = None
```

*上述情况可以看到Node类还没定义，就被当作参数来传入。这种情况可以使用特殊的方法来避免：*

```python
class Node:
    def __init__(self, prev: "Node"):
        self.prev = prev
        self.next = None
```

* 队列和字典：

来看一个例子，按照前面的说法优化一下list求和函数：

```python
def my_sum(lst: list) -> int:
    total = 0
    for i in lst:
        total += i
    return total

my_sum([0, 1, 2])
my_sum(1) # 这里传入interger会报错
my_sum([0, 1, "3"])
```

虽然我们优化了求和函数，保证传入的参数一定是一个list，但是list由于内部可包含的类型很多，即使list内部类型不同的元素也会进行一个求和。优化如下：

```python
# Python3.9以后的版本
def my_sum(lst: list[int]) -> int:
    total = 0
    for i in lst:
        total += i
    return total

my_sum([0, 1, 2])
my_sum(1) # mypy报错
my_sum([0, 1, "3"]) # mypy报错

# Python3.9之前的版本需要这样写
from typing import List # 3.9版本5年之后就要移除了

def my_sum(lst: List[int]) -> int:
    total = 0
    for i in lst:
        total += i
    return total

my_sum([0, 1, 2])
my_sum(1) # mypy报错
my_sum([0, 1, "3"]) # mypy报错
```

*但在实际的项目中，我们经常传入的并不是单纯的List，可能还传入tuple这种数据。* 可以引入如下的数据类型：

```python
from typing import Sequence

def my_sum(lst: Sequence[int]) -> int:
    total = 0
    for i in lst:
        total += i
    return total
# 下面所有程序都不会报错
my_sum([0, 1, 2]) 
my_sum((0, 1, 2))
my_sum(b"012")
my_sum(range(3))
```

那么举一反三，对于字典类型的例子如下：

```python
def my_sum(d: dict[str, int]) -> int:
    total = 0
    for i in d.values():
        total += i
    return total

my_sum({"a": 1, "b": 2, "c": 3})
```

* 针对函数类型可能有多种输入的情况：

```python
# 使用Union
from typing import Union

def f(x: Union[int, None]) -> int:
    if x is None:
        return 0
    return x
# 输入参数可以为int也可以为空
f(None)
f(0)

# 使用Optional
from typing import Optional

def f(x: Optional[int]) -> int:
    if x is None:
        return 0
    return x

f(None)
f(0)
```

#### Type Hint进阶

前面讲的都是关于函数的`Type Hint`，来看一个例子：

```python
users = []
users.append(1)
```

这样`users`可以加入任意的元素，但我们同样可以给变量添加类型：

```python
users: List[str] = []
users.append(1) # mypy报错
```

比较常用的`Any`类型，我们不给变量做类型标注的时候，自动会解释成为`Any`类型，但函数的返回值如果没有时，默认返回的是`None`：

```python
from typing import Any, List

def f(a: List) -> Any:
    a.append(1)
# 下面的代码mypy虽然没有检查出错误，但是从逻辑看一定是有问题的, i的值为None
lst: List = []
i: int = f(lst)

# 把返回的类型设置为None之后，逻辑就正确了
from typing import Any, List

def f(a: List) -> None:
    a.append(1)

lst: List = []
i: int = f(lst) # 该行mypy检查报错
print(i)
```

对于一些真的没有返回值的，比如一些检测到错误的函数：

```python
from typing import NoReturn

def error() -> NoReturn:
    raise ValueError

error()
a = 3 # 这行会暗下去表示不会执行
```

* 对于入参为一整个函数的时候，type hint需要这样调用：

```python
from typing import Callable

def my_dec(func: Callable):
    def wrapper(*args, **kwargs):
        print("start")
        ret = func(*args, **kwargs)
        print("end")
        return ret
    return wrapper

my_dec(1) # mypy报错
```

我们可以使用装饰器（decorator）进行使用：

```python
from typing import Callable

def my_dec(func: Callable):
    def wrapper(a: int, b: int) -> int:
        print(f"args = {a} , {b}")
        ret = func(a, b)
        print(f"result = {ret}")
        return ret
    return wrapper

@my_dec
def add(a: int, b: int) -> int:
    return a + b

@my_dec
def absolute(a: int) -> int:
    return abs(a)
# 目前mypy不会报错，但是运行的时候absolute的函数使用装饰器的时候会报错
add(2, 3)
absolute(-2) 
```

对内部函数也增加type int后，mypy就能检查出错误：

```python
from typing import Callable

def my_dec(func: Callable[[int, int], int]): # 增加了内部函数类型检查
    def wrapper(a: int, b: int) -> int:
        print(f"args = {a} , {b}")
        ret = func(a, b)
        print(f"result = {ret}")
        return ret
    return wrapper

@my_dec
def add(a: int, b: int) -> int:
    return a + b
# 这里在使用装饰器的时候mypy就会报错
@my_dec
def absolute(a: int) -> int:
    return abs(a)

add(2, 3)
absolute(-2)
```

* 最后介绍一下`Literal`，它代表规定传入的值必须是某些特定的值

```python
from typing import Literal

class Person:
    def __init__(
            self,
            name: str,
            gender: Literal["male", "female"]
    ):
        self.name = name
        self.gender = gender

x = "female"
y: Literal["male", "female"] = "female"

a = Person("Bob", "male")
b = Person("Alice", "woman") # mypy报错
c = Person("Alice", x) # mypy报错
d = Person("Alice", y) # 如果使用变量进行参数传入的时候，必须指定为Literal的类型
```

*但是假设这个Literal的限定值会增加，对后期代码维护会比较麻烦，改进方法如下：*

```python
from typing import Literal

Gendertype = Literal["male", "female"] # 这样写后续修改只需要修改这一行

class Person:
    def __init__(
            self,
            name: str,
            gender: Gendertype
    ):
        self.name = name
        self.gender = gender

x = "female"
y: Gendertype = "female"

a = Person("Bob", "male")
d = Person("Alice", y)
```

再来写一个比较常用的例子：

```python
from typing import Optional

ReturnType = tuple[int, Optional[str]]

def f(a) -> ReturnType:
    if a > 0:
        print(a)
        return 0, None
    else:
        return 1, "input is <= 0"

retcode, errmsg = f(0)
```

但是对于这种新的Type会产生一种新的bug：

```python
UserId = int
AttackPoint = int

class Player:
    def __init__(
            self,
            uid: UserId,
            attack: AttackPoint
    ):
        self.uid = uid
        self.attack = attack
    # 这里的更新逻辑错误原本可以通过静态检查分析出来的，现在则检查不出来了(输入的类型应该是AttackPoint，现在赋值给UserId类型也不报错)
    def update_attack(self, atk: AttackPoint):
        self.uid = atk  
```

*可以使用如下方法进行改进：*

```python
from typing import NewType
UserId = NewType("UserId", int)
AttackPoint = NewType("AttackPoint", int)

class Player:
    def __init__(
            self,
            uid: UserId,
            attack: AttackPoint
    ):
        self.uid = uid
        self.attack = attack
    
    def update_attack(self, atk: AttackPoint):
        self.uid = atk # 现在这里就会报错了
```











