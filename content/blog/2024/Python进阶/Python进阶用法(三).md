---
title: "Python进阶用法（三）"
date: 2024-07-11T18:18:05+08:00
lastmod: 2024-07-11T09:19:06+08:00
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


## Python进阶笔记（三）

### 魔法方法

来看一个例子：
```python 
print(f"{1+2 = }")
print(f"{'a' + 'b' =}")
# 1+2 = 3
# 'a' + 'b' ='ab'
```

其实其内部发生了这件事：
```python
print((x:=1).__add__(2))
print('a'.__add__('b'))
```

这种内置的方法也可以用于自定义的类中：
```python
from typing import List

class ShoppingCart:
    def __init__(self, items: List[str]):
        self.items = items

    def __add__(self, another_cart):
        new_cart = ShoppingCart(self.items + another_cart.items)
        return new_cart

cart1 = ShoppingCart(["apple", "banana"])
cart2 = ShoppingCart(["orange", "pear"])
new_cart = cart1 + cart2
print(new_cart.items)
# ['apple', 'banana', 'orange', 'pear']
```

可以看到这里的`+`号是重载了运算符，使用了我们写的`__add__`方法，你一定会觉得类和类相加很奇怪，所以这就是为什么叫它魔法方法的原因了。

同样的，如果我们直接打印new_cart，得到的为类在内存中的地址，如果我们要重新定义打印出来的内容，便可以使用另外一种魔法方法，想要显示的内容完全可以自己定义：
```python
from typing import List

class ShoppingCart:
    def __init__(self, items: List[str]):
        self.items = items

    def __add__(self, another_cart):
        new_cart = ShoppingCart(self.items + another_cart.items)
        return new_cart

    def __str__(self):
        return f"Cart({self.items})"

cart1 = ShoppingCart(["apple", "banana"])
cart2 = ShoppingCart(["orange", "pear"])
new_cart = cart1 + cart2
print(new_cart)
# Cart(['apple', 'banana', 'orange', 'pear'])
```

然后我需要打印类内items的长度，我也可以使用`__len__`魔法方法，最后将类可以像函数一样调用，则可以使用`__call__`方法：
```python
from typing import List

class ShoppingCart:
    def __init__(self, items: List[str]):
        self.items = items

    def __add__(self, another_cart):
        new_cart = ShoppingCart(self.items + another_cart.items)
        return new_cart

    def __str__(self):
        return f"Cart({self.items})"

    def __len__(self):
        return len(self.items)

    def __call__(self, *args):
        for item in args:
            self.items.append(item)


cart1 = ShoppingCart(["apple", "banana"])
cart2 = ShoppingCart(["orange", "pear"])
new_cart = cart1 + cart2
print(new_cart)
print(len(new_cart))
# Cart(['apple', 'banana', 'orange', 'pear'])
# 4
new_cart("x1", "x2")
print(new_cart)
print(len(new_cart))
# Cart(['apple', 'banana', 'orange', 'pear', 'x1', 'x2'])
# 6
```

学完了这些方法，可以使用一套题目来验证一下：
```python
class add:
    pass

addTwo = add(2)
addTwo # 2
addTwo + 5 # 7
addTwo(3) # 5
addTwo(3)(5) # 10

# test:
add(1)(2) # 3
add(1)(2)(3) # 6
add(1)(2)(3)(4) # 10
add(1)(2)(3)(4)(5) # 15
```

快来完成这个类吧，答案放在下面：
```python
class add:
    def __init__(self, num: int):
        self.num = num

    def __str__(self):
        return str(self.num)

    def __add__(self, other_num: int):
        new_add = add(self.num + other_num)
        return new_add

    def __call__(self, other_num: int):
        new_add = add(self.num + other_num)
        return new_add

addTwo = add(2)
print(addTwo) # 2
print(addTwo + 5) # 7
print(addTwo(3)) # 5
print(addTwo(3)(5)) # 10

# test:
print(add(1)(2)) # 3
print(add(1)(2)(3)) # 6
print(add(1)(2)(3)(4)) # 10
print(add(1)(2)(3)(4)(5)) # 15
```

### 为什么不应该把列表直接作为函数的参数？

看一个简答的例子：
```python
from typing import List

def add_to(num, target: List = []) -> List:
    print(id(target))
    target.append(num)
    return target

print(add_to(1))
print(add_to(2))
# 4304550208
# [1]
# 4304550208
# [1, 2]
```

你一定会很疑惑第二次调用的时候的结果，从target的id来看，它们是同一个。**这是因为默认参数作为函数的属性，函数定义时就被定义了，而不是调用的时候才定义。**

所以根据Pycharm内部的提示，我们不应该把这种可变的对象作为函数的参数。或者使用它作为参数时，函数中只取值而不对其进行修改，防止发生未预期的错误。

规范的写法如下：
```python
from typing import List, Optional

def add_to(num, target: Optional[List] = None) -> List:
    if not target:
        target = []
    print(id(target))
    target.append(num)
    return target


print(add_to(1))
print(add_to(2))
```