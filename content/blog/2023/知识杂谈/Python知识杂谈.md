---
title: "Python知识杂谈"
date: 2023-10-16T18:18:05+08:00
lastmod: 2023-10-17T09:19:06+08:00
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

#### Python中的赋值（复制）、浅拷贝、深拷贝

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

