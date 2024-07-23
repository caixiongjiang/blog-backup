---
title: "Python进阶用法（四）"
date: 2024-07-22T18:18:05+08:00
lastmod: 2024-07-22T09:19:06+08:00
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


## Python进阶笔记（四）

### 切片

废话不多说，直接看例子：

```python
# 注意都是英文冒号
receipt1 = "物品1: 苹果 数量: 5  单价: 3.00元"
receipt2 = "物品2: 香蕉 数量: 10 单价: 2.00元"
receipt3 = "物品3: 橙子 数量: 3  单价: 5.00元"

print(f"item: {receipt1[5:7]}, price: {receipt1[19:23]}")
print(f"item: {receipt2[5:7]}, price: {receipt2[19:23]}")
print(f"item: {receipt3[5:7]}, price: {receipt3[19:23]}")

# item: 苹果, price: 3.00
# item: 香蕉, price: 2.00
# item: 橙子, price: 5.00
```

上面的例子是为了从字符串中提取名称和价格，但可以看出这种写法明显是不合理的，索引使用的多了，所有的地方都需要进行修改。这里可以使用slice对象进行预先定义：
```python
receipt1 = "物品1: 苹果 数量: 5  单价: 3.00元"
receipt2 = "物品2: 香蕉 数量: 10 单价: 2.00元"
receipt3 = "物品3: 橙子 数量: 3  单价: 5.00元"

item_slice = slice(5, 7)
price_slice = slice(19, 23)

print(f"item: {receipt1[item_slice]}, price: {receipt1[price_slice]}")
print(f"item: {receipt2[item_slice]}, price: {receipt2[price_slice]}")
print(f"item: {receipt3[item_slice]}, price: {receipt3[price_slice]}")
```

切片的传参机制跟range是相同的：
当传入一个参数时代表`end`，两个参数时代表`start`和`end`，三个参数时代表`start`、`end`和`step`。

```python
# 从第五个开始的所有字符
receipt1[5:]
receipt1[slice(5, None)]
# 只需要step的时候
receipt1[slice(None, None, 5)]
receipt1[::2]
```
**上面的写法是等价的！**

### tqdm

如果一个程序需要很长时间运行，那我们在等待的过程中往往是比较焦虑的，担心程序卡在什么地方，或者网络不通等等问题。
那么tqdm可以帮你解决这个问题，也就是进度条显示，这对所有的训练类程序都是很有用处的。

下载tqdm库：
```bash
pip install tqdm
```

基本使用：
```python
import time
from tqdm import tqdm

for _ in tqdm(range(10000)):
    time.sleep(0.001)
```
上面这段程序会出现一个快速执行的进度条，tqdm使用也非常简单，仅仅将`range(10000)`包裹即可。

* trange使用：
```python
from tqdm import trange

for _ in trange(10000):
    time.sleep(0.001)
```

大部分情况下，这样使用即可，见到的大部分训练程序也都是简单使用。如果想要花里胡哨地设置进度条，可以继续看下面的内容：
```python
# 进度条前加入文字描述
for _ in tqdm(range(10000), desc="desc参数出现在这里"):
    time.sleep(0.001)
```

同样我们也可以对列表和生成器等可迭代对象使用进度条，举一个例子：
```python
students = ["a", "b", "c"]

for _ in tqdm(students):
    time.sleep(0.1)

def my_generator():
    for i in range(50):
        yield i
# 进度条不可见，因为生成器并不会预先知道长度，和列表不同
for _ in tqdm(my_generator()):
    time.sleep(0.1)

# 人为设定进度条
for _ in tqdm(my_generator(), total=50):
    time.sleep(0.1)
```

* 对于没有迭代器的场景，我们也可以手动设置进度条：
```python
with tqdm(total=100) as pbar:
    pbar.update(10)
    time.sleep(2)
    pbar.update(20)
    time.sleep(2)
    pbar.update(70)
    pbar.close()
```

* tqdm也可以配合数据处理的库进行使用：
```python
import seaborn as sns

df = sns.load_dataset("iris")
print(df.head())

# 普通写法
df.petal_width.apply(lambda x : x*2)

# 增加进度条写法: apply替换成了progress_apply
tqdm.pandas(desc="Processing iris")
out = df.petal_width.progress_apply(lambda x : x*2)
print(out)
```