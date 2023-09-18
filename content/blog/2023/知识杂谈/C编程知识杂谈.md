---
title: "C 编程知识杂谈"
date: 2023-09-15T18:18:05+08:00
lastmod: 2023-09-17T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%9F%A5%E8%AF%86%E6%9D%82%E8%B0%88/c%2B%2B_tools_title.jpg"
description: "使用C/C++编程时遇到的一些语法及知识"
tags:
- c++
categories:
- 知识点总结
series:
- 《知识杂谈》
comment : true
---

## C编程知识杂谈

#### typedef vs define

typedef和define都是替一个对象取一个别名，以此增强程序的可读性，区别如下：

* 原理不同：

`#define`是C语言中定义的语法，是预处理指令，在预处理时进行简单而机械的`字符串替换`，不作正确性检查，只有在编译已被展开的源程序时才会发现可能的错误并报错。

`typedef`是关键字，在编译时处理，有类型检查功能。它在自己的作用域内给一个已经存在的类型一个别名，但不能在一个函数定义里面使用`typedef`。**用typedef定义数组、指针、结构等类型会带来很大的方便**，不仅使程序书写简单，也使意义明确，增强可读性。

* 功能不同：

  * typedef用来定义类型的别名，起到类型易于记忆的功能。

  * 另一个功能是定义机器无关的类型。

    如定义一个REAL的浮点类型，在目标机器上它可以获得最高的精度：`typedef long double REAL`， 在不支持long double的机器上，会被看成`typedef double REAL`，在不支持double的机器上，会被看成`typedef float REAL`

  * #define不只是可以为类型取别名，还可以定义常量、变量、编译开关等。

* 作用域不同：

`#define`没有作用域的限制，只要是之前预定义过的宏，在以后的程序中都可以使用，而`typedef`有自己的作用域。

#### extern关键字

在C语言中，修饰符extern用在变量或者函数的声明前，用来说明“此变量/函数是在别处定义的，要在此处引用”。extern声明不是定义，即不分配存储空间。

也就是说，在一个文件中定义了变量和函数， 在其他文件中要使用它们， 可以有两种方式：

1.使用头文件，然后声明它们，然后其他文件去包含头文件

2.在其他文件中直接extern

> 使用场景

1. 现在要写一个c语言的模块，供以后使用（以后的项目可能是c的也可能是c++的），源文件事先编译好，编译成.so或.o都无所谓。头文件中声明函数时要用条件编译包含起来，如下：

```c++
#ifdef __cpluscplus  
extern "C" {  
#endif  
  
//some code  
  
#ifdef __cplusplus  
}  
#endif  
```

也就是将所有函数的声明放在`some code`的位置。

2. 如果这个模块已经存在了，模块的.h文件中没有extern "C"关键字，这个模块又不希望被改动的情况下，可以这样，在你的c++文件中，包含该模块的头文件时加上extern "C", 如下：

```c++
extern "C" {  
#include "test_extern_c.h"  
} 
```

#### ifstream

`std::ifstream` 是 C++ 标准库中的输入文件流类。它是 `std::basic_ifstream` 类的具体化，用于从文件中读取数据。

`std::ifstream` 类提供了一种方便的方式来打开文件，并从文件中读取数据。它继承自 `std::basic_ifstream` 并提供了一些特定于文件的操作。

通过 `std::ifstream` 类，你可以打开指定的文件，并使用提供的成员函数来读取文件中的数据。你可以按照不同的方式读取数据，例如按行读取、按字节读取或按数据类型读取等。

以下是一个简单的示例，展示如何使用 `std::ifstream` 打开文件并读取其中的内容：

```cpp
#include <fstream>
#include <iostream>

int main() {
  std::ifstream file("example.txt"); // 打开名为 "example.txt" 的文件

  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) { // 按行读取文件内容
      std::cout << line << std::endl;
    }
    file.close(); // 关闭文件
  } else {
    std::cout << "无法打开文件" << std::endl;
  }

  return 0;
}
```

在上述示例中，我们使用 `std::ifstream` 打开名为 "example.txt" 的文件，并按行读取其中的内容。如果成功打开文件，则逐行输出文件内容。如果无法打开文件，则输出错误消息。

注意，`std::ifstream` 类位于 `<fstream>` 头文件中，因此在使用 `std::ifstream` 之前需要包含该头文件。