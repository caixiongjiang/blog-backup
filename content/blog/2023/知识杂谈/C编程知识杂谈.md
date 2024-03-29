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

#### std::make_shared && std::shared_ptr

`std::make_shared` 和 `std::shared_ptr` 是 C++ 中用于管理共享所有权的智能指针。

`std::shared_ptr` 是一个模板类，用于管理动态分配的对象的所有权。它允许多个 `std::shared_ptr` 共享同一个对象，并在最后一个引用被销毁时自动释放对象的内存。这种共享所有权的机制称为引用计数。

`std::make_shared` 是一个模板函数，用于创建一个动态分配的对象并返回相应的 `std::shared_ptr` 智能指针。它能够在单个内存分配中同时创建对象和管理引用计数，从而提高了性能和内存利用率。

使用场景：

- 在需要多个智能指针共享同一个对象所有权的情况下，可以使用 `std::shared_ptr`。这样可以避免手动跟踪和释放对象的内存。
- `std::make_shared` 通常用于创建动态分配的对象并返回相应的 `std::shared_ptr`。它提供了一种方便的方式来创建智能指针并避免显式地调用 `new` 进行对象分配。

使用示例：

```cpp
#include <memory>

int main() {
    // 使用 std::shared_ptr 创建和管理对象
    std::shared_ptr<int> ptr1(new int(42)); // 通过 new 创建对象，并手动管理内存
    std::shared_ptr<int> ptr2 = ptr1; // 共享所有权
    // ...

    // 使用 std::make_shared 创建和管理对象
    std::shared_ptr<int> ptr3 = std::make_shared<int>(42); // 创建对象并管理内存
    std::shared_ptr<int> ptr4 = ptr3; // 共享所有权
    // ...

    return 0;
}
```

*无论是使用 `new` 还是 `std::make_shared`，都可以使用 `std::shared_ptr` 来管理对象的所有权。然而，使用 `std::make_shared` 通常更为推荐，因为它能够提供更好的性能和内存管理。*

#### C++工程的环境

对于日常开发来说，我们在开发C++工程时，通常只需要将环境安装在本机上，这时候将代码发给别人时，则需要一个代码包和一个Docker启动镜像，里面包含了需要的环境。这对于部署来说是非常重的，那如何摆脱这个Docker镜像，只需要一个c++编译的基础环境。

假设要使用OpenCV包：

1. 下载OpenCV源代码
2. 解压源代码到工程中
3. 配置CMake，生成适合该平台的构建文件，指定生成的库文件路径。（该步骤中不同的环境包有不同的构建方式）

```shell
$ cd opencv
$ mkdir build
$ cd build
$ cmake ..
```

4. 编译安装

```shell
$ make
$ sudo make install 
```

5. OpenCV库文件复制到您的工程中：将生成的OpenCV库文件复制到您的工程目录中的合适位置。您可以将库文件放在与您的C++代码文件相同的目录中，或者创建一个名为`lib`的子目录，并将库文件放在其中。
6. 设置编译器选项：在您的C++代码文件中，使用`#include`指令包含OpenCV的头文件，并在CMake或者Makefile工具添加适当的库路径和库文件名称。

#### C++中的各种变量类型

* 最基础的类型：
  * bool
  * char
  * unsigned short int
  * short int
  * unsigned long int
  * long int
  * int（16位）
  * int（32位）
  * unsigned int（16位）
  * unsigned int（32位）
  * float
  * double
* 根据C++中的位数分：
  * Int8 - 8位int
  * Int16 - 16位int
  * Int32 - 32位int
  * Int64 - 64位int
  * UInt8 - 无符号8位int
  * UInt16 - 无符号16位int
  * UInt32 - 无符号32位int
  * UInt64 - 无符号64位int
* int_t同类：
  * int8_t      : typedef signed char;
  *  uint8_t    : typedef unsigned char;
  *  int16_t    : typedef signed short ;
  *  uint16_t  : typedef unsigned short ;
  *  int32_t    : typedef signed int;
  *  uint32_t  : typedef unsigned int;
  *  int64_t    : typedef signed  long long;
  *  uint64_t  : typedef unsigned long long;
  
#### weak_ptr

`std::weak_ptr` 是 C++ 标准库中的一个智能指针，它是一种不拥有所指向对象的智能指针。`std::weak_ptr` 对象不会增加 `std::shared_ptr` 所指向对象的引用计数。这意味着，即使 `std::shared_ptr` 对象的引用计数变为零，所指向的对象也不会被销毁。

`std::weak_ptr` 的主要用途是打破 `std::shared_ptr` 的环形引用（circular references），这在使用 `std::shared_ptr` 时可能会导致内存泄漏。当两个对象相互引用，并且它们的 `std::shared_ptr` 引用计数都不会变为零时，就会出现环形引用。

`std::weak_ptr` 提供了一个 `std::shared_ptr` 不能提供的功能：检查所指向的对象是否仍然存在。这可以通过调用 `std::weak_ptr` 的 `lock()` 或 `expired()` 成员函数来实现。如果对象仍然存在，`lock()` 会返回一个新的 `std::shared_ptr` 指向该对象；如果对象已经被销毁，`lock()` 会返回一个空的 `std::shared_ptr`。`expired()` 则简单地返回一个布尔值，指示对象是否仍然存在。

下面是一个使用 `std::weak_ptr` 的例子：

```cpp
#include <iostream>
#include <memory>

class MyClass {
public:
    MyClass() {
        std::cout << "MyClass constructed\n";
    }
    ~MyClass() {
        std::cout << "MyClass destroyed\n";
    }
};

int main() {
    std::weak_ptr<MyClass> weakPtr;

    {
        std::shared_ptr<MyClass> sharedPtr = std::make_shared<MyClass>();
        weakPtr = sharedPtr;

        // 使用 weakPtr
        if (auto tempPtr = weakPtr.lock()) {
            std::cout << "MyClass still exists\n";
        } else {
            std::cout << "MyClass has been destroyed\n";
        }
    }

    // sharedPtr 已经被销毁，weakPtr 应该指向 nullptr
    if (auto tempPtr = weakPtr.lock()) {
        std::cout << "MyClass still exists\n";
    } else {
        std::cout << "MyClass has been destroyed\n";
    }

    return 0;
}
```

在这个例子中，`weakPtr` 被赋值为一个 `std::shared_ptr` 指向 `MyClass` 的实例。当 `sharedPtr` 离开作用域并被销毁时，`weakPtr` 仍然指向原始对象，但由于 `std::weak_ptr` 不增加引用计数，所以 `MyClass` 的析构函数会被调用，输出 "MyClass destroyed"。