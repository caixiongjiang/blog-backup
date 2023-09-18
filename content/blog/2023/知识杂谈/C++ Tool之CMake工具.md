---
title: "C++ Tool: CMake"
date: 2023-08-16T18:18:05+08:00
lastmod: 2023-08-20T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%9F%A5%E8%AF%86%E6%9D%82%E8%B0%88/c%2B%2B_tools_title.jpg"
description: "C++ Project的构建工具"
tags:
- c++
categories:
- 语言小工具
series:
- 《知识杂谈》
comment : true
---

## CMake

CMake 是一个项目构建工具，并且是跨平台的。关于项目构建我们所熟知的还有Makefile（通过 make 命令进行项目的构建），大多是IDE软件都集成了make，比如：VS 的 nmake、linux 下的 GNU make、Qt 的 qmake等，如果自己动手写 makefile，会发现，makefile 通常依赖于当前的编译平台，而且编写 makefile 的工作量比较大，解决依赖关系时也容易出错。

CMake 恰好能解决上述问题， 其允许开发者指定整个工程的编译流程，在根据编译平台，自动生成本地化的Makefile和工程文件，最后用户只需make编译即可，所以可以把CMake看成一款自动生成 Makefile的工具

### CMake的使用

#### 注释

CMake使用`#`进行`行注释`，使用`#[[]]`形式进行块注释

#### 只有源文件的情况

假设我们有这样的目录结构：

```shell
$ tree
.
├── add.c
├── div.c
├── head.h
├── main.c
├── mult.c
└── sub.c
```

那么我们为其添加的`CMakeLists.txt`如下所示：

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
add_executable(app add.c div.c main.c mult.c sub.c)
```

* `cmake_minimum_required`的作用是指定使用的cmake的最低版本

可以通过`cmake --version`来查看本机的cmake版本

* `project`：定义工程名称，**并可指定工程的版本、工程描述、web主页地址、支持的语言（默认情况支持所有语言）**，如果不需要这些都是可以忽略的，只需要指定出工程名字即可。

其语法如下：

```cmake
project(<PROJECT-NAME> [<language-name>...])
project(<PROJECT-NAME>
       [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
       [DESCRIPTION <project-description-string>]
       [HOMEPAGE_URL <url-string>]
       [LANGUAGES <language-name>...])
```

* `add_executable`，定义工程会生成的一个可执行程序

```cmake
add_executable(可执行程序名 源文件名称)
# 样式1
add_executable(app add.c div.c main.c mult.c sub.c)
# 样式2
add_executable(app add.c;div.c;main.c;mult.c;sub.c)
```

*这里的可执行程序名和project中的项目名没有任何关系。*

*源文件名可以是一个也可以是多个，如有多个可用` `或`;`间隔。*

* 执行`CMake`命令

```shell
# cmake 命令原型
$ cmake CMakeLists.txt文件所在路径
$ tree
.
├── add.c
├── CMakeLists.txt
├── div.c
├── head.h
├── main.c
├── mult.c
└── sub.c

0 directories, 7 files
$ cmake .
```

我们可以看到在对应的目录下生成了一个`makefile`文件，此时再执行make命令，就可以对项目进行构建得到所需的可执行程序了。

* 执行`make`命令

```shell
$ make
Scanning dependencies of target app
[ 16%] Building C object CMakeFiles/app.dir/add.c.o
[ 33%] Building C object CMakeFiles/app.dir/div.c.o
[ 50%] Building C object CMakeFiles/app.dir/main.c.o
[ 66%] Building C object CMakeFiles/app.dir/mult.c.o
[ 83%] Building C object CMakeFiles/app.dir/sub.c.o
[100%] Linking C executable app
[100%] Built target app

# 查看可执行程序是否已经生成
$ tree -L 1
.
├── add.c
├── app					# 生成的可执行程序
├── CMakeCache.txt
├── CMakeFiles
├── cmake_install.cmake
├── CMakeLists.txt
├── div.c
├── head.h
├── main.c
├── Makefile
├── mult.c
└── sub.c
```

最终可执行程序`app`就被编译出来了（这个名字是在`CMakeLists.txt`中指定的）。

#### 文件管理

在上述过程中会产生一大堆中间文件，以及一些奇怪的文件，那么我们可以将其全部集中到一个文件夹里面

```shell
$ mkdir build
$ cd build
$ cmake ..
```

这样就可以在build目录中执行make命令编译项目，生成的相关文件自然也就被存储到build目录中了。这样通过cmake和make生成的所有文件就全部和项目源文件隔离开了

#### VIP定制

> 定义变量

上面的例子中一共提供了5个源文件，假设这五个源文件需要反复被使用，每次都直接将它们的名字写出来确实是很麻烦，在cmake里定义变量需要使用`set`。

```cmake
# SET 指令的语法是：
# [] 中的参数为可选项, 如不需要可以不写
SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])
```

`VAR`为变量名、`VALUE`为变量值。

那么针对上面的例子可以这样写：

```cmake
# 方式1: 各个源文件之间使用空格间隔
# set(SRC_LIST add.c  div.c   main.c  mult.c  sub.c)
# 方式2: 各个源文件之间使用分号 ; 间隔
set(SRC_LIST add.c;div.c;main.c;mult.c;sub.c)

add_executable(app  ${SRC_LIST})
```

> 指定使用的C++标准

在编写C++程序的时候，可能会用到C++11、C++14、C++17、C++20等新特性，那么就需要在编译的时候在编译命令中制定出要使用哪个标准：

```shell
$ g++ *.cpp -std=c++11 -o app
```

上面的例子中通过参数`-std=c++11`指定出要使用c++11标准编译程序，C++标准对应有一宏叫做`DCMAKE_CXX_STANDARD`。在CMake中想要指定C++标准有两种方式：

* 在`CMakeLists.txt`中通过`set`命令指定:

```cmake
#增加-std=c++11
set(CMAKE_CXX_STANDARD 11)
#增加-std=c++14
set(CMAKE_CXX_STANDARD 14)
#增加-std=c++17
set(CMAKE_CXX_STANDARD 17)
```

* 在执行cmake命令的时候指定这个宏的值

```shell
#增加-std=c++11
cmake CMakeLists.txt文件路径 -DCMAKE_CXX_STANDARD=11
#增加-std=c++14
cmake CMakeLists.txt文件路径 -DCMAKE_CXX_STANDARD=14
#增加-std=c++17
cmake CMakeLists.txt文件路径 -DCMAKE_CXX_STANDARD=17
```

> 指定输出的路径

在CMake中指定可执行程序输出的路径，也对应一个宏，叫做`EXECUTABLE_OUTPUT_PATH`，它的值还是通过set命令进行设置:

```cmake
set(HOME /home/robin/Linux/Sort)
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin)
```

* 第一行：定义一个变量用于存储一个绝对路径
* 第二行：将拼接好的路径值设置给EXECUTABLE_OUTPUT_PATH宏
  * 如果这个路径中的子目录不存在，会自动生成，无需自己手动创建

由于可执行程序是基于 cmake 命令生成的 makefile 文件然后再执行 make 命令得到的，所以**如果此处指定可执行程序生成路径的时候使用的是相对路径 ./xxx/xxx，那么这个路径中的 ./ 对应的就是 makefile 文件所在的那个目录。**

#### 搜索文件

如果一个项目里边的源文件很多，在编写CMakeLists.txt文件的时候不可能将项目目录的各个文件一一罗列出来，这样太麻烦也不现实。所以，在CMake中为我们提供了搜索文件的命令，可以使用`aux_source_directory`命令或者`file`命令。

> 方式一

在 CMake 中使用`aux_source_directory`命令可以查找某个路径下的所有源文件，命令格式为：

```cmake
aux_source_directory(< dir > < variable >)
```

* `dir`：要搜索的目录
* `variable`：将从`dir`目录下搜索到的源文件列表存储到该变量中

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
# 搜索 src 目录下的源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(app  ${SRC_LIST})
```

>  方式二

```cmake
file(GLOB/GLOB_RECURSE 变量名 要搜索的文件路径和文件类型)
```

* `GLOB`：将指定目录下搜索到的满足条件的所有文件名生成一个列表，将其存储到变量中
* `GLOB_RECURSE`：递归搜索指定目录，将搜索到的满足条件的文件名生成一个列表，并将其存储到变量中。

**搜索当前目录的src目录下所有的源文件，并存储到变量中**

```cmake
file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB MAIN_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
```

* `CMAKE_CURRENT_SOURCE_DIR`宏表示当前访问的 CMakeLists.txt 文件所在的路径。

* 关于要搜索的文件路径和类型可加双引号，也可不加:

```cmake
file(GLOB MAIN_HEAD "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
```

#### 包含头文件

在编译项目源文件的时候，很多时候都需要将源文件对应的头文件路径指定出来，这样才能保证在编译过程中编译器能够找到这些头文件，并顺利通过编译。在CMake中设置要包含的目录也很简单，通过一个命令就可以搞定了，他就是`include_directories`:

```cmake
include_directories(headpath)
```

举例说明，有源文件若干，其目录结构如下：

```shell
$ tree
.
├── build
├── CMakeLists.txt
├── include
│   └── head.h
└── src
    ├── add.cpp
    ├── div.cpp
    ├── main.cpp
    ├── mult.cpp
    └── sub.cpp

3 directories, 7 files
```

`CMakeLists.txt`文件内容如下：

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
set(CMAKE_CXX_STANDARD 11)
set(HOME /home/robin/Linux/calc)         # 定义变量名
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin/) # 指定可执行程序输出的位置
include_directories(${PROJECT_SOURCE_DIR}/include) # 包含头文件
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp) # 搜某个文件件下所有的cpp文件
add_executable(app  ${SRC_LIST})         
```

其中，第六行指定就是头文件的路径，**`PROJECT_SOURCE_DIR`宏对应的值就是我们在使用cmake命令时，后面紧跟的目录，一般是工程的根目录**。

#### 制作动态库或者静态库

有些时候我们编写的源代码并不需要将他们编译生成可执行程序，而是生成一些静态库或动态库提供给第三方使用，下面来讲解在cmake中生成这两类库文件的方法。

> 制作静态库

在cmake中，如果要制作静态库，需要使用的命令如下：

```cmake
add_library(库名称 STATIC 源文件1 [源文件2] ...) 
```

在Linux中，静态库名字分为三部分：`lib`+`库名字`+`.a`，此处只需要指定出库的名字就可以了，另外两部分在生成该文件的时候会自动填充。

在Windows中虽然库名和Linux格式不同，但也只需指定出名字即可。

下面有一个目录，需要将`src`目录中的源文件编译成静态库，然后再使用：

```shell
.
├── build
├── CMakeLists.txt
├── include           # 头文件目录
│   └── head.h
├── main.cpp          # 用于测试的源文件
└── src               # 源文件目录
    ├── add.cpp
    ├── div.cpp
    ├── mult.cpp
    └── sub.cpp
```

根据上面的目录结构，可以这样编写`CMakeLists.txt`文件:

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(calc STATIC ${SRC_LIST})
```

这样最终就会生成对应的静态库文件`libcalc.a`。

> 制作动态库

在cmake中，如果要制作动态库，需要使用的命令如下：

```cmake
add_library(库名称 SHARED 源文件1 [源文件2] ...) 
```

在Linux中，动态库名字分为三部分：`lib`+`库名字`+`.so`，此处只需要指定出库的名字就可以了，另外两部分在生成该文件的时候会自动填充。

在Windows中虽然库名和Linux格式不同，但也只需指定出名字即可。

根据上面的目录结构，可以这样编写CMakeLists.txt文件:

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(calc SHARED ${SRC_LIST})
```

这样最终就会生成对应的动态库文件`libcalc.so`。

> 指定输出的路径

* 方法一：

对于生成的库文件来说和可执行程序一样都可以指定输出路径。由于**在Linux下生成的动态库默认是有执行权限的**，所以可以按照生成可执行程序的方式去指定它生成的目录：

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# 设置动态库生成路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(calc SHARED ${SRC_LIST})
```

对于这种方式来说，其实就是通过set命令给`EXECUTABLE_OUTPUT_PATH`宏设置了一个路径，这个路径就是可执行文件生成的路径。

* 方法二：

由于在Linux下生成的静态库默认不具有可执行权限，所以在指定静态库生成的路径的时候就不能使用`EXECUTABLE_OUTPUT_PATH`宏了，而应该使用`LIBRARY_OUTPUT_PATH`，这个宏对应静态库文件和动态库文件都适用。

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# 设置动态库/静态库生成路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 生成动态库
#add_library(calc SHARED ${SRC_LIST})
# 生成静态库
add_library(calc STATIC ${SRC_LIST})
```

> 动态库和静态库的区别

动态库和静态库都是编程中常用的库文件，它们的主要区别在于链接时刻和占用内存的方式。

静态库在编译时会被完整地复制到可执行文件中，因此它们会增加可执行文件的大小。在链接时刻，编译器会将静态库的代码与可执行文件的代码合并，生成一个完整的可执行文件。由于静态库已经被完整地复制到可执行文件中，因此程序在运行时不需要再加载静态库，这样可以提高程序的运行速度。但是，如果多个可执行文件都使用同一个静态库，那么静态库的代码会被复制多次，浪费空间。

动态库则是在程序运行时才被加载到内存中，因此它们不会增加可执行文件的大小。在链接时刻，编译器只会将动态库的引用信息添加到可执行文件中，而不会将动态库的代码复制到可执行文件中。程序在运行时会动态地加载动态库，并将其映射到内存中。由于多个可执行文件可以共享同一个动态库，因此动态库可以节省内存空间。但是，由于动态库需要在程序运行时才能加载，因此程序的启动速度可能会受到影响。

总的来说，**静态库适用于需要高效运行的小型程序，而动态库适用于需要共享代码和节省内存空间的大型程序。**

#### 包含库文件

在编写程序的过程中，可能会用到一些系统提供的动态库或者自己制作出的动态库或者静态库文件，cmake中也为我们提供了相关的加载动态库的命令。

> 链接静态库

```shell
src
├── add.cpp
├── div.cpp
├── main.cpp
├── mult.cpp
└── sub.cpp
```

现在我们把上面`src`目录中的`add.cpp、div.cpp、mult.cpp、sub.cpp`编译成一个静态库文件`libcalc.a`。

测试目录结构如下：

```cmake
$ tree 
.
├── build
├── CMakeLists.txt
├── include
│   └── head.h
├── lib
│   └── libcalc.a     # 制作出的静态库的名字
└── src
    └── main.cpp

4 directories, 4 files
```

在cmake中，链接静态库的命令如下：

```cmake
link_libraries(<static lib> [<static lib>...])
```

* **参数1**：指定出要链接的静态库的名字
  * 可以是全名`libxxx.a`
  * 也可以是掐头`lib`去尾`.a`之后的名字 `xxx`
* **参数2-N**：要链接的其它静态库的名字

如果该静态库不是系统提供的（自己制作或者使用第三方提供的静态库）可能出现静态库找不到的情况，此时可以将静态库的路径也指定出来：

```cmake
link_directories(<lib path>)
```

 这样，修改之后的`CMakeLists.txt`文件内容如下：

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
# 搜索指定目录下源文件
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# 包含头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)
# 包含静态库路径
link_directories(${PROJECT_SOURCE_DIR}/lib)
# 链接静态库
link_libraries(calc)
add_executable(app ${SRC_LIST})
```

> 链接动态库

在`cmake`中链接动态库的命令如下:

```cmake
target_link_libraries(
    <target> 
    <PRIVATE|PUBLIC|INTERFACE> <item>... 
    [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
```

* **target**：指定要加载动态库的文件的名字

  * 该文件可能是一个源文件
  * 该文件可能是一个动态库文件
  * 该文件可能是一个可执行文件

* **PRIVATE|PUBLIC|INTERFACE**：动态库的访问权限，默认为`PUBLIC`

  * 如果各个动态库之间没有依赖关系，无需做任何设置，`三者没有没有区别，一般无需指定，使用默认的 PUBLIC 即可`。

  * `动态库的链接具有传递性`，如果动态库 A 链接了动态库B、C，动态库D链接了动态库A，此时动态库D相当于也链接了动态库B、C，并可以使用动态库B、C中定义的方法。

  ```cmake
  target_link_libraries(A B C)
  target_link_libraries(D A)
  ```

  * PUBLIC：在public后面的库会被Link到前面的target中，并且里面的符号也会被导出，提供给第三方使用。
  * PRIVATE：在private后面的库仅被link到前面的target中，并且终结掉，第三方不能感知你调了啥库
  * INTERFACE：在interface后面引入的库不会被链接到前面的target中，只会导出符号。

* 动态库的链接和静态库是完全不同的：

  * 静态库会在生成可执行程序的链接阶段被打包到可执行程序中，所以可执行程序启动，静态库就被加载到内存中了。
  * 动态库在生成可执行程序的链接阶段不会被打包到可执行程序中，当可执行程序被启动并且调用了动态库中的函数的时候，动态库才会被加载到内存。


> 链接系统动态库

因此，在`cmake`中指定要链接的动态库的时候，`应该将命令写到生成了可执行文件之后`：

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# 添加并指定最终生成的可执行程序名
add_executable(app ${SRC_LIST})
# 指定可执行程序要链接的动态库名字
target_link_libraries(app pthread)
```

* 在target_link_libraries(app pthread)中：
  * app: 对应的是最终生成的可执行程序的名字
  * pthread：这是可执行程序要加载的`动态库`，这个库是`系统提供的线程库`，全名为`libpthread.so`，在指定的时候一般会掐头（lib）去尾（.so）。

> 链接第三方动态库

假设自己生成了一个动态库：

```cmake
$ tree 
.
├── build
├── CMakeLists.txt
├── include
│   └── head.h            # 动态库对应的头文件
├── lib
│   └── libcalc.so        # 自己制作的动态库文件
└── main.cpp              # 测试用的源文件

3 directories, 4 files
```

假设在测试文件`main.cpp`中既使用了自己制作的动态库`libcalc.so`，又使用了系统提供的线程库，此时`CMakeLists.txt`文件可以这样写：

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(app ${SRC_LIST})
target_link_libraries(app pthread calc)
```

在第六行中，`pthread、calc`都是可执行程序`app`要链接的动态库的名字。当可执行程序`app`生成之后并执行该文件，会提示有如下错误信息：

```shell
$ ./app 
./app: error while loading shared libraries: libcalc.so: cannot open shared object file: No such file or directory
```

这是因为可执行程序启动之后，去加载`calc`这个动态库，但是不知道这个动态库放在了什么位置

**在 CMake 中可以在生成可执行程序之前，可以通过命令指定出要链接的动态库的位置，指定静态库位置使用的也是这个命令**：

```cmake
link_directories(path)
```

所以修改之后的`CMakeLists.txt`文件应该是这样的：

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# 指定源文件或者动态库对应的头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)
# 指定要链接的动态库的路径
link_directories(${PROJECT_SOURCE_DIR}/lib)
# 添加并生成一个可执行程序
add_executable(app ${SRC_LIST})
# 指定要链接的动态库
target_link_libraries(app pthread calc)
```

**使用 target_link_libraries 命令就可以链接动态库，也可以链接静态库文件!**

#### 日志

在CMake中可以用用户显示一条消息，该命令的名字为`message`：

```cmake
message([STATUS|WARNING|AUTHOR_WARNING|FATAL_ERROR|SEND_ERROR] "message to display" ...)
```

* (无) ：重要消息
* STATUS ：非重要消息
* WARNING：CMake 警告, 会继续执行
* AUTHOR_WARNING：CMake 警告 (dev), 会继续执行
* SEND_ERROR：CMake 错误, 继续执行，但是会跳过生成的步骤
* FATAL_ERROR：CMake 错误, 终止所有处理过程

CMake的命令行工具会在stdout上显示`STATUS`消息，在stderr上显示其他所有消息。CMake的GUI会在它的log区域显示所有消息。

CMake警告和错误消息的文本显示使用的是一种简单的标记语言。文本没有缩进，超过长度的行会回卷，段落之间以新行做为分隔符。

```cmake
# 输出一般日志信息
message(STATUS "source path: ${PROJECT_SOURCE_DIR}")
# 输出警告信息
message(WARNING "source path: ${PROJECT_SOURCE_DIR}")
# 输出错误信息
message(FATAL_ERROR "source path: ${PROJECT_SOURCE_DIR}")
```

#### 变量操作

> 追加

有时候项目中的源文件并不一定都在同一个目录中，但是这些源文件最终却需要一起进行编译来生成最终的可执行文件或者库文件。如果我们通过`file`命令对各个目录下的源文件进行搜索，最后还需要做一个字符串拼接的操作，关于字符串拼接可以使用`set`命令也可以使用`list`命令。

* 使用set拼接：

```cmake
set(变量名1 ${变量名1} ${变量名2} ...)
```

关于上面的命令其实就是将从第二个参数开始往后所有的字符串进行拼接，最后将结果存储到第一个参数中。

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
set(TEMP "hello,world")
file(GLOB SRC_1 ${PROJECT_SOURCE_DIR}/src1/*.cpp)
file(GLOB SRC_2 ${PROJECT_SOURCE_DIR}/src2/*.cpp)
# 追加(拼接)
set(SRC_1 ${SRC_1} ${SRC_2} ${TEMP})
message(STATUS "message: ${SRC_1}")
```

* 使用list拼接：

```cmake
list(APPEND <list> [<element> ...])
```

`list`命令的功能比`set`要强大，字符串拼接只是它的其中一个功能，所以需要在它第一个参数的位置指定出我们要做的操作，`APPEND`表示进行数据追加，后边的参数和`set`就一样了。

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
set(TEMP "hello,world")
file(GLOB SRC_1 ${PROJECT_SOURCE_DIR}/src1/*.cpp)
file(GLOB SRC_2 ${PROJECT_SOURCE_DIR}/src2/*.cpp)
# 追加(拼接)
list(APPEND SRC_1 ${SRC_1} ${SRC_2} ${TEMP})
message(STATUS "message: ${SRC_1}")
```

