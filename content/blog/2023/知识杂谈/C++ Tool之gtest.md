---
title: "C++ Tool: gtest"
date: 2023-08-16T18:18:05+08:00
lastmod: 2023-08-17T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%9F%A5%E8%AF%86%E6%9D%82%E8%B0%88/c%2B%2B_tools_title.jpg"
description: "学习使用c++用于单元测试的小工具"
tags:
- c++
categories:
- 语言小工具
series:
- 《知识杂谈》
comment : true
---

### Google开源框架：gtest

`gtest`，它主要用于写单元测试，检查真自己的程序是否符合预期行为。这不是QA（测试工程师）才学的，也是每个优秀后端开发codoer的必备技能。

#### 构建依赖环境

先介绍下怎么基于CMakeLists.txt构建依赖环境。

由于Google没有为googletest/samples中的samples写CMakeLists.txt，因此，gtest从github克隆下来后，也无法直接运行这些samples。

在`loOK后端`公众号，回复「gtest」即可获取gtest压缩包。

* 查看当前目录结构：
```shell
$ tree -L 2
demo
├── CMakeLists.txt  
├── build        # 空的文件夹
├── include
│   ├── CMakeLists.txt
│   ├── gflags
│   └── googletest
├── main.cc
└── my_gtest_demo_1.cc
```

* 然后，在demo/build路径下，执行命令：
```shell
cmake .. && make -j 4
```

这些samples生成的可执行文件都在`demo/build/bin`路径下。

#### assertion

在`gtest`中，是通过断言（`assertion`）来判断代码实现的功能是否符合预期。断言的结果分为`success`、`non-fatal failture`和`fatal failture`。

根据断言失败的种类，`gtest`提供了两种断言函数：

* `success`：即断言成功，程序的行为符合预期，程序继续向下允许。

* `non-fatal failure`：即断言失败，但是程序没有直接crash，而是继续向下运行。

`gtest`提供了宏函数`EXPECT_XXX(expected, actual)`：如果`condition(expected, actual)`返回false，则`EXPECT_XXX`产生的就是`non-fatal failure`错误，并显示相关错误。

* `fatal failure`：断言失败，程序直接crash，后续的测试案例不会被运行。

`gtest`提供了宏函数`ASSERT_XXX(expected, actual)`。

在写单元测试时，更加倾向于使用`EXPECT_XXX`，因为`ASSERT_XXX`是直接crash退出的，可能会导致一些内存、文件资源没有释放，因此可能会引入一些bug。

具体的`EXPECT_XXX`、`ASSERT_XXX`函数及其判断条件，如下两个表:

**一元比较**：
| ASSERT |                        EXPECT                       | Verifies |
| :--: | :------------------------------------------------: |:------:|
|  ASSERT_TRUE(condition) |    EXPECT_TRUE(condition); | condition is true |
|  ASSERT_FALSE(condition);  |  EXPECT_FALSE(condition);  | condition is false |

**二元比较**：
| ASSERT |                        EXPECT                       | Condition |
| :--: | :------------------------------------------------: |:------:|
|  ASSERT_EQ(val1, val2); |    EXPECT_EQ(val1, val2); | val1 == val2 |
|  ASSERT_NE(val1, val2);  |  EXPECT_NE(val1, val2);  | val1 != val2 |
|  ASSERT_LT(val1, val2);  |  EXPECT_LT(val1, val2);  | val1 < val2 |
|  ASSERT_LE(val1, val2);  |  EXPECT_LE(val1, val2);  | val1 <= val2 |
|  ASSERT_GT(val1, val2);  |  EXPECT_GT(val1, val2);  | val1 > val2 |
|  ASSERT_GE(val1, val2);  |  EXPECT_GE(val1, val2);  | val1 >= val2 |

#### Quick Start

下面以`EXPECT_XXX`为例子，快速开始使用gtest吧。

对于`EXPECT_XXX`，无论条件是否满足，都会继续向下运行，但是如果条件不满足，在报错的地方会显示：

没有通过的那个`EXPECT_XXX`函数位置；
`EXPECT_XXX`第一个参数的值，即期待值
`EXPECT_XXX`第二个参数的值，即实际值
demo:
```c++
// in gtest_demo_1.cc
#include <gtest/gtest.h>

int add(int lhs, int rhs) { return lhs + rhs; }

int main(int argc, char const *argv[]) {

    EXPECT_EQ(add(1,1), 2); // PASS
    EXPECT_EQ(add(1,1), 1) << "FAILED: EXPECT: 2, but given 1";; // FAILDED
    
    return 0;
}
```
编译执行后输出如下：
```shell
$ ./gtest_demo_1
/Users/self_study/Cpp/OpenSource/demo/gtest_demo_1.cc:9: Failure
Expected equality of these values:
  add(1,1)
    Which is: 2                # 期待的值
  1                            # 给定的值
FAILED: EXPECT: 2, but given 1 # 自己添加的提示信息 
```

`gtest`允许添加自定义的描述信息，当这个语句测试未通过时就会显示，比如上面的`"FAILED: EXPECT: 2, but given 1"`。

这个`<<`和`std::ostream`接受的类型一致，即可以接受`std::ostream`可以接受的类型。

#### TEST

以`googletest/samples`中的`sample1_unittest.cc`中的demo为例，介绍如何更好地组织测试案例。

一个简单计算阶乘函数`Factorial`实现如下：
```c++
int Factorial(int n) {
  int result = 1;
  for (int i = 1; i <= n; i++) {
    result *= i;
  }

  return result;
}
```

怎么使用`gtest`来测试这个函数的行为？

从上面给出的表来看，我们需要使用`EXPECT_EQ`宏来判断：
```c++
EXPECT_EQ(1, Factorial(-5)); // 测试计算负数的阶乘
EXPECT_EQ(1, Factorial(0));   // 测试计算0的阶乘
EXPECT_EQ(6, Factorial(3));   // 测试计算正数的阶乘 
```

但是当测试案例规模变大，不好组织。

因此，为了更好的组织`test cases`，比如针对`Factorial`函数，输入是负数的cases为一组，输入是0的case为一组，正数cases为一组。gtest提供了一个宏`TEST(TestSuiteName, TestName)`，用于组织不同场景的cases，这个功能在`gtest`中称为`test suite`。

用法如下：
```c++
// 下面三个 TEST 都是属于同一个 test suite，即 FactorialTest
// 正数为一组
TEST(FactorialTest, Negative) {
  EXPECT_EQ(1, Factorial(-5));
  EXPECT_EQ(1, Factorial(-1));
  EXPECT_GT(Factorial(-10), 0);
}
// 0
TEST(FactorialTest, Zero) {
  EXPECT_EQ(1, Factorial(0));
}
// 负数为一组
TEST(FactorialTest, Positive) {
  EXPECT_EQ(1, Factorial(1));
  EXPECT_EQ(2, Factorial(2));
  EXPECT_EQ(6, Factorial(3));
  EXPECT_EQ(40320, Factorial(8));
}
```

那么需要如何运行这些`Test`呢？
在`sample1_unittest.cc`的main函数中，添加`RUN_ALL_TESTS`函数即可。
```c++
int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
```
在`build/bin`路径下，执行对应的可执行文件，输出如下：
```shell
$./sample1_unittest 
Running main() from /Users/self_study/Cpp/OpenSource/demo/include/googletest/googletest/samples/sample1_unittest.cc
[==========] Running 6 tests from 2 test suites. # 在 sample1_unittest.cc 中有两个 test suites
[----------] Global test environment set-up.    

# 第一个 test suite，即上面的 FactorialTest
[----------] 3 tests from FactorialTest     # 3 组
[ RUN      ] FactorialTest.Negative         # Negative 组输出
[       OK ] FactorialTest.Negative (0 ms)  # OK 表示 Negative 组全部测试通过
[ RUN      ] FactorialTest.Zero             # Zero组输出 
[       OK ] FactorialTest.Zero (0 ms)    
[ RUN      ] FactorialTest.Positive         # Positive组输出
[       OK ] FactorialTest.Positive (0 ms)   
[----------] 3 tests from FactorialTest (0 ms total)

# sample1_unitest 另一个测试案例的输出 ...

[----------] Global test environment tear-down  
[==========] 6 tests from 2 test suites ran. (0 ms total) 
[  PASSED  ] 6 tests.              # 全部测试结果：PASS表示全部通过 
```

在`sample2_unittest.cc`中，测试一个自定义类MyString的复制构造函数是否表现正常：
```c++
const char kHelloString[] = "Hello, world!";

// 在 TEST内部，定义变量
TEST(MyString, CopyConstructor) {
  const MyString s1(kHelloString);
  const MyString s2 = s1;
  EXPECT_EQ(0, strcmp(s2.c_string(), kHelloString));
}
```

#### TEST_F
`gtest`中更为高级的功能：`test fixture`，对应的宏函数是`TEST_F(TestFixtureName, TestName)`。

`fixture`，其语义是固定的设施，而`test fixture`在`gtest`中的作用就是为每个`TEST`都执行一些同样的操作。

比如，要测试一个队列`Queue`的各个接口功能是否正常，因此就需要向队列中添加元素。如果使用一个`TEST`函数测试`Queue`的一个接口，那么每次执行`TEST`时，都需要在`TEST`宏函数中定义一个`Queue`对象，并向该对象中添加元素，就很冗余、繁琐。

怎么避免这部分冗余的过程？

`TEST_F`就是完成这样的事情，它的第一个参数`TestFixtureName`是个类，需要继承`testing::Test`，同时根据需要实现以下两个虚函数：
* `virtual void SetUp()`：在`TEST_F`中测试案例之前运行；
* `virtual void TearDown()`：在`TEST_F`之后运行。

可以类比对象的构造函数和析构函数。这样，同一个`TestFixtureName`下的每个`TEST_F`都会先执行`SetUp`，最后执行`TearDown`。

此外，`testing::Test`还提供了两个`static`函数：
* `static void SetUpTestSuite()`：在第一个`TEST`之前运行
* `static void TearDownTestSuite()`：在最后一个`TEST`之后运行

以`sample3-inl`中实现的`class Queue`为例：
```c++
class QueueTestSmpl3 : public testing::Test { // 继承了 testing::Test
protected:  
  
  static void SetUpTestSuite() {
    std::cout<<"run before first case..."<<std::endl;
  } 

  static void TearDownTestSuite() {
    std::cout<<"run after last case..."<<std::endl;
  }
  
  virtual void SetUp() override {
    std::cout<<"enter into SetUp()" <<std::endl;
    q1_.Enqueue(1);
    q2_.Enqueue(2);
    q2_.Enqueue(3);
  }

  virtual void TearDown() override {
    std::cout<<"exit from TearDown" <<std::endl;
  }
  
  static int Double(int n) {
    return 2*n;
  }
  
  void MapTester(const Queue<int> * q) {
    const Queue<int> * const new_q = q->Map(Double);

    ASSERT_EQ(q->Size(), new_q->Size());

    for (const QueueNode<int>*n1 = q->Head(), *n2 = new_q->Head();
         n1 != nullptr; n1 = n1->next(), n2 = n2->next()) {
      EXPECT_EQ(2 * n1->element(), n2->element());
    }

    delete new_q;
  }

  Queue<int> q0_;
  Queue<int> q1_;
  Queue<int> q2_;
};
```

下面是`sample3_unittest.cc`中的`TEST_F`：
```c++
// in sample3_unittest.cc

// Tests the default c'tor.
TEST_F(QueueTestSmpl3, DefaultConstructor) {
  // !!! 在 TEST_F 中可以使用 QueueTestSmpl3 的成员变量、成员函数 
  EXPECT_EQ(0u, q0_.Size());
}

// Tests Dequeue().
TEST_F(QueueTestSmpl3, Dequeue) {
  int * n = q0_.Dequeue();
  EXPECT_TRUE(n == nullptr);

  n = q1_.Dequeue();
  ASSERT_TRUE(n != nullptr);
  EXPECT_EQ(1, *n);
  EXPECT_EQ(0u, q1_.Size());
  delete n;

  n = q2_.Dequeue();
  ASSERT_TRUE(n != nullptr);
  EXPECT_EQ(2, *n);
  EXPECT_EQ(1u, q2_.Size());
  delete n;
}

// Tests the Queue::Map() function.
TEST_F(QueueTestSmpl3, Map) {
  MapTester(&q0_);
  MapTester(&q1_);
  MapTester(&q2_);
}
```

以`TEST_F(QueueTestSmpl3, DefaultConstructor)`为例，再具体讲解下`TEST_F`的运行流程：

1.`gtest`构造一个`QueueTestSmpl3`对象`t1`；
2.`t1.setUp`初始化`t1`
3.第一个`TEST_F`即`DefaultConstructor`开始运行并结束
4.`t1.TearDwon`运行，用于清理工作
5.`t1`被析构

因此，`sample3_unittest.cc`输出如下：
```shell
% ./sample3_unittest
Running main() from /Users/self_study/Cpp/OpenSource/demo/include/googletest/googletest/samples/sample3_unittest.cc
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from QueueTestSmpl3
run before first case...    # 所有的test case 之前运行
[ RUN      ] QueueTestSmpl3.DefaultConstructor
enter into SetUp()          # 每次都会运行
exit from TearDown
[       OK ] QueueTestSmpl3.DefaultConstructor (0 ms)
[ RUN      ] QueueTestSmpl3.Dequeue
enter into SetUp()          # 每次都会运行
exit from TearDown
[       OK ] QueueTestSmpl3.Dequeue (0 ms)
[ RUN      ] QueueTestSmpl3.Map
enter into SetUp()          # 每次都会运行
exit from TearDown
[       OK ] QueueTestSmpl3.Map (0 ms)
run after last case...      # 所有test case结束之后运行
[----------] 3 tests from QueueTestSmpl3 (0 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 3 tests. 
```

`TEST_F`相比较`TEST`可以更加简洁地实现功能测试。