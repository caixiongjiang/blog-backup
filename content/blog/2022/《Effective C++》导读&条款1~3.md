---
title: "《Effective C++》导读&条款1~3"
date: 2022-03-01T18:07:05+08:00
lastmod: 2022-03-01T00:43:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/efc%2B%2B.jpg"
description: "养成更好的C++语言使用习惯"
tags:
- c++
- 《Effective C++》学习笔记
categories:
- 《Effective C++》学习笔记
series:
- 《Effective C++》学习笔记
comment : true
---

## 《Effective c++》学习笔记

### 导读部分

正式开始55条条款之前，这本书的目的是为了让c++程序员如何写出更加安全高效的程序准备的，属于进阶书籍，如果观看本书还有很多不明白的地方，还需要去学基础，推荐《c++ Primer》,看本书前需要知道的有：c++面向过程，c++面向对象，c++泛型编程，c++ STL标准库， c++多线程编程。

#### 术语

* 1.std命名空间是几乎所有c++标准程序库元素的栖息地，所以在导入c++头文件的时候，一定要在std命名空间内写程序

* 2.`size_t`是一个typedef，是c++计算个数（字符串内的个数或者STL容器内的元素个数）时使用的，它属于无符号整数

* 3.一个类的构造函数使用`explicit`关键字可以阻止它们被用来**隐式类型转换**，但他们仍然可被用于**显示类型转化**。**被声明为`explicit`关键字的构造函数通常比没有它更受欢迎，因为它们禁止编译器执行非预期的类型**

  demo：

  * 头文件（.h）:

    ```c++
    class A{
    public:
      A();   //default构造函数
    };
    
    class B{
    public:
      explicit B(int x = 0, bool b = true);//default构造函数 两个参数都已经设置默认值
    };
    
    class C{
    public:
      explicit C(int x);//不是default构造函数
    };
    ```

  * 执行文件（.cpp）:

    ```c++
    void doSomething(B bObject);  //函数声明，接收一个类型为B的对象
    
    B bObj1;
    doSomething(bObj1);
    B bObj2(28);//正确，根据int28建立一个B
    
    doSomething(28);//错误！doSomething应该接收一个B
    								//int至B之间没有隐式类型转化
    
    doSomething(B(28));//正确，使用B构造函数将int显示转换
    ```

* 4.拷贝构造和拷贝赋值的区别：

  * 拷贝构造函数是**以同型对象初始化自我对象**；而拷贝赋值是**将另一个同型对象其值拷贝到自我对象**

  * 拷贝构造一定会有一个新对象被定义，一定会有个拷贝构造函数倍调用；拷贝赋值没有新对象被定义，不回有构造函数被调用

    demo：

    ```c++
    class Widget{
    public:
      Widget();//默认构造
      Widget(const Widget& rhs);//拷贝构造
      Widget& operator=(const Widget& rhs);//拷贝赋值
    	···
    };
    
    Widget w1;
    Widget w2(w1);//拷贝构造
    w1 = w2;//默认构造
    ```

  * 注意点：**值传递**意味着调用拷贝构造函数；**最好不要**使用值传递的方式传递用户自定义类型，**常量的引用传递**往往是比较好的选择。

  

#### 命名习惯

这部分根据个人喜好来决定，不一定按照这个规则来执行！

* 参数命名：左操作数为lhs，右操作数为rhs
* 指针命名：如果为“指向一个T型对象”命名为pt，意思是“pointer to T”
* 成员函数：以mf命名

#### 关于线程

*如果不熟悉可以忽略！*

#### TR1和Boost
* TR1是一份规范，描述加入c++标准库的诸多新机能。**所有TR1组件都在命名空间tr1内，后者又嵌套于命名空间std中**
* Boost提供可移植，源码开放的c++标准库。大多数TR1机能是以Boost的工作为基础。
  

### 条款01:视c++为一个语言联邦
c++是一个同时支持过程形式，面向对象形式，函数形式，泛型形式，元编程形式的语言。

它主要可以分为四部分：

* C（面向过程）
* Object-Oriented C++(面向对象)：类（构造，析构），封装，继承，多态，virtual函数。
* Template C++（泛型）：也就是我们说的模版技术
* STL（标准库）：包括容器和迭代器，算法，函数对象等

#### Remember

**C++高效编程守则视状况而变化，取决于你使用C++的哪一部分** 

### 条款02：尽量以const,enum,inline替换 #define

这个条款也可以写成**用编辑器替换于处理器**。

由于`#define`在编译器处理之前就会被预处理器处理。而且宏定义的内容会被多个文件共同使用，可能会产生某些问题。

demo：
```c++
#define ASPECT_RATIO 1.653 //大写名称常用于宏
```
解决的方法是用一个常量替换上述的宏
```c++
const double AspectRatio = 1.653;
```

当我们用常量替换`#define`时，有两种情况值得一说：
* 由于敞亮定义式通常被放在头文件内（以便被不同的源码含入），因此**有必要将指针本身声明为const**。

demo(在文件内定义一个常量的char*字符串)：
```c++
//第一个const代表char*为一个字符串常量，第二个const代表指向char*的指针为一个常量
const char* const authorName = "Scott Meyers"; 
```
* class的专属常量：为了将常量的作用域限制于class内，你必须让它成为class的一个成员；**而为确保该常量只有一份实体，你必须让它成为一个static成员。**

demo：
头文件(.h)：
```c++
class GamePlayer{
private:
  static const int NumTurns = 5; //常量声明式
  int scores[NumTurns]; //使用该常量
  ··· 
};
```
*注意：上述声明并不是定义式子，c++的编译器必须要看到一个定义式，所以你要在指向文件里提供另外的定义式*

执行文件(.cpp):
```c++
const int GamePlayer::NumTurns //NumTurns的定义
```

**综上来说，我们无法利用`#define`创建一个class专属常量，因为它不重视作用域**

当你在class编译期间需要一个class常量值，但编译器又不允许static整数型常量在class内完成初值设定时，这时可以改用“the enum hack”补偿做法。

demo：
```c++
class GamePlayer{
private:
  enum {NumTurns = 5}; //“the enum hack”-令NumTurns
                       //成为5的一个记号名称
  int scores[NumTurns];
};
```
*注意：enum back的行为与`#define`较像，不像const。例如，取一个const的地址是合法的，但取一个enum的地址不合法，取一个`#define`的地址通常也不合法。*

**当你的宏里带有类似函数的东西，必须给实参加上小括号，否则在调用宏时会遭遇麻烦，且这里的函数不遵守作用域和访问规则。你可以使用一个“class类内的私有内联函数来代替”。**

#### Remember

* **对于单纯常量，最好以`const对象`或者`enums`替换`#define`**
* **对于形似函数的宏，最好改用内联（inline）函数替换`#define`**

### 条款03:尽可能使用const

const可修饰的对象：
* class外部修饰`global`或`namespace`作用域中的常量
* 文件，函数或区块作用域中被声明为static的对象
* class内部的`static`和`non-static`成员变量
* 指针自身或指针所指物

#### const修饰指针

注意：
* *如果关键字const出现在星号左边，表示被指物是常量；如果出现在星号左边，表示指针自身是常量；如果出现在星号两边，表示被指物和指针两者都是常量*

demo：
```c++
const char* p = greeting; //non-const pointer, non-const data
char* const p = greeting; //non-const pointer, const data
const char* const p = greeting;//const pointer, const data
```

* *const既可以写在类型之前，也可以写在类型之后，星号之前*

demo：
```c++
//f1,f2获得一个指针，指向一个常量Widget对象
void f1(const Widget* pw);
void f2(Widget const * pw);
```
#### const修饰迭代器
* 迭代器为一个常量（类似于指针是一个常量） 
* 迭代器所指的对象是一个常量（类似于指针所指的对象是一个常量）

demo:
```c++
//迭代器为常量
const vector<int>::iterator it = vec.begin();
//迭代器所指对象为常量，也就是vector容器的数据无法修改
vector<int>::const_iterator cit = vec.begin();
```

#### const和函数的关联

* const可以和`函数返回值`，`各参数`，`函数自身`（如果是成员函数）产生关联

demo（有理数的operator*的声明式）:
```c++
class Rational {...};
//第一个const代表函数返回值为const，第二，三个const代表参数为const
const Rational operator* (const Rational& lhs, const Rational& rhs);
```
* const成员函数

  使用const成员函数的好处：
  * 1.使得class的接口易于理解：知道哪个函数可以修改对象内容，哪个函数不可以
  * 2.使“操作对象”成为可能：const成员函数可以实现`以常量的引用方式`传递对象

#### const和重载
两个成员函数如果只有常量性不同，可以被重载。

demo（声明）:
```c++
class Textblock{
public:
  ···
  //const成员函数
  const char& operator[](size_t position) const{
    return text[position];
  }
  //与上述函数只有常量性不同
  char& operator[](size_t position){
    return text[position];
  } 
private:
  string text;
};
```

demo(使用):
```c++
//使用non-const成员函数
TextBlock tb("Hello");
cout << tb[0];
//使用const成员函数
const TextBloclk ctb("World");
cout << ctb[0];
```

注意：*无法通过一个const成员函数来进行赋值（写）操作*

demo:
```c++
//普通operator[]重载函数进行读写
cout << tb[0];//正确
tb[0] = 'x';//正确
//constoperator[]重载函数进行读写
cout << ctb[0];//正确
ctb[0] = 'x';//错误
```

**如果non-const operator[]的返回值是一个char，则`tb[0] =  'x';`也无法通过编译。因为函数的返回值类型为一个内置类型，那么改动函数返回值本身就不合法**

#### bitwise constness和logical constness

编译器按照bitwise constness的规则进行编译。

bitwise constness的规则：

成员函数只有在不更改对象之任何成员变量时才可以说时是const，因此const成员函数不可以更改对象内任何non—static成员变量。

#### 如何利用const成员函数修改成员变量的值

利用`mutable`关键字：

mutable释放掉non—static成员变量的bitwise constness（使const成员函数可以更改类内的成员变量）

#### const和non—const成员函数避免重复

demo（修改前）：
```c++
class TextBlock{
public:
  ···
  const char& operator[](size_t position) const{
    ···       //边界检查
    ···       //志记数据访问  
    ···       //检查数据完整性
    return text[position];
  }
  
  char& operator[](size_t position){
    ···       //边界检查
    ···       //志记数据访问 
    ···       //检查数据完整性
    return text[position];
  }
};
```

上述代码有大量重复，可以通过non-const operator[]调用其const兄弟是避免代码重复的一个安全做法。

demo（修改后）：
```c++
class TextBlock{
public:
  ···
  const char& operator[](size_t position) const{
    ···       //边界检查
    ···       //志记数据访问  
    ···       //检查数据完整性
    return text[position];
  }
  
  char& operator[](size_t position){
    return
      const_cast<char&>(static_cast<const TextBlock&>(*this)
      [position]);
  }
};
```

* const_cast:改变表达式中的常量性和易变性（移除常量性）
* static_cast:只要不包含底层的const,都可以使用static_cast。常用于non—const转const

注意：*不能反向操作：让const版本调用non—const版本来避免重复*

#### Remember
* 将某些东西声明为const可帮助编译器侦测出错误用法。const可以被施加于任何作用域内的对象，函数参数，函数返回类型，成员函数本体
* 编译器强制实施bitwise constness，但你编写程序时应该使用“概念上的常量性”。
* 当const和non—const成员函数有着等价的实现时，令non—const版本调用const版本可避免代码重复。