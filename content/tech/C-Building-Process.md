---
title: "C/C++的编译过程"
date: 2020-04-30T22:15:29+01:00
categories: [Tech,Programming]
tags: ["C/C++"]
slug: "cpp-building-process"
---

最近回顾C++，发现很多对于以前直接套用的语句讲不出个所以然，比如说头文件的预编译过程的语句我就不明白是什么。所以希望用这篇文章了解一下：如何在Linux环境下使用`g++`理解`C++`程序的编译过程。<!--more-->

## 程序的编译和链接

**编译**(Compile)和**链接**(Link)是把源代码转换成可执行文件的过程。无论是C还是C++，源文件都要被编译为中间代码文件（目标文件Object file），在Windows系统中为`.obj`文件，在UNIX下则为`.o`文件。编译完成后，将目标文件和成为可执行文件的过程叫做链接。为了方便，以下名词均按UNIX系统来说。

### 编译

编译只需要满足语法正确，函数与变量声明的正确。为了找到声明(declaration)的位置，与定义(definition)联系起来，编译器需要被告知头文件的位置，只要语法正确，编译器就可以编译出**每个**`.cpp`源文件对应的中间目标文件。如果没有声明，编译器会给出警告，但是依然可以生成`.o`文件。

在Linux系统中，[gcc](https://gcc.gnu.org/)组件是最流行的C/C++编译器组合(GNU Compiler Collections)，但是原本的gcc是设计在GNU系统使用的(GNU C Compiler)，现在的gcc组件包含很多中编译器，比如gcc和`g++`，前者用来编译C程序而后者用于编译`C++`程序，但后者实际也可以编译C程序，因为`C++`从C语言扩展而来。而对于Windows系统，gcc组件移植为MinGW。

### 链接

链接主要是链接函数和全局变量，不管函数的源文件，只管它们的目标文件。要是在指明的目标文件中，链接器找不到函数的实现语句，就会报出链接错误(linker error)。当源文件太多时，中间目标文件也会很多，管理起来非常麻烦。这时候可以给中间目标文件打包，Windows下为库文件(Library file: `.lib`)，UNIX系统下为Archive file (`.a`)。

### 工程方法

在编译`C++`源文件时，我们有两类主要的方法：

- 在终端中通过命令编译。按照需要编译的文件量的多少，又可以分为使用`gcc/g++`对少量源文件编译和使用makefile/cmake对大量的多文件编译。**使用命令行方法的优点是能够在命令行中展示出生成可执行文件的过程中发生了什么，这也是写这篇文章的首要原因**。
- 使用集成开发环境（IDE）进行管理。比如说Visual Studio，方便进行工程管理。注意VSCode不是IDE，而是纯文本编辑器（editor），所以在使用VSCode的时候需要手动装好编译器（compiler）。

而另一个常见的操作名词：**调试**，实际上是一种特殊的运行程序的操作。在IDE或者VSCode中，我以前经常的调试方法最好的习惯也只是`断点调试(F9|F5)+单步执行(F10/F11)`，虽然说好于`无调试运行(Ctrl+F5)`，但是都看不到编译和链接的过程，毕竟很多bug只会产生警告，编译会照常进行生成目标文件，写程序初期需要有`编译(Ctrl+Shift+B)`的步骤。

## `g++`执行过程

首先了解`C++`编程中相关文件以及它们的含义。

| 后缀名                       | 描述                                  |
| ---------------------------- | ------------------------------------- |
| .a                           | 打包目标文件的库文件                  |
| .c/.C/.cc/.cp/.cpp/.cxx/.c++ | 源代码文件，函数和变量的**定义/实现** |
| .h                           | 头文件，函数和变量的**声明**          |
| .ii                          | 编译预处理产生的文件                  |
| .o (.obj in Windows)         | 编译产生的中间目标文件                |
| .s                           | 编译产生的汇编语言文件                |
| .so                          | 编译产生的动态库文件                  |
| .out (.exe in Windows)       | 链接目标文件产生的可执行文件          |

部分文件产生的过程分为**四步**，包括：

1. 预处理。条件编译，头文件包含，宏替换的处理。

   > 主要包括：
   >
   > - 对全部的`#define`进行宏展开；
   > - 处理条件编译指令：`#if`, `#ifdef`, `#elif`, `#else`, `#endif`；
   > - 处理`#include`，递归过程；
   > - ... 

2. 编译。预处理后的文件转换为汇编语言。

3. 汇编。产生目标文件。

4. 链接。链接目标文件，生成可执行程序。

![产生可执行文件的过程](https://i.loli.net/2020/05/01/63SJVkP9e14XLAG.png)

### 单个源文件生成可执行程序

从头开始，在合适的文件夹中创建名为hello的文件夹，进入hello，创建hello.cpp，打开编辑加入代码。

```bash
$ vi hello.cpp
$ more hello.cpp 
#include <iostream>
int main(int argc,char *argv[]) {
  std::cout << "hello, world" << std::endl;
  return(0);
}
```

```bash
$ g++ hello.cpp
```

编译器 `g++` 通过检查命令行中指定的文件的后缀名可识别其为 C++ 源代码文件。**编译器默认的动作：编译源代码文件生成对象文件(object file)，链接对象文件和 libstdc++ 库中的函数得到可执行程序, 然后删除对象文件。**

由于命令行中未指定可执行程序的文件名，编译器采用默认的 a.out。程序可以这样来运行：

```bash
$ ./a.out
hello, world
```

更普遍的做法是通过 `-o` 选项指定可执行程序的文件名。下面的命令将产生名为 helloworld 的可执行文件：

```bash
$ g++ hello.cpp -o hello
```

在命令行中输入程序名可使之运行：

```bash
$ ./hello
hello, world
```

### 多个源文件生成可执行程序

定义一个名为ask.h的头文件，包含类的定义和成员函数的声明：

```bash
$ vi ask.h
$ more ask.h 
#include <iostream>
class Ask{
	public:
		void askAge(const char *);
};
```

定义名为ask.cpp的文件，包含此成员函数的定义：

```bash
$ vi ask.cpp
$ more ask.cpp 
#include "ask.h"
using namespace std;
void Ask::askAge(const char *str){
	cout << str << ", how old are you?" << endl;
}
```

主函数放在askme.cpp文件中：

```bash
$ vi askme.cpp
$ more askme.cpp 
#include "ask.h"
using namespace std;

int main(int argc, char *argv[])
{
	Ask ask;
	ask.askAge("Soldier");
	return(0);
}
```

组合单一的可执行程序：

```bash
$ g++ askme.cpp ask.cpp -o askme
$ ls
ask.cpp  ask.h  askme  askme.cpp  hello.cpp  hello.ii  hello.o  hello.s
$ askme
askme: command not found
$ ./askme
Soldier, how old are you?
```

我们来看看如果不声明成员函数会发生什么。发现只有预编译不会报错也不会警告，生成汇编文件或者目标文件都会报错，和之前说的仅产生警告不太一样，不确定发生了什么，**留作疑问**。

```bash
(base) jinhang:~/Documents/hello$ more ask.h
#include <iostream>
class Ask{
//	public:
//		void askAge(const char *);
};
$ g++ -E ask.cpp -o temp.ii
$ ls
ask.cpp  askme      ask.s      hello.ii  hello.s
ask.h    askme.cpp  hello.cpp  hello.o   temp.ii
$ g++ -S ask.cpp
ask.cpp:3:33: error: no ‘void Ask::askAge(const char*)’ member function declared in class ‘Ask’
 void Ask::askAge(const char *str){
                                 ^
$ g++ -c ask.cpp
ask.cpp:3:33: error: no ‘void Ask::askAge(const char*)’ member function declared in class ‘Ask’
 void Ask::askAge(const char *str){
```

### 预处理阶段

使用选项`-E`指明预处理：

```bash
$ g++ -E helloworld.cpp
```

此时不会生成预处理文件，只会在终端中打印预处理文件内容，数不清的行数，但是基本上是**清除了无关代码**。指明输出文件选项`-o`保存看看：

```bash
$ g++ -E hello.cpp -o hello.ii
$ ls
hello.cpp  hello.ii
```

### 生成汇编代码

指明选项`-S`:

```bash
$ g++ -S hello.cpp 
$ ls
hello.cpp  hello.ii  hello.s
(base) jinhang:~/Documents/hello$ more hello.s
	.file	"hello.cpp"
	.text
	.section	.rodata
	.type	_ZStL19piecewise_construct, @object
	.size	_ZStL19piecewise_construct, 1
_ZStL19piecewise_construct:
	.zero	1
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
.LC0:
	.string	"hello, world"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1493:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
--More--(19%)
```

### 生成目标文件

选项`-c`告诉编译器编辑代码但是不执行链接，输出目标文件`.o`。

```bash
$ g++ -c hello.cpp 
$ ls
hello.cpp  hello.ii  hello.o  hello.s
```

### 链接

g++可以直接利用生成的目标文件来进行链接，把多个目标文件转换为单个可执行程序。

```bash
$ g++ -c ask.cpp
$ g++ -c askme.cpp
$ g++ ask.o askme.o -o askmeagain
$ ./askmeagain 
Soldier, how old are you?
```

### 创建静态库

我们可以把多个目标文件打包为一个库（归档文件），库中的成员包括普通函数，雷定义，雷德对象实例等等。管理这种归档文件的工具叫做ar。

首先创建两个对象模块，然后利用其生成静态库。头文件ask.h包含函数askAge()的圆形和类Ask的定义：

```cpp
/* ask.h */
#include <iostream>
using namespace std;

void askme(void);

class Ask{
	private:
		string thestring;
	public:
		Ask(string str){	// Constructor
			thestring = str;
		}
		void askThis(const char *str){
			cout << str << " from a static library\n";
		}
		void askString(void);
};
```

下面是文件 ask.cpp 是我们要加入到静态库中的两个目标文件之一的源码。它包含 Ask 类中 askString() 函数的定义体；类 Ask 的一个实例 libraryask 的声明也包含在内：


```cpp
/* ask.cpp */
#include "ask.h"
using namespace std;

void Ask::askString(){
	cout << thestring << "?"  << endl;
}

Ask libraryask("Library instance of Ask");
```

源码文件 askme.cpp 是我们要加入到静态库中的第二个目标文件的源码。它包含函数 askme() 的定义：

```cpp
/* askme.cpp */
#include "ask.h"
using namespace std;

void askme()
{
	cout << "me from a static library" << endl;
}
```

将源码编译为目标文件，命令ar将它们存入库中：

```bash
$ g++ -c askme.cpp
$ g++ -c ask.cpp
ask.cpp:9:41: warning: ISO C++ forbids converting a string constant to ‘char*’ [-Wwrite-strings]
 Ask libraryask("Library instance of Ask");
                                         ^
$ ls
ask.cpp  askme       askme.cpp  ask.o  hello.cpp  hello.o  temp.ii
ask.h    askmeagain  askme.o    ask.s  hello.ii   hello.s
$ ar -r libask.a askme.o ask.o
ar: creating libask.a
```

程序 ar 配合参数 `-r` 创建一个新库 `libask.a` 并将命令行中列出的对象文件插入。采用这种方法，如果库不存在的话，参数 -r 将创建一个新的库，而如果库存在的话，将用新的模块替换原来的模块。

下面是主程序 askmain.cpp，它调用库 libask.a 中的代码：

```cpp
/* askmain.cpp */
#include "ask.h"

int main(int argc, char *argv[]){
  extern Ask libraryask; // 使用库的对象
  Ask localask = Ask("Local instance of Ask"); //使用库的类定义
  askme(); // 使用库的普通该函数
  libraryask.askThis("howdy");
  libraryask.askString();
  localask.askString();
  return(0);
}
```

该程序可以下面的命令来编译和链接，**只需要对主函数源文件和归档文件进行链接**：

```bash
$ g++ askmain.cpp libask.a -o askmain
```

程序运行时，产生以下输出：

```bash
$ ./askmain 
me from a static library
howdy from a static library
Library instance of Ask?
Local instance of Ask?
```

## 参考

- [What is the difference between gcc and g++ in Linux?](https://www.includehelp.com/c-programming-questions/difference-between-gcc-and-g.aspx)
- [Visual Studio Code 如何编写运行 C、C++ 程序？](https://www.zhihu.com/question/30315894/answer/154979413)
- [Compiling Cpp](https://wiki.ubuntu.org.cn/Compiling_Cpp)
- [C++程序编译的四个过程](https://blog.csdn.net/a1314521531/article/details/52625408)
- [C++ 编译初步](http://wulc.me/2018/11/24/C++%20%E7%BC%96%E8%AF%91%E5%88%9D%E6%AD%A5/)