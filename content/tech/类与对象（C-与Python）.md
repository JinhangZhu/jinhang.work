---
title: "类与对象（C++与Python）"
date: 2020-05-01T22:19:36+01:00
categories: [Tech,Programming]
tags: [C/C++, Python]
slug: "class-object-cpp-python"
---

作为面向对象编程(Object-Oriented Programming: OOP)的语言，C++和Python在很多应用中都很强调类的使用，这篇文章将展开类的讨论，并同时给出两种语言中的实现，**以研究细节为主**。<!--more-->

类(class)提供了将数据（data）和操作（functionality/operations）打包起来的方法。Python的类是 C++ 和Modula-3 中类机制的一个融合。就像模块（module）一样，Python的类具有动态特性：在运行的时候创建，并且可以在创建之后进行进一步的修改。

和 C++ 不一样，Python 可以将内置类型作为基类（base classes）扩展。和 C++ 一样，大多数的有特殊语法的内置运算符（operators）都可以被重载，供类的实例（instances）使用。

Python 中，同一具体的对象可以有多个名称，这在其他语言中叫**别名（aliasing）**。别名在涉及可变对象（如列表，字典和大多数其他类型）时可能非常方便。如传递一个对象实际上就是传递了一个指针，函数中如果修改了作为参数传入的对象，调用者可以在原来名字那里看到修改。

{% blockquote Eric Matthes, Python Crash Course %}

*Object-oriented programming* is one of the most effective approaches to writing soft-ware. In object-oriented programming you write *classes* that represent real-world things and situations, and you create *objects* based on these classes. When you write a class, you define the general behavior that a whole category of objects can have.

{% endblockquote %}

## 名词解释

看这篇文章，理应对类与对象有了基本的了解，但是若是要抛开代码把各个名词和原理将明白，却不是那么容易的事，所以把理论基础再说明白一点。这里我们采用来自Quora上一个解释的非常清楚的答案：

故事从上帝造人开始。上帝需要制造一个新的物种叫做人，也就是一个叫“人”的**类（class）**。上帝开始了设计，比如说人应该有手，腿，眼睛，鼻子等等，这些叫做**类的属性（attributes/properties）**。所有的人都应该可以走路，跑步，讲话，吃东西等等，这些就叫做行为和操作，也就是**类的方法（methods）**。但是现在在现实中还没有人，因为上帝的设计还在纸上，是个蓝图（blue print）。

那么上帝开始造人了，根据蓝图的设计，现实中人就被造了出来，这个过程就是**创造类的对象（objects）**。人应该在地球上有一块空间来存在，于是分配了这样的一块空间的过程就叫做**创建对象的内存（memory）**。

现在有个问题：上帝造的人怎么区分呢？上帝要是先让其中一部分去造房子怎么办？那么这就是一个**实例化（instantiation）** 的过程，上帝给某个人名字叫亚当，那么这样的**对象（人）** 就有了可供**查阅的项（reference）** 叫亚当。名叫亚当的这个对象成为了**实例（instance）** ，可以被 **喊名字（refer）** 区分了。

所以说，对象是一个统称的概念，它们物理意义上存在但是还没有被区分，实例具有身份了，可以被区别开来。我们所熟悉的整型就是类，整型变量是对象，名字叫 num 的对象就是一个实例了。OOP 语言中，归类带来了很多遍历，也比以基于操作的C语言等好理解。

> <span class='quora-content-embed' data-name='What-is-the-difference-between-object-and-instance/answer/Amandeep-Verma-16'>Read <a class='quora-content-link' data-width='560' data-height='260' href='https://www.quora.com/What-is-the-difference-between-object-and-instance/answer/Amandeep-Verma-16' data-type='answer' data-id='107835486' data-key='bc85b0b82645d881d525cd6b16215ca1' load-full-answer='False' data-embed='yiksrzn'><a href='https://www.quora.com/Amandeep-Verma-16'>Amandeep Verma</a>&#039;s <a href='/What-is-the-difference-between-object-and-instance?top_ans=107835486'>answer</a> to <a href='/What-is-the-difference-between-object-and-instance' ref='canonical'><span class="rendered_qtext">What is the difference between object and instance?</span></a></a> on <a href='https://www.quora.com'>Quora</a><script type="text/javascript" src="https://www.quora.com/widgets/content"></script></span>

## 创建一个类

设计一个类`Dog`，狗，具有`name`, `age`两种属性，`sit`, `roll_over`两种行为/方法/操作。

C++ ：

```c++
#include <iostream>
using namespace std;

class Dog
{
public:
    Dog(string, int);
    void sit();
    void roll_over();

private:
    string name;
    int age;
};

Dog::Dog(string strname, int intage)
{
    name = strname;
    age = intage;
}

void Dog::sit()
{
    cout << name << " is now sitting" << endl;
}

void Dog::roll_over()
{
    cout << name << " rolled over!" << endl;
}
```

Python ：

```python
class Dog():
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def sit(self):
        print(self.name.title() + " is now sitting.")
    
    def roll_over(self):
        print(self.name.title() + " rolled over!")
```

### 构造函数

在 C++ 中是和类名同名的无需指明返回类型的函数，而在 Python 中是 `__init__()`方法。方法在实例化时自动调用，并且自动传递`self`参数（是实例对自身的引用，其实在每个类方法中都要首先写出来），这也就是我们在实际创建实例的时候只需要传入除了self之外的参数的原因。

## 创建实例与调用方法

区别就是： C++ 先写变量类型（类名），再写变量名（实例名），自动调用构造函数，括号中需要填写传入的参数。而 Python 直接使用类名作为初始化函数，实则调用`__init__()`函数，将对象传给有具体名称，产生实例。对于调用成员变量/函数，以及私有/公有属性的区别调用，下文将会讲到。

C++ ：

```c++
#include <iostream>
#include "dog.h"

int main()
{
    Dog dog1("Alpha", 2);
    dog1.sit();
    dog1.roll_over();
}
//Alpha is now sitting
//Alpha rolled over!
```

Python：

```bash
>>> from dog import *
>>> dog2 = Dog('Beta', '3')
>>> print("My dog's name is " + dog2.name.title() + ". It is " + str(dog2.age) + " years old")
My dog's name is Beta. It is 3 years old
>>> dog2.sit()
Beta is now sitting.
>>> dog2.roll_over()
Beta rolled over!
```

### 变量类型

在 C++ 和 Python 中，变量可以这样划为三种类型: field （成员变量）, parameters （参数）, local variables （局部变量）。我们看成员变量field，它是具和class的这个对象一样具有一样的生存周期，当然field也是类中所有方法/函数都可以调用的。而后两者变量都是本地属性（保存在堆栈中），仅在函数执行时存在。三种类型变量具体区别如下：

<img src="https://i.loli.net/2020/05/02/x9yNH3fmuTbi8gk.png" alt="Screenshot from 2020-05-01 18-32-34.png" style="zoom:80%;" />

### 私有和公有属性

C++ 定义私有/公有属性/函数需要加上关键字`private:`, `public:`，保护成员则使用关键字`protected`。派生类无法访问基类的所有私有成员，但是可以访问基类的公有和保护成员，外部类只能访问当前类的公有成员。但是 Python 只需要在变量名/方法名之前加上单下划线`_`或者双下划线`__`，而非关键字。在模块中，单下划线开头的常被默认为内部或者保护变量/函数，用`from module import *` 的方式，这些单下划线开头的变量和函数不会被导入；但是`import module` 这种方式导入模块的话，可以用module._变量 的形式访问到这些变量和函数。如这样的`module.py`:

```python
# Variables
no_udl = 0      # No underline
_sgl_udl = 10   # Single underline
__dbl_udl = 20  # Double underline
```

采用两种导入方式对比：

![Screenshot from 2020-05-01 19-10-12.png](https://i.loli.net/2020/05/02/KOiRaWLpcmITzfS.png)

双下划线开头的命名形式，通常用于python的类中，用于将变量伪私有化，即用双下划线开头的属性或方法，不能被外部调用，也不会被子类继承；但实际上只是将该变量的名字做了改变，比如`__method` 变为了 `_{class}_method`，通过 `{instance}._{class}__method`的方式仍然可以调用。如在`module.py`中定义类`Module`和其子类`subModule`：

```python
class Module():
    def __init__(self):
        self.no_udl = 0
        self._sgl_udl = 3
        self.__dbl_udl = 6
    
class subModule(Module):
    def __init__(self):
        super().__init__()
```

对类成员变量的调用：

```bash
>>> from module import *
>>> module1 = Module()	# 实例化
>>> module1.no_udl	# 无私有直接通过实例调用
0
>>> module1._sgl_udl	# 即使是保护变量也可直接通过实例调用
3
>>> module1._Module__dbl_udl	# 伪私有变量无法通过实例直接调用，需要调用加上了单下划线和类名的前缀的变量
6
```

## 类的继承

在 Python 中，子类初始化函数需要调用`super()`来调用父类的初始化函数。即：`super().__init__()`。

```python
class inheritanceDog(Dog):
    def __init__(self, name, age, parent):
        super().__init__(name, age)
        self.parent = parent
```

在 C++ 中，类的继承比较复杂。**单继承**基本形式为：

```c++
class derived_class: access_specifier base_class
```

`access_specifier`：访问修饰符，公共/保护或私有，指明继承类型。通常用public继承，而不是protected和private。

当使用不同类型的继承时，遵循以下几个规则：

- **公有继承（public）：** 当一个类派生自**公有**基类时，基类的**公有**成员也是派生类的**公有**成员，基类的**保护**成员也是派生类的**保护**成员，基类的**私有**成员不能直接被派生类访问，但是可以通过调用基类的**公有**和**保护**成员来访问。
- **保护继承（protected）：**  当一个类派生自**保护**基类时，基类的**公有**和**保护**成员将成为派生类的**保护**成员。
- **私有继承（private）：** 当一个类派生自**私有**基类时，基类的**公有**和**保护**成员将成为派生类的**私有**成员。

**多继承**形式和但继承基本一致，不过冒号后面的内容要以逗号隔开。

```c++
class derived_class: access_specifier1 base_class1, access_specifier2 base_class2, ...
```

## 类的导入

Python 环境下使用`from`, `import`语句，全部导入就用`*`，导入部分就写逗号隔开的需要导入的类。采用单`import`则需要每次使用module中的类都需要加上module名的前缀：`module.`。

C++ 环境下则include类所在的头文件（含头文件所在位置信息）。

## 参考

- [The Python Tutorial >> Classes](https://docs.python.org/3/tutorial/classes.html#classes)
- [hat is the difference between object and instance?](https://www.quora.com/What-is-the-difference-between-object-and-instance/answer/Amandeep-Verma-16)
- [Summary 3 – Variables: Fields vs. Parameters vs. Local variables](https://www.rose-hulman.edu/class/csse/csse230/Summaries/3%20Variables-FieldsVsParametersVsLocalVariables.pdf)
- [Python中关于私有、公有的属性和方法](https://github.com/zshaolin/My-Python-Wikis/wiki/%23-Python%E4%B8%AD%E5%85%B3%E4%BA%8E%E7%A7%81%E6%9C%89%E3%80%81%E5%85%AC%E6%9C%89-%E7%9A%84%E5%B1%9E%E6%80%A7%E5%92%8C%E6%96%B9%E6%B3%95)
- Matthes, Eric. *Python crash course: a hands-on, project-based introduction to programming*. No Starch Press, 2015. 161-183