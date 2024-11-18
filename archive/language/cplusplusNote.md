## 面向对象编程

### 类

在C++中，类是一种自定义的数据类型，用于封装数据和方法。

以下是使用类的基本方法：

1. 定义类：
   使用关键字`class`来定义一个类。类的定义通常包括成员变量（类的属性）和成员函数（类的方法）。例如：

```cpp
class MyClass {
public:
    int myVariable; // 成员变量

    void myMethod() {
        // 成员函数实现
    }
};
```

2. 创建对象：
   使用类定义的对象是实际的**实例**，可以通过类名后面跟随括号来创建对象。例如：

```cpp
MyClass obj; // 创建一个MyClass对象(实例化)
```

3. 访问成员：
   使用成员访问运算符`.`来访问对象的成员变量和成员函数。例如：

```cpp
obj.myVariable = 10; // 设置成员变量的值
obj.myMethod(); // 调用成员函数
```

4. 构造函数和析构函数：
   **构造函数**用于初始化对象的状态，在对象创建时自动调用。**析构函数**在对象销毁时自动调用，用于清理对象使用的资源。例如：

```cpp
class MyClass {
public:
    int myVariable;

    MyClass() { // 构造函数
        myVariable = 0;
    }

    ~MyClass() { // 析构函数
        // 清理资源的代码
    }

    // 其他成员函数和变量
};
```

5. 封装性：
   类提供了封装性，可以使用访问修饰符（`public`、`private`和`protected`）来限制成员的访问权限。`public`成员可以从类的外部访问，`private`成员只能在类的内部访问。例如：

```cpp
class MyClass {
public: // 公共成员
    int publicVariable;

private: // 私有成员
    int privateVariable;

public: // 公共成员函数
    void myMethod() {
        publicVariable = 10; // 可以访问公共成员
        privateVariable = 20; // 可以在类的内部访问私有成员
    }
};
```



- C 与 C++区别
  - 语法和特性差异
    - C 语言使用的是过程性编程风格，而 C++是一种面向对象的编程语言

    - C++扩展了 C 语言的语法，引入了类、对象、继承、多态等特性

    - C++支持函数重载，即定义具有相同名称但参数类型或个数不同的多个函数

    - C++新增了一些关键字，如 class、public、private 等

  - 标准库差异
    - C++标准库相比于 C 标准库更为庞大，并且包含了很多容器、算法和函数模板等实用工具

    - C 语言只有基本的输入输出和字符串处理函数，而 C++新增了 iostream 库来进行更方便的输入输出操作

    - C++提供了丰富的字符串类和相关的函数，而在 C 语言中，字符串只是一个字符数组

  - 内存管理
    - C++支持动态内存管理的操作符 new 和 delete，方便进行对象的创建和销毁

    - C 语言使用 malloc 和 free 函数进行动态内存分配和释放

    - C++还引入了构造函数和析构函数，可以在对象创建和销毁时执行相应的操作

  - 异常处理
    - C++具有异常处理的机制，使用 try-catch 语句来捕获和处理异常

    - C 语言中没有内置的异常处理机制，通常使用错误码来表示发生的错误

  - 名字空间
    - C++引入了名字空间的概念，可用于避免命名冲突和组织代码

    - C 语言中没有名字空间的概念，所有的标识符都是全局的

  - 扩展性和兼容性
    - C++可以与 C 语言代码逐步集成，支持直接调用 C 语言的函数和使用 C 语言的库

    - C++可以使用 extern "C"关键字来声明 C 语言的函数，并指定 C 语言的调用约定


## C++

### 基础概念

#### C++的发展历史

#### 编译和执行过程

#### C++的基本语法规则

##### 一个例子

```c++
#include<iostream>
//这一行是预处理指令，用于包含iostream头文件，该头文件中包含了输入输出流的相关函数和对象。
using namespace std;
//这一行是命名空间的使用声明，使用`std`命名空间，表示使用`std`命名空间下的符号（函数、类等）而无需使用限定符（如`std::cout`）。
int main(){
//这一行是定义了主函数`main`，程序从这里开始执行。
cout << "hello world" << endl;
//这一行是利用输出流对象`cout`来输出字符串"hello world"，`<<`是输出运算符，`endl`是换行符。不同的元素用`<<`隔开,相当于python中字符串合并的的`+`
//输入运算符`>>`的用法类似于输出运算符,比如`cin>>a>>b;`,代表同时输入两个数据,中间用空格隔开
//多条代码可以写在同一行,用分号隔开
system("pause");
//这一行是调用系统命令`pause`，用于在输出结果后等待用户按下任意键后继续。
return 0;
//这一行是返回主函数的返回值，返回值为整型0，表示程序成功运行结束。
```

##### 关键字

c++中预先保留的单词(保留字)

##### 标识符命名规则

- 不能是关键字
- 标识符只能由字母,数字,下划线组成
- 第一个字符必须为字母或下划线
- 标识符中的字母区分大小写
- (类似于 python)最好要见名知意

##### 注释

- 单行注释 `// 描述信息`
- 通常放在一行代码的上方,或者一条语句的末尾,对该行代码说明
- 多行注释 `/*描述信息*/`
- 通常放在一段代码的上方,对该分段代码做整体说明

#### 标准输入输出

##### 流

流：C++统一管理 IO 的方法

- 什么是流
- 由于程序开发的需要，需要在程序间传递数据；
- 这些数据按顺序组成由若干字节组成的字节序列；
- 传递时，在内存中开辟一个内存缓冲区，用来存放流
  动中的数据。
- 缓冲区中的数据就是流；
- 流的分类：
- 输出流：从内存中送一串字节到显示器、打印机；
- 输入流：从键盘送一串字节到内存；
- 文件流：从内存送一串字节到外存；
- 与流相关的运算符：输出<<, 输入>>

##### 数据的输入

作用:用于从键盘获取数据 (其实是从缓冲区获取数据,遇到空格停止)
使用>>链接可以忽略空格,换行符,制表符
关键字:`cin`  
语法:`cin >> 变量`  
实例

```c++
#include <iostream>
using namespace std;

int main()
{
 // 整型输入
 int a = 0;
 cout << "请输入整型变量"<< endl;
 cin >> a;
 cout <<a << endl;

 // 浮点型输入
 double b = 0;
 cout << "请输入浮点型变量"<< endl;
 cin >> b;
 cout <<b << endl;

 // 字符型输入
 char c = '0';
 cout << "请输入字符型变量"<< endl;
 cin >> c;
 cout <<c << endl;

 // 字符串型
 char d[] = "0";
 cout << "请输入字符串型变量"<< endl;
 cin >> d;
 cout <<d << endl;

 // 布尔型
 bool e = true;
 cout << "请输入布尔型变量"<< endl;
 cin >> e;
 cout <<e << endl;


 system("pause");

 return 0;
}
```

- cin 流对象
- 从标准输入设备(键盘)获取数据；
- 程序中的变量通过流提取运算符“>>”从流中
  提取数据；
- 流提取符“>>”从流中提取数据时通常跳过输
  入流中的空格、tab 键、换行符等空白字符；
- 遇到无效字符或文件结束标志，停止提取；
- 只有在输入完数据再按回车键后，该行数据才
  被送入键盘缓冲区，形成输入流，提取运算符
  “>>”才能从中提取数据

#### 数据的输出

标准输出 cout

- cout (console output，控制台输出)
- cout 流在内存中对应开辟了一个缓冲区，用来
  存放流中的数据，
- 当向 cout 流插入一个 endl 时，不论缓冲区是否
  已满，都立即输出流中所有数据，并加入一个
  换行符，然后刷新流(清空缓冲区)。
- 用“cout<<”输出基本类型的数据时，不必考
  虑数据是什么类型，系统会判断数据的类型

##### 使用控制符控制输出格式

- 常量控制符(iostream)
- dex 按十进制输出
- hex 按是十六进制输出
- oct 按八进制输出
- endl 插入换行符,并刷新流
- ends 插入空字符
- 函数控制符(iomanip)
- setbase(n) 设置整数基数为 n(16/8/10)
- setfill(c) 设置填充字符 c
- setprecision(n)与 fixed 设置浮点数精度
- setw(n) 设置字段宽度 n
- setiosflags(ios::fixed) 设置浮点数以固定位数显示
- setiosflags(ios::scientific) 设置浮点数以科学计数法显示
- setiosflags(ios::left) 输出数据左对齐
- setiosflags(ios::right) 输出数据右对齐
- setiosflags(ios::skipws) 忽略前导的空格
- setiosflags(ios::uppercase) 十六进制数大写输出
- setiosflags(ios::lowercase) 十六进制数小写输出
- setiosflags(ios::showpos) 输出正数时加上+号
- resetiosflags() 终止已设置输出格式状态(**不会用**)

### 数据类型和变量

#### **变量**

给一段指定的内存空间命名,方便操作这段内存  
`数据类型 变量名 = 初始值;`  
初次定义之后,再次使用就不需要再定义数据类型了

```c++
#include <iostream>
using namespace std;

int main()
{
 int a = 10;

 cout <<"a = "<< a << endl;

 system("pause");

 return 0;
}
```

#### **常量**

用于记录程序中不可更改的数据  
两种定义方式

- #define 宏常量 `define 常量名 常量值`
- 通常在文件的上方定义,表示一个常量
- 不需要考虑变量的类型,在下面的代码中遇到它就替换
- 不需要写`;`
- const 修饰的变量 `const 数据类型 常量名 = 常量值`
- 通常在变量定义前加上关键字 const,修饰该变量为常量,不可修改
- constexpr 修饰的变量 `constexpr 数据类型 常量名 = 常量值`
- ``

```c++
#include <iostream>
using namespace std;

#define day 7

int main(){

 cout << "一周有:" << day << "天" << endl;

 system("pause");

 return 0;

}
```

```c++
#include <iostream>
using namespace std;

int main()
{
 const int day = 7;

 cout << "一周有 " << day << "天" << endl;

 system("pause");

 return 0;
}
```

c++规定在创建一个变量或者常量时,必须要指定出相应的数据类型(分配合适的内存空间),否则无法给变量分配内存

#### 整型(区别在于所占内存空间不同)

- 短整型`short` 占用空间 2 字节,取值范围(-2^15,2^15-1)
- 整型`int` 占用空间 4 字节,取值范围(-2^31,2^31-1)
- 长整型`long` windows 为 4 字节,linux 为 4 字节(32 位),8 字节(64 位),取值范围(-2^31,2^31-1)
- 长长整型`long long` 占用空间 8 字节,取值范围(-2^63,2^63-1)

> 上述数据类型都是有符号型,即 signed,它特地画出一位来表示正负
> 对应的还有无符号型,即 unsigned,它会将数据表示范围扩大(去除了表示正负的一位)  
> **补码** 负整数(对源码取反加 1)

#### sizeof 关键字

利用 sizeof 关键字可以统计数据类型所占内存大小  
`sizeof(数据类型/变量)`(选择一个输入,变量名或者数据类型)

```c++
#include <iostream>
using namespace std;

int main()
{
 int a = 10;

 cout <<sizeof(int) << endl;
 cout <<sizeof(a) << endl;

 system("pause");

 return 0;
}
```

#### 枚举类型

#### 实型(浮点型)

用于表示小数(区别在于表示的有效数字范围不同)

- 单精度 float 4 字节 7 位有效数字
- 双精度 double 8 字节 15-16 位有效数字
- 在 c++中小数一般显示 6 位有效数字,如需更多,则需要特殊配置

> 科学计数法 3e-2 `%e`
> e 之后为负数,则为 0.1 的次方,正数则为 10 的次方

```c++
#include <iostream>
using namespace std;

int main()
{
 float a = 3.14f;
 /*
 可以多个变量一起赋值
 不加f会默认为双精度小数,用float接受会多做一步转换
 在末尾加上f直接作为单精度小数被接受,减少了一次转换
 */
 double b = 3.14;

 cout <<sizeof(int) << endl;
 cout <<sizeof(a) << endl;
 cout << a << endl;
 cout << b << endl;

 system("pause");

 return 0;
}
```

#### 字符型

字符型变量用于显示单个字符  
`char ch = 'a'`

> 注意 1:在显示字符型变量时,用**单引号**将字符括起来,不要用双引号
> 注意 2:单引号中**只能有一个字符**,不能是字符串
> c 和 c++中字符型变量只占用 1 个字节
> 字符型变量并不是把字符本身放入内存中储存,而是将对应的 ASCII 编码放入储存单元(a-97,A-65)

#### 转义字符

用于表示一些不能显示出来的 ASCII 字符  
现阶段常用的转义字符有`\n \\ \t`  
转义字符形式 含义 ASCII 码值
\a x 响铃符 007
\b 退格符，将光标位置移到下一页开头 008
\f 进纸符，将光标位置移到下一页开头 012
**\n 换行符，将光标位置移到下一行开头 010**
\r 回车符，将光标位置移到本行开头 013
**\t 水平制表符，光标跳到下一个 Tab 位置 009**
\v 垂直制表符 011
\' 单引号 039
\" 双引号 034
**\\ 单反斜杠 092**
\? 问号 063
\0 空字符 000
\ooo 用 1~3 位八进制数 ooo 为码值所对应的字符 ooo（八进制）
\xhh 用 1、2 位十六进制 hh 为码符所对应的字符 hh（十六进制）

#### 字符串型

作用:用于表示一串字符

- c 风格字符串 `char 变量名[] = "字符串值"`

- > 在字符串的结尾自动加\0,表示字符串的结束

- 中括号,双引号

- 不同于字符数组,输出的时候自带一个循环

- c++风格字符串 `string 变量名 = "字符串值"`

- 用 c++风格的字符串时要包含**头文件** `#include <string>`

- **字符串要使用双引号**

- 可以使用二位数组储存字符串,多出位置补上\0

- 可以通过判断结尾是否是\0,来判断字符串的结束

```c++
#include <iostream>
using namespace std;
#include <string>

int main()
{
 char str1[] = "hello world";//一般不在[]里面写数字,要写的话要比字符串长度加1
 string str2 = "hello world";

 cout <<str1 << endl;
 cout <<str2 << endl;
 cout <<sizeof(str1) << endl;
 cout <<sizeof(str2) << endl;

 system("pause");

 return 0;
}
```

#### 布尔类型 bool

作用:布尔数据类型代表真或假的值

- `true` 真(本质为 1)
- `false` 假(本质为 0)
- 占一个字节的大小
- `bool 变量名 = true(false);`

### 运算符和表达式

作用:用于执行代码的运算

- 算术运算符-用于处理四则运算
- 赋值运算符-用于将表达式的值赋给变量
- 比较运算符-用于表达式的比较,并返回一个真值或假值
- 逻辑运算符-用于根据表达式的值来返回真值或假值

#### 算术运算符

- `+` 正号 或者 加号
- `-` 负号 或者 减号
- `*` 乘
- `/` 除
- c++中规定两个整数相除,结果仍然是整数(丢去小数部分)(整数运算)
- 两个浮点数相除,结果可有小数 (实数运算)
- `%` 取模(取余数)
- 前面为被除数,后面为除数
- 两个小数之间不可以做取模运算(只有整数可以)
- 自增自减运算符(只能作用于变量而不是表达式,优先级由他自己规定)
- `++` 前置(后置)递增(优先级)
- 前置 例如`++a`让变量加 1 先让变量+1,然后进行表达式的计算(先加后用)
- 后置 例如`b++`让变量加 1 先进行表达式的计算,再让变量+1(先用后加)
- `--` 前置(后置)递减
- 类似前者
- 优先级
- **括号>\* / % > + -**
- 还要考虑从左到右的顺序(剪刀法)

##### 运算中的类型转换

整数类型*浮点类型 转换为浮点类型  
浮点数*浮点数 按照  
整数\*整数 按照

```c++
#include <iostream>
using namespace std;

int main()
{
 int a2 = 5;
 int b2 = 2 * a2++;
 int c2 = 2 * ++a2;
 cout << b2 << endl;
 cout << c2 << endl;


 system("pause");

 return 0;
}
```

#### 赋值运算符(类似于 python)

- =
- 左右类型不一致会类型转换,右边的数据会被转换为左边的类型
- 截断(长数转短数)
- 无变化(短数转长数)
- 无符号数和符号数的赋值(保留二进制形式)
- 表达式会返回值,可以用于连续赋值(初始化变量之后)运算顺序从右到左
- +=
- -+
- \*=
- /=
- %=

#### 比较运算符

作用:用于表达式的比较,并返回一个真值或假值
最好将表达式用小括号括起来`bool flag = (a2 != b2);`

- `==`
- `!=`
- `<`
- `>`
- `<=`
- `>=`

```c++
#include <iostream>
using namespace std;

int main()
{
 int a2 = 5;
 int b2 = 2 * a2++;
 int c2 = 2 * ++a2;
 bool flag = (a2 != b2);
 cout << b2 << endl;
 cout << c2 << endl;
 cout << flag << endl;


 system("pause");

 return 0;
}

```

#### 逻辑运算符

作用:根据表达式的值来返回真值或假值(bool)

- ! 非 (取反)
- && 与 (同真则真,非同则假)
- || 或(一真则真,同假才假)
- !>&&>||(优先级)
- 当不确定的时候就加括号

#### 计算机运行优化(短路特性)

只有在必须执行下一个逻辑运算符才能求出表达式的解的时候,才执行运算符号

- 对于表达式 a&&b&&c
- 当 a 为 false
- 对于表达式 a||b||c

### 控制流程（循环、条件语句等）

- 顺序结构
- 选择结构
- 循环结构

#### 选择结构

##### if 语句

作用:执行满足条件的语句  
if 语句的三种形式

- 单行格式 if 语句 `if(条件){条件满足执行的语句}`
- 注意 if 条件之后不要加分号(会使得条件判断失效,总会执行下面的代码块)
- 多行格式 if 语句 `if(条件){条件满足执行的语句}else{条件不满足执行的语句}`
- 多条件的 if 语句 `if(条件1){条件1满足执行的语句}else if(条件2){条件2满足执行的语句}......else{都不满足执行的语句}`.
- 嵌套 if 语句(在 if 语句中再使用 if 语句对条件进行更精确的判断)

> **单一语句可以不添加大括号,多个语句使用花括号(语句组),但是强烈建议都加上**

```c++
#include <iostream>
using namespace std;
int main()
{
 int score = 0;
 cout<<"请输入一个分数"<<endl;
 cin >> score;
 cout<<"你刚输入的分数为"<< score <<endl;
 if (score > 600)
 {
  cout<<"恭喜你取得了好成绩,祝你在大学中取得自己的美好生活"<<endl;

 }
 system("pause");

 return 0;
}
```

```c++
#include <iostream>
using namespace std;

int main()
{
 int score = 0;
 cout << "请输入一个分数" << endl;
 cin >> score;
 cout << "你刚输入的分数为" << score << endl;
 if (score > 600)
 {
  cout << "恭喜你取得了好成绩,祝你在大学中取得自己的美好生活" << endl;
  //单一语句可以不添加大括号

 }
 else
 {
  cout << "别灰心,来年再来" << endl;

 }
 system("pause");

 return 0;
}
```

```c++
#include <iostream>
using namespace std;

int main()
{
 int score = 0;
 cout << "请输入一个分数" << endl;
 cin >> score;
 cout << "你刚输入的分数为" << score << endl;
 if (score > 600)
 {
  cout << "恭喜你取得了好成绩,祝你在大学中取得自己的美好生活" << endl;

 }
 else if(score > 500)
 {
  cout << "别灰心,来年再来,你是有实力的" << endl;

 }
 else
 {
  cout << "建议回家" << endl;
 }
 system("pause");

 return 0;
}
```

##### 三目运算符

作用:通过三目运算符实现简单的判断(用于赋值)  
语法:`表达式1 ? 表达式2 : 表达式3`  
解释:

- 如果表达式 1 的值为真,执行表达式 2,并返回表达式 2 的结果
- 如果表达式 1 的值为假,执行表达式 3,并返回表达式 3 的结果

> 对变量再次赋值的时候不需要声明数据类型
> 在 c++中三目运算符返回的是变量,可以继续赋值

```c++
#include <iostream>
using namespace std;

int main()
{
 int a = 0;
 int b = 10;
 cin >> a;
 // 赋值
 a = (a > 0 ? 1 : 0);
 cout << a << endl;
 // 被赋值
 (a > b? a : b) = 100;
 cout << a << endl;
 cout << b << endl;


 system("pause");

 return 0;
}
```

##### switch 语句

作用:执行多条件分支语句(具体情况,具体数值)

> switch 缺点:判断的时候只能是**整型或者字符型**,不可以是一个区间  
> 优点:结构清晰,执行效率高(速度比 if 快)  
> case 中没有 break,则程序会一直向下进行
> 语法:

```c++
switch(表达式)
// 根据表达式的结果来执行语句
{
 case 结果1://冒号是跳转,后面的语句没有分割,会一直执行
 执行语句;
 // break 退出判断过程,否则会一直执行下去
 break;

 case 结果2:
 执行语句;
 break;

 case 结果3:
 执行语句;
 break;

 // 前面的条件均不满足,执行的默认语句
 default:
 执行语句;
 break;
}
```

#### 循环结构

##### while 循环语句

作用:满足循环条件,执行循环语句  
语法:`while(循环条件){循环语句}`  
解释:只要循环条件的结果为真,就执行循环语句

> 注意要避免死循环,仔细设定循环条件
> break 这个关键字可以用在循环结构中来退出

```c++
#include <iostream>
using namespace std;

int main()
{
 int a = 0;
 while (a < 100)
 {
  a += 1;
  cout << a << endl;
 }

 system("pause");

 return 0;
}
```

```c++
#include <iostream>
using namespace std;

int main()
{
 int target = rand()%10000;
 // rand()%a 用于生成随机数 %之后的数字为其范围(不可取) [0,a)
 int answer = 0;
 while (answer = target)
 {
  cin >> answer;
  if (answer > target)
  {
   cout << "猜大了" << endl;
  }
  else if (answer < target)
  {
   cout << "猜小了" << endl;
  }
  else
  {
   break;
  }

 }
 cout << "恭喜你,猜对了" << endl;
 system("pause");

 return 0;
}

```

##### do...while 循环语句

作用:满足循环条件,执行循环语句  
语法:`do{循环语句}while(循环条件)`  
注意:与 while 的区别在于 do...while 会先执行一次循环语句,再判断循环条件  
**开始 > 执行语句 > 判断循环条件 > ...**

```c++
#include <iostream>
using namespace std;

int main()
{
 int a = 0;
 do
 {
  cout << a << endl;
  a++;
 }
 while (a < 100);
 system("pause");

 return 0;
}
```

实例:水仙花数

```c++
#include <iostream>
using namespace std;

int main()
{
 int t = 100;
 while (t < 1000)
 {
  int a = t / 100;
  int b = t / 10 - a * 10;
  int c = t - a * 100 - b * 10;
  // 小数舍弃
  if (a * a * a + b * b * b + c * c * c == t)
  {
   cout << t << endl;

  }
  t++;
 }
 system("pause");

 return 0;
}
```

##### for 循环语句

作用:满足循环条件,执行循环语句  
语法:`for(起始表达式;条件表达式;末尾循环体){循环语句;}`

- 起始表达式:不参与循环体,用于循环变量的初始化,只执行一次
- 条件表达式:终止条件,用于判断
- 末尾循环体:每一次循环之后要做的操作(语句,也可以写在循环体中)
- 起始条件和末尾循环体可以不在 for 之后出现,但是分号要保留

> while,do...while,for 都是开发中比较常用的循环语句,for 循环结构比较清晰,更加常用  
> 实例:

```c++
#include<iostream>
using namespace std;

int main()
{
 for(int i = 0; i < 10; i++>)//i有生命周期,在循环外就被销毁了
 {
  cout<< i << endl;
 }

 system("pause");

 return 0;
}
```

实例:敲桌子

```c++
#include <iostream>
using namespace std;

int main()
{
 for (int i = 1; i <= 100; i++ )
 {
  if (i % 7 == 0)
  {
   cout << "敲桌子" << endl;
  }
  else if (i / 100 == 7)
  {
   cout << "敲桌子" << endl;
  }
  else if (i / 10 - i / 100 == 7)
  {
   cout << "敲桌子" << endl;
  }
  else if (i % 10 == 7)
  {
   cout << "敲桌子" << endl;
  }
  else
  {
   cout << i << endl;
  }
 }

 system("pause");

 return 0;
}
```

```c++
// 利用逻辑运算符
#include <iostream>
using namespace std;

int main()
{
 for (int i = 1; i <= 100; i++ )
 // 这里的i只在他所属的语句块中有效,在语句块外无法访问
 {
  if (i % 7 == 0 ||i / 100 == 7||i / 10 - i / 100 == 7||i % 10 == 7)
  {
   cout << "敲桌子" << endl;
  }
  else
  {
   cout << i << endl;
  }
 }

 system("pause");

 return 0;
}
```

##### 嵌套循环

作用:在循环中再嵌套一层循环,用于解决一些实际问题  
实例(打印星星点阵图):

```c++
#include <iostream>
using namespace std;

int main()
{
 int y = 0;
 while (y <= 10)
 {
  y++;
  int x = 0;
  while (x <= 10)
  {
   x++;
   cout << "* " ;
  }
  cout <<endl;
  // endl是换行操作,并非必须的
 }
 system("pause");

 return 0;
}
```

实例(乘法口诀表):

```c++
#include <iostream>
using namespace std;

int main()
{
 for (int y = 1; y < 10; y++)
 {
  for (int x = 1; x <= y; x++)
  {
   cout << y << '*' << x << '=' << y * x << "\t";
  }
  cout << endl;
 }

 system("pause");

 return 0;
}
```

#### 跳转语句

##### break 语句

作用:用于跳出选择结构或者循环结构  
break 的使用时机:

- 出现在 switch 语句中,作用是终止 case 并跳出 switch(**必要**)
- 出现在循环语句中,作用是跳出当前的循环语句
- 出现在嵌套循环中,跳出最近的内层循环语句

##### continue 语句

作用:在循环语句中,跳过本次循环余下尚未执行的语句,继续执行下一次循环  
实例:

```c++
#include <iostream>
using namespace std;

int main()
{
 for(int i = 1;i < 100;i++)
 {
  if (i % 2 == 0)
  {
   continue;
  }
  cout << i << endl;
 }
 system("pause");

 return 0;
}
```

##### goto 语句

作用:可以无条件跳转语句  
语法:`goto 标记;`  
解释:如果标记的名称存在,执行到 goto 语句时.会跳转到标记的位置(太过于强大,使得程序结构混乱)  
实例:

```c++
#include <iostream>
using namespace std;

int main()
{
 b:
 //这是标记
 cout << "这是起点" << endl;
 for(int i = 1;i < 100;i++)
 {
  if (i % 2 == 0)
  {
   continue;
  }
  cout << i << endl;
 }
 goto b;
 // 这是goto语句
 system("pause");

 return 0;
}
```

## 函数和模块化编程

### 函数的定义,调用和声明

#### 全局变量与局部变量

全局变量在主函数之前定义,作用范围是代码全体  
局部变量在代码块或者函数的中生效  
当局部变量和全局变量命名冲突时,局部变量会屏蔽全局变量

#### 定义

作用:将一段经常使用的代码封装起来,减少重复代码

> 一个较大的程序,一般分为若干个程序块,每个模块实现特定的功能  
> 定义一个函数函数的五个步骤:

- 返回值类型;一个函数可以返回一个值,在函数定义之中
- 函数名;给这个函数起个名字
- 参数表列;使用该函数的时候,传入的数据
- 函数体语句;函数内要执行的语句
- return 表达式;和返回值类型挂钩,函数执行完后,返回相应的数据
- **定义要在主函数之前**
- 函数如果不需要返回值,则声明返回值类型的时候写 void(返回语句可写可不写)
  具体的语法结构:

```c++
#include<iostream>
using namespace std;
 int add(int num1,int num2) //返回值类型 函数名(参数表列)
 {
  int sum = num1 + num2;//语句体
  return sum;//return表达式
 }
int main()
{

 int a = 4;
 int b = 5;
 int sum = add(a,b);
 cout<<sum<<endl;
 return 0;
}
```

四种函数常见形式

- 无参无返
- 有参无返
- 无参有返
- 有参有返

#### 调用

函数的调用

- 功能:使用定义好的函数
- 语法:`函数名 (参数)`
- 附加知识点:形式参数,实际参数,实际参数的值会传递给形式参数

#### 声明

作用:提前告诉编译器函数的名称(函数存在)及如何调用函数.函数的主体可以单独定义(写在 main 函数的后面)

- 函数的声明可以多次,但是函数的定义只能有一次
- 函数的声明只有函数的返回值类型,函数名和参数列表
  `int max(int a,int b);`

#### 值传递 地址传递

- 在函数调用的时候实参将数值传入形参
- 值传递时,如果形参发生任何改变,并不会影响实参
- 在函数体内的操作都是针对于形式参数(在函数调用的时候被创建,占有与实际参数不相同的内存空间)而言的,对形式参数的操作并不会影响实际参数(除非声明全局变量)
- 利用指针作为函数的参数可以修改实参的值(**地址传递**)
- 利用地址传递可以减少内存开销,不会产生新的副本

### 函数重载

### 函数模板

#### 函数的分文件编写

作用:让代码结构更加清晰  
函数分文件编写一般有四个步骤

- 创建后缀名为.h 的头文件
- 创建后缀名为.cpp 的源文件
- 在头文件中写函数的声明
- 在源文件中写函数的定义
- 在源文件中写`#include"头文件名"`,这里使用""是因为这是自定义的头文件,这个步骤用于关联头文件和源文件
- 在头文件中写  
  `#include<iostream>`
  `using namespace std;`这样才可以使用各种语法内容
- 使用的时候只需要包含头文件即可

### 函数的递归调用

每一次调用开辟一个新的内存空间,并且保留  
直到到最后一层结束释放内存返回值再逐层释放内存空间

- 优点
- 思路简单
- 缺点
- 占用内存很大

### 命名空间

### 模块化编程的基本思想

## 数组和字符串

### 数组的定义和使用

解释:一个集合,里面存放了相同类型的数据元素  
特点:

- 数组中的每个数据元素都是**相同的数据类型**
- 数组是由连续的内存位置组成的
  一维数组的三种定义方式:
- `数据类型 数组名[数组长度];`
- `数据类型 数组名[数组长度] = {值1,值2,...};`
- `数据类型 数组名[] = {值1,值2,...};`.
- 数组名命名规范与变量名相同,但不要与变量重名
- 索引从 0 开始
- **如果在初始化的时候没有给所有的数组位置赋值,其余位置会自动填充为 0**
- 数组必须有初始长度
- **数组的长度不能为变量,即必须是固定的常数**

> 数组中的数据有先后顺序,可以通过其下标来访问(索引),从 0 起始  
> 当超出索引范围访问数组中的变量时，C++的行为是未定义的。这意味着编译器不会对这种情况进行任何错误检查或处理，程序可能会出现意想不到的结果。  
> 当超出数组索引范围的时候，可能会访问到数组之外的内存区域，这可能会引起程序崩溃、数据的不明确或不正确的行为。有时候，超出索引范围访问数组中的变量可能会导致程序执行结果的随机性。因此，应该在访问数组元素之前，先检查数组下标的合法性，确保不超出数组的索引范围，以避免这种情况的发生。

```c++
#include <iostream>
using namespace std;

int main()
{
 int list[10];
 list[0] = 1;
 list[1] = 1;
 int list1[] = { 0,1,2,3,4,5 };
 cout << list1[3] << endl;
 int list2[6] = { 0,1,2,3,4,5 };
 cout << list2[3] << endl;

 system("pause");

 return 0;
}
```

一维数组数组名的用处

- 可以统计整个数组在内存中的长度`sizeof()`
- 可以获取数组在内存中的首地址`cout<<数组名<<endl;`,得到地址编号
- 要使用取址符号`&`,若要输出数组中一个元素的地址
- 数组名是常量,不可以进行赋值操作(指向了一个首地址)

> 可以用数组内存大小和元素内存大小的比值确定数据中的最后一个数据的下标

### 多维数组

#### 二维数组的定义方式

- `数据类型 数组名[行数][列数];`
- `数据类型 数组名[行数][列数] = {{数据1,数据2},{数据3,数据4}};`
- `数据类型 数组名[行数][列数] = {数据1,数据2,数据3,数据4};`
- `数据类型 数组名[][列数] = {数据1,数据2,数据3,数据4};`
- 以上四种定义方式,利用第二种更加直观,提高代码的可读性
- 访问需要使用两组索引

```c++
int main()
{
 //二维数组的定义方式1
 int arr[2][3];
 arr[0][0] = 1;
 arr[0][1] = 2;
 arr[0][2] = 3;
 arr[1][0] = 4;
 arr[1][1] = 5;
 arr[1][2] = 6;
 cout<<arr[0][0]<<endl;
 cout<<arr[0][1]<<endl;
 cout<<arr[0][2]<<endl;
 cout<<arr[1][0]<<endl;
 cout<<arr[1][1]<<endl;
 cout<<arr[1][2]<<endl;
 return 0;
}
int main()
{
 //二维数组的定义方式2
 int arr[2][3] =
 {
  {1,2,3},
  {4,5,6}
 };
 cout<<arr[0][0]<<endl;
 cout<<arr[0][1]<<endl;
 cout<<arr[0][2]<<endl;
 cout<<arr[1][0]<<endl;
 cout<<arr[1][1]<<endl;
 cout<<arr[1][2]<<endl;
 return 0;
}
int main()
{
 //二维数组的定义方式3
 int arr[2][3] =
 {1,2,3,4,5,6}; //程序自己划分结构,第一行,第二行...
 cout<<arr[0][0]<<endl;
 cout<<arr[0][1]<<endl;
 cout<<arr[0][2]<<endl;
 cout<<arr[1][0]<<endl;
 cout<<arr[1][1]<<endl;
 cout<<arr[1][2]<<endl;
 return 0;
}
int main()
{
 //二维数组的定义方式4
 int arr[][3] =
 {1,2,3,4,5,6}; //自动划分行数,不可省略列数
 cout<<arr[0][0]<<endl;
 cout<<arr[0][1]<<endl;
 cout<<arr[0][2]<<endl;
 cout<<arr[1][0]<<endl;
 cout<<arr[1][1]<<endl;
 cout<<arr[1][2]<<endl;
 return 0;
}
```

#### 二维数组数组名

- 查看二维数组所占内存空间
- 单独使用行号和列号访问的是一行或者一列数据(可以用于计算行数和列数)
- 获取二维数组首地址(直接输出数组名),和第一行,第一列,第一个的首地址都用一样

### 字符串的基本操作（拼接、查找、替换等）

### 字符串的常见问题和解决方法

## 对象和类

### 结构体

基本概念:结构体属于用户自定义的数据类型(数据类型的集合),允许用户储存不同的数据类型  
定义及使用:

- 语法:`struct 结构体名 {结构体成员列表};`(此时 struct 不可省略)
  通过结构体创建变量的方式有 3 种(在创建变量中关键字 struct 可以省略):
- `struct 结构体名 变量名` ,此后再通过`变量名.属性`访问和修改属性
- `struct 结构体名 变量名 = {成员1值,成员2值...}`
- 定义结构体时顺便创建变量

```c++
#include<iostream>
using namespace std;
#include<string>
int main()
{
 //定义3,顺便定义
 struct Student {
  string name;
  int age;
  int score;
 }s3;
 s3.name = "kk";
 s3.age = 80;
 s3.score = 99;
 //定义1
 struct Student s1={"jack",10,100
 };
 //定义2
 struct Student s2;
 s2.name = "eric";
 s2.age = 25;
 s2.score = 89;
 cout<<s1.name<<s1.age<<s1.score<<endl;
 cout<<s2.name<<s2.age<<s2.score<<endl;
 cout<<s3.name<<s3.age<<s3.score<<endl;
 return 0;
}
```

#### 结构体数组

作用:将自定义的结构体变量放入数组中方便维护  
语法:`struct 结构体名 数组名[元素个数] ={{},{},....}`  
其余用法类似于数组的相关操作

```c++
#include<iostream>
using namespace std;
#include<string>
int main()
{
 struct Student {
  string name;
  int age;
  int score;
 };
 struct Student lst[2] =
 {
  {"jack",10,99},
  {"lucy",9,98}
 };

 return 0;
}
```

#### 结构体指针

作用:通过指针访问结构体中的成员(指针与指针指向的变量类型要相同)

- 利用操作符`->`可以通过结构体指针访问结构体的属性

```c++
#include<iostream>
using namespace std;
#include<string>
int main()
{
 struct Student
 {
  string name;
  int age;
  int score;
 };
 struct Student s1={"jack",10,100
 };
 struct Student *p = &s1;
 cout<<p->score;
}
```

#### 结构体嵌套结构体

作用:结构体中的成员可以是另一个结构体(在使用结构体之前要定义结构体)  
可以在结构体处访问嵌套结构体内的属性

```c++
#include<iostream>
using namespace std;
#include<string>
int main()
{
 struct Student
 {
  string name;
  int age;
  int score;
 };
 struct Teacher
 {
  string name;
  int age;
  struct Student stu;
  } ;

 struct Student s1={"jack",10,100
 };
 struct Student *p = &s1;
 cout<<p->score;
}
```

#### 结构体做函数参数

作用:将结构体作为参数向函数中传递(传递之前要先在函数前定义结构体)  
传递方式有两种:

- 值传递
- 地址传递
- 这两种方式看你想不想修改实参来选择
  示例:

```c++
//值传递
#include<iostream>
using namespace std;
#include<string>
struct Student
{
 string name;
 int age;
 int score;
};
void print1(struct Student stu)
{
 cout<<stu.name<<stu.age<<stu.score<<endl;
}
int main()
{

 struct Student s1={"jack",10,100
 };
 print1(s1);
}
```

```c++
//地址传递
#include<iostream>
using namespace std;
#include<string>
struct Student
{
 string name;
 int age;
 int score;
};
void print1(struct Student *p)
{
 cout<<p->name<<p->age<<p->score<<endl;
}
int main()
{

 struct Student s1={"jack",10,100
 };
 print1(&s1);
}
```

#### 结构体中使用 const

作用:用 const 来防止误操作

```c++
#include<iostream>
using namespace std;
#include<string>
struct Student
{
 string name;
 int age;
 int score;
};
void print1(const struct Student *p)//只允许读取,不允许修改,修改即报错
{
 cout<<p->name<<p->age<<p->score<<endl;
}
int main()
{
 struct Student s1={"jack",10,100};
}
```

#### 结构体案例

### 类的定义和对象的创建

### 成员变量和成员函数

### 构造函数和析构函数

### 静态成员

### 封装、继承和多态的概念

### 类的继承和派生

### 虚函数和纯虚函数

## 异常的处理

### 异常的基本概念

### try-catch 块

### 异常类型的继承关系

### 异常处理的最佳实践

## 文件操作

### 文件的打开和关闭

### 读写文件流

### 文件指针和定位

### 文件的读写模式和格式化输入输出

## 动态内存管理

### new 和 delete 运算符

### 动态数组的分配和释放

### 指针的基本概念和使用方法

指针:可以通过指针间接访问内存

- 内存编号是从 0 开始记录的,一般使用十六进制数表示
- 可以用指针变量保存地址
- 指针变量定义语法:`数据类型 * 变量名;` ;`p = &目标变量`
  使用指针:可以通过**解引用**的方式找到指针指向的内存,语法`*p`(指针前加`*`)

> 解引用之后可以对其指向的变量进行修改读取操作  
> 指针也是一种数据类型,也占用内存空间

- 在 32 位操作系统下,所有指针均占用 4 个字节空间
- 在 64 位操作系统下,所有指针均占用 8 个字节空间

#### 空指针和野指针

空指针:指针变量指向内存中编号为 0 的空间  
用途:初始化指针变量  
注意:空指针指向的内存是不可以访问的  
示例:

```c++
int main()
{
 //指针变量p指向的内存地址位0的内存空间
 int * p = NULL;
 //不可访问
 //内存编号从0到255为系统占用内存,不允许用户访问
}
```

野指针:指针指向非法的内存空间(没有被申请的,无权限的)

#### const 修饰指针

const 修饰指针有三种情况:

- const 修饰指针 --常量指针(指向常量的指针)
- `const int *p = &a;`
- 指针的指向可以修改,指针指向的值不可以修改(即不能直接解引用对内存空间里的值进行修改,但可以指向另一个内存地址,其中含有的值和原内存空间不一定相同)
- 可以通过对原变量直接修改达到相同的效果(而不是用解引用)
- const 修饰常量 --指针常量
- `int * const p = &a;`
- 指针的指向不可以修改,但是指针指向的值可以修改(通过解引用)
- const 既修饰指针,又修饰常量
- `const int * const p = &a;`
- 指针的指向和指针指向的值都不可以修改

#### 指针与数组

利用指针访问数组中的元素  
`int *p = arr`指针指向的是数组的首地址,即第一个元素

- 对于整型的数组,只需要让整型指针`p++`即可指向下一个元素(都是 4 字节)
- 指针存在溢出的情况,要注意限定范围

```c++
#include<iostream>
using namespace std;
int main()
{
 int arr[] = {1,2,3,4,5,6,7,8,9,0};
 int *p =arr;
 for (int i = 0;i<10;i++)
 {
  cout<< *p <<" ";
  p++;
 }
 return 0;
}
```

#### 指针与函数

利用指针作为函数的参数可以修改实参的值(**地址传递**)

```c++
#include<iostream>
using namespace std;
void swap(int *p1,int *p2)
{
 int temp =0;
 temp = *p1;
 *p1 = *p2;
 *p2 = temp;
 //均是通过解引用来操作实参的值
}
int main()
{
 int a = 5;
 int b =10;
 cout<<a<<" "<<b<<endl;
 swap(&a,&b);
 cout<<a<<" "<<b<<endl;
 return 0;
}
```

#### 指针数组函数综合(冒泡排序)

```c++
#include<iostream>
using namespace std;
void swap(int *p1,int *p2)
{
 int temp =0;
 if(*p1>*p2)
 {
  temp = *p1;
  *p1 = *p2;
  *p2 = temp;
 }
}
void print(int *arr,int n)
{
 for (int i = 0;i< n ;i++)
 {
  cout<<arr[i]<<" ";
 }
 cout<<endl;
}
int main()
{
 int arr[] = {10,85,45,74,85,6,2,8899,415,256,215,16,26,46,48,979,445};
 int n = sizeof(arr)/sizeof(arr[0]);
 print(arr,n);
 for (int i = 0;i<n - 1;i++)
 {
  for(int j = 0;j<n- i-1;j++)
  {
   swap(&arr[j],&arr[j+1]);
  }
 }
 print(arr,n);
 return 0;
}
```

### 内存泄漏和悬挂指针的处理

## 标准库

### 标准模板库 (STL) 的基本概念和使用方法

### 容器（vector、list、map 等）

### 算法（排序、搜索等）

## 随机数

`rand()%a` 用于生成随机数,范围为[0,a)

### 随机数种子

添加随机数种子,利用系统当前时间生成随机数,防止每次生成的随机数相同  
`srand((unsigned)time(NULL));`  
该语句需要头文件 ctime  
`#include<ctime>`

## 应试

### 浮点数精度控制

包含头文件`iomanip`
`#include<iomanip>`
输出时使用 setprecision()和 fixed 进行精度控制和补 0  
在使用时，可以提前声明，也可以直接写在输出流中  
函数 setprecision() 控制输出流的输出精度（精度控制采用四舍五入）  
注意：setprecision 单独使用是控制有效位数的，与 fixed 合用才是控制小数点后位数的

```c++
#include <iostream>
#include<iomanip>
using namespace std;
int main()
{
 float a = 3.156516;
 cout<<fixed<<setprecision(3)<<a<<endl;

 system("pause");

 return 0;
}
```

### 幂运算

使用 pow 函数时应遵循一些指导原则:
程序必须包含 cmath 头文件；  
在传递给函数的两个参数中，至少第一个参数应该是 double 类型的，当然也可以两个都是；  
因为 pow 函数返回一个 double 值，所以被赋值的任何变量也应该是 double 类型的;  
`pow(底数(double),指数)`

### ASCII

字符和 ascii 等价,如果将一个字符变量的值赋给一个整型,则完成转换

### STL

#### algorithm

##### sort 函数

```c++
#include<algorithm>
sort (a,a+n,cmp);
//a是可随机访问的容器地址,a+n是结束地址,cmp可选,是自定义比较函数
//return true的放在前面
//定义格式
bool cmp()
{
 return expression;
}
```

##### next_permutation 函数

```c++
next_permutation(a,a+n)
```

#### oi

#### sstream

sstream 库定义了三种类：istringstream、ostringstream 和 stringstream，分别用来进行流的输入、输出和输入输出操作。另外，每个类都有一个对应的宽字符集版本。注意，sstream 使用 string 对象来代替字符数组。这样可以避免缓冲区溢出的危险。而且，传入参数和目标对象的类型被自动推导出来，即使使用了不正确的格式化符也没有危险。

- `istringstream` 用于字符串输入

```c++
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
int main() {
  string good;
  istringstream is(good);
  string temp;
  while(is>>temp)
  {
    continue;
  }

  return 0;
}

```

- `stringstream`stringstream 中存放多个字符串，实现多个字符串拼接,数据类型转换
- 包含方法`stream.str()`将 stringstream 类型转换为 string 类型
- 如果想清空 stringstream，必须使用 `sstream.str("");` 方式；`clear()`方法适用于进行多次数据类型转换的场景。

```c++
//字符串拼接
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
int main() {
    stringstream sstream;

    // 将多个字符串放入 sstream 中
    sstream << "first" << " " << "string,";
    sstream << " second string";
    cout << "strResult is: " << sstream.str() << endl;

    // 清空 sstream
    sstream.str("");
    sstream << "third string";
    cout << "After clear, strResult is: " << sstream.str() << endl;

  return 0;
}

```

```c++
//数据类型转换
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>

using namespace std;

int main()
{
    stringstream sstream;
    string strResult;
    int nValue = 1000;

    // 将int类型的值放入输入流中
    sstream << nValue;
    // 从sstream中抽取前面插入的int类型的值，赋给string类型
    sstream >> strResult;

    cout << "[cout]strResult is: " << strResult << endl;
    printf("[printf]strResult is: %s\n", strResult.c_str());

    return 0;
}

```

```c++
//clear()的使用
//clear() 方法只是重置了stringstream的状态标志，并没有清空数据
#include <sstream>
#include <iostream>

using namespace std;

int main()
{
    stringstream sstream;
    int first, second;

    // 插入字符串
    sstream << "456";
    // 转换为int类型
    sstream >> first;
    cout << first << endl;

    // 在进行多次类型转换前，必须先运行clear()
    sstream.clear();

    // 插入bool值
    sstream << true;
    // 转换为int类型
    sstream >> second;
    cout << second << endl;

    return 0;
}

```

#### iostream

##### cin

- cin
- cin.get() 获取一个字符(包括空格,换行符,制表符)
- cin.get(char) 获取一个字符赋值给 char
- cin.get(ch,10,'\n') 从缓冲区读取 10-1 个字符,指定结束符为\n
- 有返回值
- getchar() 获取一个字符(不跳过任何字符,包括 EOF)
- getline()
- get 和 getline 的区别(包括空格,换行符,制表符)
- getline 遇到终止字符时结束,缓冲区指针移到终止字符之后
- get 遇到终止字符结束,缓冲区指针停在终止字符之前
  -cin.ignore() 忽略缓冲区内的一个字符
- EOF

##### cout

#### iomanip

##### setprecision

#### cmath

##### pow

##### sqrt

#### cstdlib

#### bitset

##### bitset<位数>(要表示的数)

#### string

在C++中，字符串是一种用于存储文本数据的数据类型，用于表示字符序列。C++提供了string类来处理字符串，它位于头文件string中。string类提供了丰富的字符串操作功能，包括创建、访问、修改、搜索、连接等。  

- 初始化字符串(用双引号)
- 获取字符串长度
  - 使用length()函数
  - 使用字符串的size()方法
  - 使用C风格的字符串函数strlen()（需要包含cstring头文件）
- 访问字符串中的元素
  - 索引
  - 迭代器  
- 向字符串插入元素
  - 指定位置
  - 开头
  - 结尾
- 字符串的连接
  - +连接
  - append()函数
- 删除字符串中的元素 erase()函数  
- 修改字符串中的元素
  - 索引
  - replace()
- 查找元素或者字串
  - find()函数
  - rfind()函数
- 截取字串
  - substr()函数
- 判断是否为空 empty()函数
- 字符串的比较
  - ==
  - compare()
- 字符串转数字
  - stoi()整数
  - stof()浮点数
  - stod()双精度浮点数
- 读入一行文本并赋值  getline()函数
- 字符串大小写转换
  - toupper()
  - tolower()

##### string的构造函数的形式  

```c++
string str; //生成空字符串

string s(str);//生成字符串为str的复制品

string s(str, strbegin,strlen);//将字符串str中从下标strbegin开始、长度为strlen的部分作为字符串初值

string s(cstr, char_len);//以C_string类型cstr的前char_len个字符串作为字符串s的初值

string s(num ,c);//生成num个c字符的字符串

string s(str, stridx);//将字符串str中从下标stridx开始到字符串结束的位置作为字符串初值

eg:


    string str1;               //生成空字符串
    string str2("123456789");  //生成"1234456789"的复制品
    string str3("12345", 0, 3);//结果为"123"
    string str4("012345", 5);  //结果为"01234"
    string str5(5, '1');       //结果为"11111"
    string str6(str2, 2);      //结果为"3456789"
```

1. size()和length()：返回string对象的字符个数，他们执行效果相同。

2. max_size()：返回string对象最多包含的字符数，超出会抛出length_error异常

3. capacity()：重新分配内存之前，string对象能包含的最大字符数

**string的字符串比较.**

- C ++字符串支持常见的比较操作符（>,>=,<,<=,==,!=），甚至支持string与C-string的比较(如 str<”hello”)。  
  在使用>,>=,<,<=这些操作符的时候是根据“当前字符特性”将字符按字典顺序进行逐一得 比较。字典排序靠前的字符小，  
  比较的顺序是从前向后比较，遇到不相等的字符就按这个位置上的两个字符的比较结果确定两个字符串的大小(前面减后面)
  同时，string (“aaaa”) <string(aaaaa)
- 当前缀相同的时候,长的字符串被认为是大的
- 另一个功能强大的比较函数是成员函数**compare()**。他支持多参数处理，支持用索引值和长度定位子串来进行比较
  他返回一个整数来表示比较结果，返回值意义如下：
  **0：相等**
  **1：大于**
  **-1：小于 (A的ASCII码是65，a的ASCII码是97)**

**string的插入：push_back() 和 insert().**  

```c++
void  test4()
{
    string s1;

    // 尾插一个字符
    s1.push_back('a');
    s1.push_back('b');
    s1.push_back('c');
    cout<<"s1:"<<s1<<endl; // s1:abc

    // insert(pos,char):在制定的位置pos前插入字符char
    s1.insert(s1.begin(),'1');
    cout<<"s1:"<<s1<<endl; // s1:1abc
}
```

**string拼接字符串：append() & + 操作符.**  

```c++
void test5()
{
    // 方法一：append()
    string s1("abc");
    s1.append("def");
    cout<<"s1:"<<s1<<endl; // s1:abcdef

    // 方法二：+ 操作符
    string s2 = "abc";
    /*s2 += "def";*/
    string s3 = "def";
    s2 += s3.c_str();
    cout<<"s2:"<<s2<<endl; // s2:abcdef
}
```

**string的删除：erase().**  

```c++
void test6()
{
    string s1 = "123456789";


    // s1.erase(s1.begin()+1);              // 结果：13456789
    // s1.erase(s1.begin()+1,s1.end()-2);   // 结果：189
    s1.erase(1,6);                       // 结果：189
    string::iterator iter = s1.begin();
    while( iter != s1.end() )
    {
        cout<<*iter;
        *iter++;
    }
    cout<<endl;
}
```

（1）erase(pos,n); 删除从pos开始的n个字符，比如erase(0,1)就是删除第一个字符

（2）erase(position);删除position处的一个字符(position是个string类型的迭代器)

（3）erase(first,last);删除从first到last之间的字符（first和last都是迭代器）

**迭代器.**  

```c++
/*题目描述
一个字符串的前缀是从该字符串的第一个字符起始的一个子串。例如 "carbon"的字串是: "c", "ca", "car", "carb", "carbo", 和 "carbon"。注意到这里我们不认为空串是子串, 但是每个非空串是它自身的子串. 我们现在希望能用前缀来缩略地表示单词。例如, "carbohydrate" 通常用"carb"来缩略表示. 现在给你一组单词, 要求你找到唯一标识每个单词的最短前缀
在下面的例子中，"carbohydrate" 能被缩略成"carboh", 但是不能被缩略成"carbo" (或其余更短的前缀) 因为已经有一个单词用"carbo"开始
一个精确匹配会覆盖一个前缀匹配，例如，前缀"car"精确匹配单词"car". 因此 "car" 是 "car"的缩略语是没有二义性的 , “car”不会被当成"carriage"或者任何在列表中以"car"开始的单词.

关于输入
输入包括至少2行，至多1000行. 每行包括一个以小写字母组成的单词，单词长度至少是1，至多是20.

关于输出
输出的行数与输入的行数相同。每行输出由相应行输入的单词开始，后面跟着一个空格接下来是相应单词的没有二义性的最短前缀标识符*/
// Submitted by 'Programming Grid' VS Code Extension
/*
这串代码使用了find函数,具体调用方式为s1.find(s2),该函数的返回值为s2字串在S1中出现的第一次位置的索引,如果没有,则返回string::npos
erase函数可以删除指定索引位置之后的元素
*/
#include <iostream>
#include <string>
using namespace std;
int n = 0;
struct Word {
  string self;
  string selfn;
  string hat;
  int length;
};
Word words[1000];
void find() {
  int s = 0;
  int count = 0;
  for (int i = 0; i < n; i++) {
    words[i].hat = words[i].self;
    while (true) {

      s = 0;
      count = 0;
      while ((words[s].self.find(words[i].hat) == string::npos || s == i ||
              words[s].self.find(words[i].hat) != 0) &&
             (s < n)) {
        count++;
        s++;
      }
      if (count == n) {
        words[i].self = words[i].hat;
        words[i].hat.erase(words[i].hat.size() - 1);
      } else {
        break;
      }
    }
  }
}
int main() {
  int i = 0; // i是字符串个数
  while (cin >> words[i].self) {
    words[i].hat = words[i].self;
    words[i].selfn = words[i].self;
    words[i].length = words[i].self.size();
    i++;
  }
  n = i;
  find();
  for (int k = 0; k < n; k++) {
    cout << words[k].selfn << ' ' << words[k].self << endl;
  }
  return 0;
}
```

**string的转化.**  

- stoi： string型变量转换为int型变量
- stol： string型变量转换为long型变量
- stoul：string型变量转换为unsigned long型变量
- stoll： string型变量转换为long long型变量(常用)
- stoull：string型变量转换为unsigned long long型变量
- stof： string型变量转换为float型变量
- stod： string型变量转换为double型变量(常用)
- stold：string型变量转换为long double型变量
