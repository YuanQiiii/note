## 写在前面

- 这是关于 [代码随想录](https://www.programmercarl.com/)的学习笔记,接下来也会不断更新

- 题目都是超链接(蓝色有下划线的字体),可以直接跳转至对应题目(一般都在力扣上)
- 如果有想法,也欢迎在评论区和我讨论,我将与大家一同进步
- 变量的命名习惯:`变量名_变量类型(或者模拟的类型)`,例如`node_stack`

## 第零章 算法性能分析

### 时间复杂度

- 什么是时间复杂度?
  - **时间复杂度是一个函数，它定性描述该算法的运行时间**。

- 什么是大O
  - **大O用来表示上界的**，当用它作为算法的最坏情况运行时间的上界，就是对任意数据输入的运行时间的上界。

- 不同数据规模的差异

  - ![时间复杂度，不同数据规模的差异](./codeNote.assets/20200728191447384-20230310124015324.png)

    - `O(1)常数阶 < O(logn)对数阶 < O(n)线性阶 < O(nlogn)线性对数阶 < O(n^2)平方阶 < O(n^3)立方阶 < O(2^n)指数阶`

      - 当然也要注意常数

  - ![程序超时1](./codeNote.assets/20201208231559175-20230310124325152.png)

- 递归函数的时间复杂度
  - **递归的次数 \* 每次递归中的操作次数**  

- 减少时间复杂度的方式:避免使用多层嵌套循环
- 在递归函数中正确剪枝

### 空间复杂度

- c++的内存管理

注意**固定部分**，和**可变部分**:其中固定部分的内存消耗不会随着代码运行而产生变化,可变部分是会产生变化的

![C++内存空间](./codeNote.assets/20210309165950660.png)

- 更加具体的一种分类方法

  - 栈区(Stack) ：由编译器自动分配释放，存放函数的参数值，局部变量的值等，其操作方式类似于数据结构中的栈。
  
  - 堆区(Heap) ：一般由程序员分配释放，若程序员不释放，程序结束时可能由OS收回(**自主申请的内存没有释放内存泄漏的主要原因**,但是python不需要考虑这个问题,因为虚拟机做了这些事)
  
  - 未初始化数据区(Uninitialized Data)： 存放未初始化的全局变量和静态变量

  - 初始化数据区(Initialized Data)：存放已经初始化的全局变量和静态变量
  
  - 程序代码区(Text)：存放函数体的二进制代码
  
- 数据类型的大小

![C++数据类型的大小](./codeNote.assets/20200804193045440.png)

- 注意图中有两个不一样的地方，为什么64位的指针就占用了8个字节，而32位的指针占用4个字节呢？
  - 1个字节占8个比特，那么4个字节就是32个比特，可存放数据的大小为2^32，也就是4G空间的大小，即：可以寻找4G空间大小的内存地址。
  - 大家现在使用的计算机一般都是64位了，所以编译器也都是64位的。

  - 安装64位的操作系统的计算机内存都已经超过了4G，也就是指针大小如果还是4个字节的话，就已经不能寻址全部的内存地址，所以64位编译器使用8个字节的指针才能寻找所有的内存地址。

  - 注意2^64是一个非常巨大的数，对于寻找地址来说已经足够用了。

- 内存对齐

  - **为什么会有内存对齐？**

    - 平台原因：不是所有的硬件平台都能访问任意内存地址上的任意数据，某些硬件平台只能在某些地址处取某些特定类型的数据，否则抛出硬件异常。为了同一个程序可以在多平台运行，需要内存对齐。

    - 硬件原因：经过内存对齐后，CPU访问内存的速度大大提升。
      一个具体例子

        ![内存对齐](./codeNote.assets/20200804193307347.png)
        ![非内存对齐](./codeNote.assets/20200804193353926.png)

- 好处

  - 访问效率：当数据按照对齐规则存储时，CPU可以更高效地访问内存，减少数据读取的时间开销。
  
  - 缓存性能：现代计算机通常具有多级缓存，缓存以缓存行（cache line）为单位进行数据读取。如果数据没有对齐，可能会跨越多个缓存行，导致额外的缓存读取和写入，影响缓存性能。
  
  - 平台兼容性：不同的计算机体系结构对于内存对齐有不同的要求，正确的内存对齐可以确保代码在不同平台上的可移植性和兼容性。
  
- 缺点
  
  - 内存对齐会使得原本占用小内存的数据占用了更大的内存空间
  
  - 过度的内存对齐可能导致内存空间的浪费。因此，在进行内存对齐时需要权衡对齐带来的性能提升与内存空间的消耗，并根据实际需求做出适当的选择。
  
  - 减少空间复杂度的方式:(一般来说要牺牲空间复杂度来优化时间复杂度)避免新建数组
  
## 第一章 数组

### 理论知识

- 基础
  - 数组是存放在连续内存空间上的相同类型数据的集合
  - 数组可以方便的通过下标索引的方式获取到下标下对应的数据(下标索引其实是地址的一种更方便的使用形式)
  - 数组下标都是从0开始的(在任何语言中通用)
  - 数组内存空间的地址是连续的(不能删除,只能覆盖)  **==>**  删除数组中元素时要移动其他元素

- 进阶用法

  - **二分查找**

  - **双指针**

  - **滑动窗口**

  - **模拟行为**

### 典型例题

#### [704. 二分查找](https://leetcode.cn/problems/binary-search/)

> 给定一个 `n` 个元素有序的（升序）整型数组 `nums` 和一个目标值 `target` ，写一个函数搜索 `nums` 中的 `target`，如果目标值存在返回下标，否则返回 `-1`。
>
>
> **示例 1:**
>
> ```
> 输入: nums = [-1,0,3,5,9,12], target = 9
> 输出: 4
> 解释: 9 出现在 nums 中并且下标为 4
> ```
>
> **示例 2:**
>
> ```
> 输入: nums = [-1,0,3,5,9,12], target = 2
> 输出: -1
> 解释: 2 不存在 nums 中因此返回 -1
> ```
>
>  
>
> **提示：**
>
> 1. 你可以假设 `nums` 中的所有元素是不重复的。
> 2. `n` 将在 `[1, 10000]`之间。
> 3. `nums` 的每个元素都将在 `[-9999, 9999]`之间。

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int middle = left + (right - left) / 2;
            if (target > nums[middle]) {
                left = middle + 1;
            } else if (target < nums[middle]) {
                right = middle - 1;
            } else {
                return middle;
            }
        }
        return -1;
    }
};
```

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size();
        while (left < right) {
            int middle = left + (right - left) >> 1;
            if (target > nums[middle]) {
                left = middle + 1;
            } else if (target < nums[middle]) {
                right = middle;
            } else {
                return middle;
            }
        }
        return -1;
    }
};
```

- `left`,`right`的设置取决于你的**合法区间**的设置
  - 如果是闭区间,使用`int left = 0, right = nums.size() - 1;`
  - 如果是左闭右开区间,使用`int left = 0, right = nums.size();`
- `while`的条件判断取`<`还是`<=`和上面的条件要求一致,如果可以取等即代表使用了**闭区间**
- `>>`和`<<`可能会被忽视,但是后面可以用于**状态压缩**
- 要注意所有的分支都应当返回结果
- 注意在`middle`的计算中出现了**加法**运算,要避免数据超出`int`所表示的范围(c++/c)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            middle = left + (right - left) // 2
            if target > nums[middle]:
                left = middle + 1
            elif target < nums[middle]:
                right = middle - 1
            else:
                return middle
        return -1
```

- `python`要注意缩进
- 因为`python`中数据的隐式转换更加智能,所以注意区分`/`和`//`的区别,防止得出的数据不是自己想要的数据类型
  - `/`是浮点数除法
  - `//`为整数除法(与c++一致)

#### [35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
>
> 请必须使用时间复杂度为 `O(log n)` 的算法。
>
>  
>
> **示例 1:**
>
> ```
> 输入: nums = [1,3,5,6], target = 5
> 输出: 2
> ```
>
> **示例 2:**
>
> ```
> 输入: nums = [1,3,5,6], target = 2
> 输出: 1
> ```
>
> **示例 3:**
>
> ```
> 输入: nums = [1,3,5,6], target = 7
> 输出: 4
> ```
>
>  
>
> **提示:**
>
> - `1 <= nums.length <= 10^4`
> - `-10^4 <= nums[i] <= 10^4`
> - `nums` 为 **无重复元素** 的 **升序** 排列数组
> - `-10^4 <= target <= 10^4`

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int middle = (left + right) >> 1;
            if (target > nums[middle]) { // 目标值在右区间
                left = middle + 1;
            } else if (target < nums[middle]) { // 目标值在左区间
                right = middle - 1;
            } else {
                return middle; // 目标值在中间
            }
        }
        return left; //目标值不在区间之内
        // 在所有元素之前 ==> 0 ==> 一直更新右边界 ==> left
        // 在所有元素之后 ==> num.size() ==> 一直更新左边界 ==> left
        // 在元素集合内部 ==> 更新左右边界,但是不断趋向于目标值,直到left和right翻转 ==> left 
        
    }
};
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            middle = (left + right) // 2
            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
        return left
```

- 和一般的**二分查找**没有太大区别,主要在于没有找到时的返回值上:二分查找会不断靠近目标值,停止的位置便是数据应当插入的位置,`left`在变化中会转移到`right`的后面,恰好是目标位置

- 暴力解法是直接遍历数组,没有什么技术含量

#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

> 给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。
>
> 如果数组中不存在目标值 `target`，返回 `[-1, -1]`。
>
> 你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。
>
> **示例 1：**
>
> ```
> 输入：nums = [5,7,7,8,8,10], target = 8
> 输出：[3,4]
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [5,7,7,8,8,10], target = 6
> 输出：[-1,-1]
> ```
>
> **示例 3：**
>
> ```
> 输入：nums = [], target = 0
> 输出：[-1,-1]
> ```
>
> **提示：**
>
> - `0 <= nums.length <= 105`
> - `-109 <= nums[i] <= 109`
> - `nums` 是一个非递减数组
> - `-109 <= target <= 109`

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int middle, point = -1;
        bool flag = false; // 找到目标数据则为true
        vector<int> ans(2, -1);
        while (left <= right) {
            middle = (right + left) / 2;
            if (target > nums[middle]) {
                left = middle + 1;
            } else if (target < nums[middle]) {
                right = middle - 1;
            } else {
                point = middle;
                flag = true;
                break;
            }
        }
        // 找到之后向左右延伸,这里使用了for循环来寻找,当然也可以使用while循环
        if (flag) {
            for (int i = point; i < nums.size(); i++) {
                if (nums[i] == target) {
                    right = i;
                    continue;
                } else {
                    break;
                }
            }
            ans[1] = right;
            for (int i = point; i >= 0; i--) {
                if (nums[i] == target) {
                    left = i;
                    continue;
                } else {
                    break;
                }
            }
            ans[0] = left;
        }
        return ans;
    }
};
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        length = len(nums)
        left = 0
        right = length - 1
        point = 0
        flag = False
        ans = [-1, -1]
        while left <= right:
            middle = left + (right - left) // 2
            if target > nums[middle]:
                left = middle + 1
            elif target < nums[middle]:
                right = middle - 1
            else:
                ans[0] = ans[1] = middle
                flag = True
                break
        # while循环版
        if flag:
            while ans[1] < length - 1:
                if nums[ans[1] + 1] == target:
                    ans[1] += 1
                else:
                    break
            while ans[0] > 0:
                if nums[ans[0] - 1] == target:
                    ans[0] -= 1
                else:
                    break
        return ans
```

- 此种做法采用**二分查找**的思路找到一个目标值,然后向左右**延伸**(延伸的逻辑可以使用`for`循环或者`while`循环)
- 但如果想要更加**直接**的找到两个边界`leftwall`和`rightwall`,则需要仔细思考循环过程**更新边界值**的逻辑
  - 一个在**二分查找**中普遍的现象是`left`会不断向右,(如果没有在循环内部终止)直至大于`right`,`right`会不断向左,直至小于`left`
  - 则`left`和`right`终止的位置会在目标值的两侧,也就是**未经处理的边界**
  - 一个数组会有以下三种情况
    - 目标值大于或小于全体数据 ==> 对应的一侧`leftwall`或`rightwall`不被更新
    - 目标值在数组中没有出现 ==> `leftwall`或`rightwall`的差值小于1
    - 目标值出现 ==> `leftwall`或`rightwall`一个指向目标片段的前方,一个指向后方(不包含该元素)

```c++
class Solution {
private:
    int searchRightWall(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int rightwall = -2;
        while (left <= right) {
            int middle = left + ((right - left) >> 1);
            if (nums[middle] > target) {
                right = middle - 1;
            } else {
                left = middle + 1;
                rightwall = left;
            }
        }
        return rightwall;
    }
    int searchLeftWall(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int leftwall = -2;
        while (left <= right) {
            int middle = left + ((right - left) >> 1);
            if (nums[middle] >= target) {
                right = middle - 1;
                leftwall = right;
            } else {
                left = middle + 1;
            }
        }
        return leftwall;
    }

public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int leftwall = searchLeftWall(nums, target),
            rightwall = searchRightWall(nums, target);
        if (rightwall == -2 || leftwall == -2) {
            return {-1, -1};
        }
        if (rightwall - leftwall > 1) {
            return {leftwall + 1, rightwall - 1};
        }
        return {-1, -1};
    }
};
```

```python
class Solution:
    def searchLeftWall(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        leftWall = -2
        while left <= right:
            middle = left + (right - left) // 2
            if nums[middle] >= target:
                right = middle - 1
                leftWall = right
            else:
                left = middle + 1
        return leftWall

    def searchRightWall(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        rightWall = -2
        while left <= right:
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle - 1
            else:
                left = middle + 1
                rightWall = left
        return rightWall

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        leftWall = self.searchLeftWall(nums, target)
        rightWall = self.searchRightWall(nums, target)
        if leftWall == -2 or rightWall == -2:
            return [-1, -1]
        if rightWall - leftWall > 1:
            return [leftWall + 1, rightWall - 1]
        return [-1, -1]
```

- 注意使用`self`来调用类内部的**方法** (如果不使用`self`来调用这些方法，那么它们将被视为全局函数，而不是类的方法。这将导致未定义的行为或错误。)
- 在面向对象编程中，`self`是一个约定俗成的名称，用于表示对象自身。在类的方法中，`self`是必须的，它作为第一个参数传递给方法。通过使用`self`来调用方法，可以确保该方法是在对象上调用的。

- `python`可以在函数内部定义函数
  - 这种在函数内定义函数的方式称为**嵌套函数**（nested function）或**内部函数**（inner function）。
  - 嵌套函数的定义方式与普通函数类似，只是它们位于另一个函数的内部。
  - 嵌套函数可以访问外部函数的变量和参数，这是因为它们形成了一个**闭包**（closure）。

#### [27. 移除元素](https://leetcode.cn/problems/remove-element/)

> 给你一个数组 `nums` 和一个值 `val`，你需要 **[原地](https://baike.baidu.com/item/原地算法)** 移除所有数值等于 `val` 的元素，并返回移除后数组的新长度。
>
> 不要使用额外的数组空间，你必须仅使用 `O(1)` 额外空间并 **[原地](https://baike.baidu.com/item/原地算法)修改输入数组**。
>
> 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
>
>  
>
> **说明:**
>
> 为什么返回数值是整数，但输出的答案是数组呢?
>
> 请注意，输入数组是以**「引用」**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
>
> 你可以想象内部操作如下:
>
> ```
> // nums 是以“引用”方式传递的。也就是说，不对实参作任何拷贝
> int len = removeElement(nums, val);
> 
> // 在函数里修改输入数组对于调用者是可见的。
> // 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
> for (int i = 0; i < len; i++) {
>     print(nums[i]);
> }
> ```
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [3,2,2,3], val = 3
> 输出：2, nums = [2,2]
> 解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [0,1,2,2,3,0,4,2], val = 2
> 输出：5, nums = [0,1,3,0,4]
> 解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。
> ```
>
>  
>
> **提示：**
>
> - `0 <= nums.length <= 100`
> - `0 <= nums[i] <= 50`
> - `0 <= val <= 100`

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == val) {
                int move_i = i;
                while (move_i < nums.size() - 1) {
                    nums[move_i] = nums[move_i + 1];
                    move_i++;
                }
                nums.pop_back();
                i = 0;
            }
        }
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == val) {
                int move_i = i;
                while (move_i < nums.size() - 1) {
                    nums[move_i] = nums[move_i + 1];
                    move_i++;
                }
                nums.pop_back();
                i = 0;
            }
        }
        return nums.size();
    }
};
```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        while i < len(nums):
            if nums[i] == val:
                nums.pop(i)
            i += 1
        i = 0
        while i < len(nums):
            if nums[i] == val:
                nums.pop(i)
            i += 1
        i = 0
        while i < len(nums):
            if nums[i] == val:
                nums.pop(i)
            i += 1
        return len(nums)
```

- 使用了一种非常暴力的办法,甚至可以说是愚蠢(在一次删除操作没有达成目标时,再次使用相同操作)
- 本质上是数组性质的深刻理解(数组的元素不能被直接删除,只能通过覆盖重写的方式实现删除的效果)
  - 库函数 **==>** c++的`vector`中,如果使用了`erase`方法,则数组会被重排,`size`也会随之减小(并不是`O(1)`的操作,而是`O(n)`的操作,只是将下面的操作流程封装起来使用)
    - 关于**库函数的使用**:如果可以直接用库函数解决,那么就不要使用库函数,如果库函数只是我们实现算法的一小步,并且我们知道它的实现逻辑和时空复杂度,那么是可以使用的,同时也减少了出错的概率
  - 对于一般的数组而言(没有`vector`的特性),则需要移动后续元素到前面覆盖删除位置完成删除操作以保证数组数据的地址连续性

```c++
// 一种更加简洁的暴力算法
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int length = nums.size();
        for (int i = 0; i < length; i++) {
            if (nums[i] == val) {
                for (int j = i; j < length - 1; j++) {
                    nums[j] = nums[j + 1];
                }
                length--;
                i--;
            }
        }
        return length;
    }
};
```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        length = len(nums)
        i = 0
        while i < length:
            if nums[i] == val:
                for j in range(i, length - 1):
                    nums[j] = nums[j + 1]
                length -= 1
                i -= 1
            i += 1
        return length
```

- 这里引入一种**双指针**的思路(后续有相应的章节)
  - 定义**快慢指针**,将旧数组和新数组关联起来
  - 当**快指针指向的元素**`!=`**要删除元素**时,更新慢指针指向的值为快指针指向的值,**慢指针**向后移动一次
    - 快慢指针只是在数组中存在**要删除元素时**,才会有快慢差异

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int fast = 0, slow = 0;
        for (; fast < nums.size(); fast++) {
            if (nums[fast] != val) {
                nums[slow] = nums[fast];
                slow++;
            }
        }
        return slow; // 最后一跳使得慢指针指向最后一个元素的后面
    }
};
```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        fast = slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

- `python`中还可以使用`list`的`remove(元素)`方法来实现该操作
  - `remove`方法的实现逻辑如下：
    1. 从列表的第一个元素开始遍历，直到找到第一个与`element`相等的元素。
    2. 找到相等的元素后，将其从列表中删除，并将后续元素向前移动，填补被删除元素的位置。
    3. 如果列表中存在多个与`element`相等的元素，`remove`方法只会删除第一个遇到的元素。
    4. 如果列表中不存在与`element`相等的元素，则会抛出`ValueError`异常。

#### [26.删除排序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

> 给你一个 **非严格递增排列** 的数组 `nums` ，请你**[原地](http://baike.baidu.com/item/原地算法)** 删除重复出现的元素，使每个元素 **只出现一次** ，返回删除后数组的新长度。元素的 **相对顺序** 应该保持 **一致** 。然后返回 `nums` 中唯一元素的个数。
>
> 考虑 `nums` 的唯一元素的数量为 `k` ，你需要做以下事情确保你的题解可以被通过：
>
> - 更改数组 `nums` ，使 `nums` 的前 `k` 个元素包含唯一元素，并按照它们最初在 `nums` 中出现的顺序排列。`nums` 的其余元素与 `nums` 的大小不重要。
> - 返回 `k` 。
>
> **判题标准:**
>
> 系统会用下面的代码来测试你的题解:
>
> ```
> int[] nums = [...]; // 输入数组
> int[] expectedNums = [...]; // 长度正确的期望答案
> 
> int k = removeDuplicates(nums); // 调用
> 
> assert k == expectedNums.length;
> for (int i = 0; i < k; i++) {
>     assert nums[i] == expectedNums[i];
> }
> ```
>
> 如果所有断言都通过，那么您的题解将被 **通过**。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [1,1,2]
> 输出：2, nums = [1,2,_]
> 解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [0,0,1,1,1,2,2,3,3,4]
> 输出：5, nums = [0,1,2,3,4]
> 解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
> ```
>
> **提示：**
>
> - `1 <= nums.length <= 3 * 104`
> - `-104 <= nums[i] <= 104`
> - `nums` 已按 **非严格递增** 排列

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int slow = 0, fast = 1;
        for (; fast < nums.size(); fast++) {
            if (nums[fast] != nums[slow]) {
                nums[slow + 1] = nums[fast];
                slow++;
            }
        }
        return slow + 1;
    }
};
```

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        fast = 1
        slow = 0
        for fast in range(1, len(nums)):
            if nums[fast] != nums[slow]:
                nums[slow + 1] = nums[fast]
                slow += 1
        return slow + 1
```

- 通过快慢指针的数值差异来建立新数组
  - 直到`fast`指向与`slow`不相等的值,才会把`++slow`对应的元素更新
  - 注意,`slow`在最后指向了一个不重复的元素,并没有指向新数组的最后一个元素的后面

#### [283.移动零](https://leetcode.cn/problems/move-zeroes/)

> 给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。
>
> **请注意** ，必须在不复制数组的情况下原地对数组进行操作。
>
>  
>
> **示例 1:**
>
> ```
> 输入: nums = [0,1,0,3,12]
> 输出: [1,3,12,0,0]
> ```
>
> **示例 2:**
>
> ```
> 输入: nums = [0]
> 输出: [0]
> ```
>
>  
>
> **提示**:
>
> - `1 <= nums.length <= 104`
> - `-231 <= nums[i] <= 231 - 1`
>
>  
>
> **进阶：**你能尽量减少完成的操作次数吗？

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int fast = 0, slow = 0;
        for (; fast < nums.size(); fast++) {
            if (nums[fast] != 0) {
                nums[slow] = nums[fast];
                slow++;
            }
        }
        for (int j = slow; j < nums.size(); j++) {
            nums[j] = 0;
        }
    }
};
```

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        fast = slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
        for i in range(slow, len(nums)):
            nums[i] = 0
```

- 没有什么特殊的技巧,使用了**双指针**
- 可以使用特殊的**冒泡排序**完成

#### [844.比较含退格的字符串](https://leetcode.cn/problems/backspace-string-compare/)

> 给定 `s` 和 `t` 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 `true` 。`#` 代表退格字符。
>
> **注意：**如果对空文本输入退格字符，文本继续为空。
>
>  
>
> **示例 1：**
>
> ```
> 输入：s = "ab#c", t = "ad#c"
> 输出：true
> 解释：s 和 t 都会变成 "ac"。
> ```
>
> **示例 2：**
>
> ```
> 输入：s = "ab##", t = "c#d#"
> 输出：true
> 解释：s 和 t 都会变成 ""。
> ```
>
> **示例 3：**
>
> ```
> 输入：s = "a#c", t = "b"
> 输出：false
> 解释：s 会变成 "c"，但 t 仍然是 "b"。
> ```
>
>  
>
> **提示：**
>
> - `1 <= s.length, t.length <= 200`
> - `s` 和 `t` 只含有小写字母以及字符 `'#'`

- 栈方法(后续的章节里会展开)

```c++
// 使用栈
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        stack<char> sStack, tStack;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '#') {
                if (!sStack.empty()) {
                    sStack.pop();
                }
            } else {
                sStack.push(s[i]);
            }
        }
        for (int i = 0; i < t.size(); i++) {
            if (t[i] == '#') {
                if (!tStack.empty()) {
                    tStack.pop();
                }
            } else {
                tStack.push(t[i]);
            }
        }
        if (sStack.size() != tStack.size()) {
            return false;
        } else {
            while (!sStack.empty()) {
                if (sStack.top() == tStack.top()) {
                    sStack.pop();
                    tStack.pop();
                    continue;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
};
```

- `python`中可以使用`list`这一数据结构以及相应的方法实现栈
  - `pop(index)`方法在不指定index时默认删除最后一个元素
  - `append(element)`方法可以向列表末尾**追加**一个元素
  - `listname[-1]`可以引用栈顶的元素(即末尾元素)
  - `len()`方法可以用来得到列表的长度(也可以判断列表是否为空)

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        sList = []
        tList = []
        for i in range(len(s)):
            if s[i] == "#":
                if len(sList):
                    sList.pop()
            else:
                sList.append(s[i])
        for i in range(len(t)):
            if t[i] == "#":
                if len(tList):
                    tList.pop()
            else:
                tList.append(t[i])
        if len(sList) != len(tList):
            return False
        else:
            while len(sList):
                if sList[-1] == tList[-1]:
                    sList.pop()
                    tList.pop()
                    continue
                else:
                    return False
        return True
```

- 双指针(不同数组内的快慢指针)![1](./codeNote.assets/1.gif)

```c++
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        int i = s.size() - 1, j = t.size() - 1; // 逆序遍历
        int skipS = 0, skipT = 0; // 未消耗的退格数目
        while (i >= 0 || j >= 0) {
            // 寻找s的一个合法字符
            while (i >= 0) {
                if (s[i] == '#') {
                    skipS++, i--;
                } else if (skipS > 0) {
                    skipS--, i--;
                } else {
                    break;
                }
            } 
            // 寻找t的一个合法字符
            while (j >= 0) {
                if (t[j] == '#') {
                    skipT++, j--;
                } else if (skipT > 0) {
                    skipT--, j--;
                } else {
                    break;
                }
            }
            // 判断是否相等
            if (i >= 0 && j >= 0) {
                if (s[i] != t[j]) {
                    return false;
                }
            } else {
                if (i >= 0 || j >= 0) { //有至少一个指针已经指向了字符串之外(长度不一)
                    return false;
                }
            }
            i--, j--;
        }
        return true;
    }
};
```

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        i = len(s) - 1
        j = len(t) - 1
        skipS = 0
        skipT = 0
        while i >= 0 or j >= 0:
            while i >= 0:
                if s[i] == "#":
                    skipS += 1
                    i -= 1
                elif skipS > 0:
                    skipS -= 1
                    i -= 1
                else:
                    break
            while j >= 0:
                if t[j] == "#":
                    skipT += 1
                    j -= 1
                elif skipT > 0:
                    skipT -= 1
                    j -= 1
                else:
                    break
            if i >= 0 and j >= 0:
                if s[i] != t[j]:
                    return False
            else:
                if i >= 0 or j >= 0:
                    return False
            i -= 1
            j -= 1
        return True
```

#### [977.有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)

> 给你一个按 **非递减顺序** 排序的整数数组 `nums`，返回 **每个数字的平方** 组成的新数组，要求也按 **非递减顺序** 排序。
>
> **示例 1：**
>
> ```
> 输入：nums = [-4,-1,0,3,10]
> 输出：[0,1,9,16,100]
> 解释：平方后，数组变为 [16,1,0,9,100]
> 排序后，数组变为 [0,1,9,16,100]
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [-7,-3,2,3,11]
> 输出：[4,9,9,49,121]
> ```
>
> **提示：**
>
> - `1 <= nums.length <= 104`
> - `-104 <= nums[i] <= 104`
> - `nums` 已按 **非递减顺序** 排序
>
> **进阶：**
>
> - 请你设计时间复杂度为 `O(n)` 的算法解决本问题

- 暴力做法(直接平方+排序)
  - `python`中`list`的`sort()`方法的一些参数
    - `key`：指定一个函数来用作排序的关键字。默认值为`None`，表示使用元素自身进行比较。如果指定了`key`参数，`sort()`方法将使用该函数的返回值进行排序。
    - `reverse`：指定排序顺序。默认值为`False`，表示升序排序。如果将`reverse`设置为`True`，则进行降序排序。
  - `c++`中`algorithm`头文件下`std::sort()`函数的一些参数
    - `first` 和 `last`：指定排序范围的首元素和尾元素的迭代器。排序将从 `first` 开始，直到 `last` 的前一个元素。
      - 注意是**迭代器**,对于`vector`或者`string`排序要使用相应的迭代器;对一般数组排序使用地址即可
    - `comp`（可选）：指定一个自定义的比较函数，用于确定元素的顺序。默认情况下，使用 `<` 运算符进行比较。比较函数应该接受两个参数，并返回一个布尔值，表示第一个参数是否小于第二个参数。

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        for (int i = 0; i < nums.size(); i++) {
            nums[i] = nums[i] * nums[i];
        }
        sort(nums.begin(), nums.end());
        return nums;
    }
};
```

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            nums[i] = nums[i] * nums[i]
        nums.sort()
        return nums
```

- 双指针做法

```c++
// 其实是三个指针
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int leftptr = 0, rightptr = nums.size() - 1;
        int newptr = rightptr;
        vector<int> output(nums.size(), 0);
        while (leftptr <= rightptr) {
            int a = nums[leftptr] * nums[leftptr],
                b = nums[rightptr] * nums[rightptr];
            if (a >= b) {
                output[newptr] = a;
                leftptr++;
            } else {
                output[newptr] = b;
                rightptr--;
            }
            newptr--;
        }
        return output;
    }
};
```

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        output = [0] * len(nums)
        leftptr = 0
        newptr = rightptr = len(nums) - 1
        # 提前计算出来会减少重复的计算
        for i in range(len(nums)):
            nums[i] = nums[i] ** 2
        while leftptr <= rightptr:
            if nums[leftptr] >= nums[rightptr]:
                output[newptr] = nums[leftptr]
                leftptr += 1
            else:
                output[newptr] = nums[rightptr]
                rightptr -= 1
            newptr -= 1
        return output
```

- `python`版本中如果你使用了`output = nums`这样的语句,则`output`只是对`nums`的一个新的**引用**,对于`nums`的修改会反映到`output`上

  - 在 Python 中，变量不直接存储值本身，而是存储对值的引用。这意味着当你创建一个变量时，你实际上是在创建一个指向内存中某个对象的指针。这种行为在处理可变对象（如列表、字典、集合等）时尤其重要，因为如果你有多个引用指向同一个可变对象，通过一个引用对对象的修改会影响到所有引用。

    **引用：**

    当你将一个变量赋值给另一个变量时，你只是在创建新的引用到原始对象。例如：

    ```python
    a = [1, 2, 3] # 创建一个列表 [1, 2, 3]
    b = a         # b 是对同一个列表的新引用
    ```

    在这个例子中，`a` 和 `b` 都引用相同的列表。如果你修改了 `b`（比如 `b.append(4)`），`a` 也会变化，因为它们指向的是同一个对象。

    **创建副本：**

    有时候，你可能不想要两个变量指向同一个对象，而是想要它们各自有自己的独立对象。这时候，你就需要创建一个副本。对于列表，你可以使用多种方法创建副本：

    ```python
    a = [1, 2, 3]    # 创建一个列表 [1, 2, 3]
    b = a[:]          # 使用切片操作创建 a 的一个副本
    c = list(a)       # 使用 list 构造函数创建 a 的一个副本
    d = a.copy()      # 使用列表的 copy 方法创建 a 的一个副本
    ```

    在这个例子中，`b`、`c` 和 `d` 是 `a` 的副本，它们具有相同的内容，但是是独立的对象。现在，如果你修改 `b`、`c` 或 `d`，`a` 将不会受到影响，因为它们不再指向同一个对象。

    对于其他可变数据类型，如字典和集合，你也可以使用相应的 `copy()` 方法或相应的构造函数来创建副本。

    **深拷贝与浅拷贝：**

    当对象中还包含其他对象时（例如，列表中的列表），光复制顶层对象可能不够。创建副本的方法（如切片、`list()` 构造函数、`copy()` 方法）只进行浅拷贝，即只复制对象本身和其中的直接子对象的引用，而不复制子对象本身。

    如果你需要一个完全独立的副本，其中包含的所有子对象也都是独立的副本，你需要进行深拷贝。在 Python 中，你可以使用 `copy` 模块的 `deepcopy()` 函数来实现：

    ```python
    import copy
    a = [[1, 2], [3, 4]]
    b = copy.deepcopy(a)  # 创建 a 的深拷贝
    ```

    现在，`b` 是 `a` 的一个深拷贝，所以你可以独立地修改 `b` 中的子列表，而不会影响 `a`。

#### [209.长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

> 给定一个含有 `n` 个正整数的数组和一个正整数 `target` **。**
>
> 找出该数组中满足其总和大于等于 `target` 的长度最小的 **连续子数组** `[numsl, numsl+1, ..., numsr-1, numsr]` ，并返回其长度**。**如果不存在符合条件的子数组，返回 `0` 。
>
>  
>
> **示例 1：**
>
> ```
> 输入：target = 7, nums = [2,3,1,2,4,3]
> 输出：2
> 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
> ```
>
> **示例 2：**
>
> ```
> 输入：target = 4, nums = [1,4,4]
> 输出：1
> ```
>
> **示例 3：**
>
> ```
> 输入：target = 11, nums = [1,1,1,1,1,1,1,1]
> 输出：0
> ```
>
>  
>
> **提示：**
>
> - `1 <= target <= 109`
> - `1 <= nums.length <= 105`
> - `1 <= nums[i] <= 105`

- 暴力解法,遇到特别大的数据规模会超时
  - 时间复杂度:`O(n^2)`
  - 空间复杂度:`O(1)`

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int length = nums.size();
        bool flag = false;
        for (int i = 0; i < nums.size(); i++) {
            int sum = 0;
            for (int j = i; j < nums.size(); j++) {
                sum += nums[j];
                if (sum >= target) {
                    flag = true;
                    length = min(j - i + 1, length);
                    break;
                }
            }
        }
        if (flag) {
            return length;
        } else {
            return 0;
        }
    }
};
```

- **前缀和**做法,依然超时,但是已经开了一个好头(减去了重复的和计算)
  - 时间复杂度:`O(n^2)`
  - 空间复杂度:`O(n)`

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        vector<unsigned long long int> sNums(n, 0);
        unsigned long long sum{0};
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            sNums[i] = sum;
        }
        if (sum < target) {
            return 0;
        } else {
            int length = n;
            for (int j = n - 1; j >= 0; j--) {
                for (int k = j; k >= 0; k--) {
                    if (sNums[j] - sNums[k] >= target) {
                        length = min(length, j - k);
                    }
                }
            }
            return length;
        }
    }
};
```

- **滑动窗口(双指针)**+前缀和(可以被优化,这里便于理解)
  - 时间复杂度:`O(n)`
  - 空间复杂度:`O(n)`

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        vector<unsigned long long int> sNums(n + 1, 0);
        unsigned long long sum{0};
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            sNums[i + 1] = sum;
        }
        if (sum < target) {
            return 0;
        }
        int lPtr = 0, rPtr = 1;
        // rPtr是窗口的右端点,主动更新
        // lPtr是窗口的左端点,被动更新
        int ans = n;
        bool flag = false;
        for (; rPtr <= n; rPtr++) {
            while (sNums[rPtr] - sNums[lPtr] >= target) {
                flag = true;
                lPtr++;
                ans = min(rPtr - lPtr + 1, ans);
            }
        }
        if (flag) {
            return ans;
        } else {
            return 0;
        }
    }
};
```

```python
# 优化了连续子序列和的计算方式(有点像"忒修斯之船")
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        ans = len(nums)
        lPtr = 0
        sNum = 0
        flag = False
        for rPtr in range(0, len(nums)):
            sNum += nums[rPtr]
            while sNum >= target:
                flag = True
                ans = min(rPtr - lPtr + 1, ans)
                sNum -= nums[lPtr]
                lPtr += 1
        if flag:
            return ans
        else:
            return 0
```

#### [904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)

>你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 `fruits` 表示，其中 `fruits[i]` 是第 `i` 棵树上的水果 **种类** 。
>
>你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：
>
>- 你只有 **两个** 篮子，并且每个篮子只能装 **单一类型** 的水果。每个篮子能够装的水果总量没有限制。
>- 你可以选择任意一棵树开始采摘，你必须从 **每棵** 树（包括开始采摘的树）上 **恰好摘一个水果** 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
>- 一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
>
>给你一个整数数组 `fruits` ，返回你可以收集的水果的 **最大** 数目。
>
>**示例 1：**
>
>```
>输入：fruits = [1,2,1]
>输出：3
>解释：可以采摘全部 3 棵树。
>```
>
>**示例 2：**
>
>```
>输入：fruits = [0,1,2,2]
>输出：3
>解释：可以采摘 [1,2,2] 这三棵树。
>如果从第一棵树开始采摘，则只能采摘 [0,1] 这两棵树。
>```
>
>**示例 3：**
>
>```
>输入：fruits = [1,2,3,2,2]
>输出：4
>解释：可以采摘 [2,3,2,2] 这四棵树。
>如果从第一棵树开始采摘，则只能采摘 [1,2] 这两棵树。
>```
>
>**示例 4：**
>
>```
>输入：fruits = [3,3,3,1,2,1,1,2,3,3,4]
>输出：5
>解释：可以采摘 [1,2,1,1,2] 这五棵树。
>```
>
>
>
>**提示：**
>
>- `1 <= fruits.length <= 105`
>- `0 <= fruits[i] < fruits.length`

```c++
// 还没写......
```

#### [76.最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

> 给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。
>
>  
>
> **注意：**
>
> - 对于 `t` 中重复字符，我们寻找的子字符串中该字符数量必须不少于 `t` 中该字符数量。
> - 如果 `s` 中存在这样的子串，我们保证它是唯一的答案。
>
>  
>
> **示例 1：**
>
> ```
> 输入：s = "ADOBECODEBANC", t = "ABC"
> 输出："BANC"
> 解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
> ```
>
> **示例 2：**
>
> ```
> 输入：s = "a", t = "a"
> 输出："a"
> 解释：整个字符串 s 是最小覆盖子串。
> ```
>
> **示例 3:**
>
> ```
> 输入: s = "a", t = "aa"
> 输出: ""
> 解释: t 中两个字符 'a' 均应包含在 s 的子串中，
> 因此没有符合条件的子字符串，返回空字符串。
> ```
>
>  
>
> **提示：**
>
> - `m == s.length`
> - `n == t.length`
> - `1 <= m, n <= 105`
> - `s` 和 `t` 由英文字母组成
>
>  
>
> **进阶：**你能设计一个在 `o(m+n)` 时间内解决此问题的算法吗？

```c++
// 还没写......
```

#### [59. 螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)

> 给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/spiraln.jpg)
>
> ```
> 输入：n = 3
> 输出：[[1,2,3],[8,9,4],[7,6,5]]
> ```
>
> **示例 2：**
>
> ```
> 输入：n = 1
> 输出：[[1]]
> ```
>
>  
>
> **提示：**
>
> - `1 <= n <= 20`

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> matrix(n, vector<int>(n, 0));
        int di[4] = {0, 1, 0, -1}, dj[4] = {1, 0, -1, 0}, ptr = 1, i = 0, j = 0, direction = 0;
        while (ptr <= n * n) {
            matrix[i][j] = ptr;
            int ni = i + di[direction], nj = j + dj[direction];
            if (ni < 0 || nj < 0 || ni >= n || nj >= n || matrix[ni][nj] != 0) { // 模拟,检查是否越界
                direction = (direction + 1) % 4;
            }
            i += di[direction];
            j += dj[direction];
            ptr++;
        }
        return matrix;
    }
};
```

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> matrix(n, vector<int>(n, 0));
        vector<vector<int>> doneMatrix(n + 2, vector<int>(n + 2, 1)); // 使用了一个记录矩阵
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                doneMatrix[i][j] = 0;
            }
        }
        int di[4] = {0, 1, 0, -1}, dj[4] = {1, 0, -1, 0}, ptr = 1, i = 0, j = 0,
            direction = 0;
        while (ptr <= n * n) {
            matrix[i][j] = ptr;
            doneMatrix[i + 1][j + 1] = 1; // 整体平移一波
            if (doneMatrix[i + di[direction] + 1][j + dj[direction] + 1]) {
                direction++;
                direction = direction % 4;
            } 
            i += di[direction];
            j += dj[direction];
            ptr++;
        }
        return matrix;
    }
};
```

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        ptr = 1
        matrix = [[0 for _ in range(n)] for _ in range(n)] # 列表推导式
        di = [0, 1, 0, -1]
        dj = [1, 0, -1, 0]
        direction = 0
        i = j = 0
        while ptr <= n**2:
            matrix[i][j] = ptr
            pi = i + di[direction]
            pj = j + dj[direction]
            if not (pi >= 0 and pi < n and pj >= 0 and pj < n and matrix[pi][pj] == 0):
                direction = (direction + 1) % 4
            i = i + di[direction]
            j = j + dj[direction]
            ptr += 1
        return matrix

```

- Python中`list`的生成可以使用列表推导式

  - 列表推导式（List Comprehension）是一种在Python中创建和转换列表的简洁方式。它允许我们使用一种紧凑的语法来生成新的列表，而不需要使用显式的循环。

    列表推导式的一般形式是：

    ```python
    targetList = [expression for item in iterable if condition]
    ```

    其中：

    - `expression` 是对每个元素进行操作的表达式或函数。
    - `item` 是可迭代对象中的每个元素。
    - `iterable` 是一个可迭代对象，如列表、字符串、元组等。
    - `condition` 是一个可选的条件，用于过滤元素。

    列表推导式的工作流程如下：

    1. 遍历可迭代对象中的每个元素。
    2. 对每个元素应用表达式或函数，并生成一个新的元素。
    3. 如果提供了条件，则只包含满足条件的元素。
    4. 返回一个新的列表，其中包含了生成的元素。

    以下是一些示例，演示了列表推导式的用法：

    ```python
    python复制# 生成一个包含1到5的平方的列表
    squares = [x**2 for x in range(1, 6)]
    # 输出: [1, 4, 9, 16, 25]
    
    # 生成一个仅包含偶数的列表
    evens = [x for x in range(1, 11) if x % 2 == 0]
    # 输出: [2, 4, 6, 8, 10]
    
    # 生成一个字符串中每个字符的ASCII码值的列表
    ascii_values = [ord(char) for char in "Hello"]
    # 输出: [72, 101, 108, 108, 111]
    
    # 生成一个包含两个列表对应位置元素乘积的列表
    nums1 = [1, 2, 3]
    nums2 = [4, 5, 6]
    product = [x * y for x, y in zip(nums1, nums2)]
    # 输出: [4, 10, 18]
    ```

    通过使用列表推导式，可以以一种简洁而优雅的方式生成新的列表，减少了代码量，并提高了可读性。

  - 可以通过**列表推导式**生成高维数组(矩阵,张量等)

- `_`的使用'

  - 1. 循环中的占位符：当我们在循环中不需要使用循环变量时，可以使用下划线 `_` 来表示占位符。

    ```python
    for _ in range(5):
        print("Hello")
    ```

    2. 忽略函数返回值：当函数返回多个值时，如果我们只关心其中的一部分，可以使用下划线 `_` 来忽略不需要的返回值。

    ```python
    _, result = some_function()  # 忽略第一个返回值
    ```

    3. 迭代器中的占位符：当我们使用迭代器进行循环遍历时，可以使用下划线 `_` 作为占位符来忽略迭代器中的某些值。

    ```python
    for _, value in enumerate(some_list):  # 忽略索引值
        print(value)
    ```

    4. 无用变量的占位符：当我们定义了某个变量，但后续没有使用时，可以使用下划线 `_` 来表示这个变量没有实际用途。

    ```python
    def some_function():
        result = calculate_result()
        _ = save_result_to_database(result)  # 没有使用返回值
    ```

#### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

> 给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/spiral1.jpg)
>
> ```
> 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
> 输出：[1,2,3,6,9,8,7,4,5]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/spiral.jpg)
>
> ```
> 输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
> 输出：[1,2,3,4,8,12,11,10,9,5,6,7]
> ```
>
>  
>
> **提示：**
>
> - `m == matrix.length`
> - `n == matrix[i].length`
> - `1 <= m, n <= 10`
> - `-100 <= matrix[i][j] <= 100`

```c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int i = 0, j = 0;
        int n = matrix.size();
        int m = matrix[0].size();
        vector<int> ans;
        int di[4] = {0, 1, 0, -1}, dj[4] = {1, 0, -1, 0}, direction = 0;
        int ptr = 1;
        while (ptr <= m * n) {
            ans.push_back(matrix[i][j]);
            matrix[i][j] = 101; // 注意题目中的数据范围,这里取了个巧
            int pi = i + di[direction], pj = j + dj[direction];
            if (!(pi >= 0 && pi < n && pj >= 0 && pj < m &&
                  matrix[pi][pj] != 101)) {
                direction = (direction + 1) % 4;
            }
            i = i + di[direction];
            j = j + dj[direction];
            ptr++;
        }
        return ans;
    }
};
```

- `vector`的高维数组,其实是**向量**嵌套,可以使用`arr[0].size()`来访问每一行的向量个数

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        n = len(matrix)
        m = len(matrix[0])
        ans = []
        i = j = 0
        di = [0, 1, 0, -1]
        dj = [1, 0, -1, 0]
        ptr = 1
        direction = 0
        while ptr <= m * n:
            ans.append(matrix[i][j])
            matrix[i][j] = 101
            pi = i + di[direction]
            pj = j + dj[direction]
            if not (
                pi >= 0 and pi < n and pj >= 0 and pj < m and matrix[pi][pj] != 101
            ):
                direction = (direction + 1) % 4
            i = i + di[direction]
            j = j + dj[direction]
            ptr += 1
        return ans
```

- 注意不要混淆**C++/C**和**Python**的**数组**表示方式和**逻辑连接词**

## 第二章 链表

### 理论知识

![链表1](./codeNote.assets/20200806194529815.png)

![链表2](./codeNote.assets/20200806194559317.png)

![链表4](./codeNote.assets/20200806194629603.png)

- 链表是一种通过指针串联在一起的线性结构，每一个节点由两部分组成，一个是**数据域**一个是**指针域**（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）。

- 链表基本分为**三类**
  - 单链表 ==> 单向查询
    - 指针域只能指向节点的下一个节点

    - 一个例子

      - ```c++
        // 单链表
        struct ListNode {
            int val;  // 节点上存储的元素
            ListNode *next;  // 指向下一个节点的指针
            ListNode(int x) : val(x), next(NULL) {}  // 节点的构造函数
            // 调用该函数会创建一个值为x,下一节点为空的链表节点,如果大括号中有代码,则在调用该函数的时候会执行相应语句
        };
        ```

      - ```python
        class ListNode:
            def __init__(self, val, next=None):
                self.val = val
                self.next = next

  - 双链表 ==> 双向查询
    - 每一个节点有两个指针域，一个指向下一个节点，一个指向上一个节点

  - 循环链表 ==> 可以用来解决**约瑟夫环**问题
    - 双链表的一种特殊形式,**首尾相连**
  
- 链表不同于数组,在空间上并不是连续的,通过**指针**串联在一起,一般形式为结构体`struct`

- 在没有定义**构造函数**的时候,C++会默认生成一个构造函数,但是并不会初始化任何成员变量(保持成员变量的初始值)

- 链表的操作
  - **删除节点**(删除指定节点,并将剩余节点连接起来)![链表-删除节点](./codeNote.assets/20200806195114541-20230310121459257.png)
  - **添加节点**(断开对应位置节点的连接,将新节点插入并与其他节点建立连接)![链表-添加节点](./codeNote.assets/20200806195134331-20230310121503147.png)

- 性能分析(链表/数组)
  - ![链表-链表与数据性能对比](./codeNote.assets/20200806195200276.png)

- 成员操作符
  - `.`（点操作符）：点操作符用于直接访问对象的成员变量或成员函数。它适用于对象本身（非指针）的访问，通过对象名后面跟上成员名来访问相应的成员。例如，`object.member`。
  - `->`（箭头操作符）：箭头操作符用于通过指针访问对象的成员变量或成员函数。它适用于指向对象的指针的访问，通过指针变量后面跟上箭头`->`和成员名来访问相应的成员。例如，`pointer->member`。
  
- **虚拟头节点**(哨兵节点)`dummy`

  - 方便进行**增删改**的操作

### 典型例题

#### [203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)

> 给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。
>
> **示例 1：**
>
> ![img](./codeNote.assets/removelinked-list.jpg)
>
> ```
> 输入：head = [1,2,6,3,4,5,6], val = 6
> 输出：[1,2,3,4,5]
> ```
>
> **示例 2：**
>
> ```
> 输入：head = [], val = 1
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：head = [7,7,7,7], val = 7
> 输出：[]
> ```
>
> ```c++
> // Definition for singly-linked list.
> struct ListNode {
>     int val;
>     ListNode* next;
>     ListNode() : val(0), next(nullptr) {}
>     ListNode(int x) : val(x), next(nullptr) {}
>     ListNode(int x, ListNode* next) : val(x), next(next) {}
> };
> ```
>
> **提示：**
>
> - 列表中的节点数目在范围 `[0, 104]` 内
> - `1 <= Node.val <= 50`
> - `0 <= val <= 50`

```c++
// 分类讨论
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* temp = head;
        // 目标值在头节点上
        while (head != nullptr && head->val == val) {
            head = head->next;
            delete temp;
            temp = head;
        }
        // 目标值在非头节点上
        while (temp != nullptr && temp->next != nullptr) {
            ListNode* deletePtr = head; // 释放内存
            if (temp->next->val == val) {
                deletePtr = temp->next;
                temp->next = temp->next->next;
                delete deletePtr;
            } else {
                temp = temp->next;
            }
        }
        return head;
    }
};
```

```c++
// 虚拟头节点
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* dummyHead = new ListNode(0, head); //对象的实例化
        ListNode* deletePtr = nullptr;
        ListNode* temp = dummyHead;
        while (temp->next != nullptr) {
            if (temp->next->val == val) {
                deletePtr = temp->next;
                temp->next = temp->next->next;
                delete deletePtr;
            } else {
                temp = temp->next;
            }
        }
        return dummyHead->next;
    }
};
```

- 使用**虚拟头节点(`dummyHead`)**时,可以统一节点的增加和删除操作
- 注意删除节点之后要**释放内存**,使用`delete`关键字

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummyHead = ListNode(0, head)
        temp = dummyHead
        while temp.next != None:
            if temp.next.val == val:
                temp.next = temp.next.next
            else:
                temp = temp.next
        return dummyHead.next
```

- Python中没有**指针**的概念,但是可以把**引用**看作是一种**智能指针**,对于对象的直接赋值并不会创建变量的副本,而是生成了一个指向该对象的引用
- 成员操作符只有`.`,Python中链表属于**类(class)**的一种
- 不需要释放内存,Python有智能的内存释放机制

#### [707. 设计链表](https://leetcode.cn/problems/design-linked-list/)

> 你可以选择使用单链表或者双链表，设计并实现自己的链表。
>
> 单链表中的节点应该具备两个属性：`val` 和 `next` 。`val` 是当前节点的值，`next` 是指向下一个节点的指针/引用。
>
> 如果是双向链表，则还需要属性 `prev` 以指示链表中的上一个节点。假设链表中的所有节点下标从 **0** 开始。
>
> 实现 `MyLinkedList` 类：
>
> - `MyLinkedList()` 初始化 `MyLinkedList` 对象。
> - `int get(int index)` 获取链表中下标为 `index` 的节点的值。如果下标无效，则返回 `-1` 。
> - `void addAtHead(int val)` 将一个值为 `val` 的节点插入到链表中第一个元素之前。在插入完成后，新节点会成为链表的第一个节点。
> - `void addAtTail(int val)` 将一个值为 `val` 的节点追加到链表中作为链表的最后一个元素。
> - `void addAtIndex(int index, int val)` 将一个值为 `val` 的节点插入到链表中下标为 `index` 的节点之前。如果 `index` 等于链表的长度，那么该节点会被追加到链表的末尾。如果 `index` 比长度更大，该节点将 **不会插入** 到链表中。
> - `void deleteAtIndex(int index)` 如果下标有效，则删除链表中下标为 `index` 的节点。
>
>
>
> **示例：**
>
> ```
> 输入
> ["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
> [[], [1], [3], [1, 2], [1], [1], [1]]
> 输出
> [null, null, null, null, 2, null, 3]
> 
> 解释
> MyLinkedList myLinkedList = new MyLinkedList();
> myLinkedList.addAtHead(1);
> myLinkedList.addAtTail(3);
> myLinkedList.addAtIndex(1, 2);    // 链表变为 1->2->3
> myLinkedList.get(1);              // 返回 2
> myLinkedList.deleteAtIndex(1);    // 现在，链表变为 1->3
> myLinkedList.get(1);              // 返回 3
> ```
>
>
>
> **提示：**
>
> - `0 <= index, val <= 1000`
> - 请不要使用内置的 LinkedList 库。
> - 调用 `get`、`addAtHead`、`addAtTail`、`addAtIndex` 和 `deleteAtIndex` 的次数不超过 `2000` 。

- 单链表

```c++
class MyLinkedList {
private:
    // 链表节点的定义
    struct ListNode {
        int val;
        ListNode* next;
        ListNode(int x) : val(x), next(nullptr) {}
        ListNode(int x, ListNode* next) : val(x), next(next) {}
    };
    int size;
    ListNode* head;

public:
    // 构造函数
    MyLinkedList() {
        this->size = 0;
        this->head = nullptr;
    }
 // 获取指定位置的节点值
    int get(int index) {
        if (index < 0 || index >= size) {
            return -1;
        }
        ListNode* current = head;
        for (int i = 0; i < index; i++) {
            current = current->next;
        }
        return current->val;
    }
 // 在链表头上加入一个节点
    void addAtHead(int val) {
        ListNode* newNode = new ListNode(val, head);
        head = newNode;
        size++;
    }
 // 在链表尾加入一个节点
    void addAtTail(int val) {
        if (size == 0) {
            addAtHead(val);
            return;
        }
        ListNode* current = head;
        while (current->next != nullptr) {
            current = current->next;
        }
        current->next = new ListNode(val);
        size++;
    }
    // 在指定节点前加入一个节点
    void addAtIndex(int index, int val) {
        if (index < 0 || index > size) {
            return;
        }
        if (index == 0) {
            addAtHead(val);
            return;
        }
        ListNode* current = head;
        for (int i = 0; i < index - 1; i++) {
            current = current->next;
        }
        ListNode* newNode = new ListNode(val, current->next);
        current->next = newNode;
        size++;
    }
 // 删除指定节点
    void deleteAtIndex(int index) {
        if (index < 0 || index >= size) {
            return;
        }
        ListNode* current = head;
        if (index == 0) {
            head = head->next;
            delete current;
        } else {
            for (int i = 0; i < index - 1; i++) {
                current = current->next;
            }
            ListNode* deletePtr = current->next;
            current->next = current->next->next;
            delete deletePtr;
        }
        size--;
    }
 // 析构函数
    ~MyLinkedList() {
        ListNode* current = head;
        while (current != nullptr) {
            ListNode* next = current->next;
            delete current;
            current = next;
        }
        head = nullptr;
    }
};
```

- 链表的设计细节很多,要妥善处理好链表节点之间的关系,并做好**内存管理**

```python
class ListNode:  # 含有默认参数时,不指定参数则使用默认参数
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MyLinkedList:
    def __init__(self):
        self.head = None  # self指向自身
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        current = self.head
        for _ in range(index):
            current = current.next
        return current.val

    def addAtHead(self, val: int) -> None:
        newNode = ListNode(val, self.head)
        self.head = newNode
        self.size += 1

    def addAtTail(self, val: int) -> None:
        if self.head == None:
            self.addAtHead(val)
        else:
            current = self.head
            while current.next != None:
                current = current.next
            current.next = ListNode(val)
            self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            pass
        elif index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            dummy_head = ListNode(0, self.head)
            current = dummy_head
            for _ in range(index):
                current = current.next
            newNode = ListNode(val, current.next)
            current.next = newNode
            self.size += 1
            self.head = dummy_head.next # 哑节点的后续操作

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        if index == 0:
            self.head = self.head.next
        else:
            current = self.head
            for _ in range(index - 1): # 遍历到目标节点的前一节点
                current = current.next
            current.next = current.next.next
        self.size -= 1



# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)

```

- 双向链表

```c++
class MyLinkedList {
private:
    struct ListNode {
        int val;
        ListNode* next;
        ListNode* prev;
        ListNode(int x, ListNode* prev, ListNode* next)
            : val(x), prev(prev), next(next) {}
    };
    int size;
    ListNode* head;

public:
    MyLinkedList() {
        this->size = 0;
        this->head = nullptr;
    }

    int get(int index) {
        if (index < 0 || index >= size) {
            return -1;
        }
        ListNode* current = head;
        for (int i = 0; i < index; i++) {
            current = current->next;
        }
        return current->val;
    }

    void addAtHead(int val) {
        ListNode* newNode = new ListNode(val, nullptr, head);
        head = newNode;
        size++;
    }

    void addAtTail(int val) {
        if (size == 0) {
            addAtHead(val);
            return;
        }
        ListNode* current = head;
        while (current->next != nullptr) {
            current = current->next;
        }
        current->next = new ListNode(val, current, nullptr);
        size++;
    }

    void addAtIndex(int index, int val) {
        if (index < 0 || index > size) {
            return;
        }
        if (index == 0) {
            addAtHead(val);
            return;
        }
        ListNode* current = head;
        for (int i = 0; i < index - 1; i++) {
            current = current->next;
        }
        ListNode* newNode = new ListNode(val, current, current->next);
        current->next = newNode;
        size++;
    }

    void deleteAtIndex(int index) {
        if (index < 0 || index >= size) {
            return;
        }
        ListNode* current = head;
        if (index == 0) {
            head = head->next;
            if (head != nullptr) { // 边界的细节
                head->prev = nullptr;
            }
            delete current;
        } else {
            for (int i = 0; i < index - 1; i++) {
                current = current->next;
            }
            ListNode* deletePtr = current->next;
            current->next = deletePtr->next;
            if (current->next != nullptr) { // 边界的细节
                current->next->prev = current;
            }
            delete deletePtr;
        }
        size--;
    }

    ~MyLinkedList() {
        ListNode* current = head;
        while (current != nullptr) {
            ListNode* next = current->next;
            delete current;
            current = next;
        }
        head = nullptr;
    }
};

```

```python
class ListNode:  # 含有默认参数时,不指定参数则使用默认参数
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.next = next
        self.prev = prev


class MyLinkedList:
    def __init__(self):
        self.head = None  # self指向自身
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        if self.size == 0:
            return -1
        current = self.head
        for _ in range(index):
            current = current.next
        if current != None:
            return current.val
        else:
            return -1

    def addAtHead(self, val: int) -> None:
        newNode = ListNode(val, None, self.head)
        self.head = newNode
        self.size += 1

    def addAtTail(self, val: int) -> None:
        if self.head == None:
            self.addAtHead(val)
        else:
            current = self.head
            while current.next != None:
                current = current.next
            current.next = ListNode(val, current, None)
            self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            pass
        elif index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            dummy_head = ListNode(0, None, self.head)
            current = dummy_head
            for _ in range(index):
                current = current.next
            newNode = ListNode(val, current, current.next)
            current.next = newNode
            self.size += 1
            self.head = dummy_head.next

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return

        if index == 0:
            if self.head.next:
                self.head = self.head.next
                self.head.prev = None
            else:
                self.head = None
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            current.next = current.next.next
            if current.next:
                current.next.prev = current
            self.size -= 1


"""
Your MyLinkedList object will be instantiated and called as such:

 obj = MyLinkedList()
 param_1 = obj.get(index)
 obj.addAtHead(val)
 obj.addAtTail(val)
 obj.addAtIndex(index,val)
 obj.deleteAtIndex(index)
"""
```

- 注意头节点没有`prev`,尾节点没有`next`,细节很多,注意**报错代码**

#### [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

> 给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。
>
>
>
> **示例 1：**
>
> ![img](./codeNote.assets/rev1ex1.jpg)
>
> ```
> 输入：head = [1,2,3,4,5]
> 输出：[5,4,3,2,1]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/rev1ex2.jpg)
>
> ```
> 输入：head = [1,2]
> 输出：[2,1]
> ```
>
> **示例 3：**
>
> ```
> 输入：head = []
> 输出：[]
> ```
>
> ![206_反转链表](./codeNote.assets/20210218090901207.png)
>
> **提示：**
>
> - 链表中节点的数目范围是 `[0, 5000]`
> - `-5000 <= Node.val <= 5000`
>
>
>
> **进阶：**链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？

- 双指针法

```c++
/*
  Definition for singly-linked list.
  struct ListNode {
      int val;
      ListNode *next;
      ListNode() : val(0), next(nullptr) {}
      ListNode(int x) : val(x), next(nullptr) {}
      ListNode(int x, ListNode *next) : val(x), next(next) {}
  };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = nullptr;
        while (fast != nullptr) {
            ListNode* temp = fast->next;
            fast->next = slow;
            slow = fast;
            fast = temp;
        }
        return slow;
    }
};
```

![img](./codeNote.assets/206.翻转链表.gif)

- 这里的`cur`和`pre`相当于`fast`和`slow`

```python
"""
 Definition for singly-linked list.
 class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
"""
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = head
        slow = None
        while fast!=None:
            temp = fast.next
            fast.next = slow
            slow = fast
            fast = temp
        return slow
```

#### [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

> 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/swap_ex1.jpg)
>
> ```
> 输入：head = [1,2,3,4]
> 输出：[2,1,4,3]
> ```
>
> **示例 2：**
>
> ```
> 输入：head = []
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：head = [1]
> 输出：[1]
> ```
>
>  
>
> **提示：**
>
> - 链表中节点的数目在范围 `[0, 100]` 内
> - `0 <= Node.val <= 100`

```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode dummy = ListNode(0, head);
        ListNode* prev = &dummy; // prev 初始化为 dummy 节点

        while (prev->next != nullptr && prev->next->next != nullptr) {
            ListNode* first = prev->next;        // 第一个节点
            ListNode* second = prev->next->next; // 第二个节点

            // 交换 first 和 second
            first->next = second->next;
            second->next = first;
            prev->next = second;

            // 移动 prev 节点，准备下一对交换
            prev = first;
        }

        return dummy.next; // 返回新的头节点，即 dummy 的下一个节点
    }
};
```

```python
"""
 Definition for singly-linked list.
 class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
"""
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None or head.next == None:
            return head
        dummy = ListNode(0, head)
        prev = dummy
        while prev.next != None and prev.next.next != None:
            first = prev.next
            second = prev.next.next

            first.next = second.next
            second.next = first
            prev.next = second

            prev = first
        return dummy.next

```

#### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

> 给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/remove_ex1.jpg)
>
> ```
> 输入：head = [1,2,3,4,5], n = 2
> 输出：[1,2,3,5]
> ```
>
> **示例 2：**
>
> ```
> 输入：head = [1], n = 1
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：head = [1,2], n = 1
> 输出：[1]
> ```
>
>  
>
> **提示：**
>
> - 链表中结点的数目为 `sz`
> - `1 <= sz <= 30`
> - `0 <= Node.val <= 100`
> - `1 <= n <= sz`
>
>  
>
> **进阶：**你能尝试使用一趟扫描实现吗？

```c++
/*
  Definition for singly-linked list.
  struct ListNode {
      int val;
      ListNode *next;
      ListNode() : val(0), next(nullptr) {}
      ListNode(int x) : val(x), next(nullptr) {}
      ListNode(int x, ListNode *next) : val(x), next(next) {}
  };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* current = head;
        int nodeNumber = 1;
        while (current->next != nullptr) {
            nodeNumber++;
            current = current->next;
        }
        ListNode dummy = ListNode(0, head);
        current = &dummy;
        for (int i = 0; i < nodeNumber - n; i++) {
            current = current->next;
        }
        ListNode* temp = current->next;
        current->next = current->next->next;
        delete temp;
        return dummy.next; // 本身就是链表节点内指针域的数据,数据类型为LinkNode指针
    }
};
```

```python
"""
 Definition for singly-linked list.
 class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
"""
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        nodeNumber = 1
        current = head
        while current.next != None:
            nodeNumber += 1
            current = current.next
        dummy = ListNode(0, head)
        current = dummy
        for _ in range(nodeNumber - n):
            current = current.next
        current.next = current.next.next
        return dummy.next
```

#### [面试题 02.07. 链表相交](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/)

> 给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。
>
> 图示两个链表在节点 `c1` 开始相交**：**
>
> [![img](./codeNote.assets/160_statement.png)](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)
>
> 题目数据 **保证** 整个链式结构中不存在环。
>
> **注意**，函数返回结果后，链表必须 **保持其原始结构** 。
>
>
>
> **示例 1：**
>
> [![img](./codeNote.assets/160_example_1.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)
>
> ```
> 输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
> 输出：Intersected at '8'
> 解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
> 从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。
> 在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
> ```
>
> **示例 2：**
>
> [![img](./codeNote.assets/160_example_2.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_2.png)
>
> ```
> 输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
> 输出：Intersected at '2'
> 解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
> 从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。
> 在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
> ```
>
> **示例 3：**
>
> [![img](./codeNote.assets/160_example_3.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png)
>
> ```
> 输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
> 输出：null
> 解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
> 由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
> 这两个链表不相交，因此返回 null 。
> ```
>
>
>
> **提示：**
>
> - `listA` 中节点数目为 `m`
> - `listB` 中节点数目为 `n`
> - `0 <= m, n <= 3 * 104`
> - `1 <= Node.val <= 105`
> - `0 <= skipA <= m`
> - `0 <= skipB <= n`
> - 如果 `listA` 和 `listB` 没有交点，`intersectVal` 为 `0`
> - 如果 `listA` 和 `listB` 有交点，`intersectVal == listA[skipA + 1] == listB[skipB + 1]`
>
>
>
> **进阶：**你能否设计一个时间复杂度 `O(n)` 、仅用 `O(1)` 内存的解决方案？

- 注意**对齐操作**

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        int nodeNumberA = 1, nodeNumberB = 1;
        ListNode* currentA = headA;
        while (currentA->next != nullptr) {
            currentA = currentA->next;
            nodeNumberA++;
        }
        ListNode* currentB = headB;
        while (currentB->next != nullptr) {
            currentB = currentB->next;
            nodeNumberB++;
        }
        // 得到二者长度
        currentB = headB;
        currentA = headA;
        if (nodeNumberA >= nodeNumberB) {
            for (int i = 0; i < nodeNumberA - nodeNumberB; i++) {
                currentA = currentA->next;
            }
        } else {
            for (int i = 0; i < nodeNumberB - nodeNumberA; i++) {
                currentB = currentB->next;
            }
        }
        while (currentA != currentB) {
            currentA = currentA->next;
            currentB = currentB->next;
        }
        if (currentA == nullptr) {
            return nullptr;
        } else {
            return currentA;
        }
    }
};
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA == None or headB == None:
            return None
        nodeNumberA = 1
        nodeNumberB = 1
        currentA = headA
        currentB = headB
        while currentA.next != None:
            currentA = currentA.next
            nodeNumberA += 1
        while currentB.next != None:
            currentB = currentB.next
            nodeNumberB += 1
        currentA = headA
        currentB = headB
        if nodeNumberA >= nodeNumberB:
            for _ in range(nodeNumberA - nodeNumberB):
                currentA = currentA.next
        else:
            for _ in range(nodeNumberB - nodeNumberA):
                currentB = currentB.next
        while currentA != currentB:
            currentA = currentA.next
            currentB = currentB.next
        if currentA == None:
            return None
        else:
            return currentA

```

#### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

> 给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*
>
> 如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。
>
> **不允许修改** 链表。
>
>
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/circularlinkedlist.png)
>
> ```
> 输入：head = [3,2,0,-4], pos = 1
> 输出：返回索引为 1 的链表节点
> 解释：链表中有一个环，其尾部连接到第二个节点。
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/circularlinkedlist_test2.png)
>
> ```
> 输入：head = [1,2], pos = 0
> 输出：返回索引为 0 的链表节点
> 解释：链表中有一个环，其尾部连接到第一个节点。
> ```
>
> **示例 3：**
>
> ![img](./codeNote.assets/circularlinkedlist_test3.png)
>
> ```
> 输入：head = [1], pos = -1
> 输出：返回 null
> 解释：链表中没有环。
> ```
>
>  
>
> **提示：**
>
> - 链表中节点的数目范围在范围 `[0, 104]` 内
> - `-105 <= Node.val <= 105`
> - `pos` 的值为 `-1` 或者链表中的一个有效索引
>
>  
>
> **进阶：**你是否可以使用 `O(1)` 空间解决此题？

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* detectCycle(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = head;
        ListNode* index1 = head;
        ListNode* index2 = nullptr;
        while (fast != nullptr &&
               fast->next != nullptr) { // 没有环结构在这里终止
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) { // 有环结构在这里终止
                index2 = fast;
                break;
            }
        }
        if (!(fast != nullptr && fast->next != nullptr)) {
            return nullptr;
        }
        while (index1 != index2) {
            index1 = index1->next;
            index2 = index2->next;
        }
        return index2;
    }
};
```

- ![img](./codeNote.assets/20220925103433.png)
- ![141.环形链表](./codeNote.assets/141.环形链表.gif)
- 双指针法+数学推导

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = head
        slow = head
        index1 = head
        index2 = None
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                index2 = fast
                break
        if fast == None or fast.next == None:
            return None
        while index1 != index2:
            index1 = index1.next
            index2 = index2.next
        return index2

```

## 第三章 哈希表

### 理论基础

- 用途

  - 一般用来**快速查询元素**是否存在于表中(**当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法**。)

  - 去重操作可以使用哈希表,但是如果对象可以使用别的方法去重,选择哈希表可能会带来更大的性能开销(在对象数目极大时,构建哈希表耗费的时间很长)

- 实现原理

  - 哈希函数

  - 哈希碰撞(不同元素通过哈希函数映射到了同一个位置)
    - 拉链法(挂在冲突索引的下面)
    - ![哈希表4](./codeNote.assets/20210104235015226.png)
    - 线性探测法
    - ![哈希表5](./codeNote.assets/20210104235109950.png)

- 常见的哈希结构

  | -                  | 集合   | 底层实现 | 是否有序 | 数值是否可以重复 | 能否更改数值 | 查询效率   | 增删效率 |
  | ------------------ | ------ | -------- | -------- | ---------------- | ------------ | ---------- |
  | std::set           | 红黑树 | 有序     | 否       | 否               | `O(log n)`   | `O(log n)` |
  | std::multiset      | 红黑树 | 有序     | 是       | 否               | `O(logn)`    | `O(logn)`  |
  | std::unordered_set | 哈希表 | 无序     | 否       | 否               | `O(1)`       | `O(1)`     |

  | -                  | 映射   | 底层实现 | 是否有序    | 数值是否可以重复 | 能否更改数值 | 查询效率   | 增删效率 |
  | ------------------ | ------ | -------- | ----------- | ---------------- | ------------ | ---------- |
  | std::map           | 红黑树 | key有序  | key不可重复 | key不可修改      | `O(logn)`    | `O(logn)`  |
  | std::multimap      | 红黑树 | key有序  | key可重复   | key不可修改      | `O(log n)`   | `O(log n)` |
  | std::unordered_map | 哈希表 | key无序  | key不可重复 | key不可修改      | `O(1)`       | `O(1)`     |

### 使用说明

#### C++

##### 共有方法

**tip**:这些是**container类**共有的方法,但是需要注意的是,不同的子类不一定支持以下的所有方法

1. `begin()`：返回指向容器中第一个元素的迭代器。
2. `end()`：返回指向容器中最后一个元素之后位置的迭代器。
3. `size()`：返回容器中元素的数量。
4. `empty()`：检查容器是否为空。
5. `clear()`：清空容器中的所有元素。
6. `insert()`：将元素插入容器中。
7. `erase()`：从容器中删除指定的元素或范围。
8. `find()`：在容器中查找指定元素，并返回其迭代器。**(未找到则返回迭代器`end()`)**
9. `count()`：计算容器中与指定元素值相等的元素数量。
10. `swap()`：交换两个容器的内容。

- 支持情况

1. `map`：`map` 是关联容器，不支持 `size()`和 `empty()`方法。其余方法均可使用。
2. `unordered_map`：`unordered_map` 是哈希关联容器，不支持 `size()`和 `empty()`方法。其余方法均可使用。
3. `string`：`string` 是字符串容器类，不支持 `insert()`方法和迭代器的使用，但支持其他方法。
4. `set`：`set` 是关联容器，不支持 `size()`和 `empty()`方法。其余方法均可使用。
5. `unordered_set`：`unordered_set` 是哈希关联容器，不支持 `size()`和 `empty()`方法。其余方法均可使用。
6. `vector`：`vector` 是顺序容器，支持所有列出的方法。

##### 迭代器

在 C++ 中，**迭代器**（**iterator**）是一种用于遍历容器（如数组、向量、链表、映射等）中元素的对象。迭代器提供了一种统一的方式来访问**容器**中的元素，而不需要了解容器的内部实现细节。

迭代器是一个类对象，它可以指向容器中的一个特定元素，并提供了一系列操作来访问、移动和修改容器中的元素。通过使用迭代器，你可以在容器中进行**顺序访问**、**逆序访问**、**随机访问**等操作。

迭代器的常用操作包括：

- `begin()`：返回指向容器中第一个元素的迭代器。
- `end()`：返回指向容器中最后一个元素之后位置的迭代器，也可以看作是表示结束的迭代器。
- `++`：将迭代器移动到容器中的下一个元素。
- `--`：将迭代器移动到容器中的上一个元素。
- `*`：返回迭代器当前指向的元素的引用。
- `==` 和 `!=`：用于比较两个迭代器是否相等。
- `->`：用于访问迭代器当前指向的元素的成员（如果该元素是一个对象）。

以下是一个使用迭代器遍历向量的示例：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> myVector = {1, 2, 3, 4, 5};

    // 迭代器遍历向量
    for (std::vector<int>::iterator it = myVector.begin(); it != myVector.end(); ++it) {
        std::cout << *it << " ";
    }

    return 0;
}
```

在上面的示例中，我们创建了一个整数向量 `myVector`，其中包含了一些整数。然后，我们使用迭代器 `std::vector<int>::iterator` 来遍历向量中的元素。我们初始化迭代器 `it` 为 `myVector` 的起始迭代器 `myVector.begin()`，并在循环中逐步递增迭代器，直到它等于 `myVector` 的结束迭代器 `myVector.end()`。在循环中，我们通过解引用迭代器 `*it` 来访问当前元素，并将其输出。

需要注意的是，C++ 的标准库提供了多种类型的迭代器，包括正向迭代器、反向迭代器、常量迭代器等。不同类型的迭代器具有不同的特性和能力，可以在不同的情况下使用。另外，C++11 引入了一种更简洁的迭代器写法，称为范围基于的 `for` 循环，使得遍历容器更加方便。

关于迭代器的种类，C++ 标准库提供了多种类型的迭代器，每种迭代器都具有不同的特性和能力，可以在不同的情况下使用。以下是一些常见的迭代器种类：

1. 正向迭代器（Forward Iterator）：只能向前遍历容器中的元素，支持 `++`、`*`、`==` 和 `!=` 等操作。例如，`std::vector` 的迭代器就是正向迭代器。
2. 双向迭代器（Bidirectional Iterator）：除了支持正向迭代器的操作外，还支持 `--` 操作，可以向后遍历容器中的元素。例如，`std::list` 的迭代器就是双向迭代器。
3. 随机访问迭代器（Random Access Iterator）：除了支持双向迭代器的操作外，还支持随机访问，可以通过 `+`、`-`、`[]` 操作来访问容器中的任意元素。例如，`std::vector` 和 `std::array` 的迭代器就是随机访问迭代器。
4. 输入迭代器（Input Iterator）：只能用于读取容器中的元素，不支持修改操作。例如，`std::istream_iterator` 就是输入迭代器。
5. 输出迭代器（Output Iterator）：只能用于向容器中写入元素，不支持读取和修改操作。例如，`std::ostream_iterator` 就是输出迭代器。

不同种类的迭代器可用于不同类型的容器，并提供了不同的功能和操作。在选择迭代器时，应根据具体的需求和容器的特性来做出选择。

##### `auto`声明

在 C++11 中引入了 `auto` 关键字，它可以用来自动推导变量的类型。使用 `auto` 声明变量时，编译器会根据等号右侧的表达式推导出变量的类型，从而省去了显式指定类型的麻烦。使用 `auto` 可以简化代码，使其更具可读性和灵活性。

以下是使用 `auto` 声明变量的示例：

```cpp
auto x = 42;  // 推导为 int 类型
auto message = "Hello, World!";  // 推导为 const char* 类型
auto pi = 3.14f;  // 推导为 float 类型
```

在上面的示例中，我们使用 `auto` 关键字声明了变量 `x`、`message` 和 `pi`，并根据等号右侧的表达式推导出了它们的类型。

#### Python

##### **set**

1. 创建一个`set`对象：

   ```python
   my_set = {1, 2, 3}  # 使用花括号创建一个set对象
   my_set = set() # 使用set()函数创建一个空的set对象
   # 使用空的花括号创建的是空字典而非空集合,创建空集合有且仅有这一种方式,创建空字典有两种方式
   ```

2. 添加元素到`set`：

   - `add(element)`: 将单个元素添加到`set`中。没有返回值。
   - `update(iterable)`: 将一个可迭代对象的元素添加到`set`中。没有返回值。

3. 删除元素从`set`：

   - `remove(element)`: 从`set`中删除指定元素，如果元素不存在会引发`KeyError`异常。没有返回值。
   - `discard(element)`: 从`set`中删除指定元素，如果元素不存在也不会引发异常。没有返回值。
   - `pop()`: 从`set`中随机删除并返回一个元素。

4. 查询元素在`set`中的存在：

   - `element in my_set`: 使用`in`关键字来检查元素是否存在于`set`中。返回布尔值。

5. 获取`set`的长度：

   - `len(my_set)`: 返回`set`中元素的数量。

6. 清空`set`：

   - `my_set.clear()`: 清空`set`中的所有元素。没有返回值。

7. 复制`set`：

   - `new_set = my_set.copy()`: 创建一个新的`set`对象，包含与原始`set`相同的元素。

8. 集合操作：

   - `union(other_set)`: 返回一个新的`set`，包含原始`set`和`other_set`中的所有元素。
     - `&`：交集运算符，返回两个集合的交集。
   - `intersection(other_set)`: 返回一个新的`set`，包含原始`set`和`other_set`中共有的元素。
     - `|`：并集运算符，返回两个集合的并集。
   - `difference(other_set)`: 返回一个新的`set`，包含原始`set`中存在但`other_set`中不存在的元素。
     - `-`：差集运算符，返回两个集合的差集（即从第一个集合中删除在第二个集合中存在的元素）。
   - `symmetric_difference(other_set)`: 返回一个新的`set`，包含原始`set`和`other_set`中不重复的元素。
     - `^`：对称差运算符，返回两个集合的对称差集（即返回两个集合中不重复的元素）。

### 典型例题

#### [242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

> 给定两个字符串 `*s*` 和 `*t*` ，编写一个函数来判断 `*t*` 是否是 `*s*` 的字母异位词。
>
> **注意：**若 `*s*` 和 `*t*` 中每个字符出现的次数都相同，则称 `*s*` 和 `*t*` 互为字母异位词。
>
>  
>
> **示例 1:**
>
> ```
> 输入: s = "anagram", t = "nagaram"
> 输出: true
> ```
>
> **示例 2:**
>
> ```
> 输入: s = "rat", t = "car"
> 输出: false
> ```
>
>  
>
> **提示:**
>
> - `1 <= s.length, t.length <= 5 * 104`
> - `s` 和 `t` 仅包含小写字母
>
>  
>
> **进阶:** 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        unordered_set<char> ss, ts;
        ss.insert(s.begin(), s.end());
        ts.insert(t.begin(), t.end());
        return ss == ts;
    }
};
```

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        ss = set(s)
        ts = set(t)
        return ss==ts
```

- C++中`unordered_set`头文件提供了基于**哈希表**的**无序集合**数据类型,位于C++标准库STL中
- Python中`set`是基于哈希表的无序集合

#### [349. 两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)

> 给定两个数组 `nums1` 和 `nums2` ，返回 *它们的交集* 。输出结果中的每个元素一定是 **唯一** 的。我们可以 **不考虑输出结果的顺序** 。
>
> **示例 1：**
>
> ```
> 输入：nums1 = [1,2,2,1], nums2 = [2,2]
> 输出：[2]
> ```
>
> **示例 2：**
>
> ```
> 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
> 输出：[9,4]
> 解释：[4,9] 也是可通过的
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums1.length, nums2.length <= 1000`
> - `0 <= nums1[i], nums2[i] <= 1000`

```c++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> sans;
        unordered_set<int> temp(nums1.begin(), nums1.end());
        for (int num : nums2) {
            if (temp.find(num) != temp.end()) {
                sans.insert(num);
            }
        }
        vector<int> ans(sans.begin(), sans.end());
        return ans;
    }
};
```

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        s1 = set(nums1)
        s2 = set(nums2)
        return list(s1 & s2)
```

[350. 两个数组的交集 II](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)

> 给你两个整数数组 `nums1` 和 `nums2` ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums1 = [1,2,2,1], nums2 = [2,2]
> 输出：[2,2]
> ```
>
> **示例 2:**
>
> ```
> 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
> 输出：[4,9]
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums1.length, nums2.length <= 1000`
> - `0 <= nums1[i], nums2[i] <= 1000`
>
>  
>
> ***\*进阶\**：**
>
> - 如果给定的数组已经排好序呢？你将如何优化你的算法？
> - 如果 `nums1` 的大小比 `nums2` 小，哪种方法更优？
> - 如果 `nums2` 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

```c++
```

```python

```

#### [202. 快乐数](https://leetcode.cn/problems/happy-number/)

> 编写一个算法来判断一个数 `n` 是不是快乐数。
>
> **「快乐数」** 定义为：
>
> - 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
> - 然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。
> - 如果这个过程 **结果为** 1，那么这个数就是快乐数。
>
> 如果 `n` 是 *快乐数* 就返回 `true` ；不是，则返回 `false` 。
>
>  
>
> **示例 1：**
>
> ```
> 输入：n = 19
> 输出：true
> 解释：
> 1^2 + 9^2 = 82
> 8^2 + 2^2 = 68
> 6^2 + 8^2 = 100
> 1^2 + 0^2 + 0^2 = 1
> ```
>
> **示例 2：**
>
> ```
> 输入：n = 2
> 输出：false
> ```
>
>  
>
> **提示：**
>
> - `1 <= n <= 2^31 - 1`

```c++
class Solution {
public:
    bool isHappy(int n) {
        unordered_set<int> result;
        int temp = 0, sum = 0;
        while (result.insert(sum).second) {
            sum = 0;
            while (n != 0) {
                temp = n % 10;
                sum += temp * temp;
                n = n / 10;
            }
            if (sum == 1) {
                return true;
            }
            n = sum;
        }
        return false;
    }
};
```

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        temp = 0
        sumNum = 0
        result = set()
        while sumNum not in result:
            result.add(sumNum)
            sumNum = 0
            while n != 0:
                temp = n % 10
                sumNum += temp**2
                n = n // 10
            if sumNum == 1:
                return True
            n = sumNum
        return False

```

#### [1. 两数之和](https://leetcode.cn/problems/two-sum/)

> 给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。
>
> 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
>
> 你可以按任意顺序返回答案。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [2,7,11,15], target = 9
> 输出：[0,1]
> 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [3,2,4], target = 6
> 输出：[1,2]
> ```
>
> **示例 3：**
>
> ```
> 输入：nums = [3,3], target = 6
> 输出：[0,1]
> ```
>
>  
>
> **提示：**
>
> - `2 <= nums.length <= 104`
> - `-109 <= nums[i] <= 109`
> - `-109 <= target <= 109`
> - **只会存在一个有效答案**
>
>  
>
> **进阶：**你可以想出一个时间复杂度小于 `O(n2)` 的算法吗？

```c++
// 暴力解法
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }
        return {-1, -1};
    }
};
// map || unordered_map
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> umap;
        for (int i = 0; i < nums.size(); i++) {
            auto iter = umap.find(target - nums[i]);
            if (iter != umap.end()) {
                return {iter->second, i};
            }
            umap.insert(pair<int, int>(nums[i], i));
        }
        return {-1, -1};
    }
};
```

- `std::pair`是C++标准库中的一个模板类，用于存储两个不同类型的值(可用构造键值对`{key,value}`)。`std::pair`的使用方法如下：

  1. 包含头文件：要使用`std::pair`，需要包含头文件 `<utility>`。

  2. 创建`std::pair`对象：可以使用以下方式创建`std::pair`对象：

     ```cpp
     std::pair<T1, T2> myPair; // 创建一个空的pair对象，其中T1和T2分别是两个元素的类型
     std::pair<T1, T2> myPair(value1, value2); // 创建一个带有初始值的pair对象
     std::make_pair(value1, value2); // 使用make_pair函数创建一个带有初始值的pair对象
     ```

  3. 访问`std::pair`中的元素：可以使用以下方式访问`std::pair`中的元素：

     ```cpp
     myPair.first; // 访问第一个元素
     myPair.second; // 访问第二个元素
     ```

  4. 修改`std::pair`中的元素：可以使用以下方式修改`std::pair`中的元素：

     ```cpp
     myPair.first = newValue1; // 修改第一个元素的值
     myPair.second = newValue2; // 修改第二个元素的值
     ```

  5. 比较`std::pair`对象：可以使用以下方式比较`std::pair`对象：

     ```cpp
     myPair1 == myPair2; // 比较两个pair对象是否相等
     myPair1 != myPair2; // 比较两个pair对象是否不相等
     ```

  6. 使用`std::pair`作为函数的返回值：`std::pair`常用于函数返回多个值的情况，例如：

     ```cpp
     std::pair<int, int> getMinMax(const std::vector<int>& nums) {
         int minVal = *std::min_element(nums.begin(), nums.end());
         int maxVal = *std::max_element(nums.begin(), nums.end());
         return std::make_pair(minVal, maxVal);
     }
     
     // 使用返回的pair对象
     std::pair<int, int> result = getMinMax(nums);
     int minValue = result.first;
     int maxValue = result.second;
     ```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_dict = dict()
        for i in range(len(nums)):
            temp = num_dict.get(target - nums[i])
            if temp is not None:
                return [temp, i]
            num_dict[nums[i]] = i
        return [-1, -1]
```

#### [454. 四数相加 II](https://leetcode.cn/problems/4sum-ii/)

> 给你四个整数数组 `nums1`、`nums2`、`nums3` 和 `nums4` ，数组长度都是 `n` ，请你计算有多少个元组 `(i, j, k, l)` 能满足：
>
> - `0 <= i, j, k, l < n`
> - `nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`
>
> **示例 1：**
>
> ```
>输入：nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
> 输出：2
> 解释：
> 两个元组如下：
> 1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
> 2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
> ```
>
> **示例 2：**
>
> ```
>输入：nums1 = [0], nums2 = [0], nums3 = [0], nums4 = [0]
> 输出：1
> ```
>
> **提示：**
>
> - `n == nums1.length`
>- `n == nums2.length`
> - `n == nums3.length`
>- `n == nums4.length`
> - `1 <= n <= 200`
> - `-2^28^ <= nums1[i], nums2[i], nums3[i], nums4[i] <= 2^28`

- 暴力法:  Time Complexity : O(n^4^)
- 拆分哈希表:  Time Complexity : O(m\*n + o\*p)

```c++
// 暴力解法(超时)
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3,
                     vector<int>& nums4) {
        int ans = 0;
        for (int i = 0; i < nums1.size(); i++) {
            for (int j = 0; j < nums2.size(); j++) {
                for (int k = 0; k < nums3.size(); k++) {
                    for (int l = 0; l < nums4.size(); l++) {
                        if (nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0) {
                            ans++;
                        }
                    }
                }
            }
        }
        return ans;
    }
};
// unordered_map 对暴力法降次
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3,
                     vector<int>& nums4) {
        unordered_map<int, int> umap;
        for (int i = 0; i < nums1.size(); i++) {
            for (int j = 0; j < nums2.size(); j++) {
                int sum = nums1[i] + nums2[j];
                if (umap.find(sum) == umap.end()) {
                    umap[sum] = 1;
                } else {
                    umap[sum] += 1;
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < nums3.size(); i++) {
            for (int j = 0; j < nums4.size(); j++) {
                int sum = nums3[i] + nums4[j];
                if (umap.find(0 - sum) != umap.end()) {
                    ans += umap[0 - sum];
                }
            }
        }
        return ans;
    }
};
// 另一写法
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> umap; //key:a+b的数值，value:a+b数值出现的次数
        // 遍历大A和大B数组，统计两个数组元素之和，和出现的次数，放到map中
        for (int a : A) {
            for (int b : B) {
                umap[a + b]++; // 如果没有该键值对,在这里原始默认值为0(int类型)
            }
        }
        int count = 0; // 统计a+b+c+d = 0 出现的次数
        // 在遍历大C和大D数组，找到如果 0-(c+d) 在map中出现过的话，就把map中key对应的value也就是出现次数统计出来。
        for (int c : C) {
            for (int d : D) {
                if (umap.find(0 - (c + d)) != umap.end()) {
                    count += umap[0 - (c + d)];
                }
            }
        }
        return count;
    }
};

```

```python
class Solution:
    def fourSumCount(
        self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]
    ) -> int:
        numDict = dict()
        for i in nums1:
            for j in nums2:
                sumNum = i + j
                if numDict.get(sumNum) == None:
                    numDict[sumNum] = 1
                else:
                    numDict[sumNum] += 1
        ans = 0
        for i in nums3:
            for j in nums4:
                sumNum = i + j
                if numDict.get(0 - sumNum) != None:
                    ans += numDict[0 - sumNum]
        return ans
```

#### [383. 赎金信](https://leetcode.cn/problems/ransom-note/)

> 给你两个字符串：`ransomNote` 和 `magazine` ，判断 `ransomNote` 能不能由 `magazine` 里面的字符构成。
>
> 如果可以，返回 `true` ；否则返回 `false` 。
>
> `magazine` 中的每个字符只能在 `ransomNote` 中使用一次。
>
>  
>
> **示例 1：**
>
> ```
> 输入：ransomNote = "a", magazine = "b"
> 输出：false
> ```
>
> **示例 2：**
>
> ```
> 输入：ransomNote = "aa", magazine = "ab"
> 输出：false
> ```
>
> **示例 3：**
>
> ```
> 输入：ransomNote = "aa", magazine = "aab"
> 输出：true
> ```
>
> **提示：**
>
> - `1 <= ransomNote.length, magazine.length <= 105`
>- `ransomNote` 和 `magazine` 由小写英文字母组成

```c++
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        int arr1[26]{0}, arr2[26]{0};
        for (char iter : ransomNote) {
            arr1[iter - 'a']++;
        }
        for (char iter : magazine) {
            arr2[iter - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (arr2[i] - arr1[i] < 0) {
                return false;
            }
        }
        return true;
    }
};
```

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        list1 = [0 for _ in range(26)]
        list2 = [0 for _ in range(26)]
        # ord(char) -> int 可以返回字符的ascii码对应整型
        # char(int) -> str 相对的
        for item in ransomNote:
            list1[ord(item) - ord('a')] += 1
        for item in magazine:
            list2[ord(item) - ord('a')] += 1
        for i in range(26):
            if list2[i] - list1[i] < 0:
                return False
        return True
    
# python风格
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        r_set = set(ransomNote)
        r_list = list(r_set)
        for char in r_list:
            if char  not in magazine or ransomNote.count(char) > magazine.count(char):
                return False
        return True    
```

- 在Python中，以下类型支持`count()`方法：

  1. 字符串（`str`）：`count()`方法用于计算字符串中指定子字符串的出现次数。
  2. 列表（`list`）：`count()`方法用于计算列表中指定元素的出现次数。
  3. 元组（`tuple`）：`count()`方法用于计算元组中指定元素的出现次数。

  以下是示例代码，展示了不同类型中的`count()`方法的使用：

  ```python
  # 字符串
  string = "Hello, World!"
  print(string.count('o'))  # 2
  print(string.count('l'))  # 3
  
  # 列表
  my_list = [1, 2, 3, 2, 4, 2]
  print(my_list.count(2))  # 3
  print(my_list.count(5))  # 0
  
  # 元组
  my_tuple = (1, 2, 3, 2, 4, 2)
  print(my_tuple.count(2))  # 3
  print(my_tuple.count(5))  # 0
  ```

  注意：对于字符串、列表和元组来说，`count()`方法都是区分大小写的。如果要进行不区分大小写的计数，可以先将字符串转换为小写或大写进行处理。

#### [15. 三数之和](https://leetcode.cn/problems/3sum/)

> 给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请
>
> 你返回所有和为 `0` 且不重复的三元组。
>
> **注意：**答案中不可以包含重复的三元组。
>
> **示例 1：**
>
> ```
> 输入：nums = [-1,0,1,2,-1,-4]
> 输出：[[-1,-1,2],[-1,0,1]]
> 解释：
> nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
> nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
> nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
> 不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
> 注意，输出的顺序和三元组的顺序并不重要。
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [0,1,1]
> 输出：[]
> 解释：唯一可能的三元组和不为 0 。
> ```
>
> **示例 3：**
>
> ```
> 输入：nums = [0,0,0]
> 输出：[[0,0,0]]
> 解释：唯一可能的三元组和为 0 。
> ```
>
>
>
> **提示：**
>
> - `3 <= nums.length <= 3000`
> - `-105 <= nums[i] <= 105`

- 在 C++ 中，`std::tuple` 是一个非常灵活的工具，它允许你将不同类型的值组合成单一对象。这在 C++11 及其之后的版本中得到了引入和支持。`tuple` 是标准库头文件 `<tuple>` 中定义的。

  `std::tuple` 可以用来存储任意数量和类型的元素（只要这些类型是可复制或可移动的），每个元素都有它自己的类型。`tuple` 对于将多个值作为一个单一返回值从函数返回特别有用。

  这里是一些使用 `std::tuple` 的基本示例：

  - 创建和初始化

  ```cpp
  #include <tuple>
  
  // 创建一个 tuple
  std::tuple<int, std::string, float> my_tuple(42, "Hello", 3.14);
  
  // 使用 std::make_tuple 来创建 tuple
  auto another_tuple = std::make_tuple(42, "Hello", 3.14);
  ```

  - 访问元素

  使用 `std::get<>()` 来访问元素：

  ```cpp
  // 获取第一个元素
  int my_int = std::get<0>(my_tuple);
  
  // 获取第二个元素
  std::string my_string = std::get<1>(my_tuple);
  ```

  - 解包（Unpacking）

  使用 `std::tie` 来解包 tuple 中的元素：

  ```cpp
  int my_int;
  std::string my_string;
  float my_float;
  
  // 解包 tuple 到已存在的变量
  std::tie(my_int, my_string, my_float) = my_tuple;
  ```

  或者使用结构化绑定（C++17 及更新版本）：

  ```cpp
  auto [my_int, my_string, my_float] = my_tuple;
  ```

  - 元素数量

  使用 `std::tuple_size` 来获取 tuple 中的元素数量：

  ```cpp
  constexpr size_t num_elements = std::tuple_size<decltype(my_tuple)>::value;
  ```

  - 比较操作

  可以比较两个 tuple，前提是它们的每个对应元素都可以比较：

  ```cpp
  if (my_tuple == another_tuple) {
      // ...
  }
  ```

  `std::tuple` 支持 `==`、`!=`、`<`、`<=`、`>` 和 `>=` 操作符。

  - 使用 std::apply

  在 C++17 中，你可以使用 `std::apply` 来调用一个函数，并将 tuple 的元素作为参数：

  ```cpp
  auto my_tuple = std::make_tuple(1, 2, 3);
  
  auto sum = [](int a, int b, int c) { return a + b + c; };
  
  // 应用 sum 函数，将 tuple 的元素作为参数
  int result = std::apply(sum, my_tuple);
  ```

- 前缀运算符和后缀运算符

  - 在 C++ 中，前缀递增运算符 (`++i`) 和后缀递增运算符 (`i++`) 的作用略有不同，尽管它们都将变量的值增加 1。

    前缀递增 (`++i`) 先将变量的值增加 1，然后返回增加后的值。而后缀递增 (`i++`) 则先返回变量当前的值，然后再将变量的值增加 1。由于后缀递增需要存储原始值以便返回，这可能需要额外的存储空间和操作，特别是当变量是复杂的对象时（比如迭代器或自定义对象），这会造成性能上的开销。

    在不需要使用变量增加前的值的情况下，通常推荐使用前缀递增，因为它可能会更高效，特别是在循环或频繁调用的场景中。对于基本数据类型（如 `int`），现代编译器通常能够优化这两个操作，使得它们的性能差异不大。但习惯上，C++ 程序员倾向于默认使用前缀递增，除非确实需要后缀递增的语义。

    简而言之，选择前缀递增是出于对性能的考虑，即使在许多情况下这种性能差异非常微小，但它是一种更安全的默认行为。

  - 递减运算符同理

```c++
// hash去重 (很慢)
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        set<vector<int>> result_set;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size() && nums[i] <= 0; i++) {
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                if (nums[i] + nums[left] + nums[right] > 0) {
                    right--;
                } else if (nums[i] + nums[left] + nums[right] < 0) {
                    left++;
                } else {
                    result_set.insert({nums[i], nums[left], nums[right]});
                    left++;
                    right--;
                }
            }
        }
        vector<vector<int>> result(result_set.begin(), result_set.end());
        return result;
    }
};
// 不使用hash去重 O(n^2)
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size() && nums[i] <= 0; ++i) {
            // 跳过重复的数
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                const int sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) {
                    --right;
                } else if (sum < 0) {
                    ++left;
                } else {
                    result.push_back({nums[i], nums[left], nums[right]});
                    // 跳过所有相同的左值和右值
                    while (left < right && nums[left] == nums[left + 1]) {
                        ++left;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        --right;
                    }
                    ++left;
                    --right;
                }
            }
        }
        return result;
    }
};
```

- **可哈希化**

  - 在C++和Python中，可哈希化的数据类型是那些**不可变（immutable）**的数据类型。这些数据类型的对象在其生命周期内不会改变，因此可以用作哈希表的键（例如，在C++的`std::unordered_map`中或Python的`dict`和`set`中）。以下是两种语言中可哈希化数据类型的概览：

    - C++

    在C++中，标准库提供的可哈希化数据类型包括：

    1. **整数类型**：如`int`, `char`, `size_t`, `uint8_t`等。
    2. **浮点数类型**：如`float`, `double`（但请注意，由于浮点精度问题，直接使用浮点数作为键可能会导致不准确的查找）。
    3. **指针类型**：原生指针（如`int*`）可以哈希化，但使用指针作为键需要谨慎，确保指针的生命周期和上下文适合这种用途。
    4. `std::string`：字符串类型在C++中也是可哈希的。
    5. `std::pair`、`std::tuple`：对于少量固定大小的组合类型，标准库提供了哈希实现。

    用户自定义类型可以通过特化`std::hash`模板来实现可哈希化。例如：

    ```cpp
    struct MyType {
        int x;
        bool y;
    };
    
    namespace std {
        template <>
        struct hash<MyType> {
            size_t operator()(const MyType& my_type) const {
                return std::hash<int>()(my_type.x) ^ std::hash<bool>()(my_type.y);
            }
        };
    }
    ```

    - Python

    在Python中，以下是内置的可哈希化数据类型：

    1. **数值类型**：如整数（`int`）、浮点数（`float`）、复数（`complex`，虽然实际上不常用作哈希键）。
    2. **字符串类型**：`str`，是可哈希的。
    3. **元组类型**：`tuple`，但它必须包含的全部元素也都是可哈希的（即元组内部不能有列表、字典或其他集合类型）。
    4. **冻结集合**：`frozenset`，是不可变集合，可以哈希化。

    通常，Python中任何实现了`__hash__()`方法并返回一个整数的对象都可以被哈希化。如果一个对象还定义了`__eq__()`方法，Python会确保哈希表能正确地处理键的相等性。如果一个类定义了`__eq__()`但没有定义`__hash__()`，它的实例将不是可哈希的，因为这会破坏哈希表的不变性要求。

    例如，你可以定义一个可哈希的类：

    ```python
    class MyType:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
        def __hash__(self):
            return hash((self.x, self.y))
    
        def __eq__(self, other):
            return isinstance(other, MyType) and self.x == other.x and self.y == other.y
    ```

    在创建哈希表时，例如一个字典或集合，Python会使用`__hash__()`方法来获取对象的哈希值，并使用`__eq__()`方法来解决哈希冲突。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = set()
        for i in range(len(nums)):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                sumNum = nums[i] + nums[left] + nums[right]
                if sumNum > 0:
                    right -= 1
                elif sumNum < 0:
                    left += 1
                else:
                    ans.add((nums[i], nums[left], nums[right]))
                    left += 1
                    right -= 1
        return list(ans)
```

#### [18. 四数之和](https://leetcode.cn/problems/4sum/)

> 给你一个由 `n` 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且**不重复**的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：
>
> - `0 <= a, b, c, d < n`
> - `a`、`b`、`c` 和 `d` **互不相同**
> - `nums[a] + nums[b] + nums[c] + nums[d] == target`
>
> 你可以按 **任意顺序** 返回答案 。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [1,0,-1,0,-2,2], target = 0
> 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [2,2,2,2,2], target = 8
> 输出：[[2,2,2,2]]
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums.length <= 200`
> - `-109 <= nums[i] <= 109`
> - `-109 <= target <= 109`

```c++

```

```python

```

## 第四章 字符串

### 理论基础

**打基础的时候，不要太迷恋于库函数**

#### KMP算法

- 相比于传统的暴力算法`O(m*n)`,**KMP算法**的时间复杂度为`O(m+n)`,空间复杂度为`O(m)`

- **当出现字符串不匹配时，可以记录一部分之前已经匹配的文本内容，利用这些信息避免从头再去做匹配。**

  - 什么是KMP
    - Knuth，Morris和Pratt三人发明了该算法，所以取了三位学者名字的首字母。所以叫做KMP
  - KMP有什么用
    - **字符串匹配**
  - 什么是前缀表（prefix table）
    - 前缀表是用来回退的，它记录了**模式串**与**主串(文本串)**不匹配的时候，模式串应该从哪里开始重新匹配。
  - 为什么一定要用前缀表
    - 前缀:不包含最后一个字符的所有以第一个字符为首的连续子串
    - 后缀:不包含第一个字符的所有以最后一个字符为尾的连续子串
    - 寻找**模式串的最长相等前后缀的长度**作为**前缀表**(`profix/next`),这个表指出了在**模式串匹配失败之后==回滚==的位置**
  - 如何计算前缀表
    - 如图![KMP精讲8](./codeNote.assets/KMP精讲8.png)
    - 实现模式(不匹配的时候回滚到前面计算的前缀表中最大值为索引的位置,图中这个例子是2)![KMP精讲2](./codeNote.assets/KMP精讲2.gif)
  - 前缀表与next数组
    - 本质上并没有太大的差别
  
  - 使用next数组来匹配
    - 与`profix`数组类似
  
  - 时间复杂度分析
    - 其中n为文本串长度，m为模式串长度，因为在匹配的过程中，根据前缀表不断调整匹配的位置，可以看出匹配的过程是O(n)，之前还要单独生成next数组，时间复杂度是O(m)。所以整个KMP算法的时间复杂度是O(n+m)的。
  
  - 构造next数组(构造next数组其实就是计算模式串s，前缀表的过程)
    - `void getNext(int* next, const string& s)`
      - 初始化
      - 处理前后缀不相同的情况
      - 处理前后缀相同的情况
  
  - 使用next数组来做匹配
  - 前缀表统一减一 C++代码实现
  - 前缀表（不减一）C++实现
  - 总结
  
### 典型例题

#### [344. 反转字符串](https://leetcode.cn/problems/reverse-string/)

> 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 `s` 的形式给出。
>
>不要给另外的数组分配额外的空间，你必须**[原地](https://baike.baidu.com/item/原地算法)修改输入数组**、使用 O(1) 的额外空间解决这一问题。
>
>
>
>**示例 1：**
>
>```
>输入：s = ["h","e","l","l","o"]
>输出：["o","l","l","e","h"]
>```
>
>**示例 2：**
>
>```
>输入：s = ["H","a","n","n","a","h"]
>输出：["h","a","n","n","a","H"]
>```
>
>
>
>**提示：**
>
>- `1 <= s.length <= 105`
>- `s[i]` 都是 [ASCII](https://baike.baidu.com/item/ASCII) 码表中的可打印字符

```c++
class Solution {
public:
    void reverseString(vector<char>& s) {
        int i = 0, j = s.size() - 1;
        while (i <= j) {
            char temp = s[j];
            s[j] = s[i];
            s[i] = temp;
            i++;
            j--;
        }
    }
};
// 简洁版
class Solution {
public:
    void reverseString(vector<char>& s) {
        for (int i = 0, j = s.size() - 1; i <= j; i++, j--) {
            swap(s[i], s[j]);
        }
    }
};
```

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i = 0
        j = len(s) - 1
        while i <= j:
            temp = s[i]
            s[i] = s[j]
            s[j] = temp
            i += 1
            j -= 1
```

- `swap(a,b)`的实现逻辑

  - 一种就是常见的交换数值：(普遍)

    ```cpp
    int tmp = s[i];
    s[i] = s[j];
    s[j] = tmp;
    ```

  - 一种就是通过位运算：(适用于可以表示为二进制形式的数据)

    ```cpp
    s[i] ^= s[j];
    s[j] ^= s[i];
    s[i] ^= s[j];
    ```

#### [541. 反转字符串 II](https://leetcode.cn/problems/reverse-string-ii/)

> 给定一个字符串 `s` 和一个整数 `k`，从字符串开头算起，每计数至 `2k` 个字符，就反转这 `2k` 字符中的前 `k` 个字符。
>
> - 如果剩余字符少于 `k` 个，则将剩余字符全部反转。
> - 如果剩余字符小于 `2k` 但大于或等于 `k` 个，则反转前 `k` 个字符，其余字符保持原样。
>
>  
>
> **示例 1：**
>
> ```
> 输入：s = "abcdefg", k = 2
> 输出："bacdfeg"
> ```
>
> **示例 2：**
>
> ```
> 输入：s = "abcd", k = 2
> 输出："bacd"
> ```
>
>  
>
> **提示：**
>
> - `1 <= s.length <= 104`
> - `s` 仅由小写英文组成
> - `1 <= k <= 104`

```c++
class Solution {
private:
    void reverse(string& s, int forward_ptr, int behind_ptr) {
        while (forward_ptr < behind_ptr) {
            char temp = s[forward_ptr];
            s[forward_ptr] = s[behind_ptr];
            s[behind_ptr] = temp;
            ++forward_ptr;
            --behind_ptr;
        }
    }

public:
    string reverseStr(string s, int k) {
        int begin_ptr = 0;
        while (begin_ptr < s.size()) {
            if (s.size() - begin_ptr >= 2 * k) {
                reverse(s, begin_ptr, begin_ptr + k - 1);

            } else if (s.size() - begin_ptr >= k) {
                reverse(s, begin_ptr, begin_ptr + k - 1);
            } else {
                reverse(s, begin_ptr, s.size() - 1);
            }
            begin_ptr += 2 * k;
        }

        return s;
    }
};
// 细节比较多,注意题目要求
```

- Python中对于不可变对象,是不能通过**引用传递**修改的
  - 整数（int）
  - 浮点数（float）
  - 复数（complex）
  - 布尔值（bool）
  - 字符串（str）
  - 元组（tuple）
  - 冻结集合（frozenset）
  - 字节（bytes）
- 这些对象在创建后不能被修改。如果对这些对象进行修改操作，实际上是创建了一个新的对象
- 当你对一个不可变对象进行修改操作时，Python会创建一个新的对象来保存修改后的值，并将新对象的引用返回给你。原来的对象保持不变，并且在没有引用指向它时，会被垃圾回收机制自动释放。

```python
class Solution:
    def reverse(self, s: str, startIndex: int, endIndex: int) -> str:
        s_list = list(s)  # 将字符串转换为列表
        while startIndex < endIndex:
            s_list[startIndex], s_list[endIndex] = s_list[endIndex], s_list[startIndex]
            startIndex += 1
            endIndex -= 1
        return ''.join(s_list)  # 将列表转换回字符串

    def reverseStr(self, s: str, k: int) -> str:
        startIndex = 0
        while startIndex < len(s):
            if len(s) - startIndex >= k:
                s = self.reverse(s, startIndex, startIndex + k - 1)
            else:
                s = self.reverse(s, startIndex, len(s) - 1)
            startIndex += 2 * k
        return s

```

#### [替换数字](https://kamacoder.com/problempage.php?pid=1064)

> 时间限制：1.000S 空间限制：128MB
>
> ###### 题目描述
>
> 给定一个字符串 s，它包含小写字母和数字字符，请编写一个函数，将字符串中的字母字符保持不变，而将每个数字字符替换为number。 例如，对于输入字符串 "a1b2c3"，函数应该将其转换为 "anumberbnumbercnumber"。
>
> ###### 输入描述
>
> 输入一个字符串 s,s 仅包含小写字母和数字字符。
>
> ###### 输出描述
>
> 打印一个新的字符串，其中每个数字字符都被替换为了number
>
> ###### 输入示例
>
> ```
> a1b2c3
> ```
>
> ###### 输出示例
>
> ```
> anumberbnumbercnumber
> ```
>
> ###### 提示信息
>
> 数据范围：
> 1 <= `s.length` < 10000。

```c++
#include <iostream>
#include <string>
using namespace std;
int main() {
  string temp, out;
  cin >> temp;
  for (auto iter : temp) {
    if (iter <= '9' && iter >= '0') {
      out = out + "number";
    } else {
      out = out + iter;
    }
  }
  cout << out << endl;
}
// 时间复杂度更小的算法

#include<iostream>
using namespace std;
int main() {
    string s;
    while (cin >> s) {
        int count = 0; // 统计数字的个数
        int sOldSize = s.size();
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                count++;
            }
        }
        // 扩充字符串s的大小，也就是每个空格替换成"number"之后的大小
        s.resize(s.size() + count * 5);
        int sNewSize = s.size();
        // 从后先前将空格替换为"number"
        for (int i = sNewSize - 1, j = sOldSize - 1; j < i; i--, j--) {
            if (s[j] > '9' || s[j] < '0') {
                s[i] = s[j];
            } else {
                s[i] = 'r';
                s[i - 1] = 'e';
                s[i - 2] = 'b';
                s[i - 3] = 'm';
                s[i - 4] = 'u';
                s[i - 5] = 'n';
                i -= 5;
            }
        }
        cout << s << endl;
    }
}

```

```python
temp = input()
out =  str()
nums = ['1','2','3','4','5','6','7','8','9','0']
for i in temp:
    if i in nums:
        out = out+"number"
    else:
        out = out + i
print(out)
```

#### [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/)

>**综合考察字符串操作的好题。**
>
>给你一个字符串 `s` ，请你反转字符串中 **单词** 的顺序。
>
>**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。
>
>返回 **单词** 顺序颠倒且 **单词** 之间用单个空格连接的结果字符串。
>
>**注意：**输入字符串 `s`中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
>
>
>
>**示例 1：**
>
>```
>输入：s = "the sky is blue"
>输出："blue is sky the"
>```
>
>**示例 2：**
>
>```
>输入：s = "  hello world  "
>输出："world hello"
>解释：反转后的字符串中不能存在前导空格和尾随空格。
>```
>
>**示例 3：**
>
>```
>输入：s = "a good   example"
>输出："example good a"
>解释：如果两个单词间有多余的空格，反转后的字符串需要将单词间的空格减少到仅有一个。
>```
>
>
>
>**提示：**
>
>- `1 <= s.length <= 104`
>- `s` 包含英文大小写字母、数字和空格 `' '`
>- `s` 中 **至少存在一个** 单词
>
>**进阶：**如果字符串在你使用的编程语言中是一种可变数据类型，请尝试使用 `O(1)` 额外空间复杂度的 **原地** 解法。

```c++
// 比较水的一种做法,没有体现所学的内容
class Solution {
public:
    string reverseWords(string s) {
        istringstream is(s);
        vector<string> s_container;
        string temp;
        while (is >> temp) {
            s_container.push_back(temp);
        }
        int forward = 0, behind = s_container.size() - 1;
        while (forward < behind) {
            swap(s_container[forward], s_container[behind]);
            forward++;
            behind--;
        }
        string ans;
        for (int i = 0; i < s_container.size(); i++) {
            if (i == 0) {
                ans = s_container[i];
            } else {
                ans += " " + s_container[i];
            }
        }
        return ans;
    }
};
```

```c++
// 移除多余空格
// 翻转字符串
// 翻转单词片段

```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(s.strip().split()[::-1])
```

- 切片操作
- 字符串相关

#### [右旋字符串](https://kamacoder.com/problempage.php?pid=1065)

> ###### 题目描述
>
> 字符串的右旋转操作是把字符串尾部的若干个字符转移到字符串的前面。给定一个字符串 s 和一个正整数 k，请编写一个函数，将字符串中的后面 k 个字符移到字符串的前面，实现字符串的右旋转操作。
>
> 例如，对于输入字符串 "abcdefg" 和整数 2，函数应该将其转换为 "fgabcde"。
>
> ###### 输入描述
>
> 输入共包含两行，第一行为一个正整数 k，代表右旋转的位数。第二行为字符串 s，代表需要旋转的字符串。
>
> ###### 输出描述
>
> 输出共一行，为进行了右旋转操作后的字符串。
>
> ###### 输入示例
>
> ```
> 2
> abcdefg
> ```
>
> ###### 输出示例
>
> ```
> fgabcde
> ```
>
> ###### 提示信息
>
> 数据范围：
> 1 <= k < 10000,
> 1 <= `s.length` < 10000;

```c++
#include <iostream>
#include <string>
using namespace std;
int main() {
  string temp;
  int k = 0;
  cin >> k;
  cin >> temp;
  int startIndex = temp.size() - k;
  string targetS = temp.substr(startIndex, k);
  temp.erase(startIndex, k);
  string ans = targetS + temp;
  cout << ans << endl;
  return 0;
}
```

```python
k = int(input())
s = str(input())
n = len(s)
s = s[n-k:len(s):]+s[0:n-k:]
print(s)
```

#### [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

> 给你两个字符串 `haystack` 和 `needle` ，请你在 `haystack` 字符串中找出 `needle` 字符串的第一个匹配项的下标（下标从 0 开始）。如果 `needle` 不是 `haystack` 的一部分，则返回 `-1` 。
>
>  
>
> **示例 1：**
>
> ```
> 输入：haystack = "sadbutsad", needle = "sad"
> 输出：0
> 解释："sad" 在下标 0 和 6 处匹配。
> 第一个匹配项的下标是 0 ，所以返回 0 。
> ```
>
> **示例 2：**
>
> ```
> 输入：haystack = "leetcode", needle = "leeto"
> 输出：-1
> 解释："leeto" 没有在 "leetcode" 中出现，所以返回 -1 。
> ```
>
>  
>
> **提示：**
>
> - `1 <= haystack.length, needle.length <= 104`
> - `haystack` 和 `needle` 仅由小写英文字符组成

```c++
```

```python
```

#### [459. 重复的子字符串](https://leetcode.cn/problems/repeated-substring-pattern/)

> 给定一个非空的字符串 `s` ，检查是否可以通过由它的一个子串重复多次构成。
>
>  
>
> **示例 1:**
>
> ```
> 输入: s = "abab"
> 输出: true
> 解释: 可由子串 "ab" 重复两次构成。
> ```
>
> **示例 2:**
>
> ```
> 输入: s = "aba"
> 输出: false
> ```
>
> **示例 3:**
>
> ```
> 输入: s = "abcabcabcabc"
> 输出: true
> 解释: 可由子串 "abc" 重复四次构成。 (或子串 "abcabc" 重复两次构成。)
> ```
>
>  
>
> **提示：**
>
>
>
> - `1 <= s.length <= 104`
> - `s` 由小写英文字母组成

```c++

```

```python
```

## 第五章 栈与队列

### 理论基础

#### 容器和容器适配器

> 在 C++ 标准库中，容器适配器是一种特殊的**类模板**，它提供了一种特定的接口和行为，这些接口和行为是通过封装一个底层容器类（如 `vector`, `deque`, 或 `list`）来实现的。简而言之，容器适配器是对现有容器类的封装，以提供不同的功能或接口。
>
> 容器适配器和基础容器之间的主要关系可以总结如下：
>
> ### 容器 (Container)
>
> - 容器是数据结构的一种实现，它可以存储元素并提供对它们的访问。标准库中的容器包括序列容器（如 `vector`, `deque`, `list`）和关联容器（如 `set`, `map`, `unordered_set`, `unordered_map`）。
> - 每个容器都提供了一套丰富的成员函数，用于插入、删除、遍历和访问元素。
> - 它们的成员函数有相似性
>
> ### 容器适配器 (Container Adapters)
>
> - 容器适配器提供了一种不同的接口，使得基础容器的行为更符合特定的数据结构如栈（stack）、队列（queue）和优先队列（priority_queue）。
> - 标准库中有三种容器适配器：`stack`, `queue`, `priority_queue`。
> - 容器适配器通常限制了对其底层容器的直接访问，只提供了与其模拟的数据结构相匹配的一组操作。例如，`stack` 提供 `push`, `pop`, `top`, `size`, 和 `empty` 操作，但它不允许对底层容器的随机访问。
> - 容器适配器在创建时可以指定一个底层容器类型（默认通常是 `deque`,即在不指定的时候），并且可以通过特定的成员函数来操作这个底层容器。
>
> ### 关系
>
> - 容器适配器并不直接实现数据存储和管理的逻辑本身，而是依赖于一个底层的容器来实现这些功能。
> - 容器适配器通过其接口提供了对底层容器的封装，并限定了可以执行的操作集合。
> - 开发者可以选择使用底层容器直接操作更丰富的接口，或者使用容器适配器来获得更简单、更特定的操作集。
> - 不同的容器会使得容器适配器的性能有所不同
>
>   - 容器适配器的性能在很大程度上取决于它所封装的底层容器的性能特性。
>
>     每种容器都有其特定的性能权衡，例如：
>
>     - `vector` 在尾端添加或移除元素非常快，但在中间或开始位置插入或删除元素就比较慢。
>     - `deque` 支持高效地在两端添加或移除元素，但是在中间进行操作的效率较低。
>     - `list` 是一个双向链表，可以在任何位置快速插入和删除元素，但它不支持随机访问。
>
>     当一个容器适配器使用不同的底层容器时，其性能会受到以下因素的影响：
>
>## `stack`
>
>     - `stack` 默认使用 `deque` 作为其底层容器，也可以使用 `vector` 或 `list`。因为 `stack` 的操作主要是在一端添加或移除元素，所以 `deque` 和 `list` 都是不错的选择。如果选择了 `vector` 作为底层容器，尽管在尾端操作非常高效，但如果 `vector` 因容量不足而需要扩容，可能会导致较大的性能开销。
>
>## `queue`
>
>     - `queue` 也默认使用 `deque`，它在两端都进行操作，因此 `deque` 是一个合适的选择，因为它在两端操作都很高效。如果使用 `list` 作为底层容器，也能获得类似的性能，但不能使用 `vector`，因为 `vector` 在头部添加或移除元素的性能很差。
>
>## `priority_queue`
>
>     - `priority_queue` 默认使用 `vector` 作为底层容器，并在此基础上提供一个**二叉堆**(将在下一章介绍)的接口。`vector` 在这种情况下是一个很好的选择，因为二叉堆需要随机访问容器的元素，并且 `vector` 提供了最佳的随机访问性能。虽然 `deque` 也可以用作底层容器，但通常 `vector` 会更优。
>                                                                                                                                        
>     因此，根据不同的应用场景和性能要求，选择恰当的底层容器对于容器适配器的性能至关重要。这也是为什么在设计容器适配器时，标准库允许开发者指定底层容器类型的原因之一。
>
>
> 总之，容器适配器的设计允许程序员依据具体需求选择更合适的数据结构，同时能够复用底层容器的实现，提高了代码的复用性和抽象层次。

**栈是以底层容器完成其所有的工作，对外提供统一的接口，底层容器是可插拔的（也就是说我们可以控制使用哪种容器来实现栈的功能）。**

- **栈(stack) : 先进先出**
- **队列(queue) : 先进后出**

![栈与队列理论1](./codeNote.assets/20210104235346563.png)

这只是一种数据的结构,就算不使用**STL**,也可以通过基本的数据类型设计配合适当的函数实现相应的方法

![栈与队列理论3](./codeNote.assets/20210104235459376.png)

#### 栈(先进后出)

**递归的实现是栈：每一次递归调用都会把函数的局部变量、参数值和返回地址等压入调用栈中**，然后递归返回的时候，从栈顶弹出上一次递归的各项参数，所以这就是递归为什么可以返回上一层位置的原因。

- 基本方法
  -  删除栈顶的一个元素
  - 向栈中压入一个元素
  - 返回对于栈顶元素的引用
  - 清空栈
- 拓展方法
  - 返回栈中元素的个数
  - 返回栈是否为空

#### 队列(先进先出)

- 基本方法

  - 将元素item添加到队列的末尾。
  - 从队列的头部移除并返回元素。
  - 返回队列头部的元素，但不对队列进行修改。
  - 清空队列，将队列中所有元素移除

- 拓展方法

  - 检查队列是否为空，如果队列为空则返回true，否则返回false。
  - 返回队列中元素的个数。

### 使用细节

#### `std::stack`

1. 构造函数：`std::stack` 可以通过不同的构造函数进行初始化：
   - `std::stack<T>`：创建一个空的堆栈，其中 `T` 是堆栈中元素的类型。
   - `std::stack<T, Container>`：使用指定的容器类型 `Container` 创建一个空的堆栈。
2. `push(element)`：将元素压入堆栈的顶部（即栈顶）。
3. `pop()`：从堆栈顶部（栈顶）移除元素，没有返回值。
4. `top()`：返回堆栈顶部（栈顶）的元素，但不从堆栈中移除它。
5. `empty()`：检查堆栈是否为空，返回 `true` 如果堆栈为空，否则返回 `false`。
6. `size()`：返回堆栈中元素的数量。

#### `std::queue`(注意这是一种容器)

1. 构造函数：`std::queue` 可以通过不同的构造函数进行初始化：
   - `std::queue<T>`：创建一个空的队列，其中 `T` 是队列中元素的类型。
   - `std::queue<T, Container>`：使用指定的容器类型 `Container` 创建一个空的队列。
2. `push(element)`：将元素压入队列的尾部。
3. `pop()`：从队列的头部移除元素，没有返回值。
4. `front()`：返回队列的头部元素，但不从队列中移除它。
5. `back()`：返回队列的尾部元素，但不从队列中移除它。
6. `empty()`：检查队列是否为空，返回 `true` 如果队列为空，否则返回 `false`。
7. `size()`：返回队列中元素的数量。

#### `std::deque`

1. 构造函数：`std::deque` 可以通过不同的构造函数进行初始化：
   - `std::deque<T>`：创建一个空的双端队列，其中 `T` 是队列中元素的类型。
   - `std::deque<T>(size, value)`：创建一个包含 `size` 个复制的 `value` 的双端队列。
   - `std::deque<T>(first, last)`：创建一个包含从迭代器 `first` 到 `last` 的元素的双端队列。
2. `push_back(element)`：将元素插入到双端队列的尾部。
3. `push_front(element)`：将元素插入到双端队列的头部。
4. `pop_back()`：从双端队列的尾部移除元素，没有返回值。
5. `pop_front()`：从双端队列的头部移除元素，没有返回值。
6. `back()`：返回双端队列的尾部元素，但不从双端队列中移除它。
7. `front()`：返回双端队列的头部元素，但不从双端队列中移除它。
8. `empty()`：检查双端队列是否为空，返回 `true` 如果双端队列为空，否则返回 `false`。
9. `size()`：返回双端队列中元素的数量。
10. `clear()`：移除双端队列中的所有元素。

#### `std::priority_queue`

1. 构造函数：`std::priority_queue` 可以通过不同的构造函数进行初始化：
   - `std::priority_queue<T>`：创建一个空的优先级队列，其中 `T` 是队列中元素的类型。
   - `std::priority_queue<T>(compare)`：创建一个空的优先级队列，并指定一个比较函数来定义元素的排序顺序。
   - `std::priority_queue<T>(compare, container)`：创建一个优先级队列，并使用指定的比较函数和容器来初始化队列。
2. `push(element)`：将元素插入到优先级队列中。插入的元素会根据比较函数进行排序。
3. `pop()`：从优先级队列中移除顶部（最大或最小）的元素。
4. `top()`：返回优先级队列中顶部（最大或最小）的元素，但不从队列中移除它。
5. `empty()`：检查优先级队列是否为空，返回 `true` 如果优先级队列为空，否则返回 `false`。
6. `size()`：返回优先级队列中元素的数量。
7. `swap(other)`：交换两个优先级队列的内容。
8. 自定义比较函数

### 典型例题

#### [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

  > 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）：
  >
  > 实现 `MyQueue` 类：
  >
  > - `void push(int x)` 将元素 x 推到队列的末尾
  > - `int pop()` 从队列的开头移除并返回元素
  > - `int peek()` 返回队列开头的元素
  > - `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`
  >
  > **说明：**
  >
  > - 你 **只能** 使用标准的栈操作 —— 也就是只有 `push to top`, `peek/pop from top`, `size`, 和 `is empty` 操作是合法的。
  > - 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
  >
  >  
  >
  > **示例 1：**
  >
  > ```
  > 输入：
  > ["MyQueue", "push", "push", "peek", "pop", "empty"]
  > [[], [1], [2], [], [], []]
  > 输出：
  > [null, null, null, 1, 1, false]
  > 
  > 解释：
  > MyQueue myQueue = new MyQueue();
  > myQueue.push(1); // queue is: [1]
  > myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
  > myQueue.peek(); // return 1
  > myQueue.pop(); // return 1, queue is [2]
  > myQueue.empty(); // return false
  > ```
  >
  >  
  >
  > **提示：**
  >
  > - `1 <= x <= 9`
  > - 最多调用 `100` 次 `push`、`pop`、`peek` 和 `empty`
  > - 假设所有操作都是有效的 （例如，一个空的队列不会调用 `pop` 或者 `peek` 操作）
  >
  >  
  >
  > **进阶：**
  >
  > - 你能否实现每个操作均摊时间复杂度为 `O(1)` 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 `O(n)` ，即使其中一个操作可能花费较长时间。

  ```c++
  class MyQueue {
  private:
      stack<int> one;
  
  public:
      MyQueue() {}
  
      void push(int x) { one.push(x); }
  
      int pop() {
          stack<int> temp;
          while (!one.empty()) {
              temp.push(one.top());
              one.pop();
          }
          int ans = temp.top();
          temp.pop();
          while (!temp.empty()) {
              one.push(temp.top());
              temp.pop();
          }
          return ans;
      }
  
      int peek() {
          stack<int> temp;
          while (!one.empty()) {
              temp.push(one.top());
              one.pop();
          }
          int ans = temp.top();
          while (!temp.empty()) {
              one.push(temp.top());
              temp.pop();
          }
          return ans;
      }
  
      bool empty() { return one.empty(); }
  };
  /**
   * Your MyQueue object will be instantiated and called as such:
   * MyQueue* obj = new MyQueue();
   * obj->push(x);
   * int param_2 = obj->pop();
   * int param_3 = obj->peek();
   * bool param_4 = obj->empty();
   */
  ```

- 但是上述办法时间复杂度上不占优势,因此推荐使用以下方式

  - 使用栈来模式队列的行为，如果仅仅用一个栈，是一定不行的，所以需要两个栈**一个输入栈，一个输出栈**，这里要注意输入栈和输出栈的关系。

  - gif模拟![](./codeNote.assets/232.用栈实现队列版本2.gif)

  - 在push数据的时候，只要数据放进输入栈就好，**但在pop的时候，操作就复杂一些，输出栈如果为空，就把进栈数据全部导入进来（注意是全部导入）**，再从出栈弹出数据，如果输出栈不为空，则直接从出栈弹出数据就可以了。

    最后如何判断队列为空呢？**如果进栈和出栈都为空的话，说明模拟的队列为空了。**

  - 在两个栈中的转换改变了数据的顺序(相当于把栈翻转)

```c++
class MyQueue {
private:
    stack<int> stack_in;
    stack<int> stack_out;

public:
    MyQueue() {}

    void push(int x) { stack_in.push(x); }

    int pop() {
        if (stack_out.empty()) {
            while (!stack_in.empty()) {
                stack_out.push(stack_in.top());
                stack_in.pop();
            }
        }
        int result = stack_out.top();
        stack_out.pop();
        return result;
    }

    int peek() {
        int result = this->pop();
        stack_out.push(result);
        return result;
    }

    bool empty() { return stack_in.empty() && stack_out.empty(); }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

- python可以使用list完成队列或者栈的模拟

  ```python
  class MyQueue:
      def __init__(self):
          self.stack = []
  
      def push(self, x: int) -> None:
          self.stack.append(x)
  
      def pop(self) -> int:
          return self.stack.pop(0)
  
      def peek(self) -> int:
          return self.stack[0]
  
      def empty(self) -> bool:
          return len(self.stack) == 0
  
  
  # Your MyQueue object will be instantiated and called as such:
  # obj = MyQueue()
  # obj.push(x)
  # param_2 = obj.pop()
  # param_3 = obj.peek()
  # param_4 = obj.empty()
  ```

#### [225. 用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

  > 请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（`push`、`top`、`pop` 和 `empty`）。
  >
  > 实现 `MyStack` 类：
  >
  > - `void push(int x)` 将元素 x 压入栈顶。
  > - `int pop()` 移除并返回栈顶元素。
  > - `int top()` 返回栈顶元素。
  > - `boolean empty()` 如果栈是空的，返回 `true` ；否则，返回 `false` 。
  >
  >  
  >
  > **注意：**
  >
  > - 你只能使用队列的基本操作 —— 也就是 `push to back`、`peek/pop from front`、`size` 和 `is empty` 这些操作。
  > - 你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
  >
  >  
  >
  > **示例：**
  >
  > ```
  > 输入：
  > ["MyStack", "push", "push", "top", "pop", "empty"]
  > [[], [1], [2], [], [], []]
  > 输出：
  > [null, null, null, 2, 2, false]
  > 
  > 解释：
  > MyStack myStack = new MyStack();
  > myStack.push(1);
  > myStack.push(2);
  > myStack.top(); // 返回 2
  > myStack.pop(); // 返回 2
  > myStack.empty(); // 返回 False
  > ```
  >
  >  
  >
  > **提示：**
  >
  > - `1 <= x <= 9`
  > - 最多调用`100` 次 `push`、`pop`、`top` 和 `empty`
  > - 每次调用 `pop` 和 `top` 都保证栈不为空
  >
  >  
  >
  > **进阶：**你能否仅用一个队列来实现栈。

  ```c++
  class MyStack {
  private:
      queue<int> one;
  
  public:
      MyStack() {}
  
      void push(int x) { one.push(x); }
  
      int pop() {
          queue<int> temp;
          int ans = one.back();
          while (one.size() != 1) {
              temp.push(one.front());
              one.pop();
          }
          one.pop();
          while (!temp.empty()) {
              one.push(temp.front());
              temp.pop();
          }
          return ans;
      }
  
      int top() { return one.back(); }
  
      bool empty() { return one.empty(); }
  };
  
  /**
   * Your MyStack object will be instantiated and called as such:
   * MyStack* obj = new MyStack();
   * obj->push(x);
   * int param_2 = obj->pop();
   * int param_3 = obj->top();
   * bool param_4 = obj->empty();
   */
  ```

  ```python
  class MyStack:
      def __init__(self):
          self.stack = []
  
      def push(self, x: int) -> None:
          self.stack.append(x)
  
      def pop(self) -> int:
          return self.stack.pop(-1)
  
      def top(self) -> int:
          return self.stack[-1]
  
      def empty(self) -> bool:
          return len(self.stack) == 0
  
  
  # Your MyStack object will be instantiated and called as such:
  # obj = MyStack()
  # obj.push(x)
  # param_2 = obj.pop()
  # param_3 = obj.top()
  # param_4 = obj.empty()
  ```

- 进阶思考:使用一个队列来完成栈的模拟

```c++
// 单一队列实现
class MyStack {
private:
    queue<int> one;

public:
    MyStack() {}

    void push(int x) { one.push(x); }

    int pop() {
        int one_size = one.size();
        // 将除了最后一个元素插入到队列尾部
        one_size--;
        while (one_size--) {
            one.push(one.front());
            one.pop();
        }
        int result = one.front();
        one.pop();
        return result;
    }

    int top() { return one.back(); }

    bool empty() { return one.empty(); }
};
```

#### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

> 给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。
>
> 有效字符串需满足：
>
> 1. 左括号必须用相同类型的右括号闭合。
> 2. 左括号必须以正确的顺序闭合。
> 3. 每个右括号都有一个对应的相同类型的左括号。
>
>  
>
> **示例 1：**
>
> ```
> 输入：s = "()"
> 输出：true
> ```
>
> **示例 2：**
>
> ```
> 输入：s = "()[]{}"
> 输出：true
> ```
>
> **示例 3：**
>
> ```
> 输入：s = "(]"
> 输出：false
> ```
>
>  
>
> **提示：**
>
> - `1 <= s.length <= 104`
> - `s` 仅由括号 `'()[]{}'` 组成

```c++
class Solution {
private:
    bool isPair(char down, char up) {
        return (down == '(' && up == ')') || (down == '[' && up == ']') ||
               (down == '{' && up == '}');
    }

public:
    bool isValid(string s) {
        stack<char> temp;
        for (auto iter : s) {
            if (!temp.empty()) {
                if (isPair(temp.top(), iter)) {
                    temp.pop();
                } else {
                    temp.push(iter);
                }
            } else {
                temp.push(iter);
            }
        }
        return temp.empty();
    }
};
```

```python
class Solution:
    def isPair(self, down: str, up: str) -> bool:
        return (
            (down == "(" and up == ")")
            or (down == "[" and up == "]")
            or (down == "{" and up == "}")
        )

    def isValid(self, s: str) -> bool:
        stack = []
        for item in s:
            if len(stack) != 0:
                if self.isPair(stack[-1], item):
                    stack.pop(-1)
                else:
                    stack.append(item)
            else:
                stack.append(item)
        return len(stack) == 0

```

#### [1047. 删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

> 给出由小写字母组成的字符串 `S`，**重复项删除操作**会选择两个相邻且相同的字母，并删除它们。
>
> 在 S 上反复执行重复项删除操作，直到无法继续删除。
>
> 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
>
>  
>
> **示例：**
>
> ```
> 输入："abbaca"
> 输出："ca"
> 解释：
> 例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
> ```
>
>  
>
> **提示：**
>
> 1. `1 <= S.length <= 20000`
> 2. `S` 仅由小写英文字母组成。

```c++
class Solution {
public:
    string removeDuplicates(string s) {
        stack<char> temp;
        for (auto iter : s) {
            if (!temp.empty()) {
                if (iter == temp.top()) {
                    temp.pop();
                } else {
                    temp.push(iter);
                }
            } else {
                temp.push(iter);
            }
        }
        string ans;
        while (!temp.empty()) {
            ans.push_back(temp.top());
            temp.pop();
        }
        int left = 0, right = ans.size() - 1;
        while (left < right) {
            swap(ans[left], ans[right]);
            left++;
            right--;
        }
        return ans;
    }
};
// 使用string完成栈的模拟
class Solution {
public:
    string removeDuplicates(string s) {
        string result;
        for (char c : s) {
            if (!result.empty() && result.back() == c) {
                // If the current character is the same as the last character
                // in the result, remove the last character from the result.
                result.pop_back();
            } else {
                // If it's not a duplicate(重复), append it to the result.
                result.push_back(c);
            }
        }
        return result;
    }
};
```

```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        ans = []
        for item in s:
            if len(ans) != 0 and ans[-1] == item:
                ans.pop(-1)
            else:
                ans.append(item)
        return "".join(ans)

```

#### [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

> 给你一个字符串数组 `tokens` ，表示一个根据 [逆波兰表示法](https://baike.baidu.com/item/逆波兰式/128437) 表示的算术表达式。
>
> 请你计算该表达式。返回一个表示表达式值的整数。
>
> **注意：**
>
> - 有效的算符为 `'+'`、`'-'`、`'*'` 和 `'/'` 。
> - 每个操作数（运算对象）都可以是一个整数或者另一个表达式。
> - 两个整数之间的除法总是 **向零截断** 。
> - 表达式中不含除零运算。
> - 输入是一个根据逆波兰表示法表示的算术表达式。
> - 答案及所有中间计算结果可以用 **32 位** 整数表示。
>
>  
>
> **示例 1：**
>
> ```
> 输入：tokens = ["2","1","+","3","*"]
> 输出：9
> 解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
> ```
>
> **示例 2：**
>
> ```
> 输入：tokens = ["4","13","5","/","+"]
> 输出：6
> 解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
> ```
>
> **示例 3：**
>
> ```
> 输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
> 输出：22
> 解释：该算式转化为常见的中缀算术表达式为：
>   ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
> = ((10 * (6 / (12 * -11))) + 17) + 5
> = ((10 * (6 / -132)) + 17) + 5
> = ((10 * 0) + 17) + 5
> = (0 + 17) + 5
> = 17 + 5
> = 22
> ```
>
>  
>
> **提示：**
>
> - `1 <= tokens.length <= 104`
> - `tokens[i]` 是一个算符（`"+"`、`"-"`、`"*"` 或 `"/"`），或是在范围 `[-200, 200]` 内的一个整数
>
>  
>
> **逆波兰表达式：**
>
> 逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。
>
> - 平常使用的算式则是一种中缀表达式，如 `( 1 + 2 ) * ( 3 + 4 )` 。
> - 该算式的逆波兰表达式写法为 `( ( 1 2 + ) ( 3 4 + ) * )` 。
>
> 逆波兰表达式主要有以下两个优点：
>
> - 去掉括号后表达式无歧义，上式即便写成 `1 2 + 3 4 + *`也可以依据次序计算出正确结果。
> - 适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中

```c++
class Solution {
private:
    int calu(int left, int right, char symbol) {
        switch (symbol) {
        case '+':
            return left + right;
        case '-':
            return left - right;
        case '*':
            return left * right;
        case '/':
            return left / right;
        }
        return 0;
    }

public:
    int evalRPN(vector<string>& tokens) {
        stack<int> temp;
        for (auto iter : tokens) {
            if ((iter[0] >= '0' && iter[0] <= '9') || (iter.size() > 1)) {
                temp.push(stoi(iter));
            } else {
                int right = temp.top();
                temp.pop();
                int left = temp.top();
                temp.pop();
                temp.push(calu(left, right, iter[0]));
            }
        }
        return temp.top();
    }
};
```

```python
class Solution:
    def calu(self, left: int, right: int, symbol: str) -> int:
        if symbol == "+":
            return left + right
        elif symbol == "-":
            return left - right
        elif symbol == "*":
            return left * right
        else:
            return int(left / right) # 实现整数除法,向0取整;如果是//,则向下取整,与c++中不符

    def evalRPN(self, tokens: List[str]) -> int:
        temp = []
        flag = ["+", "-", "*", "/"]
        for item in tokens:
            if item in flag:
                ans = self.calu(temp[-2], temp[-1], item)
                temp.pop(-1)
                temp.pop(-1)
                print(ans, "\n")
                temp.append(ans)
            else:
                temp.append(int(item))
        return temp[0]
```

#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

> 给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。
>
> 返回 *滑动窗口中的最大值* 。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
> 输出：[3,3,5,5,6,7]
> 解释：
> 滑动窗口的位置                最大值
> ---------------               -----
> [1  3  -1] -3  5  3  6  7       3
>  1 [3  -1  -3] 5  3  6  7       3
>  1  3 [-1  -3  5] 3  6  7       5
>  1  3  -1 [-3  5  3] 6  7       5
>  1  3  -1  -3 [5  3  6] 7       6
>  1  3  -1  -3  5 [3  6  7]      7
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [1], k = 1
> 输出：[1]
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums.length <= 105`
> - `-104 <= nums[i] <= 104`
> - `1 <= k <= nums.length`

```c++
// 超时做法(直观暴力做法)
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int head = 0, tail = k - 1;
        vector<int> window_max;
        while (tail < nums.size()) {
            int temp = INT_MIN;
            for (int i = head; i <= tail; i++) {
                temp = max(temp, nums[i]);
            }
            window_max.push_back(temp);
            head++;
            tail++;
        }
        return window_max;
    }
};


// 单调队列
class Solution {
private:
    class MyQueue {
    private:
        deque<int> que;

    public:
        void pop(int value) {
            if (!que.empty() && value == que.front()) {
                que.pop_front();
            }
        }
        void push(int value) {
            while (!que.empty() && value > que.back()) {
                que.pop_back();
            }
            que.push_back(value);
        }
        int front() { return que.front(); }
    };

public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        MyQueue que;
        vector<int> ans;
        for (int i = 0; i < k; i++) {
            que.push(nums[i]);
        }
        ans.push_back(que.front());
        for (int i = k; i < nums.size(); i++) {
            que.pop(nums[i - k]);
            que.push(nums[i]);
            ans.push_back(que.front());
        }
        return ans;
    }
};
```

```python
class MyQueue:
    def __init__(self):
        self.que = []

    def pop(self, value: int):
        if value == self.que[0] and len(self.que) != 0:
            self.que.pop(0)

    def push(self, value: int):
        while len(self.que) > 0 and value > self.que[-1]:
            self.que.pop(-1)
        self.que.append(value)

    def front(self) -> int:
        return self.que[0]


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        que = MyQueue()
        for i in range(k):
            que.push(nums[i])
        ans.append(que.front())
        for i in range(k, len(nums)):
            que.pop(nums[i - k])
            que.push(nums[i])
            ans.append(que.front())
        return ans

```

#### [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

> 给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。
>
>  
>
> **示例 1:**
>
> ```
> 输入: nums = [1,1,1,2,2,3], k = 2
> 输出: [1,2]
> ```
>
> **示例 2:**
>
> ```
> 输入: nums = [1], k = 1
> 输出: [1]
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums.length <= 105`
> - `k` 的取值范围是 `[1, 数组中不相同的元素的个数]`
> - 题目数据保证答案唯一，换句话说，数组中前 `k` 个高频元素的集合是唯一的
>
>  
>
> **进阶：**你所设计算法的时间复杂度 **必须** 优于 `O(n log n)` ，其中 `n` 是数组大小。

```c++
class Solution {
private:
    static bool cmp(pair<int, int> a, pair<int, int> b) { return a.second > b.second; }
/*
静态成员函数的一个重要特点是它们不依赖于特定的对象实例，因此它们可以直接通过类名访问(Solution::cmp)。这使得静态成员函数成为类的全局函数，可以在没有对象实例的情况下使用。
*/
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> pq_map;
        for (auto iter : nums) {
            pq_map[iter]++;
        }
        // 建立映射关系
        vector<pair<int, int>> ans;
        for (auto iter : pq_map) {
            ans.push_back(iter);
        }
        // 将pair元素放在vector中,便于排序
        sort(ans.begin(), ans.end(), this->cmp);
        vector<int> res;
        for (int i = 0; i < k; i++) {
            res.push_back(ans[i].first);
        }
        return res;
    }
};
/*
这段代码的时间复杂度为O(nlogn)，其中n是nums数组的长度。
首先，遍历nums数组并使用unordered_map统计每个元素的出现频率的时间复杂度为O(n)。
然后，将unordered_map中的元素转存到vector中的时间复杂度为O(n)。
最后，使用sort函数对vector进行排序的时间复杂度为O(nlogn)。
因此，整个代码的时间复杂度为O(nlogn)。
*/
```

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        pq_dict = dict()
        for item in nums:
            pq_dict[item] = pq_dict.get(item, 0) + 1
            # 在Python中，如果尝试访问字典中不存在的键，会抛出 KeyError 异常。因此，在没有为字典中的值指定默认值的情况下，访问字典中不存在的键会引发异常。

        ans = []
  # 自定义的sort比较函数
        def custom_cmp(item):
            return item[1]

        for number, times in pq_dict.items():
            # 使用pq_dict.items() 方法来获取字典的键值对
            ans.append((number, times))
        ans.sort(key=custom_cmp, reverse=True)
        res = []
        for item in ans[:k]:
            res.append(item[0])
        return res

```

- 还有一种使用**优先队列**的方式

```c++
class Solution {
public:
    // 小顶堆
    class mycomparison {
    public:
        bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
            return lhs.second > rhs.second;
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 要统计元素出现频率
        unordered_map<int, int> map; // map<nums[i],对应出现的次数>
        for (int i = 0; i < nums.size(); i++) {
            map[nums[i]]++;
        }

        // 对频率排序
        // 定义一个小顶堆，大小为k
        priority_queue<pair<int, int>, vector<pair<int, int>>, mycomparison> pri_que;

        // 用固定大小为k的小顶堆，扫面所有频率的数值
        for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
            pri_que.push(*it);
            if (pri_que.size() > k) { // 如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                pri_que.pop();
            }
        }

        // 找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
        vector<int> result(k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pri_que.top().first;
            pri_que.pop();
        }
        return result;

    }
};
// 来自Carl的解法
```

> 这里有一段来自GPT4.0的解析(关于重载运算符)
>
> 在 C++ 中，通过重载运算符，我们可以给特定的类或结构体赋予自定义的运算能力。在这个例子中，重载的是小于运算符 `operator<`，以便它可以用在 `priority_queue` 的上下文中，这样 `priority_queue` 就会使用这个运算符来决定两个元素的顺序。
>
> 这里是一个简单的解释：
>
> ### 1. 重载运算符的类定义 (`mycomparison`)
>
> ```cpp
> class mycomparison {
> public:
>     bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
>         return lhs.second > rhs.second;
>     }
> };
> ```
>
> - `class mycomparison` 定义了一个可调用的对象，它的作用是比较两个 `pair<int, int>` 类型的对象。
> - `operator()` 是一个函数调用运算符，可以让对象像函数一样被调用。
> - 在这个重载的 `operator()` 中，比较的是 `pair` 中的 `second` 值，它代表元素频率。
> - 返回值为 `true` 表示第一个参数 `lhs`（左手边的元素）应该位于第二个参数 `rhs`（右手边的元素）之后，即 `lhs` 在优先级队列中的优先级比 `rhs` 低。
> - 因为 `priority_queue` 默认是大顶堆（最大元素总是在顶部），对于 `pair`，默认比较的是 `first` 值，并且使用 `less` 运算符，这意味着较大的 `first` 会被放在前面。但在这个例子中，我们希望基于 `second` 的值创建一个小顶堆，并且希望频率较小的元素在堆的顶部，所以我们需要重载小于运算符来反转比较的逻辑。
>
> ### 2. 如何在 `priority_queue` 中使用
>
> ```cpp
> priority_queue<pair<int, int>, vector<pair<int, int>>, mycomparison> pri_que;
> ```
>
> - 这行代码创建了一个 `priority_queue`，它使用了 `mycomparison` 对象来决定元素的顺序。
> - 这意味着 `priority_queue` 会使用 `mycomparison` 类的 `operator()` 方法来决定两个元素的顺序。
> - 因为我们定义了 `operator()` 让它返回 `lhs.second > rhs.second`，`priority_queue` 就会把频率较小的元素放在顶部，实现小顶堆的效果。
>
> ### 3. 实际应用
>
> 在 `topKFrequent` 方法的循环中：
>
> ```cpp
> for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
>     pri_que.push(*it);
>     if (pri_que.size() > k) {
>         pri_que.pop();
>     }
> }
> ```
>
> - 每当一个新的元素（一个 `pair`）被插入到 `priority_queue` 中时，`priority_queue` 将使用 `mycomparison` 类的 `operator()` 方法来保持元素的正确顺序。
> - 如果队列的大小超过了 `k`，它将弹出顶部的元素（即频率最小的元素），因为我们希望保留频率最高的 `k` 个元素。
>
> 通过这种方式，当完成所有插入后，`priority_queue` 中将保留有 `k` 个频率最高的元素，而且由于是小顶堆的性质，顶部的元素将是这 `k` 个元素中频率最小的。最后，可以将这些元素弹出堆并存储到结果 `vector` 中，它们将按照频率从高到低的顺序排列。

## 第六章 二叉树

### 理论基础

**二叉树离不开递归!!!**

![二叉树大纲](./codeNote.assets/20210219190809451.png)

#### 二叉树的种类

- 满二叉树:如果一棵二叉树只有**度**为0的结点和度为2的结点，并且度为0的结点在同一层上，则这棵二叉树为满二叉树。(在图论和树的数据结构中，一个节点的“度”指的是它有多少个子节点。在二叉树中，每个节点的度可以是 0、1 或 2，也就是说，一个节点可以有最多两个子节点。)
  - ![img](./codeNote.assets/20200806185805576.png)
- 完全二叉树:在完全二叉树中，除了**最底层节点可能没填满外**，其余每层节点数都达到最大值，并且**最下面一层的节点都集中在该层最左边的若干位置**。若最底层为第 h 层（h从1开始），则该层包含 1~ 2^(h-1) 个节点。
  - ![img](./codeNote.assets/20200920221638903.png)

> 二叉树是一种树形数据结构，其中每个节点最多有两个子节点，通常称为左子节点和右子节点。下面是二叉树的一些基本概念和性质：
>
> 1. **节点（Node）**：
>    - **根节点（Root）**：没有父节点的节点是根节点，它是二叉树的起始节点。
>    - **叶节点（Leaf）**：度为 0 的节点，即没有子节点的节点称为叶节点。
>    - **内部节点（Internal node）**：至少有一个子节点的节点称为内部节点。
>
> 2. **子节点（Children）**：一个节点的直接后继者称为它的子节点。
>
> 3. **父节点（Parent）**：一个节点的直接前驱者称为它的父节点。
>
> 4. **兄弟节点（Siblings）**：具有相同父节点的节点称为兄弟节点。
>
> 5. **祖先节点（Ancestors）**：从根到该节点的路径上的所有节点都是该节点的祖先。
>
> 6. **后代节点（Descendants）**：该节点下的所有子树中的节点都是它的后代。
>
> 7. **深度（Depth）**：从根节点到一个节点的唯一路径上的边的数量。
>
> 8. **高度（Height）**：从一个节点到其任何叶子的最长路径上的边的数量。一个节点的高度也可以看作是它的所有子树的最大高度加一。叶节点的高度为 0。
>
> 二叉树可以特化为多种形式：
>
> - **完全二叉树（Complete Binary Tree）**：除了最后一层外，每一层都被完全填满，并且所有节点都尽可能地靠左排列。
>
> - **满二叉树（Full Binary Tree）**：每个节点都有 0 或 2 个子节点的二叉树。
>
> - **完美二叉树（Perfect Binary Tree）**：所有内部节点都有两个子节点，并且所有叶子都在同一层级上的二叉树。
>
> - **平衡二叉树（Balanced Binary Tree）**：任意两个叶子的深度差不会超过一定的阈值（例如 AVL 树）。
>
> - **二叉搜索树（Binary Search Tree, BST）**：对于树中的每个节点，其左子树中的所有值都小于该节点的值，其右子树中的所有值都大于该节点的值。
>
> 二叉树在计算机科学中应用广泛，主要用于数据的组织和管理，如搜索、排序和索引等操作。由于二叉树的结构特性，它的许多操作可以在对数时间内完成，这使得二叉树成为高效算法的基础。

- **二叉搜索树**(类似于二分查找的记录表)

  - 前面介绍的树，都没有数值的，而二叉搜索树是有数值的了，**二叉搜索树是一个有序树**。

    - **若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值；**
    - **若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值；**
    - **它的左、右子树也分别为二叉排序树**

    下面这两棵树都是搜索树

    - ![img](./codeNote.assets/20200806190304693.png)

- **平衡二叉搜索树**

  平衡二叉搜索树：又被称为AVL（Adelson-Velsky and Landis）树，且具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

  如图：

  ![img](./codeNote.assets/20200806190511967.png)

  最后一棵 不是平衡二叉树，因为它的左右两个子树的高度差的绝对值超过了1。

  **C++中map、set、multimap，multiset的底层实现都是平衡二叉搜索树**，所以map、set的增删操作时间时间复杂度是`logn` ，注意我这里没有说unordered_map、unordered_set，unordered_map、unordered_set底层实现是哈希表。

#### 二叉树的存储方式

  **二叉树可以链式存储，也可以顺序存储。**

  那么链式存储方式就用指针， 顺序存储的方式就是用数组。

  顾名思义就是顺序存储的元素在内存是连续分布的，而链式存储则是通过指针把分布在各个地址的节点串联一起。

  链式存储如图：

  ![img](./codeNote.assets/2020092019554618.png)

  链式存储是大家很熟悉的一种方式，那么我们来看看如何顺序存储呢？

  其实就是用数组来存储二叉树，顺序存储的方式如图：

  ![img](./codeNote.assets/20200920200429452.png)

  用数组来存储二叉树如何遍历的呢？

  **如果父节点的数组下标是 i，那么它的左孩子就是 i \* 2 + 1，右孩子就是 i \* 2 + 2。**(有明显的数学逻辑关系,可以据此写出表达式)

  但是用链式表示的二叉树，更有利于我们理解，所以一般我们都是用链式存储二叉树。

  **所以大家要了解，用数组依然可以表示二叉树。**

#### 二叉树的遍历方式

**如果是模拟前中后序遍历就用栈，如果是适合层序遍历就用队列，当然还是其他情况，那么就是 先用队列试试行不行，不行就用栈。**

  关于二叉树的遍历方式，要知道二叉树遍历的基本方式都有哪些。

  二叉树主要有两种遍历方式：

  1. 深度优先遍历：先往深走，遇到叶子节点再往回走。
  2. 广度优先遍历：一层一层的去遍历。

  **这两种遍历是图论中最基本的两种遍历方式**，后面在介绍图论的时候 还会介绍到。

  那么从深度优先遍历和广度优先遍历进一步拓展，才有如下遍历方式：

- 深度优先遍历
  - 前序遍历（递归法，迭代法）
  - 中序遍历（递归法，迭代法）
  - 后序遍历（递归法，迭代法）
  - **以上三种做法如果要手写的话既可以使用顺写法,也可以使用插入法(先写出最开始的三个节点,把剩下的节点插入进去,这种方法也便于劈分遍历结果还原二叉树)**
    - 第一种适合简单二叉树，第二种适合复杂的二叉树，不容易出错
- 广度优先遍历
  - 层次遍历（迭代法）

  在深度优先遍历中：有三个顺序，前中后序遍历， 有同学总分不清这三个顺序，经常搞混，我这里教大家一个技巧。

  **这里前中后，其实指的就是中间节点的遍历顺序**，只要大家记住**前中后序指的就是中间节点的位置**就可以了。

  看如下中间节点的顺序，就可以发现，中间节点的顺序就是所谓的遍历方式

- 前序遍历：中左右
- 中序遍历：左中右
- 后序遍历：左右中

  大家可以对着如下图，看看自己理解的前后中序有没有问题。

  ![img](./codeNote.assets/20200806191109896.png)

  最后再说一说二叉树中深度优先和广度优先遍历实现方式，我们做二叉树相关题目，经常会使用递归的方式来实现深度优先遍历，也就是实现前中后序遍历，使用递归是比较方便的。

  **之前我们讲栈与队列的时候，就说过栈其实就是递归的一种实现结构**，也就说前中后序遍历的逻辑其实都是可以借助栈使用递归的方式来实现的。

  而广度优先遍历的实现一般使用队列来实现，这也是队列先进先出的特点所决定的，因为需要先进先出的结构，才能一层一层的来遍历二叉树。

  **这里其实我们又了解了栈与队列的一个应用场景了。**

#### 二叉树的定义

刚刚我们说过了二叉树有两种存储方式顺序存储，和链式存储，顺序存储就是用数组来存，这个定义没啥可说的，我们来看看链式存储的二叉树节点的定义方式。

C++代码如下：

```cpp
// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right)
        : val(x), left(left), right(right) {}
};
```

大家会发现二叉树的定义 和链表是差不多的，相对于链表 ，二叉树的节点里多了一个指针， 有两个指针，指向左右孩子。

这里要提醒大家要注意二叉树节点定义的书写方式。

**在现场面试的时候 面试官可能要求手写代码，所以数据结构的定义以及简单逻辑的代码一定要锻炼白纸写出来。**

因为我们在刷`leetcode`的时候，节点的定义默认都定义好了，真到面试的时候，需要自己写节点定义的时候，有时候会一脸懵逼！

Python代码如下:

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
```

### 知识点&&典型例题

#### 二叉树的递归遍历

- **递归的三要素**

  1. **确定递归函数的参数和返回值：** 确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数， 并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。

  2. **确定终止条件：** 写完了递归算法, 运行的时候，经常会遇到栈溢出的错误，就是没写终止条件或者终止条件写的不对，操作系统也是用一个栈的结构来保存每一层递归的信息，如果递归没有终止，操作系统的内存栈必然就会溢出。

  3. **确定单层递归的逻辑：** 确定每一层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程。

- **示例**
  1. **确定递归函数的参数和返回值**：因为要打印出前序遍历节点的数值，所以参数里需要传入vector来放节点的数值，除了这一点就不需要再处理什么数据了也不需要有返回值，所以递归函数返回类型就是void，代码如下：

	```cpp
	void traversal(TreeNode* cur, vector<int>& vec)
	// 均使用地址传递,可以修改vec的值
	```
	
	2. **确定终止条件**：在递归的过程中，如何算是递归结束了呢，当然是当前遍历的节点是空了，那么本层递归就要结束了，所以如果当前遍历的这个节点是空，就直接return，代码如下：
	
	```cpp
	if (cur == NULL) return;
	```
	
	3. **确定单层递归的逻辑**：前序遍历是中左右的循序，所以在单层递归的逻辑，是要先取中节点的数值，代码如下：
	
	```cpp
	vec.push_back(cur->val);    // 中
	traversal(cur->left, vec);  // 左
	traversal(cur->right, vec); // 右
	// 结果保存时也会按照这个逻辑
	```

	​	单层递归的逻辑就是按照中左右的顺序来处理的，这样二叉树的前序遍历，基本就写完了，再看一下完整代码：

	​	前序遍历：
	
	```cpp
	class Solution {
	public:
	    void traversal(TreeNode* cur, vector<int>& vec) {
	        if (cur == NULL) return;
	        vec.push_back(cur->val);    // 中
	        traversal(cur->left, vec);  // 左
	        traversal(cur->right, vec); // 右
	    }
	    vector<int> preorderTraversal(TreeNode* root) {
	        vector<int> result;
	        traversal(root, result);
	        return result;
	    }
	};
	```

	​	那么前序遍历写出来之后，中序和后序遍历就不难理解了，代码如下：

	​	中序遍历：
	
	```cpp
	void traversal(TreeNode* cur, vector<int>& vec) {
	    if (cur == NULL) return;
	    traversal(cur->left, vec);  // 左
	    vec.push_back(cur->val);    // 中
	    traversal(cur->right, vec); // 右
	}
	```

	​	后序遍历：
	
	```cpp
	void traversal(TreeNode* cur, vector<int>& vec) {
	    if (cur == NULL) return;
	    traversal(cur->left, vec);  // 左
	    traversal(cur->right, vec); // 右
	    vec.push_back(cur->val);    // 中
	}
	```

##### [144. 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)

> 给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/inorder_1.jpg)
>
> ```
> 输入：root = [1,null,2,3]
> 输出：[1,2,3]
> ```
>
> **示例 2：**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：root = [1]
> 输出：[1]
> ```
>
> **示例 4：**
>
> ![img](./codeNote.assets/inorder_5.jpg)
>
> ```
> 输入：root = [1,2]
> 输出：[1,2]
> ```
>
> **示例 5：**
>
> ![img](./codeNote.assets/inorder_4.jpg)
>
> ```
> 输入：root = [1,null,2]
> 输出：[1,2]
> ```
>
>  
>
> **提示：**
>
> - 树中节点数目在范围 `[0, 100]` 内
> - `-100 <= Node.val <= 100`
>
>  
>
> **进阶：**递归算法很简单，你可以通过迭代算法完成吗？

```c++

class Solution {
private:
    void traversal(TreeNode* cur, vector<int>& vec) {
        if (cur == nullptr) {
            return;
        }
        vec.push_back(cur->val);
        traversal(cur->left, vec);
        traversal(cur->right, vec);
    }

public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> ans;
        traversal(root, ans);
        return ans;
    }
};
```

```python
def traversal(cur: Optional[TreeNode], vec: List) -> None:
    if cur == None:
        return
    vec.append(cur.val)
    traversal(cur.left, vec)
    traversal(cur.right, vec)

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        traversal(root, ans)
        return ans
```



##### [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)

> 给你一棵二叉树的根节点 `root` ，返回其节点值的 **后序遍历** 。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/pre1.jpg)
>
> ```
> 输入：root = [1,null,2,3]
> 输出：[3,2,1]
> ```
>
> **示例 2：**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：root = [1]
> 输出：[1]
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数目在范围 `[0, 100]` 内
> - `-100 <= Node.val <= 100`
>
>  
>
> **进阶：**递归算法很简单，你可以通过迭代算法完成吗？

```c++
class Solution {
private:
    void traversal(TreeNode* cur, vector<int>& vec) {
        if (cur == nullptr) {
            return;
        } 
        traversal(cur->left, vec);
        traversal(cur->right, vec);
        vec.push_back(cur->val);
    }

public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        traversal(root, ans);
        return ans;
    }
};
```

```python
def traversal(cur: Optional[TreeNode], vec: List) -> None:
    if cur == None:
        return
    traversal(cur.left, vec)
    traversal(cur.right, vec)
    vec.append(cur.val)


class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        traversal(root, ans)
        return ans

```



##### [94.二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

>给定一个二叉树的根节点 `root` ，返回 *它的 **中序** 遍历* 。
>
> 
>
>**示例 1：**
>
>![img](./codeNote.assets/inorder_1-1707709908198-9.jpg)
>
>```
>输入：root = [1,null,2,3]
>输出：[1,3,2]
>```
>
>**示例 2：**
>
>```
>输入：root = []
>输出：[]
>```
>
>**示例 3：**
>
>```
>输入：root = [1]
>输出：[1]
>```
>
> 
>
>**提示：**
>
>- 树中节点数目在范围 `[0, 100]` 内
>- `-100 <= Node.val <= 100`
>
> 
>
>**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？

```c++
class Solution {
private:
    void traversal(TreeNode* cur, vector<int>& vec) {
        if (cur == nullptr) {
            return;
        } 
        traversal(cur->left, vec);
        vec.push_back(cur->val);
        traversal(cur->right, vec);
    }

public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        traversal(root, ans);
        return ans;
    }
};
```

```python
def traversal(cur: Optional[TreeNode], vec: List) -> None:
    if cur == None:
        return

    traversal(cur.left, vec)
    vec.append(cur.val)
    traversal(cur.right, vec)

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        traversal(root, ans)
        return ans
```

#### 二叉树的迭代遍历

> 遍历还可以用非递归的方式!!!(递归的底层实现依靠栈)

##### 前序遍历和后序遍历（迭代法）

前序遍历的顺序是中左右，先访问的元素是中间节点，要处理的元素也是中间节点，所以才能写出相对简洁的代码，**因为要访问的元素和要处理的元素顺序是一致的，都是中间节点。**

1. **处理：将元素放进result数组中**
2. **访问：遍历节点**

前序遍历是中左右，每次先处理的是中间节点，那么先将根节点放入栈中，然后将右孩子加入栈，再加入左孩子。

为什么要先加入 右孩子，再加入左孩子呢？ 因为这样出栈的时候才是中左右的顺序。(倒序压入栈中)

动画如下：

![二叉树前序遍历（迭代法）](./codeNote.assets/二叉树前序遍历（迭代法）.gif)

**后序遍历**我们只需要调整一下先序遍历的代码顺序，就变成中右左的遍历顺序，然后在反转result数组，输出的结果顺序就是左右中了，如下图：

![前序到后序](./codeNote.assets/20200808200338924.png)

使用该方法重新解决上面的**前序遍历**和**后续遍历**习题

```c++
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> tree_stack;
        vector<int> ans;
        if (root == nullptr) {
            return ans;
        }
        tree_stack.push(root);
        TreeNode* cur = tree_stack.top();
        while (!tree_stack.empty()) {
            cur = tree_stack.top();
            ans.push_back(cur->val); // 中
            tree_stack.pop();
            if (cur->right) {
                tree_stack.push(cur->right); // 压入右
            }
            if (cur->left) {
                tree_stack.push(cur->left); // 压入左
            }
        }
        return ans;
    }
};
```

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        if root == None:
            return ans
        node_stack.append(root)
        while len(node_stack) != 0:
            cur = node_stack[-1]
            ans.append(cur.val)
            node_stack.pop(-1) # 令人疑惑,但是知道了原理才明白Python设计的巧妙之处
            if cur.right:
                node_stack.append(cur.right)
            if cur.left:
                node_stack.append(cur.left)
        return ans
```

> 解析:
>
> 在你提供的代码中，`cur` 是一个变量，它在每次循环中被赋值为 `node_stack` 列表中的最后一个元素。当执行 `node_stack.pop(-1)` 时，确实会从 `node_stack` 中删除最后一个元素，但是重要的是要理解 `cur` 此时已经持有了那个元素的引用。
>
> 在 Python 中，删除列表中的元素并不会删除对象本身，只是删除了该列表对该对象的引用。只要还有其他变量引用该对象，该对象就不会被垃圾回收机制回收。在这段代码中，`cur` 仍然引用了 `pop` 出来的 `TreeNode` 对象，因此可以安全地访问其成员变量 `val`、`left` 和 `right`。
>
> Python 的变量更像是指向对象的指针，而不是对象本身。当你从列表中移除一个对象时，你只是移除了列表中的一个“指针”，而不是对象本身。**只有当一个对象不再被任何变量引用时，Python 的垃圾回收机制才可能会销毁这个对象。**
>
> 因此，在上述代码中，尽管 `node_stack` 的最后一个元素被 `pop` 了，但由于我们有一个名为 `cur` 的变量仍然引用着该元素，该对象就不会被销毁，我们仍然可以访问 `cur.val`、`cur.left` 和 `cur.right`。

```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        stack<TreeNode*> tree_stack;
        vector<int> ans;
        if (root == nullptr) {
            return ans;
        }
        tree_stack.push(root);
        TreeNode* cur = tree_stack.top();
        while (!tree_stack.empty()) {
            cur = tree_stack.top();
            ans.push_back(cur->val); // 中
            tree_stack.pop();
            if (cur->left) {
                tree_stack.push(cur->left); // 压入左
            }
            if (cur->right) {
                tree_stack.push(cur->right); // 压入右
            }
        }
        return vector<int>(ans.rbegin(), ans.rend());
    }
};
```

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        if root == None:
            return ans
        node_stack.append(root)
        while len(node_stack) != 0:
            cur = node_stack[-1]
            ans.append(cur.val)
            node_stack.pop(-1)
            if cur.left:
                node_stack.append(cur.left)
            if cur.right:
                node_stack.append(cur.right)
        return ans[::-1]  # 切片操作会生成一个ans的翻转副本,ans本身没有改变
```

##### 中序遍历(迭代法)

中序遍历是左中右，**先访问**的是二叉树顶部的节点，然后一层一层向下访问，直到到达树左面的最底部，再开始处理节点（也就是在把节点的数值放进result数组中），这就造成了**处理顺序和访问顺序是不一致的。**

那么**在使用迭代法写中序遍历，就需要借用指针的遍历来帮助访问节点，栈则用来处理节点上的元素。**

动画如下：

![二叉树中序遍历（迭代法）](./codeNote.assets/二叉树中序遍历（迭代法）.gif)

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> node_stack;
        vector<int> ans;
        if (root == nullptr) {
            return ans;
        }
        TreeNode* cur = root;
        while (!node_stack.empty() || cur != nullptr) {
            if (cur != nullptr) {
                node_stack.push(cur);
                cur = cur->left;
            } else {
                cur = node_stack.top();
                node_stack.pop();
                ans.push_back(cur->val);
                cur = cur->right;
            }
        }
        return ans;
    }
};
```

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        cur = root
        if cur == None:
            return ans
        while len(node_stack) != 0 or cur != None:
            if cur != None:
                node_stack.append(cur)
                cur = cur.left
            else:
                cur = node_stack[-1]
                ans.append(cur.val)
                node_stack.pop(-1)
                cur = cur.right
        return ans
```

![中序遍历](./codeNote.assets/e2gxfxwqcz.gif)

> 怎么感觉使用代码更容易记住?



#### 二叉树的统一迭代遍历

> 虽然我觉得复杂,但是统一写法确实很整齐

我们以中序遍历为例，使用栈的话，**无法同时解决访问节点（遍历节点）和处理节点（将元素放进结果集）不一致的情况**。

**那我们就将访问的节点放入栈中，把要处理的节点也放入栈中但是要做标记。**

如何标记呢，**就是要处理的节点放入栈之后，紧接着放入一个空指针作为标记。(标记的节点始终是中间节点)** 这种方法也可以叫做**标记法**。

- 迭代法中序遍历

中序遍历代码如下：（详细注释）

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> st;
        if (root != NULL) st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            if (node != NULL) {
                st.pop(); // 将该节点弹出，避免重复操作，下面再将右中左节点添加到栈中
                if (node->right) st.push(node->right);  // 添加右节点（空节点不入栈）

                st.push(node);                          // 添加中节点
                st.push(NULL); // 中节点访问过，但是还没有处理，加入空节点做为标记。

                if (node->left) st.push(node->left);    // 添加左节点（空节点不入栈）
            } else { // 只有遇到空节点的时候，才将下一个节点放进结果集
                st.pop();           // 将空节点弹出
                node = st.top();    // 重新取出栈中元素
                st.pop();
                result.push_back(node->val); // 加入到结果集
            }
        }
        return result;
    }
};
```

看代码有点抽象我们来看一下动画(中序遍历)：

![中序遍历迭代（统一写法）](./codeNote.assets/中序遍历迭代（统一写法）.gif)

动画中，result数组就是最终结果集。

可以看出我们将访问的节点直接加入到栈中，但如果是处理的节点则后面放入一个空节点， 这样只有空节点弹出的时候，才将下一个节点放进结果集。(中间节点有一个空指针的帽子)

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        if root != None:
            node_stack.append(root)
        while len(node_stack) != 0:
            node = node_stack[-1]
            if node != None:
                node_stack.pop(-1)
                if node.right:
                    node_stack.append(node.right)
                node_stack.append(node)
                node_stack.append(None)
                if node.left:
                    node_stack.append(node.left)
            else:
                node_stack.pop(-1)
                node = node_stack[-1]
                node_stack.pop(-1)
                ans.append(node.val)
        return ans
```

此时我们再来看前序遍历代码。

- 迭代法前序遍历

迭代法前序遍历代码如下： (**注意此时我们和中序遍历相比仅仅改变了两行代码的顺序**)

```cpp
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> st;
        if (root != NULL) st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            if (node != NULL) {
                st.pop();
                if (node->right) st.push(node->right);  // 右
                if (node->left) st.push(node->left);    // 左
                st.push(node);                          // 中
                st.push(NULL);
            } else {
                st.pop();
                node = st.top();
                st.pop();
                result.push_back(node->val);
            }
        }
        return result;
    }
};
```

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        if root != None:
            node_stack.append(root)
        while len(node_stack) != 0:
            node = node_stack[-1]
            if node != None:
                node_stack.pop(-1)
                if node.right:
                    node_stack.append(node.right)
                if node.left:
                    node_stack.append(node.left)
                node_stack.append(node)
                node_stack.append(None)
            else:
                node_stack.pop(-1)
                node = node_stack[-1]
                node_stack.pop(-1)
                ans.append(node.val)
        return ans
```



- 迭代法后序遍历

后续遍历代码如下： (**注意此时我们和中序遍历相比仅仅改变了两行代码的顺序**)

```cpp
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> st;
        if (root != NULL) st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            if (node != NULL) {
                st.pop();
                st.push(node);                          // 中
                st.push(NULL);

                if (node->right) st.push(node->right);  // 右
                if (node->left) st.push(node->left);    // 左

            } else {
                st.pop();
                node = st.top();
                st.pop();
                result.push_back(node->val);
            }
        }
        return result;
    }
};
```

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        node_stack = []
        ans = []
        if root != None:
            node_stack.append(root)
        while len(node_stack) != 0:
            node = node_stack[-1]
            if node != None:
                node_stack.pop(-1)
                node_stack.append(node)
                node_stack.append(None)
                if node.right:
                    node_stack.append(node.right)
                if node.left:
                    node_stack.append(node.left)               
            else:
                node_stack.pop(-1)
                node = node_stack[-1]
                node_stack.pop(-1)
                ans.append(node.val)
        return ans
```

#### 二叉树的层序遍历

> 广度优先搜索(BFS)登场!

层序遍历一个二叉树。就是从左到右一层一层的去遍历二叉树。这种遍历的方式和我们之前讲过的都不太一样。

需要借用一个辅助数据结构即队列来实现，**队列先进先出，符合一层一层遍历的逻辑，而用栈先进后出适合模拟深度优先遍历也就是递归的逻辑。**

**而这种层序遍历方式就是图论中的广度优先遍历，只不过我们应用在二叉树上。**

使用队列实现二叉树广度优先遍历，动画如下：

![102二叉树的层序遍历](./codeNote.assets/102二叉树的层序遍历.gif)

这样就实现了层序从左到右遍历二叉树。

##### [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

> 给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。
>
>  
>
> **示例 1：**
>
> <img src="C:\Users\exqin\Desktop\Blog\codeNote.assets\tree1.jpg" alt="img" style="zoom:33%;" />
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：[[3],[9,20],[15,7]]
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [1]
> 输出：[[1]]
> ```
>
> **示例 3：**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
>  
>
> **提示：**
>
> - 树中节点数目在范围 `[0, 2000]` 内
> - `-1000 <= Node.val <= 1000`

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> node_queue;
        vector<vector<int>> ans;

        if (root == nullptr) {
            return ans;
        }
        int size = 1;
        vector<int> path;
        node_queue.push(root);
        while (!node_queue.empty()) {
            TreeNode* node = node_queue.front();
            node_queue.pop();
            path.push_back(node->val);
            size--;
            if (node->left) {
                node_queue.push(node->left);
            }
            if (node->right) {
                node_queue.push(node->right);
            }
            if (size == 0) {
                ans.push_back(path);
                size = node_queue.size();
                path.clear();
                path.resize(0);
            }
        }
        return ans;
    }
};

// 简便方法
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        if (root != nullptr) {
            que.push(root);
        }
        vector<vector<int>> result;
        while (!que.empty()) {
            int size = que.size();
            vector<int> vec;
            // 这里一定要使用固定大小size，不要使用que.size()，因为que.size是不断变化的
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                vec.push_back(node->val);
                if (node->left)
                    que.push(node->left);
                if (node->right)
                    que.push(node->right);
            }
            result.push_back(vec);
        }
        return result;
    }
};
```

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        node_queue = []
        if root == None:
            return ans
        node_queue.append(root)
        while len(node_queue) != 0:
            path = []
            size = len(node_queue)
            for i in range(size):
                node = node_queue[0]
                node_queue.pop(0)
                path.append(node.val)
                if node.left:
                    node_queue.append(node.left)
                if node.right:
                    node_queue.append(node.right)
            ans.append(path)
        return ans

```

- 还可以使用**递归**实现

```c++
class Solution {
public:
    void bfs(TreeNode* cur, int depth, vector<vector<int>>& ans) {
        if (cur == nullptr) {
            return;
        }
        if (ans.size() == depth) { //如果当前层不为空,则会增加一层来使用(depth从0开始,当depth和ans.size()相等的时候,说明层数少一,应当加一层)
            ans.push_back(vector<int>());
        }
        ans[depth].push_back(cur->val);
        bfs(cur->left, depth + 1, ans);
        bfs(cur->right, depth + 1, ans);
    }
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        bfs(root, 0, ans);
        return ans;
    }
};
```

```python
class Solution:
    def bfs(self, cur: Optional[TreeNode], depth: int, ans: List[List[int]]) -> None:
        if cur == None:
            return
        if len(ans) == depth:
            ans.append([])
        ans[depth].append(cur.val)
        self.bfs(cur.left, depth + 1, ans)
        self.bfs(cur.right, depth + 1, ans)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        self.bfs(root, 0, ans)
        return ans
```

##### [107. 二叉树的层序遍历 II](https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/)

> 给你二叉树的根节点 `root` ，返回其节点值 **自底向上的层序遍历** 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/tree1-1707752897746-80.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：[[15,7],[9,20],[3]]
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [1]
> 输出：[[1]]
> ```
>
> **示例 3：**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
>  
>
> **提示：**
>
> - 树中节点数目在范围 `[0, 2000]` 内
> - `-1000 <= Node.val <= 1000`

```c++
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        queue<TreeNode*> node_queue;
        vector<vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            int size = node_queue.size();
            vector<int> path;
            for (int i = 0; i < size; i++) {
                TreeNode* cur = node_queue.front();
                path.push_back(cur->val);
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
            ans.push_back(path);
        }
        return vector<vector<int>>(ans.rbegin(), ans.rend());
    }
};
```

```python
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        node_queue = []
        ans = []
        if root == None:
            return ans
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            path = []
            for i in range(size):
                cur = node_queue[0]
                path.append(cur.val)
                node_queue.pop(0)
                if cur.left:
                    node_queue.append(cur.left)
                if cur.right:
                    node_queue.append(cur.right)
            ans.append(path)
        return ans[::-1]
```

```c++
// 递归法
class Solution {
public:
    void bfs(vector<vector<int>>& ans, int depth, TreeNode* cur) {
        if (cur == nullptr) {
            return;
        }
        if (depth == ans.size()) {
            ans.push_back(vector<int>());
        }
        ans[depth].push_back(cur->val);
        bfs(ans, depth + 1, cur->left);
        bfs(ans, depth + 1, cur->right);
    }
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> ans;
        bfs(ans, 0, root);
        return vector<vector<int>>(ans.rbegin(), ans.rend());
    }
};
```

```python
class Solution:
    def bfs(self, cur: Optional[TreeNode], depth: int, ans: List[List[int]]) -> None:
        if cur == None:
            return
        if len(ans) == depth:
            ans.append([])
        ans[depth].append(cur.val)
        self.bfs(cur.left, depth + 1, ans)
        self.bfs(cur.right, depth + 1, ans)

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        self.bfs(root, 0, ans)
        return ans[::-1]
```

##### [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

> 给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
>
>  
>
> **示例 1:**
>
> ![img](./codeNote.assets/tree.jpg)
>
> ```
> 输入: [1,2,3,null,5,null,4]
> 输出: [1,3,4]
> ```
>
> **示例 2:**
>
> ```
> 输入: [1,null,3]
> 输出: [1,3]
> ```
>
> **示例 3:**
>
> ```
> 输入: []
> 输出: []
> ```
>
>  
>
> **提示:**
>
> - 二叉树的节点个数的范围是 `[0,100]`
> - `-100 <= Node.val <= 100` 

```c++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> ans;
        queue<TreeNode*> node_queue;
        if (root == nullptr) {
            return ans;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode* cur = node_queue.front();
                if (i == size - 1) {
                    ans.push_back(cur->val);
                }
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
        }
        return ans;
    }
};
```

```python
# 使用了递归法,迭代法没有难度
class Solution:
    def bfs(self, cur: Optional[TreeNode], depth: int, ans: List[List[int]]) -> None:
        if cur == None:
            return
        if len(ans) == depth:
            ans.append([])
        ans[depth].append(cur.val)
        self.bfs(cur.left, depth + 1, ans)
        self.bfs(cur.right, depth + 1, ans)

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        self.bfs(root, 0, ans)
        out = []
        for item in ans:
            out.append(item[-1])
        return out

```

##### [637. 二叉树的层平均值](https://leetcode.cn/problems/average-of-levels-in-binary-tree/)

> 给定一个非空二叉树的根节点 `root` , 以数组的形式返回每一层节点的平均值。与实际答案相差 `10-5` 以内的答案可以被接受。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/avg1-tree.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：[3.00000,14.50000,11.00000]
> 解释：第 0 层的平均值为 3,第 1 层的平均值为 14.5,第 2 层的平均值为 11 。
> 因此返回 [3, 14.5, 11] 。
> ```
>
> **示例 2:**
>
> ![img](./codeNote.assets/avg2-tree.jpg)
>
> ```
> 输入：root = [3,9,20,15,7]
> 输出：[3.00000,14.50000,11.00000]
> ```
>
>  
>
> **提示：**
>
> 
>
> - 树中节点数量在 `[1, 104]` 范围内
> - `-231 <= Node.val <= 231 - 1`

```c++
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        queue<TreeNode*> node_queue;
        vector<double> ans;
        if (root == nullptr) {
            return ans;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            TreeNode* cur;
            double sum = 0;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                sum += cur->val;
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
            ans.push_back(sum / size);
        }
        return ans;
    }
};
```

```python
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        ans = []
        node_queue = []
        if root == None:
            return ans
        node_queue.append(root)
        while len(node_queue) != 0:
            addnum = 0
            size = len(node_queue)
            for i in range(size):
                node = node_queue[0]
                node_queue.pop(0)
                addnum += node.val
                if node.left:
                    node_queue.append(node.left)
                if node.right:
                    node_queue.append(node.right)
            ans.append(addnum / size)
        return ans

```

##### [429. N 叉树的层序遍历](https://leetcode.cn/problems/n-ary-tree-level-order-traversal/)

> 给定一个 N 叉树，返回其节点值的*层序遍历*。（即从左到右，逐层遍历）。
>
> 树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/narytreeexample.png)
>
> ```
> 输入：root = [1,null,3,2,4,null,5,6]
> 输出：[[1],[3,2,4],[5,6]]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/sample_4_964.png)
>
> ```
> 输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
> 输出：[[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]
> ```
>
>  
>
> **提示：**
>
> - 树的高度不会超过 `1000`
> - 树的节点总数在 `[0, 10^4]` 之间

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
public:
    vector<vector<int>> levelOrder(Node* root) {
        queue<Node*> node_queue;
        vector<vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            Node* cur;
            int size = node_queue.size();
            vector<int> path;
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                path.push_back(cur->val);
                node_queue.pop();
                for (auto iter : cur->children) {
                    node_queue.push(iter);
                }
            }
            ans.push_back(path);
        }
        return ans;
    }
};
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Solution:
    def levelOrder(self, root: "Node") -> List[List[int]]:
        ans = []
        node_queue = []
        if root == None:
            return ans
        node_queue.append(root)
        while len(node_queue) != 0:
            path = []
            size = len(node_queue)
            for i in range(size):
                node = node_queue[0]
                node_queue.pop(0)
                path.append(node.val)
                for item in node.children:
                    node_queue.append(item)
            ans.append(path)
        return ans
```

> 对于**递归法**,只需要修改以下代码
>
> ```python
> def bfs(self, cur: Optional[TreeNode], depth: int, ans: List[List[int]]) -> None:
>         if cur == None:
>             return
>         if len(ans) == depth:
>             ans.append([])
>         ans[depth].append(cur.val)
>         self.bfs(cur.left, depth + 1, ans)
>         self.bfs(cur.right, depth + 1, ans)
> ```
>
> 其中的7,8行修改
>
> ```python
> def bfs(self, cur: Optional[TreeNode], depth: int, ans: List[List[int]]) -> None:
>         if cur == None:
>             return
>         if len(ans) == depth:
>             ans.append([])
>         ans[depth].append(cur.val)
>         for item in node.children:
>             self.bfs(item, depth + 1, ans)
> ```
>
> C++的修改类似,在这里不在赘述
>
> Python的具体递归解法为:
>
> ```python
> class Solution:
>     def bfs(self, cur: "Node", depth: int, ans: List[List[int]]) -> None:
>         if cur == None:
>             return
>         if len(ans) == depth:
>             ans.append([])
>         ans[depth].append(cur.val)
>         for item in cur.children:
>             self.bfs(item, depth + 1, ans)
> 
>     def levelOrder(self, root: "Node") -> List[List[int]]:
>         ans = []
>         self.bfs(root, 0, ans)
>         return ans
> ```

##### [515. 在每个树行中找最大值](https://leetcode.cn/problems/find-largest-value-in-each-tree-row/)

> 给定一棵二叉树的根节点 `root` ，请找出该二叉树中每一层的最大值。
>
>  
>
> **示例1：**
>
> ![img](./codeNote.assets/largest_e1.jpg)
>
> ```
> 输入: root = [1,3,2,5,3,null,9]
> 输出: [1,3,9]
> ```
>
> **示例2：**
>
> ```
> 输入: root = [1,2,3]
> 输出: [1,3]
> ```
>
>  
>
> **提示：**
>
> - 二叉树的节点个数的范围是 `[0,104]`
> - `-2^31 <= Node.val <= 2^31 - 1`

```c++
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        queue<TreeNode*> node_queue;
        vector<int> ans;
        if (root == nullptr) {
            return ans;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            TreeNode* cur;
            int maxNum = INT_MIN;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                maxNum = max(cur->val, maxNum);
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
            ans.push_back(maxNum);
        }
        return ans;
    }
};	
```

```python
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        node_queue = []
        ans = []
        if root == None:
            return ans
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            maxNum = -inf
            for i in range(size):
                cur = node_queue[0]
                maxNum = max(maxNum, cur.val)
                node_queue.pop(0)
                if cur.left:
                    node_queue.append(cur.left)
                if cur.right:
                    node_queue.append(cur.right)
            ans.append(maxNum)
        return ans

```

##### [116. 填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

> 给定一个 **完美二叉树** ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
>
> ```
> struct Node {
>   int val;
>   Node *left;
>   Node *right;
>   Node *next;
> }
> ```
>
> 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL`。
>
> 初始状态下，所有 next 指针都被设置为 `NULL`。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/116_sample.png)
>
> ```
> 输入：root = [1,2,3,4,5,6,7]
> 输出：[1,#,2,3,#,4,5,6,7,#]
> 解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
> ```
>
> 
>
> **示例 2:**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数量在 `[0, 212 - 1]` 范围内
> - `-1000 <= node.val <= 1000`
>
>  
>
> **进阶：**
>
> - 你只能使用常量级额外空间。
> - 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(NULL), right(NULL), next(NULL) {}

    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
*/

class Solution {
public:
    Node* connect(Node* root) {
        queue<Node*> node_queue;
        if (root == nullptr) {
            return root;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            Node* cur;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                node_queue.pop();
                if (i != size - 1) {
                    cur->next = node_queue.front();
                }
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
        }
        return root;
    }
};
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""


class Solution:
    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        node_queue = []
        if root == None:
            return root
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            for i in range(size):
                cur = node_queue[0]
                node_queue.pop(0)
                if i != size - 1:
                    cur.next = node_queue[0]
                if cur.left:
                    node_queue.append(cur.left)
                if cur.right:
                    node_queue.append(cur.right)
        return root

```

##### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/)

> 给定一个二叉树：
>
> ```
> struct Node {
>   int val;
>   Node *left;
>   Node *right;
>   Node *next;
> }
> ```
>
> 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL` 。
>
> 初始状态下，所有 next 指针都被设置为 `NULL` 。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/117_sample.png)
>
> ```
> 输入：root = [1,2,3,4,5,null,7]
> 输出：[1,#,2,3,#,4,5,7,#]
> 解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化输出按层序遍历顺序（由 next 指针连接），'#' 表示每层的末尾。
> ```
>
> **示例 2：**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
>  
>
> **提示：**
>
> - 树中的节点数在范围 `[0, 6000]` 内
> - `-100 <= Node.val <= 100`
>
> **进阶：**
>
> - 你只能使用常量级额外空间。
> - 使用递归解题也符合要求，本题中递归程序的隐式栈空间不计入额外空间复杂度。

**竟然和上一道题的答案一模一样,不在赘述**

##### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树 `root` ，返回其最大深度。
>
> 二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/tmp-tree.jpg)
>
>  
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：3
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [1,null,2]
> 输出：2
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数量在 `[0, 104]` 区间内。
> - `-100 <= Node.val <= 100`

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*> node_queue;
        int depth = 0;
        if (root == nullptr) {
            return depth;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            TreeNode* cur;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
            depth++;
        }
        return depth;
    }
};
```

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        node_queue = []
        depth = 0
        if root == None:
            return depth
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            for i in range(size):
                cur = node_queue[0]
                node_queue.pop(0)
                if cur.left:
                    node_queue.append(cur.left)
                if cur.right:
                    node_queue.append(cur.right)
            depth += 1
        return depth

```

##### [111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最小深度。
>
> 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
>
> **说明：**叶子节点是指没有子节点的节点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/ex_depth.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：2
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [2,null,3,null,4,null,5,null,6]
> 输出：5
> ```
>
>  
>
> **提示：**
>
> - 树中节点数的范围在 `[0, 105]` 内
> - `-1000 <= Node.val <= 1000`

```c++
class Solution {
public:
    int minDepth(TreeNode* root) {
        int depth = 0;
        if (root == nullptr) {
            return depth;
        }
        queue<TreeNode*> node_queue;
        node_queue.push(root);
        while (!node_queue.empty()) {
            TreeNode* cur;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                node_queue.pop();
                if (!cur->left && !cur->right) {
                    return ++depth;
                } else {
                    if (cur->left) {
                        node_queue.push(cur->left);
                    }
                    if (cur->right) {
                        node_queue.push(cur->right);
                    }
                }
            }
            ++depth;
        }
        return depth; // 虽然没用,但是没有会报错
    }
};
```

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        node_queue = []
        depth = 0
        if root == None:
            return depth
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            for i in range(size):
                cur = node_queue[0]
                node_queue.pop(0)
                if cur.left == None and cur.right == None:
                    return depth + 1
                else:
                    if cur.left:
                        node_queue.append(cur.left)
                    if cur.right:
                        node_queue.append(cur.right)
            depth += 1

```

#### [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

> 给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/invert1-tree.jpg)
>
> ```
> 输入：root = [4,2,7,1,3,6,9]
> 输出：[4,7,2,9,6,3,1]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/invert2-tree.jpg)
>
> ```
> 输入：root = [2,1,3]
> 输出：[2,3,1]
> ```
>
> **示例 3：**
>
> ```
> 输入：root = []
> 输出：[]
> ```
>
>  
>
> **提示：**
>
> - 树中节点数目范围在 `[0, 100]` 内
> - `-100 <= Node.val <= 100`

```c++
class Solution {
public:
    void reverse(TreeNode* cur) {
        if (cur == nullptr) {
            return;
        }
        TreeNode* temp = cur->right;
        cur->right = cur->left;
        cur->left = temp;
        reverse(cur->left);
        reverse(cur->right);
    }
    TreeNode* invertTree(TreeNode* root) {
        reverse(root);
        return root;
    }
};
```

- 或者使用**前序遍历**

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return root;
        }
        stack<TreeNode*> node_stack;
        node_stack.push(root);
        while (!node_stack.empty()) {
            TreeNode* cur = node_stack.top();
            node_stack.pop();
            swap(cur->left, cur->right);
            if (cur->right) {
                node_stack.push(cur->right);
            }
            if (cur->left) {
                node_stack.push(cur->left);
            }
        }
        return root;
    }
};
```

- 或者使用**统一迭代法**,这里使用**前序**为例子

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return root;
        }
        stack<TreeNode*> node_stack;
        node_stack.push(root);
        TreeNode* cur = root;
        while (!node_stack.empty()) {
            cur = node_stack.top();
            if (cur != nullptr) {
                node_stack.pop(); // 先把节点拿出来
                swap(cur->left, cur->right);
                if (cur->left) {
                    node_stack.push(cur->left);
                }
                if (cur->right) {
                    node_stack.push(cur->right);
                }
                node_stack.push(cur);
                node_stack.push(nullptr);
            } else {
                node_stack.pop();
                node_stack.pop();
            }
        }
        return root;
    }
};
```

```python
# 只用回溯法写,其他方法不再赘述
class Solution:
    def reverse(self, cur: Optional[TreeNode]) -> None:
        if cur == None:
            return
        cur.left, cur.right = cur.right, cur.left
        self.reverse(cur.left)
        self.reverse(cur.right)

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.reverse(root)
        return root
```

#### [589. N 叉树的前序遍历](https://leetcode.cn/problems/n-ary-tree-preorder-traversal/)

> 给定一个 n 叉树的根节点 `root` ，返回 *其节点值的 **前序遍历*** 。
>
> n 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 `null` 分隔（请参见示例）。
>
> 
> **示例 1：**
>
> ![img](./codeNote.assets/narytreeexample-1707807420886-5.png)
>
> ```
> 输入：root = [1,null,3,2,4,null,5,6]
> 输出：[1,3,5,6,2,4]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/sample_4_964-1707807420886-7.png)
>
> ```
> 输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
> 输出：[1,2,3,6,7,11,14,4,8,12,5,9,13,10]
> ```
>
>  
>
> **提示：**
>
> - 节点总数在范围 `[0, 104]`内
> - `0 <= Node.val <= 104`
> - n 叉树的高度小于或等于 `1000`
>
>  
>
> **进阶：**递归法很简单，你可以使用迭代法完成此题吗?

```c++
class Solution {
public:
    // 使用一波统一迭代法(递归和普通迭代没有难度)
    vector<int> preorder(Node* root) {
        stack<Node*> node_stack;
        vector<int> ans_vector;
        if (root == nullptr) {
            return ans_vector;
        }
        node_stack.push(root);
        while (!node_stack.empty()) {
            Node* cur = node_stack.top();
            if (cur != nullptr) {
                node_stack.pop();
                for (auto iter : vector<Node*>(cur->children.rbegin(),
                                               cur->children.rend())) {
                    node_stack.push(iter);
                }
                node_stack.push(cur);
                node_stack.push(nullptr);
            } else {
                node_stack.pop();
                cur = node_stack.top();
                ans_vector.push_back(cur->val);
                node_stack.pop();
            }
        }
        return ans_vector;
    }
};
```

```python
class Solution:
    def order(self, cur: "Node", ans_list: List[int]) -> None:
        if cur == None:
            return
        ans_list.append(cur.val)
        for item in cur.children:
            self.order(item, ans_list)

    def preorder(self, root: "Node") -> List[int]:
        ans_list = []
        self.order(root, ans_list)
        return ans_list

```

#### [590. N 叉树的后序遍历](https://leetcode.cn/problems/n-ary-tree-postorder-traversal/)

> 给定一个 n 叉树的根节点 `root` ，返回 *其节点值的 **后序遍历*** 。
>
> n 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 `null` 分隔（请参见示例）。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/narytreeexample-1707807492794-11.png)
>
> ```
> 输入：root = [1,null,3,2,4,null,5,6]
> 输出：[5,6,3,2,4,1]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/sample_4_964-1707807492795-13.png)
>
> ```
> 输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
> 输出：[2,6,14,11,7,3,12,8,4,13,9,10,5,1]
> ```
>
>  
>
> **提示：**
>
> - 节点总数在范围 `[0, 104]` 内
> - `0 <= Node.val <= 104`
> - n 叉树的高度小于或等于 `1000`
>
>  
>
> **进阶：**递归法很简单，你可以使用迭代法完成此题吗?

```c++
class Solution {
public:
    // 使用普通迭代法
    vector<int> postorder(Node* root) {
        vector<int> ans_vector;
        if (root == nullptr) {
            return ans_vector;
        }
        stack<Node*> node_stack;
        node_stack.push(root);
        while (!node_stack.empty()) {
            Node* cur = node_stack.top();
            node_stack.pop();
            ans_vector.push_back(cur->val);
            for (auto iter : cur->children) {
                node_stack.push(iter);
            }
        }
        return vector<int>(ans_vector.rbegin(), ans_vector.rend());
    }
};
```

```python
class Solution:
    def order(self, cur: "Node", ans_list: List[int]) -> None:
        if cur == None:
            return
        ans_list.append(cur.val)
        for item in cur.children[::-1]:
            self.order(item, ans_list)
        

    def postorder(self, root: "Node") -> List[int]:
        ans_list = []
        self.order(root, ans_list)
        return ans_list[::-1]
```

#### [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

> 给你一个二叉树的根节点 `root` ， 检查它是否轴对称。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/1698026966-JDYPDU-image.png)
>
> ```
> 输入：root = [1,2,2,3,4,4,3]
> 输出：true
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/1698027008-nPFLbM-image.png)
>
> ```
> 输入：root = [1,2,2,null,3,null,3]
> 输出：false
> ```
>
>  
>
> **提示：**
>
> - 树中节点数目在范围 `[1, 1000]` 内
> - `-100 <= Node.val <= 100`
>
>  
>
> **进阶：**你可以运用递归和迭代两种方法解决这个问题吗？

```c++
class Solution {
public:
    // 思路:层序遍历,检查每层是否对称(迭代)
    bool isSymmetric(TreeNode* root) {
        queue<TreeNode*> node_queue;
        if (root == nullptr) {
            return true;
        }
        node_queue.push(root);
        while (!node_queue.empty()) {
            int size = node_queue.size();
            vector<int> path_vector;
            TreeNode* cur;
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                node_queue.pop();
                if (cur == nullptr) {
                    path_vector.push_back(INT_MAX);
                    continue;
                }
                path_vector.push_back(cur->val);
                // 自认为比较妙的一步(使用占位的nullptr,补充空节点)而且并不会陷入死循环,空节点只会补充一次左右节点,作为中间节点的时候不会填充空节点
                node_queue.push(cur->left);
                node_queue.push(cur->right);
            }
            int left = 0, right = path_vector.size() - 1;
            while (left < right) {
                if (path_vector[left] != path_vector[right]) {
                    return false;
                }
                left++;
                right--;
            }
        }
        return true;
    }
};
```

```python
class Solution:
    # 递归
    def check(self, left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        if left and right:
            if left.val != right.val:
                return False
            else:
                return self.check(left.left, right.right) and self.check(
                    left.right, right.left
                )
        elif left == None and right == None:
            return True
        else:
            return False

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root == None:
            return True
        else:
            return self.check(root.left, root.right)
```

> 当然也可以使用队列,但是要注意压入顺序
>
> 通过队列来判断根节点的左子树和右子树的内侧和外侧是否相等，如动画所示：
>
> ![101.对称二叉树](./codeNote.assets/101.对称二叉树.gif)



#### [100. 相同的树](https://leetcode.cn/problems/same-tree/)

> 给你两棵二叉树的根节点 `p` 和 `q` ，编写一个函数来检验这两棵树是否相同。
>
> 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/ex1.jpg)
>
> ```
> 输入：p = [1,2,3], q = [1,2,3]
> 输出：true
> ```
>
> **示例 2：**
>
> ![img](C:\Users\exqin\Desktop\Blog\codeNote.assets\ex2.jpg)
>
> ```
> 输入：p = [1,2], q = [1,null,2]
> 输出：false
> ```
>
> **示例 3：**
>
> ![img](./codeNote.assets/ex3.jpg)
>
> ```
> 输入：p = [1,2,1], q = [1,1,2]
> 输出：false
> ```
>
>  
>
> **提示：**
>
> - 两棵树上的节点数目都在范围 `[0, 100]` 内
> - `-104 <= Node.val <= 104`

```c++
// 准备使用栈来解决(匹配问题)
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        stack<TreeNode*> node_stack;
        if (p == nullptr && q == nullptr) {
            return true;
        } else if (p && q) {
            node_stack.push(p);
            node_stack.push(q);
            while (!node_stack.empty()) {
                TreeNode* left = node_stack.top();
                node_stack.pop();
                TreeNode* right = node_stack.top();
                node_stack.pop();
                if (left == nullptr && right == nullptr) {
                    continue;
                }
                if ((left || right) && (!(left && right))) {
                    return false;
                }
                if (left->val != right->val) {
                    return false;
                }
                node_stack.push(left->left);
                node_stack.push(right->left);
                node_stack.push(left->right);
                node_stack.push(right->right);
            }
        } else {
            return false;
        }
        return true;
    }
};
```

```python
# 递归
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not (p or q):
            return True
        if (p or q) and (not (p and q)):
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

#### [572. 另一棵树的子树](https://leetcode.cn/problems/subtree-of-another-tree/)

> 给你两棵二叉树 `root` 和 `subRoot` 。检验 `root` 中是否包含和 `subRoot` 具有相同结构和节点值的子树。如果存在，返回 `true` ；否则，返回 `false` 。
>
> 二叉树 `tree` 的一棵子树包括 `tree` 的某个节点和这个节点的所有后代节点。`tree` 也可以看做它自身的一棵子树。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/subtree1-tree.jpg)
>
> ```
> 输入：root = [3,4,5,1,2], subRoot = [4,1,2]
> 输出：true
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/subtree2-tree.jpg)
>
> ```
> 输入：root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
> 输出：false
> ```
>
>  
>
> **提示：**
>
> - `root` 树上的节点数量范围是 `[1, 2000]`
> - `subRoot` 树上的节点数量范围是 `[1, 1000]`
> - `-104 <= root.val <= 104`
> - `-104 <= subRoot.val <= 104`

```c++
class Solution {
private:
    bool samecheck(TreeNode* first, TreeNode* second) {
        if ((first || second) && !(first && second)) {
            return false;
        } else if (!(first || second)) {
            return true;
        } else {
            if (first->val != second->val) {
                return false;
            } else {
                return samecheck(first->left, second->left) &&
                       samecheck(first->right, second->right);
            }
        }
    }

public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        if (!root) {
            return false;
        }
        if (samecheck(root, subRoot)) {
            return true;
        }
        return isSubtree(root->left, subRoot) ||
               isSubtree(root->right, subRoot);
    }
};
```

```python
class Solution:
    def samecheck(self, first: Optional[TreeNode], second: Optional[TreeNode]) -> bool:
        if first == None and second == None:
            return True
        elif not (first and second) and (first or second):
            return False
        else:
            # 前序遍历
            node_stack = []
            node_stack.append(first)
            node_stack.append(second)
            while len(node_stack) != 0:
                left = node_stack[-1]
                node_stack.pop(-1)
                right = node_stack[-1]
                node_stack.pop(-1)
                if left == None and right == None:
                    pass
                elif not (left and right) and (left or right):
                    return False
                elif left.val != right.val:
                    return False
                else:
                    node_stack.append(left.right)
                    node_stack.append(right.right)
                    node_stack.append(left.left)
                    node_stack.append(right.left)
        return True

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if root == None:
            return False
        else:
            return (
                self.samecheck(root, subRoot)
                or self.isSubtree(root.left, subRoot)
                or self.isSubtree(root.right, subRoot)
            )
```

#### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树 `root` ，返回其最大深度。
>
> 二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/tmp-tree-1708246144773-1.jpg)
>
>  
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：3
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [1,null,2]
> 输出：2
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数量在 `[0, 104]` 区间内。
> - `-100 <= Node.val <= 100`

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*> node_queue;
        int depth = 0;
        if (root == nullptr) {
            return depth;
        }
        node_queue.push(root);
        // 层序遍历
        while (!node_queue.empty()) {
            TreeNode* cur;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                cur = node_queue.front();
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
            depth++;
        }
        return depth;
    }
};

// 递归法
class Solution {
private:
    int result = 0;
    void getmaxdepth(TreeNode* cur, int depth) {
        result = depth > result ? depth : result;
        if (cur->left == nullptr && cur->right == nullptr) {
            return;
        }
        if (cur->left) {
            depth++;
            getmaxdepth(cur->left, depth);  // 这里可以直接把数据变化放在形参depth的位置上
            depth--;
        }
        if (cur->right) {
            depth++;
            getmaxdepth(cur->right, depth);
            depth--;
        }
        return;
    }

public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return result;
        }
        getmaxdepth(root, 1);
        return result;
    }
};
// 迭代法 (后序遍历)
class Solution {
public:
    int maxDepth(TreeNode* root) {
        int result = 0;
        int depth = 0;
        stack<TreeNode*> node_stack;
        if (root == nullptr) {
            return result;
        }
        node_stack.push(root);
        while (!node_stack.empty()) {
            TreeNode* cur = node_stack.top();
            if (cur) {
                node_stack.pop();
                node_stack.push(cur);
                node_stack.push(nullptr);
                if (cur->right) {
                    node_stack.push(cur->right);
                }
                if (cur->left) {
                    node_stack.push(cur->left);
                }
                depth++;
            } else {
                node_stack.pop();
                node_stack.pop();
                depth--;
            }
            result = max(result, depth);
        }
        return result;
    }
};
```

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        node_queue = []
        depth = 0
        if root == None:
            return depth
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            for i in range(size):
                cur = node_queue[0]
                node_queue.pop(0)
                if cur.left:
                    node_queue.append(cur.left)
                if cur.right:
                    node_queue.append(cur.right)
            depth += 1
        return depth
```

#### [559. N 叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-n-ary-tree/)

> 给定一个 N 叉树，找到其最大深度。
>
> 最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
>
> N 叉树输入按层序遍历序列化表示，每组子节点由空值分隔（请参见示例）。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/narytreeexample-1708246485795-4.png)
>
> ```
> 输入：root = [1,null,3,2,4,null,5,6]
> 输出：3
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/sample_4_964-1708246485795-6.png)
>
> ```
> 输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
> 输出：5
> ```
>
>  
>
> **提示：**
>
> - 树的深度不会超过 `1000` 。
> - 树的节点数目位于 `[0, 104]` 之间。

```c++
// 迭代法
class Solution {
public:
    int maxDepth(Node* root) {
        int depth = 0;
        if (root == nullptr) {
            return depth;
        }
        queue<Node*> node_queue;
        node_queue.push(root);
        depth++;
        while (!node_queue.empty()) {
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                Node* cur = node_queue.front();
                node_queue.pop();
                for (auto iter : cur->children) {
                    if (iter) {
                        node_queue.push(iter);
                    }
                }
            }
            depth++;
        }
        return depth - 1;
    }
};
```

```python
# 递归法
class Solution:
    # 后序遍历(得到左右子树的高度最大值+1,即左右中)
    def getdepth(self, cur: "Node") -> int:
        if cur == None:
            return 0
        maxdp = 0
        for item in cur.children:
            maxdp = max(maxdp, self.getdepth(item))
        return maxdp + 1

    def maxDepth(self, root: "Node") -> int:
        return self.getdepth(root)
```

#### [111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最小深度。
>
> 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
>
> **说明：**叶子节点是指没有子节点的节点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/ex_depth-1708252330150-10.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：2
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [2,null,3,null,4,null,5,null,6]
> 输出：5
> ```
>
>  
>
> **提示：**
>
> - 树中节点数的范围在 `[0, 105]` 内
> - `-1000 <= Node.val <= 1000`

- 我认为这道题可以使用**层序遍历**,**前序遍历**,**后续遍历**,不适合使用**后序遍历**
  - 其中层序边遍历可以将空节点放入queue中
- 递归也是一种好的办法
- 要正确理解最浅节点的特性;即**叶子节点**

```c++
// 递归法(规律不好总结)
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }

        int leftDepth = minDepth(root->left);
        int rightDepth = minDepth(root->right);

        // 当左子树或右子树有一个为空时，返回非空子树的最小深度加一
        if (root->left == nullptr || root->right == nullptr) {
            return max(leftDepth, rightDepth) + 1;
        }

        // 当左右子树都非空时，返回较小的深度加一
        return min(leftDepth, rightDepth) + 1;
    }
};
```

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        # 尝试使用层序遍历
        node_queue = []
        depth = 0
        if root == None:
            return depth
        node_queue.append(root)
        while len(node_queue) != 0:
            size = len(node_queue)
            for i in range(size):
                cur = node_queue[0]
                node_queue.pop(0)
                if cur.left == None and cur.right == None:
                    return depth + 1
                if cur.left:
                    node_queue.append(cur.left)
                if cur.right:
                    node_queue.append(cur.right)
            depth += 1
        return depth
# 如果存在一个节点,它没有子节点,说明是一个端点,是最浅深度的候选
```

#### [222. 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/)

> 给你一棵 **完全二叉树** 的根节点 `root` ，求出该树的节点个数。
>
> [完全二叉树](https://baike.baidu.com/item/完全二叉树/7773232?fr=aladdin) 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 `h` 层，则该层包含 `1~ 2h` 个节点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/complete.jpg)
>
> ```
> 输入：root = [1,2,3,4,5,6]
> 输出：6
> ```
>
> **示例 2：**
>
> ```
> 输入：root = []
> 输出：0
> ```
>
> **示例 3：**
>
> ```
> 输入：root = [1]
> 输出：1
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数目范围是`[0, 5 * 104]`
> - `0 <= Node.val <= 5 * 104`
> - 题目数据保证输入的树是 **完全二叉树**
>
>  
>
> **进阶：**遍历树来统计节点是一种时间复杂度为 `O(n)` 的简单解决方案。你可以设计一个更快的算法吗？

- 如果直接考虑遍历所有节点的话直观解决就可以了,但是如果利用**完全二叉树**的特性的话,这个题目可以有更好的解决办法
  - ![222.完全二叉树的节点个数](./codeNote.assets/20201124092543662.png)
  - ![222.完全二叉树的节点个数1](./codeNote.assets/20201124092634138.png)
  - 向左和向右遍历深度,深度相同就代表是是满二叉树

```c++
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        int left_depth = 0, right_depth = 0;
        TreeNode* left_ptr = root->left;
        TreeNode* right_ptr = root->right;
        while (left_ptr != nullptr) {
            left_ptr = left_ptr->left;
            left_depth++;
        }
        while (right_ptr != nullptr) {
            right_ptr = right_ptr->right;
            right_depth++;
        }
        if (left_depth == right_depth) {
            return (2 << left_depth) - 1;
        } // 这里要深入理解为什么会向下递归
        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};
```

```python
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        left_depth = 0
        right_depth = 0
        left_ptr = root.left
        right_ptr = root.right
        while left_ptr != None:
            left_ptr = left_ptr.left
            left_depth += 1
        while right_ptr != None:
            right_ptr = right_ptr.right
            right_depth += 1
        if left_depth == right_depth:
            return (2 << left_depth) - 1
        return self.countNodes(root.left) + self.countNodes(root.right) + 1
```

> 递归是一种重要思想,`return`是一种特别的终止方法

#### [110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。
>
> 本题中，一棵高度平衡二叉树定义为：
>
> > 一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/balance_1.jpg)
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：true
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/balance_2.jpg)
>
> ```
> 输入：root = [1,2,2,3,3,null,null,4,4]
> 输出：false
> ```
>
> **示例 3：**
>
> ```
> 输入：root = []
> 输出：true
> ```
>
>  
>
> **提示：**
>
> - 树中的节点数在范围 `[0, 5000]` 内
> - `-104 <= Node.val <= 104`

```c++
class Solution {
private:
    int getHeight(TreeNode* cur) {
        if (cur == nullptr) { //终止条件
            return 0;
        }
        return max(getHeight(cur->left), getHeight(cur->right)) + 1; // 递归逻辑
    }

public:
    bool isBalanced(TreeNode* root) {
        // 考虑递归函数
        if (root == nullptr) {
            return true;
        }
        return (abs(getHeight(root->left) - getHeight(root->right)) < 2) &&
               isBalanced(root->left) && isBalanced(root->right);
    }
};
// 这种方法对于同一个节点可能存在访问的情况,可以考虑使用记忆化操作优化
class Solution {
private:
    unordered_map<TreeNode*, int> height_map; // 记忆化操作
    int getHeight(TreeNode* cur) {
        auto it = height_map.find(cur); // 如果存在该键,则迭代器指向对应的键值对,可以用name.second访问值,不存在该键则指向height_map的尾迭代器
        if (it != height_map.end()) {
            return it->second;
        }
        if (cur == nullptr) { //终止条件
            return 0;
        }
        int height = max(getHeight(cur->left), getHeight(cur->right)) + 1;
        return height_map[cur] = height; // 可以之间使用下标操作符[]来访问和修改对应的值
        // 递归逻辑
    }

public:
    bool isBalanced(TreeNode* root) {
        // 考虑递归函数
        if (root == nullptr) {
            return true;
        }
        return (abs(getHeight(root->left) - getHeight(root->right)) < 2) &&
               isBalanced(root->left) && isBalanced(root->right);
    }
};
```

```python
class Solution:
    def __init__(self): # 初始化函数在不需要类外参数的时候形式参数只有self
        self.height_dict = {}

    def getHeight(self, cur: Optional[TreeNode]) -> int:
        if cur == None:
            return 0
        if self.height_dict.get(cur, 0) != 0: # get方法得到的值为键值对中的值,并且设置了默认的返回值0
            return self.height_dict[cur]
        height = max(self.getHeight(cur.left), self.getHeight(cur.right)) + 1
        self.height_dict[cur] = height
        return height

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if root == None:
            return True
        return (
            -2 < self.getHeight(root.left) - self.getHeight(root.right) < 2
            and self.isBalanced(root.left)
            and self.isBalanced(root.right)
        )
```

#### [257. 二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/)

> 给你一个二叉树的根节点 `root` ，按 **任意顺序** ，返回所有从根节点到叶子节点的路径。
>
> **叶子节点** 是指没有子节点的节点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/paths-tree.jpg)
>
> ```
> 输入：root = [1,2,3,null,5]
> 输出：["1->2->5","1->3"]
> ```
>
> **示例 2：**
>
> ```
> 输入：root = [1]
> 输出：["1"]
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数目在范围 `[1, 100]` 内
> - `-100 <= Node.val <= 100!`

![257.二叉树的所有路径](./codeNote.assets/20210204151702443.png)

```c++
class Solution {
private:
    vector<string> ans;
    // 注意string对于+运算符的特殊支持(简化代码)
    void backtracing(TreeNode* cur, string path) {
        if (cur->left == nullptr && cur->right == nullptr) {
            path += to_string(cur->val); // 将当前节点值添加到路径字符串中
            ans.push_back(path);
            return;
        }
        if (cur->left) {
            backtracing(cur->left, path + to_string(cur->val) +
                                       "->"); // 在递归调用前更新路径
        }
        if (cur->right) {
            backtracing(cur->right, path + to_string(cur->val) +
                                        "->"); // 在递归调用前更新路径
        }
    }

public:
    vector<string> binaryTreePaths(TreeNode* root) {
        if (root == nullptr) {
            return ans;
        }
        string path;
        backtracing(root, path);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.ans = []

    def findway(self, cur: Optional[TreeNode], path: str) -> None:
        if cur.left == None and cur.right == None:
            self.ans.append(path + str(cur.val))
            return
        if cur.left:
            self.findway(cur.left, path + str(cur.val) + "->")
        if cur.right:
            self.findway(cur.right, path + str(cur.val) + "->")

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if root == None:
            return ans
        path = ""
        self.findway(root, path)
        return self.ans
```

- 注意将整型转换为字符串类型 
  - `to_string()`
  - `str()`
- 注意格式
- 本题的回溯藏在了参数里,并没有显式地展现出来,每次调用函数,会生成相应的临时变量`path`,只做了内容拷贝,没有做引用操作
  - 这里体现了是否取址对于函数作用的区别



#### [404. 左叶子之和](https://leetcode.cn/problems/sum-of-left-leaves/)

> 给定二叉树的根节点 `root` ，返回所有左叶子之和。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/leftsum-tree.jpg)
>
> ```
> 输入: root = [3,9,20,null,null,15,7] 
> 输出: 24 
> 解释: 在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
> ```
>
> **示例 2:**
>
> ```
> 输入: root = [1]
> 输出: 0
> ```
>
>  
>
> **提示:**
>
> - 节点数在 `[1, 1000]` 范围内
> - `-1000 <= Node.val <= 1000`

```c++
// 注意是叶子节点
class Solution {
private:
    int ans = 0;
    void visit(TreeNode* cur) {
        if (cur->left == nullptr && cur->right == nullptr) {
            return;
        }
        if (cur->left) {
            if (cur->left->left == nullptr && cur->left->right == nullptr) {
                ans += cur->left->val;
            }
            visit(cur->left);
        }
        if (cur->right) {
            visit(cur->right);
        }
    }

public:
    int sumOfLeftLeaves(TreeNode* root) {
        if (root == nullptr) {
            return ans;
        }
        visit(root);
        return ans;
    }
};
// 也可以使用迭代法
```

```python
class Solution:
    def __init__(self):
        self.ans = 0

    def visit(self, cur: Optional[TreeNode]) -> None:
        if cur.left == None and cur.right == None:
            return
        if cur.left:
            if not (cur.left.left or cur.left.right):
                self.ans += cur.left.val
            self.visit(cur.left)

        if cur.right:
            self.visit(cur.right)

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return self.ans
        self.visit(root)
        return self.ans
```

#### [513. 找树左下角的值](https://leetcode.cn/problems/find-bottom-left-tree-value/)

> 给定一个二叉树的 **根节点** `root`，请找出该二叉树的 **最底层 最左边** 节点的值。
>
> 假设二叉树中至少有一个节点。
>
>  
>
> **示例 1:**
>
> ![img](./codeNote.assets/tree1-1708419922156-17.jpg)
>
> ```
> 输入: root = [2,1,3]
> 输出: 1
> ```
>
> **示例 2:**
>
> ![img](./codeNote.assets/tree2.jpg)
>
> ```
> 输入: [1,2,3,4,null,5,6,null,null,7]
> 输出: 7
> ```
>
>  
>
> **提示:**
>
> - 二叉树的节点个数的范围是 `[1,104]`
> - `-231 <= Node.val <= 231 - 1` 

```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        // 使用层序遍历,最下层的第一个
        queue<TreeNode*> node_queue;
        vector<vector<int>> ans;
        node_queue.push(root);
        while (!node_queue.empty()) {
            vector<int> path;
            int size = node_queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode* cur = node_queue.front();
                path.push_back(cur->val);
                node_queue.pop();
                if (cur->left) {
                    node_queue.push(cur->left);
                }
                if (cur->right) {
                    node_queue.push(cur->right);
                }
            }
            ans.push_back(path);
        }
        return ans[ans.size() - 1][0];
    }
};
```

```python
class Solution:
    def __init__(self):
        self.result = 0
        self.depth = 0

    def findvalue(self, cur: Optional[TreeNode], depth: int) -> None:
        if not (cur.left or cur.right):
            if self.depth < depth:
                self.depth = depth
                self.result = cur.val
            return
        if cur.left: # 前序遍历
            self.findvalue(cur.left, depth + 1)
        if cur.right:
            self.findvalue(cur.right, depth + 1)

    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        # 考虑使用递归方法
        self.result = root.val
        self.findvalue(root, 0)
        return self.result
```

#### [112. 路径总和](https://leetcode.cn/problems/path-sum/)

> 给你二叉树的根节点 `root` 和一个表示目标和的整数 `targetSum` 。判断该树中是否存在 **根节点到叶子节点** 的路径，这条路径上所有节点值相加等于目标和 `targetSum` 。如果存在，返回 `true` ；否则，返回 `false` 。
>
> **叶子节点** 是指没有子节点的节点。
>
>  
>
> **示例 1：**
>
> ![img](./codeNote.assets/pathsum1.jpg)
>
> ```
> 输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
> 输出：true
> 解释：等于目标和的根节点到叶节点路径如上图所示。
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/pathsum2.jpg)
>
> ```
> 输入：root = [1,2,3], targetSum = 5
> 输出：false
> 解释：树中存在两条根节点到叶子节点的路径：
> (1 --> 2): 和为 3
> (1 --> 3): 和为 4
> 不存在 sum = 5 的根节点到叶子节点的路径。
> ```
>
> **示例 3：**
>
> ```
> 输入：root = [], targetSum = 0
> 输出：false
> 解释：由于树是空的，所以不存在根节点到叶子节点的路径。
> ```
>
>  
>
> **提示：**
>
> - 树中节点的数目在范围 `[0, 5000]` 内
> - `-1000 <= Node.val <= 1000`
> - `-1000 <= targetSum <= 1000`

```c++
class Solution {
private:
    int target;
    bool flag = false;
    void backtracing(TreeNode* cur, int current_sum) {
        current_sum += cur->val;
        if (cur->left == nullptr && cur->right == nullptr) {
            if (current_sum == target) {
                flag = true;
            }
            return;
        }
        if (cur->left) {
            backtracing(cur->left, current_sum);
        }
        if (cur->right) {
            backtracing(cur->right, current_sum);
        }
    }

public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        // 考虑使用回溯
        if (root == nullptr) {
            return false;
        }
        target = targetSum;
        backtracing(root, 0);
        return flag;
    }
};
```

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        node_stack = []
        # 别忘了前序遍历
        if root == None:
            return False
        node_stack.append((root, root.val))
        while len(node_stack) != 0:
            cur = node_stack[-1]
            node_stack.pop(-1)
            if cur[0].left == None and cur[0].right == None and cur[1] == targetSum:
                return True
            if cur[0].right:
                node_stack.append((cur[0].right, cur[1] + cur[0].right.val))
            if cur[0].left:
                node_stack.append((cur[0].left, cur[1] + cur[0].left.val))
        return False
```

#### [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

> 给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。
>
> **叶子节点** 是指没有子节点的节点。
>
> 
>
> **示例 1：**
>
> ![img](./codeNote.assets/pathsumii1.jpg)
>
> ```
> 输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
> 输出：[[5,4,11,2],[5,8,4,5]]
> ```
>
> **示例 2：**
>
> ![img](./codeNote.assets/pathsum2-1708427766269-27.jpg)
>
> ```
> 输入：root = [1,2,3], targetSum = 5
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：root = [1,2], targetSum = 0
> 输出：[]
> ```
>
> 
>
> **提示：**
>
> - 树中节点总数在范围 `[0, 5000]` 内
> - `-1000 <= Node.val <= 1000`
> - `-1000 <= targetSum <= 1000`

![113.路径总和ii](./codeNote.assets/20210203160922745.png)

```c++
class Solution {
private:
    vector<vector<int>> ans;
    void backtracing(TreeNode* cur, vector<int> path, int current_num,
                     int targetSum) {

        if (!cur->left && !cur->right) {
            if (current_num == targetSum) {
                ans.push_back(path);
            }
            return;
        }
        if (cur->left) {
            path.push_back(cur->left->val);
            backtracing(cur->left, path, current_num + cur->left->val,
                        targetSum);
            path.pop_back();
        }
        if (cur->right) {
            path.push_back(cur->right->val);
            backtracing(cur->right, path, current_num + cur->right->val,
                        targetSum);
            path.pop_back();
        }
    }

public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return ans;
        }
        vector<int> path;
        path.push_back(root->val);
        backtracing(root, path, root->val, targetSum);
        return ans;
    }
};
// 这段代码其实也可以只使用一个path,放在private声明之下,做好递归操作
// 也不使用current_sum了,使用targetSum做差得0即可
```
```c++
// carl的写法
class solution {
private:
    vector<vector<int>> result;
    vector<int> path;
    // 递归函数不需要返回值，因为我们要遍历整个树
    void traversal(TreeNode* cur, int count) {
        if (!cur->left && !cur->right && count == 0) { // 遇到了叶子节点且找到了和为sum的路径
            result.push_back(path);
            return;
        }

        if (!cur->left && !cur->right) return ; // 遇到叶子节点而没有找到合适的边，直接返回

        if (cur->left) { // 左 （空节点不遍历）
            path.push_back(cur->left->val);
            count -= cur->left->val;
            traversal(cur->left, count);    // 递归
            count += cur->left->val;        // 回溯
            path.pop_back();                // 回溯
        }
        if (cur->right) { // 右 （空节点不遍历）
            path.push_back(cur->right->val);
            count -= cur->right->val;
            traversal(cur->right, count);   // 递归
            count += cur->right->val;       // 回溯
            path.pop_back();                // 回溯
        }
        return ;
    }

public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        result.clear();
        path.clear();
        if (root == NULL) return result;
        path.push_back(root->val); // 把根节点放进路径
        traversal(root, sum - root->val);
        return result;
    }
};
```

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        ans = []
        if root == None:
            return ans
        node_stack = [(root, [root.val])]
        while node_stack:  # 非常简单的一种写法
            cur, path = node_stack.pop()  # 在不指明删除元素索引时,默认删除最后一个元素,并将它返回,此外该句还使用了解包赋值
            if (not cur.left) and (not cur.right) and sum(path) == targetSum: # sum()内置函数可以为path中的元素求和
                ans.append(path)
            if cur.left:
                node_stack.append((cur.left, path + [cur.left.val])) # 使用+操作符连接两个列表
            if cur.right:
                node_stack.append((cur.right, path + [cur.right.val]))
        return ans
```

#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

> 给定两个整数数组 `inorder` 和 `postorder` ，其中 `inorder` 是二叉树的中序遍历， `postorder` 是同一棵树的后序遍历，请你构造并返回这颗 *二叉树* 。
>
> 
>
> **示例 1:**
>
> ![img](./codeNote.assets/tree-1708653879157-3.jpg)
>
> ```
> 输入：inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
> 输出：[3,9,20,null,null,15,7]
> ```
>
> **示例 2:**
>
> ```
> 输入：inorder = [-1], postorder = [-1]
> 输出：[-1]
> ```
>
> 
>
> **提示:**
>
> - `1 <= inorder.length <= 3000`
> - `postorder.length == inorder.length`
> - `-3000 <= inorder[i], postorder[i] <= 3000`
> - `inorder` 和 `postorder` 都由 **不同** 的值组成
> - `postorder` 中每一个值都在 `inorder` 中
> - `inorder` **保证**是树的中序遍历
> - `postorder` **保证**是树的后序遍历

- 个人认为这道题相当有挑战性,既考虑了二叉树的多种遍历方式的特征,又考验了数组的操作
  1. 当后续数组为空时,没有节点
  2. 后续数组的最后一个元素是节点元素
  3. 寻找中序数组的相应位置作为切割点
  4. 切割中序数组
  5. 根据切割中序数组的结果切割后序数组
  6. 递归处理左右区间

```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if (postorder.empty()) {
            return nullptr;
        }
        int rootvalue =
            postorder[postorder.size() -
                      1]; // 后序遍历数组的最后一个元素是当前的中间节点
        TreeNode* root = new TreeNode(rootvalue);
        int spiltIndex = 0;
        for (; spiltIndex < inorder.size(); spiltIndex++) {
            if (inorder[spiltIndex] == rootvalue) {
                break;
            }
        }
        // 丢弃最后一个元素
        postorder.resize(postorder.size() - 1);
        vector<int> inorderRight(inorder.begin() + spiltIndex + 1,
                                 inorder.end());
        vector<int> inorderLeft(inorder.begin(), inorder.begin() + spiltIndex);
        // 此时有一个很重的点，就是中序数组大小一定是和后序数组的大小相同的（这是必然）。
        vector<int> postorderLeft(postorder.begin(),
                                  postorder.begin() + inorderLeft.size());
        vector<int> postorderRight(postorder.begin() + inorderLeft.size(),
                                   postorder.end());
        root->left = buildTree(inorderLeft, postorderLeft);
        root->right = buildTree(inorderRight, postorderRight);
        return root;
    }
};
```

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if len(postorder) == 0:
            return None
        root = TreeNode(postorder[-1])
        spiltIndex = 0
        while spiltIndex < len(inorder):
            if inorder[spiltIndex] == root.val:
                break
            spiltIndex += 1
        postorder.pop()
        inorder_left = inorder[:spiltIndex:]
        inorder_right = inorder[spiltIndex + 1 : :]
        postorder_left = postorder[: len(inorder_left) :]
        postorder_right = postorder[len(inorder_left) : :]
        root.left = self.buildTree(inorder_left, postorder_left)
        root.right = self.buildTree(inorder_right, postorder_right)
        return root
```

#### **备考,暂时对该模块收工**



## 第七章 回溯算法

### 理论基础

#### 浅拷贝和深拷贝

在Python中，对象赋值、浅拷贝和深拷贝都涉及到对象及其数据的复制过程，但它们在复制的深度上有所不同。

**对象赋值：**

- 当你将一个对象赋给一个变量时，你只是在创建一个新的引用到原始对象。如果你通过新变量或原始变量改变了对象，这个变化会反映在另一个上，因为它们都指向同一个对象。

**浅拷贝（Shallow Copy）：**

- 浅拷贝创建一个新对象，但只复制原始对象中的顶级结构，不复制内嵌的对象。换句话说，浅拷贝只复制对象的第一层数据结构，如果源对象包含如列表、字典等其他复杂对象的引用，则复制的是这些内部对象的引用而不是对象本身。
- 在Python中，可以使用`copy`模块的`copy()`函数来进行浅拷贝。

**深拷贝（Deep Copy）：**

- 深拷贝不仅复制原始对象的顶级结构，还会递归地复制所有内嵌的对象。最终，你得到一个完全独立于原始对象的新对象。
- 在Python中，可以使用`copy`模块的`deepcopy()`函数来进行深拷贝。

**何时使用浅拷贝或深拷贝：**

1. *引用足够时*：
   - 如果你不需要独立的对象，或者仅仅是想要一个新的变量来引用同一个对象，那么你只需要简单地赋值。

2. *使用浅拷贝*：
   - 当你需要一个新对象，但是你知道你不会改变任何内嵌的子对象时，浅拷贝是有用的。
   - **浅拷贝通常在处理不可变数据类型（如字符串、元组等）或简单数据结构时使用，它们不会被修改或可以确保内部对象不会被外部引用修改。**

3. *使用深拷贝*：
   - 当你有一个复杂的对象，如包含列表或字典等其他对象，并且你需要完全独立的副本时，就需要使用深拷贝。这样，无论你如何修改新对象或原始对象，它们都不会相互影响。
   - 当对象之间有循环引用时，使用深拷贝可以正确处理，而浅拷贝可能会导致问题。

实例说明：

```python
import copy

# 原始对象
original = [1, 2, [3, 4]]

# 赋值操作
assigned = original

# 浅拷贝
shallow = copy.copy(original)

# 深拷贝
deep = copy.deepcopy(original)

# 改变原始数据结构中的内嵌列表
original[-1].append(5)

# 结果分析
print(assigned)  # [1, 2, [3, 4, 5]] - 完全相同，因为它只是引用
print(shallow)   # [1, 2, [3, 4, 5]] - 内嵌列表也被改变了，因为它是引用
print(deep)      # [1, 2, [3, 4]] - 内嵌列表没有改变，因为它是一个独立的拷贝
```

在上面的例子中，改变`original`的内嵌列表会影响`assigned`和`shallow`，因为它们都指向同一个内嵌列表对象。而`deep`是一个完全独立的拷贝，所以内嵌列表的改变不会影响它。

在选择使用浅拷贝还是深拷贝时，需要考虑应用场景和性能开销。深拷贝会递归复制对象，可能会消耗更多的时间和内存，特别是当对象结构复杂或对象层次很深时。因此，你应该根据需要来选择最合适的复制类型。

> **可变对象和不可变对象**
>
> 在Python中，对象可以被分类为不可变（immutable）和可变（mutable），这一分类决定了对象是否可以被直接修改。**理解这一点对于预测对象在多个引用之间的行为至关重要。**下面分别解释这两种情况：
>
> - **对于不可变对象的“修改”**
>
> **不可变对象包括整数、浮点数、字符串、元组等。**==当你对不可变对象进行修改时，实际上是在创建一个新的对象，并改变了变量的引用。==原始对象本身并未改变，因为它是不可变的。
>
> ```python
> a = 3
> b = a
> a = a + 2  # 这里不是修改了3，而是创建了一个新的整数5，并让a引用它
> print(b)  # b仍然是3
> ```
>
> 在上面的例子中，`b`的值不会改变，因为`3`这个对象是不可变的，`a = a + 2`语句创建了一个新的整数对象`5`，然后将`a`的引用从`3`改变到了`5`。
>
> - **对于可变对象的修改**
>
> **可变对象包括列表、字典、集合等。**==它们可以被直接修改，而不需要创建一个新的对象。如果你有多个引用指向同一个可变对象，对该对象的任何修改都会反映在所有引用上。==
>
> ```python
> a = [1, 2, 3]
> b = a
> a.append(4)  # 直接修改了对象 [1, 2, 3]
> print(b)  # b现在是 [1, 2, 3, 4]
> ```
>
> 在这个例子中，`a`和`b`都指向同一个列表对象。通过`a`对列表进行修改（添加了`4`），`b`也反映了这一改变，因为它们是指向同一个对象的引用。
>
> 因此，对于不可变对象，通常所说的“修改”实际上是创建了新的对象并改变了引用；而对于可变对象，真正的内容修改是会影响到所有指向这个对象的引用的。

#### 解决问题的类型

回溯法抽象为树形结构后，其遍历过程就是：**for循环横向遍历，递归纵向遍历，回溯不断调整结果集**。

回溯法，一般可以解决如下几种问题：

- 组合问题：N个数里面按一定规则找出k个数的集合(不强调顺序)
- 切割问题：一个字符串按一定规则有几种切割方式
- 子集问题：一个N个数的集合里有多少符合条件的子集
- 排列问题：N个数按一定规则全排列，有几种排列方式
- 棋盘问题：N皇后，解数独等等

### 典型例题

#### [77. 组合](https://leetcode.cn/problems/combinations/)

> 给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。
>
> 你可以按 **任何顺序** 返回答案。
>
>  
>
> **示例 1：**
>
> ```
> 输入：n = 4, k = 2
> 输出：
> [
>   [2,4],
>   [3,4],
>   [2,3],
>   [1,2],
>   [1,3],
>   [1,4],
> ]
> ```
>
> **示例 2：**
>
> ```
> 输入：n = 1, k = 1
> 输出：[[1]]
> ```
>
>  
>
> **提示：**
>
> - `1 <= n <= 20`
> - `1 <= k <= n`

```c++
class Solution {
private:
    vector<int> path;
    vector<vector<int>> ans;
    void backtracing(int n, int k, int start_index) {
        if (k == 0) {
            ans.push_back(path);
            return;
        }
        for (int i = start_index; i <= n; i++) {
            path.push_back(i);
            backtracing(n, k - 1, i + 1);
            path.pop_back();
        }
        return;
    }
    
public:
    vector<vector<int>> combine(int n, int k) {
        if (n < k) {
            return ans;
        }
        backtracing(n, k, 1);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.ans = []
        self.path = []

    def backtracing(self, n: int, k: int, start_index: int) -> None:
        if k == 0:
            self.ans.append(self.path[::])
            return
        i = start_index
        while i <= n:
            self.path.append(i)
            self.backtracing(n, k - 1, i + 1)
            self.path.pop()
            i += 1
        return

    def combine(self, n: int, k: int) -> List[List[int]]:
        if n < k:
            return self.ans
        self.backtracing(n, k, 1)
        return self.ans

```

>
> 在Python中，列表是可变的数据结构，如果你直接将列表对象本身添加进另一个列表，实际上添加的是对原列表的引用，而非其值的拷贝。这意味着，如果你后续修改了原列表，那么这个引用所对应的内容也会跟着变化。
>
> 在这段代码中，self.path记录了当前递归路径中的组合。当找到一个长度为k的组合时，通过self.path[:]这种方式实际上创建了self.path的一个浅拷贝，它复制了列表中的所有元素。这样，即使后续self.path被修改（例如通过pop操作），self.ans中已经存储的组合不会受到影响。
>
> 如果你只使用self.path，就只是在self.ans中添加了一个对当前路径的引用。当self.path在递归过程中被改变时，self.ans中的所有引用都会指向最后的self.path状态，这显然是不对的。
>
> 简单来说，path[:]确保了每次向`self.ans`添加的是当前路径的一个独立副本，这样每个组合才是互相独立的，递归回溯时各个路径的变化不会互相影响。

**记住什么时候使用的是引用什么时候使用的是拷贝的副本**

#### 关于上一题的优化

> 对于树状遍历的优化,一般来说是剪枝

![77.组合4](./codeNote.assets/20210130194335207.png)

如图所示,余下的节点至少是`k - current`个,少于这个数量可以直接终止函数

```c++
class Solution {
private:
    vector<int> path;
    vector<vector<int>> ans;
    void backtracing(int n, int k, int start_index) {
        // 在剩余元素不足的直接返回
        if (n - start_index + 1 < k) {
            return;
        }
        if (k == 0) {
            ans.push_back(path);
            return;
        }
        for (int i = start_index; i <= n; i++) {
            path.push_back(i);
            backtracing(n, k - 1, i + 1);
            path.pop_back();
        }
        return;
    }

public:
    vector<vector<int>> combine(int n, int k) {
        if (n < k) {
            return ans;
        }
        backtracing(n, k, 1);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.ans = []
        self.path = []

    def backtracing(self, n: int, k: int, start_index: int) -> None:
        if n - start_index + 1 < k:
            return 
        if k == 0:
            self.ans.append(self.path[::])
            return
        i = start_index
        while i <= n:
            self.path.append(i)
            self.backtracing(n, k - 1, i + 1)
            self.path.pop()
            i += 1
        return

    def combine(self, n: int, k: int) -> List[List[int]]:
        if n < k:
            return self.ans
        self.backtracing(n, k, 1)
        return self.ans

```

#### [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

> 找出所有相加之和为 `n` 的 `k` 个数的组合，且满足下列条件：
>
> - 只使用数字1到9
> - 每个数字 **最多使用一次** 
>
> 返回 *所有可能的有效组合的列表* 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。
>
> 
>
> **示例 1:**
>
> ```
> 输入: k = 3, n = 7
> 输出: [[1,2,4]]
> 解释:
> 1 + 2 + 4 = 7
> 没有其他符合的组合了。
> ```
>
> **示例 2:**
>
> ```
> 输入: k = 3, n = 9
> 输出: [[1,2,6], [1,3,5], [2,3,4]]
> 解释:
> 1 + 2 + 6 = 9
> 1 + 3 + 5 = 9
> 2 + 3 + 4 = 9
> 没有其他符合的组合了。
> ```
>
> **示例 3:**
>
> ```
> 输入: k = 4, n = 1
> 输出: []
> 解释: 不存在有效的组合。
> 在[1,9]范围内使用4个不同的数字，我们可以得到的最小和是1+2+3+4 = 10，因为10 > 1，没有有效的组合。
> ```
>
> 
>
> **提示:**
>
> - `2 <= k <= 9`
> - `1 <= n <= 60`
> - ![216.组合总和III1](./codeNote.assets/2020112319580476.png)

```c++
class Solution {
private:
    vector<int> path;
    vector<vector<int>> ans;
    void backtracing(vector<int>& visit, int start_index, int k, int n) {
        if (k == 0 && n == 0) {
            ans.push_back(path);
            return;
        }
        for (int i = start_index; i <= 9; i++) {
            if (n >= i && !visit[i]) { // 剪枝
                path.push_back(i);
                visit[i] = 1;
                backtracing(visit, i + 1, k - 1, n - i);
                path.pop_back();
                visit[i] = 0;
            }else{
                break;
	}
        }
        return;
    }

public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<int> visit(n + 1, 0);
        backtracing(visit, 1, k, n);
        return ans;
    }
};


// 在确定元素顺序之后,保存访问数组其实没用
class Solution {
private:
    vector<int> path;
    vector<vector<int>> ans;
    void backtracing(int start_index, int k, int n) {
        if (k == 0 && n == 0) {
            ans.push_back(path);
            return;
        }
        for (int i = start_index; i <= 9; i++) {
            if (n >= i) {
                path.push_back(i);
                backtracing(i + 1, k - 1, n - i);
                path.pop_back();
            } else {
                break;
            }
        }
        return;
    }

public:
    vector<vector<int>> combinationSum3(int k, int n) {
        backtracing(1, k, n);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.path = []
        self.ans = []

    def backtracing(self, start_index: int, n: int, k: int) -> None:
        if k == 0 and n == 0:
            self.ans.append(self.path[::])
            return
        for i in range(start_index, 10):
            if n >= i:
                self.path.append(i)
                self.backtracing(i + 1, n - i, k - 1)
                self.path.pop()
        return

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        self.backtracing(1, n, k)
        return self.ans

```

#### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

> 给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。
>
> 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
>
> ![img](./codeNote.assets/200px-telephone-keypad2svg.png)
>
> 
>
> **示例 1：**
>
> ```
> 输入：digits = "23"
> 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
> ```
>
> **示例 2：**
>
> ```
> 输入：digits = ""
> 输出：[]
> ```
>
> **示例 3：**
>
> ```
> 输入：digits = "2"
> 输出：["a","b","c"]
> ```
>
> 
>
> **提示：**
>
> - `0 <= digits.length <= 4`
> - `digits[i]` 是范围 `['2', '9']` 的一个数字。

```c++
class Solution {
private:
    vector<string> ans;
    string path;
    vector<string> function = {"",    "",    "abc",  "def", "ghi",
                               "jkl", "mno", "pqrs", "tuv", "wxyz"};
    void backtracing(const string& digits, int index) {
        if (index == digits.size()) {
            if (!path.empty()) {
                ans.push_back(path);
            }
            return;
        }
        for (int i = 0; i < function[digits[index] - '0'].size(); i++) {
            path.push_back(function[digits[index] - '0'][i]);
            backtracing(digits, index + 1);
            path.pop_back();
        }
        return;
    }

public:
    vector<string> letterCombinations(string digits) {
        backtracing(digits, 0);
        return ans;
    }
};
```

```python
function = [" ", " ", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]


class Solution:
    def __init__(self):
        self.ans = []
        self.path = []

    def backtracing(self, index: int, digits: str) -> None:
        if index == len(digits):
            if len(self.path) != 0:
                self.ans.append("".join(self.path[::]))
            return
        for i in range(len(function[int(digits[index])])):
            self.path.append(function[int(digits[index])][i])
            self.backtracing(index + 1, digits)
            self.path.pop()
        return

    def letterCombinations(self, digits: str) -> List[str]:
        self.backtracing(0, digits)
        return self.ans

```

#### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

> 给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。
>
> `candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
>
> 对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。
>
>  
>
> **示例 1：**
>
> ```
> 输入：candidates = [2,3,6,7], target = 7
> 输出：[[2,2,3],[7]]
> 解释：
> 2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
> 7 也是一个候选， 7 = 7 。
> 仅有这两种组合。
> ```
>
> **示例 2：**
>
> ```
> 输入: candidates = [2,3,5], target = 8
> 输出: [[2,2,2,2],[2,3,3],[3,5]]
> ```
>
> **示例 3：**
>
> ```
> 输入: candidates = [2], target = 1
> 输出: []
> ```
>
>  
>
> **提示：**
>
> - `1 <= candidates.length <= 30`
> - `2 <= candidates[i] <= 40`
> - `candidates` 的所有元素 **互不相同**
> - `1 <= target <= 40`

```c++
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> path;
    void backtracing(const vector<int>& candidates, int target,
                     int start_index) {
        if (target == 0) {
            ans.push_back(path);
            return;
        }
        for (int i = start_index; i < candidates.size(); i++) {
            if (target >= candidates[i]) {
                path.push_back(candidates[i]);
                backtracing(candidates, target - candidates[i],
                            i); // 控制它不用跳转的这么快,跟随i的变化而变化
                path.pop_back();
            }
        }
        return;
    }

public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backtracing(candidates, target, 0);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.ans = []
        self.path = []

    def backtracing(self, candidates: List[int], start_index: int, target: int) -> None:
        if target == 0:
            self.ans.append(self.path[::])
            return
        for i in range(start_index, len(candidates)):
            if target >= candidates[i]:
                self.path.append(candidates[i])
                self.backtracing(candidates, i, target - candidates[i])
                self.path.pop()

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.backtracing(candidates, 0, target)
        return self.ans
```

#### [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

> 给定一个候选人编号的集合 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。
>
> `candidates` 中的每个数字在每个组合中只能使用 **一次** 。
>
> **注意：**解集不能包含重复的组合。 
>
>  
>
> **示例 1:**
>
> ```
> 输入: candidates = [10,1,2,7,6,1,5], target = 8,
> 输出:
> [
> [1,1,6],
> [1,2,5],
> [1,7],
> [2,6]
> ]
> ```
>
> **示例 2:**
>
> ```
> 输入: candidates = [2,5,2,1,2], target = 5,
> 输出:
> [
> [1,2,2],
> [5]
> ]
> ```
>
>  
>
> **提示:**
>
> - `1 <= candidates.length <= 100`
> - `1 <= candidates[i] <= 50`
> - `1 <= target <= 30`
> - ![40.组合总和II](./codeNote.assets/20230310000918.png)

```c++
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> path;
    void backtracing(vector<int> visited, const vector<int>& candidates,
                     int target, int start_index) {
        if (target == 0) {
            ans.push_back(path);
            return;
        }
        for (int i = start_index; i < candidates.size(); i++) {
            if (target >= candidates[i]) {
                // 注意这一步的逻辑是前一个元素没选,且上一个元素与该元素相同(要求数组排序过)
                if (i > 0 && candidates[i] == candidates[i - 1] &&
                    !visited[i - 1]) {
                    continue;
                }
                path.push_back(candidates[i]);
                visited[i] = 1;
                backtracing(visited, candidates, target - candidates[i], i + 1);
                visited[i] = 0;
                path.pop_back();
            }
        }
        return;
    }

public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<int> visited(candidates.size(), 0);
        backtracing(visited, candidates, target, 0);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.ans = []
        self.path = []

    def backtracing(
        self, candidates: List[int], visited: List[int], start_index: int, target: int
    ) -> None:
        if target == 0:
            self.ans.append(self.path[::])
            return
        for i in range(start_index, len(candidates)):

            if target >= candidates[i]:
                if i > 0 and candidates[i] == candidates[i - 1] and visited[i - 1] == 0:
                    continue
                visited[i] = 1
                self.path.append(candidates[i])
                self.backtracing(candidates, visited, i + 1, target - candidates[i])
                visited[i] = 0
                self.path.pop()

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        visited = [0 for _ in range(len(candidates))]
        self.backtracing(candidates, visited, 0, target)
        return self.ans
```

#### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

> 给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串**。返回 `s` 所有可能的分割方案。
>
> **示例 1：**
>
> ```
> 输入：s = "aab"
> 输出：[["a","a","b"],["aa","b"]]
> ```
>
> **示例 2：**
>
> ```
> 输入：s = "a"
> 输出：[["a"]] 
> ```
>
> **提示：**
>
> - `1 <= s.length <= 16`
> - `s` 仅由小写英文字母组成

```c++
class Solution {
private:
    vector<vector<string>> ans;
    vector<string> way;
    bool isPalindrome(string s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            if (s[left] != s[right]) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    void backtracing(const string& mother, int start_index) {
        if (start_index == mother.size()) {
            ans.push_back(way);
            return;
        }
        for (int end_index = start_index; end_index < mother.size();
             end_index++) {
            string temp =
                mother.substr(start_index, end_index - start_index + 1);
            if (isPalindrome(temp)) {
                way.push_back(temp);
                backtracing(mother, end_index + 1);
                way.pop_back();
            }
        }
        return;
    }

public:
    vector<vector<string>> partition(string s) {
        backtracing(s, 0);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.ans = []
        self.path = []

    def isValid(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    def backtracing(self, s: str, start_index: int) -> None:
        if start_index == len(s):
            self.ans.append(self.path[::])
            return
        for i in range(start_index, len(s)):
            temp = s[start_index : i + 1 : 1]
            if self.isValid(temp):
                self.path.append(temp)
                self.backtracing(s, i + 1)
                self.path.pop()
        return

    def partition(self, s: str) -> List[List[str]]:
        self.backtracing(s, 0)
        return self.ans
```

> 这道题可以使用动态规划来加速回文串的计算的速度
>
> ```c++
> class Solution {
> private:
>     vector<vector<string>> result;
>     vector<string> path; // 放已经回文的子串
>     vector<vector<bool>> isPalindrome; // 放事先计算好的是否回文子串的结果
>     void backtracking (const string& s, int startIndex) {
>         // 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
>         if (startIndex >= s.size()) {
>             result.push_back(path);
>             return;
>         }
>         for (int i = startIndex; i < s.size(); i++) {
>             if (isPalindrome[startIndex][i]) {   // 是回文子串
>                 // 获取[startIndex,i]在s中的子串
>                 string str = s.substr(startIndex, i - startIndex + 1);
>                 path.push_back(str);
>             } else {                                // 不是回文，跳过
>                 continue;
>             }
>             backtracking(s, i + 1); // 寻找i+1为起始位置的子串
>             path.pop_back(); // 回溯过程，弹出本次已经添加的子串
>         }
>     }
>     void computePalindrome(const string& s) {
>         // isPalindrome[i][j] 代表 s[i:j](双边包括)是否是回文字串 
>         isPalindrome.resize(s.size(), vector<bool>(s.size(), false)); // 根据字符串s, 刷新布尔矩阵的大小
>         for (int i = s.size() - 1; i >= 0; i--) { 
>             // 需要倒序计算, 保证在i行时, i+1行已经计算好了
>             for (int j = i; j < s.size(); j++) {
>                 if (j == i) {isPalindrome[i][j] = true;}
>                 else if (j - i == 1) {isPalindrome[i][j] = (s[i] == s[j]);}
>                 else {isPalindrome[i][j] = (s[i] == s[j] && isPalindrome[i+1][j-1]);}
>             }
>         }
>     }
> public:
>     vector<vector<string>> partition(string s) {
>         result.clear();
>         path.clear();
>         computePalindrome(s);
>         backtracking(s, 0);
>         return result;
>     }
> };
> ```
>
> 标记一下,到时候到动态规划的时候做一下

#### [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)

> **有效 IP 地址** 正好由四个整数（每个整数位于 `0` 到 `255` 之间组成，且不能含有前导 `0`），整数之间用 `'.'` 分隔。
>
> - 例如：`"0.1.2.201"` 和` "192.168.1.1"` 是 **有效** IP 地址，但是 `"0.011.255.245"`、`"192.168.1.312"` 和 `"192.168@1.1"` 是 **无效** IP 地址。
>
> 给定一个只包含数字的字符串 `s` ，用以表示一个 IP 地址，返回所有可能的**有效 IP 地址**，这些地址可以通过在 `s` 中插入 `'.'` 来形成。你 **不能** 重新排序或删除 `s` 中的任何数字。你可以按 **任何** 顺序返回答案。
>
>  
>
> **示例 1：**
>
> ```
> 输入：s = "25525511135"
> 输出：["255.255.11.135","255.255.111.35"]
> ```
>
> **示例 2：**
>
> ```
> 输入：s = "0000"
> 输出：["0.0.0.0"]
> ```
>
> **示例 3：**
>
> ```
> 输入：s = "101023"
> 输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
> ```
>
>  
>
> **提示：**
>
> - `1 <= s.length <= 20`
> - `s` 仅由数字组成

```c++
class Solution {
private:
    vector<string> ans;
    string path;
    bool isvalid(string temp) {
        if (temp.size() <= 1) {
            return true;
        } else if (temp.size() <= 3) {
            if (temp[0] == '0') {
                return false;
            }
            if (stoi(temp) > 255) {
                return false;
            }
            return true;
        } else {
            return false;
        }
    }
    void backtracing(const string& s, int start_index, int k) {
        if (start_index == s.size() && k == 0) {
            ans.push_back(path);
            return;
        }
        if (start_index == s.size() || k == 0) {
            return;
        }
        for (int end_index = start_index; end_index < s.size(); end_index++) {
            if (end_index - start_index + 1 > 3) {
                break;
            }
            string temp = s.substr(start_index, end_index - start_index + 1);
            if (isvalid(temp)) {
                string save_str = path;
                if (path.size() != 0) {
                    path += ('.' + temp);
                } else {
                    path += temp;
                }
                backtracing(s, end_index + 1, k - 1);
                path = save_str;
            }
        }
        return;
    }

public:
    vector<string> restoreIpAddresses(string s) {
        backtracing(s, 0, 4);
        return ans;
    }
};
```

```python
class Solution:
    def __init__(self):
        self.path = ""
        self.ans = []

    def isValid(self, s: str) -> bool:
        if len(s) <= 1:
            return True
        elif len(s) <= 3:
            if s[0] == "0" or int(s) > 255:
                return False
            else:
                return True
        else:
            return False

    def backtracing(self, s: str, start_index: int, n: int) -> None:
        if start_index == len(s) and n == 0:
            self.ans.append(self.path[::])
            return
        if start_index == len(s) or n == 0:
            return
        for i in range(start_index, len(s)):
            temp = s[start_index : i + 1 : 1]
            if self.isValid(temp):
                save = self.path
                if len(self.path) == 0:
                    self.path += temp
                else:
                    self.path += "." + temp
                self.backtracing(s, i + 1, n - 1)
                self.path = save
        return

    def restoreIpAddresses(self, s: str) -> List[str]:
        self.backtracing(s, 0, 4)
        return self.ans
```

#### [78. 子集](https://leetcode.cn/problems/subsets/)

> 给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的
>
> 子集
>
> （幂集）。
>
> 
>
> 解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [1,2,3]
> 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [0]
> 输出：[[],[0]]
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums.length <= 10`
> - `-10 <= nums[i] <= 10`
> - `nums` 中的所有元素 **互不相同**

```C++
```

```python

```

#### [90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

> 给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的<span data-keyword="subset">子集</span>（幂集）。
>
> 解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。
>
>  
>
> **示例 1：**
>
> ```
> 输入：nums = [1,2,2]
> 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
> ```
>
> **示例 2：**
>
> ```
> 输入：nums = [0]
> 输出：[[],[0]]
> ```
>
>  
>
> **提示：**
>
> - `1 <= nums.length <= 10`
> - `-10 <= nums[i] <= 10`

```c++
```

```python

```



## 第八章 贪心算法

## 第九章 动态规划

## 第十章 单调栈

## 第十一章 图论
