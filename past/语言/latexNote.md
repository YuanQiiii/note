### 小写希腊字母

```latex
%α 
\alpha
%ν 
\nu
%β 
\beta
%γ 
\gamma	
%δ 
\delta
%π 
\pi
%ϵ 
\epsilon
%ρ 
\rho
%ζ 
\zeta
%σ 
\sigma
%η 
\eta			
%τ 
\tau
%θ 
\theta
%υ 
\upsilon
%ι 
\iota
%ϕ 
\phi
%κ 
\kappa
%χ 
\chi
%λ 
\lambda	
%ψ 
\psi
%μ 
\mu
%ω 
\omega
```

### 大写希腊字母

> 将原有的小写希腊字母的首字母大写即可

```latex
%A 
\Alpha
%B 
\Beta
%Γ 
\Gamma
%Λ 
\Lambda
%Σ 
\Sigma
%Ψ 
\Psi
%Δ 
\Delta
%Υ 
\Upsilon
%Ω 
\Omega
%Θ 
\Theta
%Ξ 
\Xi
%Π 
\Pi
%Φ 
\Phi
```

### 运算符 & 空格

普通字符在数学公式中含义一样，除了 # $ % & ~ _ ^ \ { } 若要在数学环境中表示这些符号# $ % & _ { }，需要分别表示为# $ % & _ { }，即在个字符前加上`\ `。

```latex
%单空格 
a \quad b
%双空格 
a \qquad b
%乘号
\times
%#
#
%$
\$	
%%	 
\%
%&
\&
%_
\_
%–
-
```

### 上下标

对于上标使用 下划线表示“ _ ” ；对于上标使用 “ ^ ”表示。

在此需要注意的是：$LaTex$​表达式默认的是 “ _ ” “ ^ ” 之后的**一位**才是上下标的内容，对于超过一个字母的上下标需要使用 { } 将它括起来

```latex
\hat{a}
```
$\hat{a}$

```latex
\acute{a}
```
$\acute{a}$

```latex
\grave{a}
```
$\grave{a}$

```latex
\breve{a}
```
$\breve{a}$

```latex
\bar{a}
```
$\bar{a}$

```latex
\widetilde{a}
```
$\widetilde{a}$

```latex
\check{a}
```
$\check{a}$

```latex
\tilde{a}
```
$\tilde{a}$

```latex
\dot{a}
```
$\dot{a}$

```latex
\ddot{a}
```
$\ddot{a}$

```latex
\vec{a}
```
$\vec{a}$

```latex
\widehat{a}
```
$\widehat{a}$

这些重音符号可用于表示变量的不同数学和科学含义，如导数、向量和其他特殊功能。在撰写数学或物理学文献时，使用这些修饰符可以帮助传达精确的概念。

### log

$log_{e}{xy}$

```latex
log_{e}{xy}
```



### 括号

$$LaTex$$表达式中的 ( ) 、 [ ] 均可以正常使用，但是对于 { } 需要使用[转义字符](https://so.csdn.net/so/search?q=转义字符&spm=1001.2101.3001.7020)使用，即使用 “\{” 和 “\}” 表示 { }

```latex
\left( A \right)
```
$\left( A \right)$

```latex
\vert A \vert
```
$\vert A \vert$

```latex
\Vert A \Vert
```
$\Vert A \Vert$

```latex
\langle A \rangle
```
$\langle A \rangle$

```latex
\lceil A \rceil
```
$\lceil A \rceil$

```latex
\lfloor A \rfloor
```
$\lfloor A \rfloor$

对于需要调整大小的标准数学符号，如圆括号、方括号等，您可以使用 `\left` 和 `\right` 命令，或者 `\bigl`, `\Bigl`, `\biggl`, `\Biggl` 等命令来调整左边符号的大小，以及 `\bigr`, `\Bigr`, `\biggr`, `\Biggr` 等命令来调整右边符号的大小。下面是一些例子：

```latex
\Biggl( \biggl( \Bigl( \bigl( A \bigr) \Bigr) \biggr) \Biggr)
```
$\Biggl( \biggl( \Bigl( \bigl( A \bigr) \Bigr) \biggr) \Biggr)$

**请注意，`\left` 和 `\right` 命令总是成对出现，用以自动调整与中间内容相匹配的大小。而 `\bigl`, `\Bigl`, `\biggl`, `\Biggl` 等命令需要与 `\bigr`, `\Bigr`, `\biggr`, `\Biggr` 等对应的命令成对使用，以确保左右符号的大小相匹配**

在撰写具有复杂分隔符的数学文档时，这些命令非常有用，因为它们能帮助清晰地表示嵌套的数学结构。

$f(x)=\begin{cases} x = \cos(t) \\y = \sin(t) \\ z = \frac xy \end{cases}$

```latex
f(x)=\begin{cases} x = \cos(t) \\y = \sin(t) \\ z = \frac xy \end{cases}
```

$f(x)=\begin{cases} 0& \text{x=0}\\1& \text{x!=0} \end{cases}$

```latex
f(x)=\begin{cases} 0& \text{x=0}\\1& \text{x!=0} \end{cases}
```

### 矩阵

$\begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix}$

```latex
\begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix}
```

$\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\\$

```latex
\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\\
```

$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$

```latex
\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
```

$\begin{Bmatrix} 1 & 0 \\ 0 & -1 \end{Bmatrix}$

```latex
\begin{Bmatrix} 1 & 0 \\ 0 & -1 \end{Bmatrix}
```

$\begin{vmatrix} a & b \\ c & d \end{vmatrix}$

```latex
\begin{vmatrix} a & b \\ c & d \end{vmatrix}
```

$\begin{Vmatrix} i & 0 \\ 0 & -i \end{Vmatrix}$

```latex
\begin{Vmatrix} i & 0 \\ 0 & -i \end{Vmatrix}
```

### 求和&积分

$\sum$ 

```latex
\sum
```

$\int$ 

```latex
\int
```

$\sum_1^n$

```latex
\sum_1^n
```

$\sum_{i=0}^\infty i^2$

```latex
\sum_{i=0}^\infty i^2
```

$\prod_{k=1}^n k = n!$

```latex
\prod_{k=1}^n k = n!
```

$\infty\bigcup\bigcap\iint\iiint$

```latex
\infty
\bigcup
\bigcap
\iint
\iiint % 类推到任意多重积分
```

### 开方

$\sqrt{x^3}$

```latex
\sqrt{x^3}
```

$\sqrt[3]{\frac xy}$

```latex
\sqrt[3]{\frac xy} 
% \frac是分数的分割线,分割后面紧跟的两个元素
% \sqrt 后面可以跟中括号[n],其中n是开n次方根
```

### 分数

```latex
\frac

\frac{a+1}{b+1} 
{a+1\over b+1}

\cfrac {a}{b} % 适用于复杂分数表示式,上述三种适用于简单分数表达式
```

> `\cfrac`和`\frac`都是用来表示分数的LaTeX命令，但它们有一些区别：
>
> 1. `\cfrac`: `\cfrac{a}{b}`表示一个带有水平线的分数，水平线会根据分子和分母的大小自动调整长度，适用于较大的分数表达式。
> 2. `\frac`: `\frac{a}{b}`表示一个普通的分数，水平线的长度会固定为分子和分母的长度，适用于简单的分数表达式。
>
> 如果您需要显示一个较大的分数表达式，可以考虑使用`\cfrac`命令，如果是简单的分数表达式，则可以使用`\frac`命令。

### 特殊函数

> - 间距
>   - 插入正间距`\,`
>   - 插入负间距`\i`

$\lim\,\,\,\,\,\lim_{x\to 0}\,\,\,\,\,\sin x\,\,\,\,\,\cos x\,\,\,\,\,\hat x\,\,\,\,\,\widehat{xy}\,\,\,\,\,\bar x\,\,\,\,\,\overline{xyz}\,\,\,\,\,\vec x\,\,\,\,\,\overrightarrow{xyz}\,\,\,\,\,\overleftrightarrow{xyz}\,\,\,\,\,\stackrel{F.T}{\longrightarrow}\,\,\,\,\,\dot x\,\,\,\,\,\ddot x$

```latex
\lim

\lim_{x\to 0}	% \to是右箭头

\sin x

\cos x

\hat x
 
\widehat{xy}
 
\bar x
 
\overline{xyz}
 
\vec x
 
\overrightarrow{xyz} % 顶上的右箭头
 
\overleftrightarrow{xyz} % 顶上的左右箭头
 
\stackrel{F.T}{\longrightarrow}

\dot x

\ddot x
```

### 特殊符号和符号

```latex
\lt
```
$<$

```latex
\gt
```
$>$

```latex
\le
```
$\leq$

```latex
\leqq
```
$\leqq$

```latex
\leqslant
```
$\leqslant$

```latex
\ge
```
$\geq$

```latex
\geqq
```
$\geqq$

```latex
\geqslant
```
$\geqslant$

```latex
\neq
```
$\neq$

```latex
\not\lt
```
$\not\lt$

```latex
\not
```
(not applicable as it is an operation rather than a symbol)

```latex
\times
```
$\times$

```latex
\div
```
$\div$

```latex
\pm
```
$\pm$

```latex
\mp
```
$\mp$

```latex
\cdot
```
$\cdot$

```latex
\cup
```
$\cup$

```latex
\cap
```
$\cap$

```latex
\setminus
```
$\setminus$

```latex
\subset
```
$\subset$

```latex
\subseteq
```
$\subseteq$

```latex
\subsetneq
```
$\subsetneq$

```latex
\supset
```
$\supset$

```latex
\in
```
$\in$

```latex
\notin
```
$\notin$

```latex
\emptyset
```
$\emptyset$

```latex
\varnothing
```
$\varnothing$

```latex
\choose
```
(not applicable as it is part of a command rather than a standalone symbol)

```latex
\to
```
$\to$

```latex
\rightarrow
```
$\rightarrow$

```latex
\leftarrow
```
$\leftarrow$

```latex
\Rightarrow
```
$\Rightarrow$

```latex
\Leftarrow
```
$\Leftarrow$

```latex
\mapsto
```
$\mapsto$

```latex
\land
```
$\land$

```latex
\lor
```
$\lor$

```latex
\lnot
```
$\lnot$

```latex
\forall
```
$\forall$

```latex
\exists
```
$\exists$

```latex
\top
```
$\top$

```latex
\bot
```
$\bot$

```latex
\vdash
```
$\vdash$

```latex
\vDash
```
$\vDash$

```latex
\star
```
$\star$

```latex
\ast
```
$\ast$

```latex
\oplus
```
$\oplus$

```latex
\circ
```
$\circ$

```latex
\bullet
```
$\bullet$

```latex
\approx
```
$\approx$

```latex
\sim
```
$\sim$

```latex
\simeq
```
$\simeq$

```latex
\cong
```
$\cong$

```latex
\equiv
```
$\equiv$

```latex
\prec
```
$\prec$

```latex
\lhd
```
$\lhd$

```latex
\therefore
```
$\therefore$

```latex
\infty
```
$\infty$

```latex
\aleph_0
```
$\aleph_0$

```latex
\nabla
```
$\nabla$

```latex
\partial
```
$\partial$

```latex
\Im
```
$\Im$

```latex
\Re
```
$\Re$

```latex
\equiv\ \pmod{n}
```
$a \equiv b \pmod{n}$

```latex
\ldots
```
$\ldots$

```latex
\cdots
```
$\cdots$

```latex
\epsilon
```
$\epsilon$

```latex
\varepsilon
```
$\varepsilon$

```latex
\phi
```
$\phi$

```latex
\varphi
```
$\varphi$

```latex
\ell
```
$\ell$

>  Please note that some commands such as `\not` are not symbols themselves but rather are combined with other symbols to create a negated version (e.g., `\not\lt` creates a "not less than" symbol). Also, the `\choose` command is typically part of a larger construct to create binomial coefficients, so I've provided the `\binom` version instead for a correct standalone command.

### 字体

```latex
\mathbb{ABCDE}
```
$\mathbb{ABCDE}$

```latex
\mathbf{abcde}
```
$\mathbf{abcde}$

```latex
\mathtt{ABCDE}
```
$\mathtt{ABCDE}$

```latex
\mathrm{ABCDE}
```
$\mathrm{ABCDE}$

```latex
\mathsf{ABCDE}
```
$\mathsf{ABCDE}$

```latex
\mathcal{ABCDE}
```
$\mathcal{ABCDE}$

```latex
\mathscr{ABCDE}
```
$\mathscr{ABCDE}$

```latex
\mathfrak{ABCDE}
```
$\mathfrak{ABCDE}$

> 请注意 `\Bbb` 命令在许多 LaTeX 系统中已经过时，取而代之的是 `\mathbb`。另外， `\mathscr` 需要 `mathrsfs` 包，而 `\mathfrak` 需要 `amssymb` 或者 `amsfonts` 包。如果你在使用 LaTeX 的时候这些命令没有正常工作，可能需要在文档的序言部分添加相应的包。
