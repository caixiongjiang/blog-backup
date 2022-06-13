---
title: "LaTeX入门教程"
date: 2022-06-13T18:18:05+08:00
lastmod: 2022-06-13T18:20:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/title_img.jpg"
description: "论文排版搞得我很头疼，mac下的office又不是很好用，索性学习一下latex"
tags:
- LaTeX
categories:
- 论文排版工具
comment : true
---



## LaTeX入门教程

### 环境安装和配置的几种方案

* mac环境

  * 1. MacTex发行版：[https://www.tug.org/mactex/](https://www.tug.org/mactex/) 
    2. texpad(付费，当然某些网站有破解版)

    texpad不需要任何复杂的配置

  * 1. MacTex发行版：[https://www.tug.org/mactex/](https://www.tug.org/mactex/) 
    2. vscode+LaTeX Workshop插件

    vscode方案装好插件之后还需要配置.json

    ```json
    //注释掉的部分属于代码提示的功能，还需要装插件
    {
        "latex-workshop.latex.tools": [
            {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-pdf",
            "%DOC%"
            ]
            },
            {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
                ]
            },          
            {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
            ]
            },
            {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
            "%DOCFILE%"
            ]
            }
        ],
    "latex-workshop.latex.recipes": [
            {
            "name": "xelatex",
            "tools": [
            "xelatex"
                        ]
                    },
            {
            "name": "latexmk",
            "tools": [
            "latexmk"
                        ]
            },
    
            {
            "name": "pdflatex -> bibtex -> pdflatex*2",
            "tools": [
            "pdflatex",
            "bibtex",
            "pdflatex",
            "pdflatex"
                        ]
            },
            {  
              "name": "xelatex->bibtex->xelatex->xelatex",  
              "tools": [  
                "xelatex",  
                "bibtex",  
                "xelatex",  
              ]  
            }  
        ],
        
    "latex-workshop.view.pdf.viewer": "tab",  
    "latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.ist",
        "*.fls",
        "*.log",
        "*.fdb_latexmk"
        ],
       
    
    //下面这段是语法检查模块
    //{
    // "ltex.enabled": true, // 启用插件
    // "ltex.language": "en-US",// 设置语言，这里是德语
    // // 要英语就下载对应 English Support，然后这里填 en, 或者 en-US,en-GB 等*/
    //  "ltex.de.dictionary": ["Niubility", "Zhihu"], 
    //  //注意根据要对应语言，ltex.<LANGUAGE>.dictionary
    // "ltex.environments.ignore": [
    //     "lstlisting",
    //     "verbatim"
    // ],
    // "ltex.commands.ignore": [
    //     "\\documentclass[]{}",
    //     "\\renewcommand*{}[]{}"
    // ],
    // "editor.fontSize": 18,
    //}语法检查功能在这里结束
    
    
    
    "latex-workshop.view.pdf.viewer": "external",
    "latex-workshop.view.pdf.external.synctex.command": "/Applications/Skim.app/Contents/SharedSupport/displayline",
    "latex-workshop.view.pdf.external.synctex.args": [
    "-r",
    "%LINE%",
    "%PDF%",
    "%TEX%"
    ],
    "latex-workshop.view.pdf.external.viewer.command": "/Applications/Skim.app/Contents/MacOS/Skim",
    "latex-workshop.view.pdf.external.viewer.args": [
        "%PDF%"
    ],
    "window.zoomLevel": 2,
    "editor.fontSize": 16,
    
    
    }
    ```

    

  * 当然也可以直接使用latex在线编辑网站overleaf：[https://cn.overleaf.com/](https://cn.overleaf.com/)

*Windows环境下的选择就很多了，网上教程五花八门。在mac下还是推荐overleaf，虽然texpad有定位功能，但是它的宏包下载实在是太慢了，不推荐。*

* 初始编译设置

  将默认编译器设置为XeLaTeX，默认的编码格式设置成utf-8。在texpad的初始配置如下图：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img1.jpg)

  而在vscode中配置完json就可以完全使用了，使用场景如下：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img2.jpg)

  点击左侧的recipe xelatex就是编译，view in VSCode tab可以在右边实时查看编译生成的pdf。



### LaTeX的中文处理

Demo:

```latex
% 导言区
\documentclass{article} % 样式：book, report, letter

\usepackage{xeCJK}% 中文处理

% 定义新的命令
\newcommand\degree{^\circ}

\title{\heiti 杂谈勾股定理} % 标题
\author{\heiti 蔡雄江} % 作者
\date{\today} % 时间


% 正文区（文稿区）{*工作环境*}
\begin{document}
	\maketitle
	Hello World!
	
	% 行内公式
	Let $f(x)$ be defined by the formula
	$f(x)=3x^2+x-1$	
	% 行间公式
	$$f(x)=3x^2+x-1$$ 
	
	
	% 中文示例
	勾股定理可以用现代语言表述如下：
	
	直角三角形斜边的平方等于两腰的平方和
	
	设直角三角形$ABC$，其中$\angle C=90\degree$,则有：
	\begin{equation} % equation用于产生带编号的行间公式
	AB^2 = BC^2 + AC^2.	
	\end{equation}
	
\end{document}}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img3.jpg)

### LaTeX的字体字号

Demo:

```latex
% 导言区 []里代表的是normal size的大小（10，11，12pt）
\documentclass[12pt]{article}

\usepackage{ctex}

% 自定义字体
\newcommand{\myfont}{\textbf{\textsf{Fancy Text}}}

\begin{document}
    % 字体族设置 (罗马字体、无衬线字体、打字机字体)
    \textrm{Roman Family}  \textsf{Sans Serif Family} \texttt{Typewriter Family}
    
    {\rmfamily Roman Family} {\sffamily Sans Serif Family} {\ttfamily Typewriter Family}
    
    % 大括号用于限定声明的范围
    {\sffamily who you are? you find self on everyone around.
    take you as the same as others!}
    
    {\rmfamily who you are? you find self on everyone around.
    take you as the same as others!}
    
    % 字体系列设置（粗细宽度）
    \textmd{Medium Series} \textbf{Boldface Series}
    
    {\mdseries Medium Series} {\bfseries Boldface Series}
    
    % 字体形状（直立、斜体、伪斜体、小型大写）
    \textup{Upright Shape} \textit{Italic Shape}
    \textsl{Slanted Shape} \textsc{Small Caps Shape}
    
    {\upshape Upright Shape} {\itshape Italic Shape} {\slshape Slanted Shape}
    {\scshape Small Caps Shape}
    
    % 中文字体 \quad代表空格
    {\songti 宋体} \quad {\heiti 黑体} \quad {\fangsong 仿宋}
    \quad {\kaishu 楷书}
    
    中文字的\textbf{粗体}与\textit{斜体}
    
    % 字体大小 \\代表换行
    {\tiny Hello}\\
    {\scriptsize Hello}\\
    {\footnotesize Hello}\\
    {\small Hello}\\
    {\normalsize Hello}\\
    {\large Hello}\\
    {\Large Hello}\\
    {\huge Hello}\\
    {\Huge Hello}\\
    
    % 中文字号设置命令 具体细节查看ctex文档
    % ctex文档查阅 在termimal输入texdoc ctex
    \zihao{5} 你好！
    
    \myfont
    
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img4.jpg)

### LaTeX的篇章结构

Demo:

```latex
% 导言区
\documentclass{ctexbook}


%====内容和格式分离
\ctexset {
  part/pagestyle = empty,
  chapter = {
    format    = \raggedright,
    pagestyle = empty,
  },
section = {
name = {第,节},
number = \chinese{section},
} }

% 正文区
\begin{document}
    % 生成文档的目录
    \tableofcontens 
    
    \chapter{绪论}
    \section{引言}
    % 正文的格式不受section等命令的影响
    Jarson Cai是一个帅哥！第一步是创建一个新的 LATEX 项目。你可以在自己的电脑上创建 .tex 文件，也可以 在 Overleaf 中启动新项目。让我们从最简单的示例开始
    
    % \\代表换行但没有缩进 \par 代表新的段落，带缩进
    近年来，唐山大人！\\近年来，真实事件\par 近年来，真的很真实
    
    \section{实验方法}
    \section{实验结果}
    \subsection{数据}
    \subsubsection{实验条件}
    \subsubsection{实验过程}
    \subsection{图表}
    \subsection{结果分析}
    \section{结论}
    \section{致谢}
    
    \chapter{实验与结果分析}
    \section{实验方法}
    \section{实验结果}
    \subsection{数据}
    \subsubsection{实验条件}
    \subsubsection{实验过程}
    \subsection{图表}
    \subsection{结果分析}
    \section{结论}
    \section{致谢}
    
\end{document}  
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img5.jpg)

### LaTeX的特殊字符

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}

\begin{document}
    \section{空白符号}
    %空行分段，多个空行等同一个
    %自动缩进，绝对不能使用空格代替
    %英文中多个空格处理为1个空格，中文空格会被忽略
    %汉字与其他字符的间距会自动由XeLaTex处理
    %禁止使用中文全角空格
    采用平台扫描仪获取叶片的数字图像,建立运用数字图像处理技术测定蔬菜叶面积的方法,同时与目前较常用的交叉网格法,CID仪器法,复印称重法和生产上常用的直尺法进行比较分析,结果表明,图像处理方法和上述传统的叶面积测定方法的测定结果呈极显著的线性相关关系,适用于叶面积的测量工作。
    
    Use a platform scanner to obtain digital images of the blades, establish a method of using digital image processing technology to determine the area of vegetable leaves. The comparative analysis shows that the measurement method of the image processing method and the above -mentioned traditional leaf area measurement method shows a very significant linear correlation, which is suitable for the measurement work of the leaf area.
    
    % 1em（当前字体中M的宽度）
    a\quad b
    
    % 2em
    a\qquad b 
    
    % 约为1/6个em
    a\,b a\thinspace b
    
    % 0.5个em
    a\enspace b
    
    % 空格
    a\ b
    
    % 硬空格
    a~b
    
    % 弹性长度空白
    a\hfill b
    
    \section{\LaTeX 控制符}
    %转译字符需要加\ \textbackslash代表反斜杠
    \#\quad \$\quad  \%\quad  \{\quad  \}\quad \~{}\quad \^{}\quad \textbackslash\quad  \&
    

    \section{排版符号}
    \S \P \dag \ddag \copyright \pounds
    
    \section{\TeX 标志符号}
    
    %基本符号
    \TeX{} \LaTeX{} \LaTeXe{}
    
    \section{引号}
    %单引号 + 双引号
    `你好' ``你好''
    
    \section{连字符}
    % 根据长度分别生成三种长度的连字符
    - -- ---
    
    \section{非英文字符}
    \oe \OE \ae \AE \aa \AA \o 
    
    \section{重音符号（以o为例）}
    \`o \'o \^o \''o \~o \=o \.o 
     
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img6.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img7.jpg)

### LaTeX的插图

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}

% 插图
% 导言区：\usepackage{graphicx}
% 语法：\includegraphics[<选项>]{<文件名>}
% 格式：EPS,PDF,PNG,JPEG,BMP
\usepackage{graphicx}
\graphicspath{{figures/},{pics/}} % 图片在当前目录下的figures目录和pics目录，路径之间使用{}进行分组


% 正文区
\begin{document}
    \LaTeX{} 中的插图：
    
    % scale代表指定缩放因子
    \includegraphics[scale=0.3]{x1.jpg}
    
    \includegraphics[width=10cm]{x1.jpg}
    
    \includegraphics[height=10cm]{x1.jpg}
    
    % 指定相对高度和相对宽度
    \includegraphics[height=0.1\textheight]{x1.jpg}
    
    \includegraphics[width=0.2\textwidth]{x1.jpg}
    
    % 指定旋转angle角度
    \includegraphics[angle=45,width=0.2\textwidth]{x1.jpg}
    
    \includegraphics[angle=-45,width=0.2\textwidth]{x1.jpg}
    
    %具体细节可以在terminal输入texdoc graphicx
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img8.jpg)

### LaTeX的表格

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}

\begin{document}
    %{}指定左、中、右对齐, |产生竖线, ||产生竖线，p{} 指定宽度
    \begin{tabular}{|l||c|c|c|p{1.5cm}|}
        % \hline命令产生横线
        \hline
        姓名 & 语文 & 数学 & 外语 & 备注\\ 
        \hline \hline % 双横线
        张三 & 87 & 100 & 93 & 优秀\\
        \hline
        李四 & 75 & 64 & 52 & 补考另行通知\\
        \hline
        王五 & 80 & 82 & 78 & 良好\\
        \hline
    \end{tabular}
    
    % texdoc booktab 三线表
    % texdoc longtab 跨页长表格
    % 建议使用在线表格工具比较方便
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img9.jpg)

### LaTeX的浮动体

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}
\usepackage{graphicx}
\graphicspath{{figures/}} % 图片路径

% 浮动体排版位置 默认为tbp
% h, 代码所在的上下文位置
% t, 页顶-代码所在页面或之后页面的顶部
% b, 页底-代码所在页面或之后页面的底部
% p, 独立一页-浮动页面

% 标题控制（caption）
% 并排与子图表（subcaption，subfig,floatrow等宏包）
% 绕排（picinpar、wrapfig等宏包）

\begin{document}
    \LaTeX{}中的mac OS的桌面图片见图\ref{fig-desktop}： % 利用标签实现交叉引用
    %{}中figure代表图片的浮动体环境
    \begin{figure}[htbp] % 指定浮动体的排版位置
        \centering % 居中排版
        \includegraphics[scale=0.2]{x1.jpg}
        \caption{\TeX mac OS的桌面图片} % 设置插图标题
        \label{fig-desktop}% 为浮动体设置标签
    \end{figure}
    
    在\LaTeX{}中的学生成绩见表\ref{tab-score}：
    %{}中figure代表表格的浮动体环境
    \begin{table}[h]
        \centering %居中排版
        % 标签一般跟在标题后面
        \caption{学生考试成绩}\label{tab-score}
        \begin{tabular}{|l||c|c|c|r|}
            % \hline命令产生横线
            \hline
            姓名 & 语文 & 数学 & 外语 & 备注\\ 
            \hline \hline % 双横线
            张三 & 87 & 100 & 93 & 优秀\\
            \hline
            李四 & 75 & 64 & 52 & 补考另行通知\\
            \hline
            王五 & 80 & 82 & 78 & 良好\\
            \hline
        \end{tabular}
    \end{table}
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img10.jpg)

### LaTeX的数学公式

#### LaTeX数学公式初步

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}
\usepackage{amsmath}

\begin{document}
    \section{简介}
    
    \LaTeX{}将排版内容分为文本模式和数学模式。文本模式用于普通文本排版，数学模式用于数学公式排版。
    
    \section{行内公式}
    \subsection{美元符号}
    交换律是 $a+b=b+a$,如$1+2=2+1=3$
    \subsection{小括号}
    交换律是\(a+b=b+a\),如\(1+2=2+1=3\)
    \subsection{math环境}
    交换律是 \begin{math}a+b=b+a\end{math},如\begin{math}1+2=2+1=3\end{math}
    
    \section{上下标}
    \subsection{上标}
    $3x^{20} - x + 2 = 0$
    
    $3x^{x+1} - x + 2 = 0$
    \subsection{下标}
    $a_0, a_1, a_2,...,a_{100}$
    
    \section{希腊字母}
    $\alpha$
    $\beta$
    $\gamma$
    $\epsilon$
    $\pi$
    $\omega$
    
    $\Gamma$
    $\Delta$
    $\Theta$
    $\Pi$
    $\Omega$
    
    $\alpha^3 + \beta^3 + \gamma = 0$
    
    \section{数学函数}
    $\log$
    $\sin$
    $\cos$
    $\arcsin$
    $\arccos$
    $\ln$
    
    $\sin^2 x+ \cos^2 x= 1$
    $y = \arcsin x$
    
    $y = \sin^{-1} x$
    
    $y = \log_2 x$
    
    $y = \ln x$
    
    $\sqrt{2}$
    $\sqrt{x^2 + y^2}$
    $\sqrt[4]{x}$ % []里代表几次方根
    
    \section{分式}
    大约是原体积的$3/4$ \quad
    大约是原体积的$\frac{3}{4}$
    
    \section{行间公式}
    \subsection{美元符号}
    交换律是
    $$a+b=b+a$$
    如
    $$1+2=2+1=3$$
    \subsection{中括号}
    交换律是
    \[a+b=b+a\]
    如
    \[1+2=2+1=3\]
    \subsection{displaymath环境}
    交换律是
    \begin{displaymath}
    a+b=b+a
    \end{displaymath}
    如
    \begin{displaymath}
    1+2=2+1=3
    \end{displaymath}
    \subsection{自动编号公式equation环境}
    交换律见式子\ref{eq:commutative}
    \begin{equation}
        a+b=b+a \label{eq:commutative}
    \end{equation}
    
    \subsection{不编号公式equation*环境}
    % 需要引入amsmath宏包
    %交换律见式子\ref{eq:commutative2}
    \begin{equation*}
        a+b=b+a %\label{eq:commutative2}
    \end{equation*}
    
    公式的编号与交叉引用也是自动实现的，在排版中，需要习惯于采用自动化的方式处理诸如图、表、公式的编号和交叉引用。再如公式\ref{eq:pol}
    \begin{equation}
        x^5 - 7x^3 + 4x = 0 \label{eq:pol}
    \end{equation}
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img11.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img12.jpg)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img13.jpg)

#### LaTeX数学模式中的矩阵

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}
\usepackage{amsmath}

% 正文区
\begin{document}
    % 使用矩阵环境需要引入amsmath环境
    % 矩阵环境，用&分隔列，用\\分隔行
    $ % 没有括号
    \begin{matrix}
        0 & 1\\
        1 & 0
    \end{matrix}
    $ \quad
    $ % 小括号
    \begin{pmatrix}
        0 & 1\\
        1 & 0
    \end{pmatrix}
    $ \quad
    $ % 中括号
    \begin{bmatrix}
        0 & 1\\
        1 & 0
    \end{bmatrix}
    $ \quad
    $ % 大括号
    \begin{Bmatrix}
        0 & 1\\
        1 & 0
    \end{Bmatrix}
    $ \quad
    $ % 单竖线
    \begin{vmatrix}
        0 & 1\\
        1 & 0
    \end{vmatrix}
    $ \quad
    $ % 双竖线
    \begin{Vmatrix}
        0 & 1\\
        1 & 0
    \end{Vmatrix}
    $ 
    
    
    % 也可以使用上下标
    $$
    A = \begin{pmatrix}
        a_{11}^2 & a_{12}^2 & a_{13}^2 \\
        0 & a_{22} & a_{23} \\
        0 & 0 & a_{23}
    \end{pmatrix}
    $$
    
    % 常用的省略号：\dots、\vdots、\ddots 分别代表横竖斜省略号
    $$ % 在数学模式中可以使用\times命令排版乘号
    A = \begin{bmatrix}
        a_{11} & \dots & a_{1n} \\
        & \ddots & \vdots \\
        0 & & a_{nn}
    \end{bmatrix}_{n \times n}
    $$
    
    % 分块矩阵
    $$ % \text代表在数学模式临时切换至文本模式
    \begin{bmatrix}
    \begin{matrix} 1 & 0 \\0 & 1 \end{matrix}
    & \text{\Large 0} \\
    \text{\Large 0} & \begin{matrix}
    1 & 0 \\ 0 & 1\end{matrix}
    \end{bmatrix}
    $$
    
    % 三角矩阵
    $$ % \multicolumn代表合并多列，2代表2列，c代表在中间，指定高度
    \begin{bmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
           & a_{22} & \cdots & a_{2n} \\
           &        & \ddots & \vdots \\
    \multicolumn{2}{c}{\raisebox{1.3ex}[0pt]{\Huge 0}}
    &  &a_{nn}
    \end{bmatrix}
    $$
    
    % 跨列省略号：\hdotsfor{<列数>}
    $$
    \begin{pmatrix}
    1 & \frac 12 & \dots & \frac 1n \\
    \hdotsfor{4} \\
    m & \frac m2 & \dots & \frac mn
    \end{pmatrix}
    $$
    
    % 行内小矩阵:smallmatrix环境
    复数$z = (x,y)$也可用矩阵
    \begin{math}
    \left( % 需要手动加上左括号
    \begin{smallmatrix}
    x & -y \\ y & x
    \end{smallmatrix}
    \right) % 需要手动加上右括号
    \end{math}来表示
    
    % array环境（类似于表格环境tabular）
    $
    \begin{array}{|r|r|}
    \hline
    \frac{1}{2} & 0 \\
    \hline
    0 & -\frac a{bc} \\
    \hline
    \end{array}
    $
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img14.jpg)

#### LaTeX数学公式中的多行公式

Demo:

```latex
% 导言区
\documentclass{article}

\usepackage{ctex}
\usepackage{amsmath}
\usepackage{amssymb}

% 正文区
\begin{document}
    % gather和gather*环境都可以使用\\换行
    % gather带编号
    \begin{gather}
        a + b + c = b + a + c \\
        = b + c + a
    \end{gather}
    
    % gather*不带编号
    \begin{gather*}
        a + b + c = b + a + c \\
        = b + c + a
    \end{gather*}
    
    % 在\\前使用\notag阻止编号
    \begin{gather}
        a + b + c = b + a + c  \\
        = b + c + a \notag
    \end{gather}
    
    % align 和 align*环境（用&进行对齐）
    % 带编号
    \begin{align}
        x &= t + \cos t + 1 \\
        y &= 2\sin t
    \end{align}
    % 不带编号
    \begin{align*}
        x &= t & x &= \cos t & x & = t \\
        y &= 2t & y &= \sin(t + 1) & y & = \sin t  
    \end{align*}
    
    % split环境（对齐采用align环境的方式，编号垂直中间）
    \begin{equation}
        \begin{split}
            \cos 2x &= \cos^2 x - sin^2 x \\
                    &= 2\cos^2 x - 1
        \end{split}
    \end{equation}
    
    % cases环境
    % 每行公式中使用&分隔为两部分
    % 通常表示值和后面的条件
    \begin{equation} % \in代表属于的数学符号 
    D(x) = \begin{cases}
        1, & \text{如果} x \in \mathbb{Q}; \\
        0, & \text{如果} x \in \mathbb{R}\setminus\mathbb{Q}.
    \end{cases}
    \end{equation}
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img15.jpg)

### LaTeX中的参考文献BibTex

Demo:

```latex
\documentclass{article}

\usepackage{ctex}

\begin{document}
    % 一次管理，一次使用
    % 参考文献格式：
    % \begin{thebibliography}{编号样本}
    %   \bibitem[记号]{引用标志}文献条目1
    %   \bibitem[记号]{应用标志}文献条目2
    %   ...
    %   \end{thebibliography}
    % 其中文献条目包括：作者，题目，出版社，年代，版本，页码等。
    
    % 引用时候可以采用:\cite{引用标志1，引用标志2，...}
    引用一篇文章\cite{article1}  引用一本书\cite{book1}等等
    
    \begin{thebibliography}{99} % 99是为了参考文献对齐
        \bibitem{article1}陈立辉，苏伟，蔡川，陈晓云.\emph{基于LaTex的Web数学公式提取研究}[J].计算机科学.2014(06)
        \bibitem{book1}William H.Press,Saul A.Teukolsky,William T. Vetterling,Brian P. Flannery,\emph{Numerical Recipes 3rd Edition:The Art of Scientific Computing}
        Cambridge University Press, New York,2007.
    
    \end{thebibliography}
    
    % 也可以使用外部导入bib文件的方式进行
    这里需要zetero的配合这部分会另外讲解
    
    
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img16.jpg)

### LaTeX中的参考文献BibLaTeX

Demo:

```latex
\documentclass{ctexart}

%\usepackage{ctex}
% biblatex/biber
% 新的TEX参考文献排版引擎
% 样式文件（参考文献样式文件 --bbx文件，引用样式文件--cbx）使用LATEX编写
% 支持根据本地化排版，如：
%       biber -l zh_pinyin texfile,用于指定按拼音排序
%       biber -l zh_stroke texfile,用于按笔画排序
\usepackage[style=numeric,backend=biber]{biblatex}
\addbibresource{mybibliography.bib} % 添加参考文献数据库
\begin{document}
    % 一次管理，多次应用
    无格式引用\cite{biblatex}
    
    带括号的引用\parencite{a1-1}
    
    上标引用\supercite{6-1}
    
    % 输出参考文献列表
    \printbibliography[title = {参考文献}]
    
\end{document}
```

其中我的`bib文件`内容如下：

```latex
@online{6-1,
    title = {PACS-L: the public-access computer systems forum},
    type = {EB/OL},
    location = {Houston, Tex},
    publisher = {University of Houston Libraries},
    year = {1989},
    url = {http://info.lib.uh.edu/pacsl.html},
    language = {english},
}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img17.jpg)

### LaTeX中的自定义命令和环境

Demo:

```latex
% 导言区
\documentclass{ctexart}

% \newcommand-定义命令
% 命令只能由字母组成，不能以\end 开头
% \newcommand<命令>[<参数个数>][<首参数默认值>]{<具体定义>}

% \newcommand可以是简单字符串替换，例如：
% 使用\PRC相当于 People's Republic of \emph{China} 这一串内容
\newcommand\PRC{People's Republic of \emph{China}}

% \newcommand也可以使用参数
% 参数个数可以从1到9，使用时间 #1,#2,......,#9 表示
% #1：代表第一个参数 #2：代表第二个参数 []里代表参数的个数
\newcommand\loves[2]{#1 喜欢 #2}
\newcommand\hatedby[2]{#2 不受 #1 喜欢}

% \newcommand的参数也可以有默认值
% 指定参数个数的同时指定了首个参数的默认值，那么这个命令的
% 第一个参数就称为可选的参数（要使用中括号指定）
\newcommand\love[3][喜欢]{#2#1#3}

% \newenvironment与\newcommand语法类似，感觉用不到，暂时不学了 

% 正文区
\begin{document}
    \PRC
    
    \loves{猫儿}{鱼}
    
    \hatedby{猫儿}{萝卜}
    
    % 默认参数可以省略
    \love{猫儿}{鱼}
    
    % 默认参数可以改变，但要使用中括号[]
    \love[最爱]{猫儿}{鱼}
\end{document}
```

结果如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/latex/img18.jpg)
