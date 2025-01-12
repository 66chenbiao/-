- [ 开源项目](#head1)
- [Python 项目](#head2)
- [ Pyqt5](#head3)
- [ 机器学习](#head4)

[目录生成网站](https://toc.codepie.fun/)

[shields.io](https://shields.io/category/activity)

[octicons](https://github.com/primer/octicons/)
[GitHub 项目徽章的添加和设置](https://lpd-ios.github.io/2017/05/03/GitHub-Badge-Introduction/)


## 开源项目
[Awesome Open Source](https://awesomeopensource.com/categories/machine-learning)：汇集了GitHub上比较有名的开源项目，是个推荐好开源的地方

## Python 项目
1、[python-cheatsheet](https://github.com/gto76/python-cheatsheet#list)：全面且实用的 Python 备忘录。这个东西特别适合我这个上了年纪的人，比如：忘记怎么用 Python 写正则、要搞个进度条忘记库的名字和基本用法、用 pandas 处理数据忘记方法需要的参数等等。正当我觉得自己需要“回炉重学”的时候发现了这个项目，有了它上面的问题都可以找到拿来即用的代码片段。我又是那个快乐的 Pythoneer 了，示例代码：

```python3
>>> from collections import Counter
>>> colors = ['blue', 'blue', 'blue', 'red', 'red']
>>> counter = Counter(colors)
>>> counter['yellow'] += 1
Counter({'blue': 3, 'red': 2, 'yellow': 1})
>>> counter.most_common()[0]
('blue', 3)
```

2、[geeksforgeeks](https://www.geeksforgeeks.org/enumerate-in-python/?ref=leftbar-rightbar)：关注于Python 字符串的一些操作

```python3
# Python program to illustrate 
# enumerate function 
l1 = ["eat","sleep","repeat"] 
s1 = "geek"
  
# creating enumerate objects 
obj1 = enumerate(l1) 
obj2 = enumerate(s1) 
  
print "Return type:",type(obj1) 
print list(enumerate(l1)) 
  
# changing start index to 2 from 0 
print list(enumerate(s1,2)) 

# result
Return type: < type 'enumerate' >
[(0, 'eat'), (1, 'sleep'), (2, 'repeat')]
[(2, 'g'), (3, 'e'), (4, 'e'), (5, 'k')]
```

3、Python源码：[codingdict](http://codingdict.com/sources/py/all)

4、[CPython-Internals](https://github.com/zpoint/CPython-Internals/blob/master/README_CN.md)
> 图文并茂的 Python 源码阅读笔记项目。阅读的是比较新的 CPython 3.8 版本，重点是项目一直在更新维护

5、[Pokemon-Terminal](https://github.com/LazoCoder/Pokemon-Terminal) 
> 适用于多种终端的口袋妖怪主题工具。支持 iTerm2、ConEmu、Terminology、Windows 的终端，已经收集了 719 个小精灵

![](https://github.com/66chenbiao/Network-resource-collection/blob/main/images/Pokemon-Terminal.gif)



## Pyqt5
[PyQt5-Chinese-tutorial](https://github.com/maicss/PyQt5-Chinese-tutorial)：PyQt5中文教程，翻译自 zetcode，项目地址：https://github.com/maicss/PyQt5-Chinese-tutoral

这个教程比较好的地方是，能讲解每一段代码的含义。

虽然PyQt的函数命名已经非常语义化了，但是对于新手来说，有这一步还是更好的。

所以我选择了翻译这篇教程，希望能给刚入门的你带来帮助。

[DearPyGui](https://github.com/hoffstadt/DearPyGui)：DearPyGui是一个易于使用（但功能强大）的Python GUI框架。 DearPyGui提供了Dear ImGui的包装，该包装模拟了传统的保留模式GUI（与Dear ImGui的立即模式范例相反）
![](https://github.com/hoffstadt/DearPyGui/raw/assets/linuxthemes.PNG?raw=true)
示例代码：
```
from dearpygui import core, simple

def save_callback(sender, data):
    print("Save Clicked")

with simple.window("Example Window"):
    core.add_text("Hello world")
    core.add_button("Save", callback=save_callback)
    core.add_input_text("string")
    core.add_slider_float("float")

core.start_dearpygui()
```
![](https://github.com/hoffstadt/DearPyGui/raw/assets/BasicUsageExample1.PNG?raw=true)

[PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/cookbook/)：从PySimpleGUI开始，你将对创建自定义GUI带来很大的飞跃。 复制并粘贴这些PySimpleGUI之一，并对其进行修改以符合您的要求。 研究它们以了解一些可以遵循的设计模式。示例代码：
```
import PySimpleGUI as sg

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Some text on Row 1')],
            [sg.Text('Enter something on Row 2'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')] ]

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    print('You entered ', values[0])

window.close()
```
![](https://user-images.githubusercontent.com/46163555/68713283-7cb38200-056b-11ea-990a-aa1603af5a11.png)


## 机器学习
1、[pumpkin-book](https://github.com/datawhalechina/pumpkin-book)

![](https://img.shields.io/github/forks/datawhalechina/pumpkin-book?style=social) ![](https://img.shields.io/github/watchers/datawhalechina/pumpkin-book?style=social ) ![](https://img.shields.io/github/stars/datawhalechina/pumpkin-book?color=green&style=social)

> 《机器学习公式详解》西瓜书公式推导解析。[在线阅读](https://datawhalechina.github.io/pumpkin-book/#/)

![](https://github.com/66chenbiao/Network-resource-collection/blob/main/images/nangua.jpg)


2、[examples](https://github.com/pytorch/examples)：关于视觉、本文等方面的 PyTorch 的示例集合。包含：使用 Convnets 的图像分类（MNIST）、生成对抗网络（DCGAN）等

3、[EasyOCR](https://github.com/JaidedAI/EasyOCR)：支持多种语言的即用型的 Python OCR 库，包括中文、日文、韩文等。示例代码：

```python3
import easyocr
reader = easyocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
result = reader.readtext('chinese.jpg')
# 输出
[([[189, 75], [469, 75], [469, 165], [189, 165]], '愚园路', 0.3754989504814148),
 ([[86, 80], [134, 80], [134, 128], [86, 128]], '西', 0.40452659130096436),
 ([[517, 81], [565, 81], [565, 123], [517, 123]], '东', 0.9989598989486694),
 ([[78, 126], [136, 126], [136, 156], [78, 156]], '315', 0.8125889301300049),
 ([[514, 126], [574, 126], [574, 156], [514, 156]], '309', 0.4971577227115631),
 ([[226, 170], [414, 170], [414, 220], [226, 220]], 'Yuyuan Rd.', 0.8261902332305908),
 ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),
 ([[529, 173], [569, 173], [569, 213], [529, 213]], 'E', 0.8405593633651733)]
 ```

4、[DeepLearningProject](https://github.com/Spandan-Madan/DeepLearningProject)：哈佛大学开源的深度学习教程





## C 项目
[ucore](https://github.com/kiukotsu/ucore)统课程，配套实验项目。推荐给有计算机结构原理、C 和汇编、数据结构基础并对操作系统感兴趣的同学，项目中包含教学视频、练习题、实验指导书等

## 其他
1、[CopyTranslator](https://github.com/CopyTranslator/CopyTranslator)：支持网页和 PDF 的划词翻译工具。有了它就可以解决阅读 PDF 文件时，要翻译大段内容情况下的乱码、换行、翻译不准的问题
![](https://camo.githubusercontent.com/fd39fcd1241c6e66a13c5f083bdc6bf4ce0386f264c331a598437a179acc2b69/68747470733a2f2f73312e617831782e636f6d2f323031382f31312f33302f466d724e46532e676966)


2、[styleguide](https://github.com/google/styleguide)：谷歌的代码风格指南。每个大型项目都有自己的代码风格，当代码的风格统一时将更容易被理解。本项目是谷歌项目的代码风格说明，包含：C++、C#、Swift、Python、Java 等语言

3、[PathPlanning](https://github.com/zhm-real/PathPlanning)：常见的路径规划算法集合。项目包含了 Python 代码实现、运行过程动画以及相关论文
![PathPlanning](https://user-images.githubusercontent.com/44183747/150706233-de3aa198-efed-45c6-b269-def832041b0e.gif)



